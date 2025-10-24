# Copyright (c) 2025
# G-Transformer: Energy-Efficient Transformer based on GIT
# Author: Syamsuddin B. Ideris, S.Pd.MM

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception as e:
    raise ImportError(
        "Harap instal transformers >= 4.40.0. "
        "pip install transformers"
    ) from e


# ----------------------------
# Konfigurasi
# ----------------------------
class GTransformerConfig(PretrainedConfig):
    model_type = "gtransformer"

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 8192,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        max_position_embeddings: int = 65536,
        hidden_act: str = "swiglu",
        layer_norm_epsilon: float = 1e-5,
        attention_dropout: float = 0.05,
        hidden_dropout_prob: float = 0.05,
        rotary_emb_base: int = 10000,
        use_flash_attention: bool = True,
        use_low_rank_ffn: bool = True,
        use_entropy_gate: bool = True,
        use_moe: bool = False,
        num_experts: int = 0,
        top_k_experts: int = 0,
        fp8_precision: bool = False,
        dvfs_enabled: bool = False,
        informational_constant_kI: float = 2.612e-20,
        energy_per_token_target_J: float = 0.07,
        delta_I_gate: float = 0.75,
        local_window: int = 512,
        global_rank: int = 64,
        kv_compression_rank: int = 64,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.rotary_emb_base = rotary_emb_base

        self.use_flash_attention = use_flash_attention
        self.use_low_rank_ffn = use_low_rank_ffn
        self.use_entropy_gate = use_entropy_gate

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts

        self.fp8_precision = fp8_precision
        self.dvfs_enabled = dvfs_enabled

        self.informational_constant_kI = informational_constant_kI
        self.energy_per_token_target_J = energy_per_token_target_J

        self.delta_I_gate = delta_I_gate
        self.local_window = local_window
        self.global_rank = global_rank
        self.kv_compression_rank = kv_compression_rank

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


# ----------------------------
# Utilitas
# ----------------------------
def swiglu(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1) * x2


def build_activation(name: str):
    if name.lower() == "swiglu":
        return swiglu
    return getattr(F, name)


# Rotary posisi sederhana
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: [B, H, T, D]
    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x_rot
    q_rot = (q * cos) + (rotate(q) * sin)
    k_rot = (k * cos) + (rotate(k) * sin)
    return q_rot, k_rot


# ----------------------------
# IA-Attention
# ----------------------------
class InformationalAttention(nn.Module):
    """
    Atensi hemat energi.
    1. Atensi lokal dengan jendela w.
    2. Seleksi token global berbasis skor informasi.
    3. Proyeksi low-rank untuk jalur global.
    """

    def __init__(self, config: GTransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0

        self.w_qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim)

        # Proyeksi low rank global
        self.rank = config.global_rank
        self.Pk = nn.Linear(self.head_dim, self.rank, bias=False)
        self.Pv = nn.Linear(self.head_dim, self.rank, bias=False)
        self.Uo = nn.Linear(self.rank, self.head_dim, bias=False)

        # Skorer informasi
        self.info_scorer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4, bias=False),
            nn.GELU(),
            nn.Linear(self.d_model // 4, 1, bias=False),
        )

        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

        self.local_window = config.local_window
        self.delta_I_gate = config.delta_I_gate
        self.use_entropy_gate = config.use_entropy_gate

    def _causal_local_mask(self, T: int, w: int, device) -> torch.Tensor:
        idxs = torch.arange(T, device=device)
        mask = idxs[None, :] - idxs[:, None]
        # izinkan hanya masa lalu dalam jendela lokal
        mask = (mask > 0) | (mask < -(w - 1))
        return mask  # True berarti masked

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        qkv = self.w_qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        cos, sin = self.rotary(q, T)
        q, k = apply_rotary(q, k, cos, sin)

        # Tambah cache jika ada
        if past_key_value is not None:
            pk, pv = past_key_value  # [B, H, T_past, D]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            T_total = k.size(2)
        else:
            T_total = T

        # Atensi lokal
        w = min(self.local_window, T_total)
        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.einsum("bhtd,bhSd->bhtS", q, k) * scale  # S = T_total
        # Mask kausal lokal
        local_mask = self._causal_local_mask(T_total, w, x.device)  # [T_total, T_total]
        local_mask = local_mask[-T:]  # baris untuk query saat ini
        attn_scores = attn_scores.masked_fill(local_mask[None, None, :, :], float("-inf"))
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask  # bentuk harus broadcastable

        attn_w_local = F.softmax(attn_scores, dim=-1)
        attn_w_local = self.attn_drop(attn_w_local)
        ctx_local = torch.einsum("bhtS,bhSd->bhtd", attn_w_local, v)

        # Seleksi global berbasis informasi
        # Skor informasi dari representasi x
        with torch.no_grad():
            info_score = self.info_scorer(x).squeeze(-1)  # [B, T]
            # skala ke 0..1 via sigmoid
            info_score = torch.sigmoid(info_score)
        if self.use_entropy_gate:
            gate = (info_score > self.delta_I_gate).float()  # [B, T]
        else:
            gate = torch.ones_like(info_score)

        # Proyeksi low rank untuk jalur global hanya pada token bergated
        # Bentuk sederhana: kompres k,v ke rank kecil lalu atensi penuh pada subset
        # Buat mask indeks global per batch
        ctx_global = torch.zeros_like(ctx_local)
        if gate.sum() > 0:
            # kompres k,v
            k_r = self.Pk(k)  # [B,H,T_total,R]
            v_r = self.Pv(v)  # [B,H,T_total,R]
            q_r = self.Pk(q)  # reuse Pk untuk q

            # gunakan atensi penuh pada subset dengan gate
            # bentuk sederhana, gunakan semua posisi, tapi bobot query di-skala gate query
            gate_q = gate[:, -T:].unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
            attn_scores_g = torch.einsum("bhtr,bhsr->bhts", q_r, k_r) * (scale * D / self.rank)
            attn_w_g = F.softmax(attn_scores_g, dim=-1)
            attn_w_g = self.attn_drop(attn_w_g)
            ctx_g_r = torch.einsum("bhts,bhsr->bhtr", attn_w_g, v_r)
            ctx_g = self.Uo(ctx_g_r)  # [B,H,T,D]
            ctx_global = ctx_g * gate_q

        ctx = ctx_local + ctx_global
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        out = self.w_o(ctx)
        out = self.proj_drop(out)

        present = (k, v) if use_cache else None
        return out, present


# ----------------------------
# Low-Rank FFN
# ----------------------------
class LowRankFFN(nn.Module):
    def __init__(self, config: GTransformerConfig):
        super().__init__()
        d = config.hidden_size
        i = config.intermediate_size
        act = build_activation(config.hidden_act)
        self.act = act
        # Faktorisasi: d -> i -> d, dengan bottleneck rank r_ffn
        r_ffn = max(128, i // 8)
        self.w1a = nn.Linear(d, r_ffn, bias=False)
        self.w1b = nn.Linear(d, r_ffn, bias=False)
        self.w2 = nn.Linear(r_ffn, d, bias=False)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SWiGLU low-rank
        u = self.w1a(x)
        v = self.w1b(x)
        h = swiglu(torch.cat([u, v], dim=-1))
        out = self.w2(h)
        return self.drop(out)


# ----------------------------
# MoE Router opsional
# ----------------------------
class EntropyMoE(nn.Module):
    def __init__(self, config: GTransformerConfig):
        super().__init__()
        assert config.num_experts > 0
        self.num_experts = config.num_experts
        self.top_k = max(1, config.top_k_experts)
        d = config.hidden_size
        i = config.intermediate_size

        self.router = nn.Sequential(
            nn.Linear(d, d // 2, bias=False),
            nn.GELU(),
            nn.Linear(d // 2, self.num_experts, bias=False),
        )
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, i), nn.GELU(), nn.Linear(i, d)) for _ in range(self.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        logits = self.router(x)  # [B,T,E]
        probs = F.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=self.top_k, dim=-1)
        idx = topk.indices  # [B,T,K]
        wgt = topk.values   # [B,T,K]

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            sel = idx[..., k]  # [B,T]
            # kumpulkan untuk tiap expert
            for e in range(self.num_experts):
                mask = (sel == e).float().unsqueeze(-1)  # [B,T,1]
                if mask.sum() == 0:
                    continue
                xe = x * mask
                ye = self.experts[e](xe)
                out = out + ye * (wgt[..., k].unsqueeze(-1))
        return out


# ----------------------------
# Blok Transformer
# ----------------------------
class GTransformerBlock(nn.Module):
    def __init__(self, config: GTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = InformationalAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        if config.use_moe and config.num_experts > 0:
            self.ff = EntropyMoE(config)
        else:
            self.ff = LowRankFFN(config) if config.use_low_rank_ffn else nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, present = self.attn(self.ln1(x), attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x, present


# ----------------------------
# Model dasar
# ----------------------------
class GTransformerModel(PreTrainedModel):
    config_class = GTransformerConfig

    def __init__(self, config: GTransformerConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GTransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:

        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        new_past = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]
            x, present = layer(x, attention_mask=attention_mask, past_key_value=pkv, use_cache=use_cache)
            if use_cache:
                new_past.append(present)

        x = self.ln_f(x)
        return x, new_past


# ----------------------------
# Causal LM
# ----------------------------
class GTransformerForCausalLM(PreTrainedModel):
    config_class = GTransformerConfig

    def __init__(self, config: GTransformerConfig):
        super().__init__(config)
        self.transformer = GTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embed_tokens = new_embeddings

    def tie_weights(self):
        # opsional tidak diikat agar stabil FP8
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        hidden_states, new_past = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Regularisasi informasi sederhana
            if self.config.use_entropy_gate:
                with torch.no_grad():
                    probs = F.softmax(shift_logits, dim=-1)
                    logp = torch.log(probs + 1e-9)
                    H = -(probs * logp).sum(dim=-1).mean()
                # target penurunan entropi moderat
                loss = loss + 1e-4 * H

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def generate_simple(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        self.eval()
        past = None
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out[:, -1:].contiguous(), use_cache=True, past_key_values=past).logits
            past = self(out[:, -1:].contiguous(), use_cache=True, past_key_values=past).past_key_values
            next_token = torch.distributions.Categorical(logits=logits[:, -1, :] / max(1e-6, temperature)).sample()
            out = torch.cat([out, next_token.unsqueeze(-1)], dim=1)
            if int(next_token[0].item()) == self.config.eos_token_id:
                break
        return out
