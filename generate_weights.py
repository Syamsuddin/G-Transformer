from src.model import GTransformerConfig, GTransformerForCausalLM
from safetensors.torch import save_file
import torch

# Inisialisasi konfigurasi
config = GTransformerConfig(
    vocab_size=65536,
    hidden_size=1024,            # Ubah ke 8192 untuk versi penuh
    intermediate_size=4096,
    num_hidden_layers=12,        # Ubah ke 48 untuk versi penuh
    num_attention_heads=16,      # Ubah ke 64 untuk versi penuh
    use_low_rank_ffn=True,
    use_entropy_gate=True,
    use_flash_attention=True,
    informational_constant_kI=2.612e-20,
)

# Buat model kosong
model = GTransformerForCausalLM(config)

# Inisialisasi bobot dengan skema default (Xavier normal)
for name, param in model.named_parameters():
    if param.dim() > 1:
        torch.nn.init.xavier_normal_(param)
    else:
        torch.nn.init.zeros_(param)

# Simpan ke format .safetensors
state_dict = model.state_dict()
save_file(state_dict, "pytorch_model.safetensors")

print("✅ File bobot berhasil dibuat: pytorch_model.safetensors")
print(f"Total parameter: {sum(p.numel() for p in model.parameters())/1e6:.2f} juta")
