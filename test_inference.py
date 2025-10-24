import torch
from src.model import GTransformerConfig, GTransformerForCausalLM
from safetensors.torch import load_file

# Load konfigurasi dan model
config = GTransformerConfig()
model = GTransformerForCausalLM(config)
model.load_state_dict(load_file("pytorch_model.safetensors"))
model.eval()

# Token input contoh
input_ids = torch.tensor([[1, 11, 12, 2]])  # <s> information energy </s>

# Inferensi sederhana
with torch.no_grad():
    outputs = model(input_ids)
    print("✅ Output logits shape:", outputs.logits.shape)
    print("Token terakhir prediksi:", outputs.logits[0, -1, :5])
