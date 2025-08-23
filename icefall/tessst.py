import torch
from zip_models.model import Zipformer

# ====== Fake config để khởi tạo model ======
config = {
    "conv_embeded": {
        "in_channels": 80,
        "out_channels": 192,
        "layer1_channels": 8,
        "layer2_channels": 32,
        "layer3_channels": 128,
        "dropout": 0.1,
    },
    "decoder": {
        "n_layers": 4,
        "d_model": 256,
        "ff_size": 512,
        "h": 4,
        "p_dropout": 0.1,
    },
    "vocab_size": 50,  # ví dụ giả định vocab = 50 token
}

# ====== Khởi tạo model ======
model = Zipformer(config)

# ====== Tạo dữ liệu test ======
batch_size = 3
max_len = 500
feat_dim = 80
inp = torch.randn(batch_size, max_len, feat_dim)         # (B, T, F)
input_lengths = torch.tensor([500, 300, 330])            # độ dài thật

# Fake decoder input (giả sử độ dài target 40, token int từ vocab)
dec_input = torch.randint(0, config["vocab_size"], (batch_size, 17))

# ====== Chạy model ======
encoder_out, decoder_out, encoder_out_lens = model(inp, input_lengths, dec_input)

print("Encoder output shape:", encoder_out.shape)      # (B, T', D)
print("Encoder output lengths:", encoder_out_lens)     # (B,)
print("Decoder output shape:", decoder_out.shape)      # (B, target_len, vocab_size)
