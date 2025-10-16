import torch 
import torch.nn as nn   
import torch.nn.functional as F
import math

class SwooshR(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = 1.0
        self.alpha = 0.08
        self.bias = 0.313261687

    def forward(self, x):
        return torch.log1p(torch.exp(x - self.offset)) - self.alpha * x - self.bias


class SwooshL(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = 4.0
        self.alpha = 0.08
        self.bias = 0.035

    def forward(self, x):
        return torch.log1p(torch.exp(x - self.offset)) - self.alpha * x - self.bias

class BiasNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        # Learnable bias (b): shape (1, 1, D)
        self.bias = nn.Parameter(torch.zeros(1, 1, num_channels))
        # Learnable log-scale (γ): scalar
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, T, D)
        rms = torch.sqrt(torch.mean((x - self.bias) ** 2, dim=-1, keepdim=True) + self.eps)

        x = x / rms * torch.exp(self.scale)
        return x