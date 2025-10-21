import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
from typing import Optional
import torch

class SelfAttention(nn.Module):
    def __init__(
        self, d_in: int, heads: int, d_h) -> None:
        super().__init__()
        self.linear = nn.ModuleList([
            nn.Linear(d_in, heads * d_h, bias=True),
            nn.Linear(heads * d_h, d_in, bias=True),
        ])

    def forward(self, x, atten_weight):
        """
        Args:
          x: (B, T, D) 
         attn_weights: (B, h, T, T)
        Returns:
           a tensor with the same shape as x.
        """
        B, T, D = x.shape

        atten_weight = atten_weight.permute(1, 0, 2, 3)  # (h, B, T, T)
        num_heads = atten_weight.shape[0]

        x = self.linear[0](x)  # (B, T, h * d_h)
        x = x.reshape(B, T, num_heads, -1).permute(2, 0, 1, 3) # (h, B, T, d_in)

        value_head_dim = x.shape[-1]

        x = torch.matmul(atten_weight, x)

        x = (
            x.permute(1, 2, 0, 3)
            .contiguous()
            .view(x.shape[1], x.shape[2], num_heads * value_head_dim)
        )

        x = self.linear[1](x)  # (B, T, D)

        return x

class MultiHeadAttentionWeight(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask= None):
        query = self.w_q(q) 
        key = self.w_k(k) 

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, h, T, T]
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = scores.softmax(dim=-1)  # [B, h, T, T]
        attn_weights = self.dropout(attn_weights)

        return attn_weights