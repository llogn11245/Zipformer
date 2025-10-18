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

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))

class DownsampleLayer(nn.Module):
    def __init__(self, k_factor: int):
        super().__init__()
        self.k_factor = k_factor
        self.weights = nn.Parameter(torch.zeros(k_factor))
        
    def forward(self, x):
        """
        x: (B, T, D)
        Output: (B, ceil(T / k_factor), D)
        """
        B, T, D = x.shape
        k = self.k_factor
        
        # Nếu T không chia hết cho k -> pad thêm các frame cuối (bằng frame cuối cùng)
        pad_len = (k - (T % k)) % k
        if pad_len > 0:
            pad = x[:, -1:, :].expand(B, pad_len, D)
            x = torch.cat([x, pad], dim=1)
        
        # Chia thành các nhóm độ dài k
        seq_len = x.shape[1] // k
        x = x.view(B, seq_len, k, D)  # (B, n_group, k, D)
        
        w = self.weights.softmax(dim=0).view(1, 1, k, 1)
        
        # Gộp theo trọng số -> mỗi nhóm thành 1 frame
        y = (x * w).sum(dim=2)  # (B, n_group, D)
        
        return y

class UpsampleLayer(torch.nn.Module):
    def __init__(self, upsample: int):
        super().__init__()
        self.upsample = upsample

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T * upsample, D)
        """
        B, T, D = x.shape
        u = self.upsample

        # (B, T, D) -> (B, T, u, D)
        x = x.unsqueeze(2).expand(B, T, u, D)
        x = x.reshape(B, T * u, D)

        return x

class NonLinearAttention(nn.Module):
    def __init__(self, d_in):
        super(NonLinearAttention, self).__init__()
        self.d_out = d_in * 3 // 4
        self.linear = nn.ModuleList([
            nn.Linear(d_in, self.d_out * 3),
            nn.Linear(self.d_out, d_in),
        ])
        self.tanh = nn.Tanh()
        

    def forward(self, x, atten_weight):
        """
        Non-Linear Attention:
        args:
            x: (B, T, D)
            attention_weight: (B, h, T, T)
        return:
            output: (B, T, D)
        """
        # Reshape atten weight to (h, B, T, T)
        atten_weight = atten_weight.permute(1, 0, 2, 3)  # (h, B, T, T)
        x = self.linear[0](x)

        # Chia input thành 3 lớp linear vói d_out = 3/4 d_in
        s, x, y = x.chunk(3, dim=-1)

        # s sẽ đi qua tanh, x truyền thẳng, y là residual 
        s = self.tanh(s)

        x = x * s

        num_heads = atten_weight.shape[0]
        B, T, D = x.shape
        x = x.reshape(B, T, num_heads, -1).permute(2, 0, 1, 3) # (h, B, T, D_head)
        
        x = torch.matmul(atten_weight, x)  # (h, B, T, D_head)
        x = x.permute(1, 2, 0, 3).reshape(B, T, -1)  # (B, T, D)

        x = x * y
        x = self.linear[1](x)

        return x
        
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
# class BypassModule(nn.Module):
#     """
#     An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
#     layer-skipping.  The bypass is limited during early stages of training to be close to
#     "straight-through", i.e. to not do the bypass operation much initially, in order to
#     force all the modules to learn something.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         skip_rate: FloatLike = 0.0,
#         straight_through_rate: FloatLike = 0.0,
#         scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
#         scale_max: FloatLike = 1.0,
#     ):
#         super().__init__()
#         self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
#         self.skip_rate = copy.deepcopy(skip_rate)
#         self.straight_through_rate = copy.deepcopy(straight_through_rate)
#         self.scale_min = copy.deepcopy(scale_min)
#         self.scale_max = copy.deepcopy(scale_max)

#     def _get_bypass_scale(self, batch_size: int):
#         # returns bypass-scale of shape (num_channels,),
#         # or (batch_size, num_channels,).  This is actually the
#         # scale on the non-residual term, so 0 corresponds to bypassing
#         # this module.
#         if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
#             return self.bypass_scale
#         else:
#             ans = limit_param_value(
#                 self.bypass_scale, min=float(self.scale_min), max=float(self.scale_max)
#             )
#             skip_rate = float(self.skip_rate)
#             if skip_rate != 0.0:
#                 mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
#                 ans = ans * mask
#                 # now ans is of shape (batch_size, num_channels), and is zero for sequences
#                 # on which we have randomly chosen to do layer-skipping.
#             straight_through_rate = float(self.straight_through_rate)
#             if straight_through_rate != 0.0:
#                 mask = (
#                     torch.rand((batch_size, 1), device=ans.device)
#                     < straight_through_rate
#                 )
#                 ans = torch.maximum(ans, mask.to(ans.dtype))
#             return ans

#     def forward(self, src_orig: Tensor, src: Tensor):
#         """
#         Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
#         Returns: something with the same shape as src and src_orig
#         """
#         bypass_scale = self._get_bypass_scale(src.shape[1])
#         return src_orig + (src - src_orig) * bypass_scale
