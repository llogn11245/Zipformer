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

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ConvolutionalModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super(ConvolutionalModule, self).__init__()
        self.layer_norm = LayerNormalization(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, dim)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, dim, time)
        x = self.pointwise_conv1(x)  # (batch, 2*dim, time)
        x = self.glu(x)  # (batch, dim, time)
        x = self.depthwise_conv(x)  # (batch, dim, time)
        x = self.batch_norm(x)  # (batch, dim, time)
        x = self.swish(x)  # (batch, dim, time)
        x = self.pointwise_conv2(x)  # (batch, dim, time)
        x = self.dropout(x)  # (batch, dim, time)
        return x.transpose(1, 2)  # (batch, time, dim)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)
        

class BypassModule(nn.Module):
    def __init__(self, input_dim, initial_min=0.9, initial_max=1.0, 
                 final_min=0.2, change_step=20000):
        super().__init__()
        self.channels = input_dim
        self.initial_min = initial_min
        self.initial_max = initial_max
        self.final_min = final_min
        self.change_step = change_step
        
        self.c = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, x, y, current_step=None):
        # Nếu đang ở chế độ eval -> luôn dùng final_min
        if not self.training:
            min_val = self.final_min
        else:
            # Nếu training, điều chỉnh theo current_step
            if current_step is None:
                raise ValueError("current_step phải được truyền vào khi training.")
            min_val = self.initial_min if current_step < self.change_step else self.final_min
        
        c_clamped = torch.clamp(self.c, min_val, 1.0)
        c_clamped = c_clamped.view(1, 1, -1)  # (B, T, D)
        
        output = (1 - c_clamped) * x + c_clamped * y
        
        return output