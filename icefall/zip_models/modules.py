import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Type, List
import torch.nn.init as init


def calc_data_len(
    result_len: int,
    pad_len,
    data_len,
    kernel_size: int,
    stride: int,
):
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len

def get_mask_from_lens(lengths, max_len: int):
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        residual: bool = False,
        conv_module: Type[nn.Module] = nn.Conv2d,
        activation: Callable = nn.LeakyReLU,  # 👉 Dùng LeakyReLU
        norm: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            conv_stride = stride if i == num_layers - 1 else 1
            conv = conv_module(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=conv_stride,
                dilation=dilation,
                padding=(kernel_size // 2)
            )
            layers.append(conv)
            if norm:
                layers.append(norm(out_channels))  # Gọi instance
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)
        self.residual = residual

        if residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm(out_channels) if norm else nn.Identity(),
                nn.Dropout(dropout),
            )
        elif residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x, mask):
        B, C, T, F = x.shape
        residual_input = x  

        for layer in self.main:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                s = layer.stride[0]
                d = layer.dilation[0]
                p = layer.padding[0]
                out_T = (T + 2 * p - d * (k - 1) - 1) // s + 1
                pad_len = T - mask.sum(dim=1)
                data_len = mask.sum(dim=1)
                new_len = calc_data_len(
                    result_len=out_T,
                    pad_len=pad_len,
                    data_len=data_len,
                    kernel_size=k,
                    stride=s,
                )
                mask = get_mask_from_lens(new_len, out_T)
                T = out_T

        if self.residual:
            shortcut = self.shortcut(residual_input)  # 👉 fix chỗ này
            x = x + shortcut

        return x, mask


class ConvolutionFrontEnd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blocks: int,
        num_layers_per_block: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        residuals: List[bool],
        activation: Callable = nn.LeakyReLU, 
        norm: Optional[Callable] = nn.BatchNorm2d, 
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []

        for i in range(num_blocks):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels[i],
                num_layers=num_layers_per_block,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                residual=residuals[i],
                activation=activation,
                norm=norm,
                dropout=dropout
            )
            blocks.append(block)
            in_channels = out_channels[i]

        self.model = nn.ModuleList(blocks)

    def forward(self, x, mask):
        for i, block in enumerate(self.model):
            x, mask = block(x, mask)
        return x, mask



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


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            return self.norm(x + self.dropout(sublayer(x)))

class ResidualForTASA(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))

class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # batch ,seqlen, d_model -> batch, seqlen, vocab_size
        return torch.log_softmax(self.proj(x), dim = -1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
    def get_pe(self, seq_len: int) -> torch.Tensor:
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, self.d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        # x is of shape (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.get_pe(seq_len).to(x.device)

        x = x + pe
        return x

class Conv2dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor):
        x = x.unsqueeze(1)  # (batch, 1, time, dim)
        B, C, T, F = x.shape
        for layer in self.sequential:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                s = layer.stride[0]
                d = layer.dilation[0]
                p = layer.padding[0]
                out_T = (T + 2 * p - d * (k - 1) - 1) // s + 1
                pad_len = T - input_lengths
                data_len = input_lengths
                new_len = calc_data_len(
                    result_len=out_T,
                    pad_len=pad_len,
                    data_len=data_len,
                    kernel_size=k,
                    stride=s,
                )
                T = out_T
        B, C, T, F = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, C * F)
        return x, new_len
    
class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
