from .modules import (
    SwooshR, SwooshL, 
    BiasNorm, FeedForwardBlock, NonLinearAttention,
    # DownsampleLayer, UpsampleLayer, BypassModule
)
from .atten import (
    MultiHeadAttentionWeight,
)
from utils import calculate_mask
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmbeded(nn.Module):
    def __init__(self, input_dim, output_dim, conv_dim):
        super(ConvEmbeded, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= 1, 
                      out_channels=conv_dim[0], 
                      kernel_size= (3, 3), 
                      stride= (1, 2), 
                      padding= (0, 1),
                      ),
            SwooshR(),
            nn.Conv2d(in_channels= conv_dim[0], 
                      out_channels=conv_dim[1], 
                      kernel_size= (3, 3), 
                      stride= (2, 2), 
                      padding= 0,
                      ),
            SwooshR(),
            nn.Conv2d(in_channels= conv_dim[1], 
                      out_channels=conv_dim[2], 
                      kernel_size= (3, 3), 
                      stride= (1, 2), 
                      padding= 0,
                      ),
            SwooshR(),
        )

        self.convnext = nn.Sequential(
            nn.Conv2d(in_channels= conv_dim[2], 
                      out_channels=conv_dim[2],
                      groups= conv_dim[2], 
                      kernel_size= (7, 7), 
                      stride= (1, 1), 
                      padding= (3, 3),
                      ),
            nn.Conv2d(in_channels= conv_dim[2], 
                      out_channels= conv_dim[3], 
                      kernel_size= (1, 1), 
                      stride= (1, 1), 
                      padding= 0,
                      ),
            SwooshL(),
            nn.Conv2d(in_channels= conv_dim[3], 
                      out_channels=conv_dim[2], 
                      kernel_size= (1, 1), 
                      stride= (1, 1), 
                      padding= 0,
                      ),
        )
        
        with torch.no_grad():
            temp = torch.zeros(1, 1, 100, input_dim) # (B, 1, T, D)
            temp = self.conv(temp) # (B, C, T', D')
            _, C, T, D = temp.shape
            temp = self.convnext(temp) # (B, C, T', D')
            _, C, T, D = temp.shape
            self.temp = C * D
        
        self.linear = nn.Linear(self.temp, output_dim)
        self.bias_norm = BiasNorm(output_dim)

    def calculate_output_length(self, input_length):
        """Tính độ dài output sau khi đi qua các conv layers"""
        # Conv1: stride=(1,2), padding=(0,1), kernel=(3,3) -> T unchanged
        length = input_length
        
        # Conv2: stride=(2,2), padding=0, kernel=(3,3) -> T = (T-3+0)//2 + 1
        length = (length - 3) // 2 + 1
        
        # Conv3: stride=(1,2), padding=0, kernel=(3,3) -> T = (T-3+0)//1 + 1 = T-2
        length = length - 2
        
        return length

    def forward(self, x):
        """
        Convolution Embeded:
        args:
            x: (B, T, D)
        return:
            output: (B, T', D')
        """
        x = x.unsqueeze(1)  # (B, 1, T, D)
        x = self.conv(x)    # (B, C, T', D')
        resi = x
        x = self.convnext(x) # (B, C, T', D')
        x = x + resi
        B, C, T, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * D) # (B, T', C*D)
        x = self.linear(x)  # (B, T', D')
        x = self.bias_norm(x)

        return x

class ZipformerBlock(nn.Module):
    def __init__(self, input_dim, ff_size, h, value_head_dim, p_dropout):
        super(ZipformerBlock, self).__init__()
        self.ffn = FeedForwardBlock(d_model=input_dim, d_ff=ff_size, dropout=p_dropout)
        self.mhaw = MultiHeadAttentionWeight(d_model=input_dim, h=h, dropout=p_dropout)
        self.nla = NonLinearAttention(d_in=input_dim)
        self.sat = SelfAttention(d_in=input_dim, heads=h, d_h=value_head_dim)

    def forward(self, x, x_mask):
        """
        Zipformer Block:
        args:
            x: (T, B, D)
            x_mask: (B, 1, T)
        """
        atten_weight = self.mhaw(x, x, x_mask)

        x = x + self.ffn(x)
        x = x + self.nla(x, atten_weight)
        x = x + self.sat(x, atten_weight)

        return x, atten_weight

class ZipformerEncoder(nn.Module):
    def __init__(self, config, vocab_size):
        super(ZipformerEncoder, self).__init__()
        self.input_dim = config['conv_embeded']['input_dim']
        self.output_dim = config['conv_embeded']['output_dim']
        self.conv_dim = config['conv_embeded']['conv_dim']
        self.ff_size = config['enc']['ff_size']
        self.h = config['enc']['h']
        self.p_dropout = config['enc']['dropout']
        self.value_head_dim = config['enc']['value_head_dim']

        self.conv_embeded = ConvEmbeded(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            conv_dim=self.conv_dim,
        )

        self.zipblock = ZipformerBlock(
            input_dim= 192,
            ff_size= self.ff_size,
            h= self.h,
            value_head_dim= self.value_head_dim,
            p_dropout= self.p_dropout
        )
    def forward(self, x, fbank_len):
        """
        Zipformer Encoder:
        args:
            x: (B, T, D)
            x_mask: (B, 1, T)
        return:
            output: (B, T', D)
            output_mask: (B, 1, T')
        """
        x = self.conv_embeded(x)  # (B, T', D')
        B, T, D = x.shape
        x_len = torch.tensor([self.conv_embeded.calculate_output_length(length.item()) for length in fbank_len])
        x_mask = calculate_mask(x_len, T)  # (B, T')

        x, atten_w = self.zipblock(x, x_mask)
        return x, x_mask, atten_w