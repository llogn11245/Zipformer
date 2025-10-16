from .modules import SwooshR, SwooshL, BiasNorm
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
    def __init__(self, dim, num_heads, ffn_dim, dropout, conv_kernel_size, reduction_factor):
        super(ZipformerBlock, self).__init__()
        self.norm1 = BiasNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = BiasNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            SwooshR(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = BiasNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim * 2, kernel_size=1, stride=1, padding=0),
            SwooshR(),
            nn.Conv1d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=conv_kernel_size, stride=1, padding=(conv_kernel_size - 1) // 2, groups=dim * 2),
            SwooshR(),
            nn.Conv1d(in_channels=dim * 2, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.Dropout(dropout),
        )
        self.reduction_factor = reduction_factor

    def forward(self, x, x_mask):
        """
        Zipformer Block:
        args:
            x: (T, B, D)
            x_mask: (B, 1, T)
        return:
            output: (T', B, D)
            output_mask: (B, 1, T')
        """
        # Multi-Head Attention
        resi = x
        x = self.norm1(x)
        x2 = x.permute(1, 0, 2)  # (B, T, D)
        x2 = x2.masked_fill(x_mask.transpose(1, 2) == 0, float('-inf'))
        attn_output, _ = self.attn(x2.transpose(0, 1), x2.transpose(0, 1), x2.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)  # (T, B, D)
        x = resi + attn

class ZipformerEncoder(nn.Module):
    def __init__(self, config, vocab_size):
        super(ZipformerEncoder, self).__init__()
        self.input_dim = config['conv_embeded']['input_dim']
        self.output_dim = config['conv_embeded']['output_dim']
        self.conv_dim = config['conv_embeded']['conv_dim']

        self.conv_embeded = ConvEmbeded(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            conv_dim=self.conv_dim,
        )

    def forward(self, x, fbank_len, x_mask):
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

        return x, x_mask