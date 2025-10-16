from .modules import SwooshR, SwooshL, BiasNorm
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmbeded(nn.Module):
    def __init__(self, conv_dim):
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
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * D)

        return x