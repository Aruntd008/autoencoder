import torch
import torch.nn as nn
import torch.nn.functional as F

        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downscale=None, activation=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downscale = downscale
        self.activation = activation if activation is not None else nn.SiLU()
        
        self.blocks = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True),
            self.activation,
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
            self.activation,
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
                
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
            
    def forward(self, x):
        return self.residual(x) + self.blocks(x)
        

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=0):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), 'constant', value=0)
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0):
        super(UpsampleBlock, self).__init__()

        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear")
        return self.conv(x)
    
class NonLocalBlock(nn.Module):
    
    """
    This technique was introduced in the paper "Attention is All You Need" 
    The need for scaling in attention mechanisms:
    
    We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax 
    function into regions where it has extremely small gradients (to the point that the updates are negligible). 
    To counteract this effect, we scale the dot products by 1/sqrt(d_k).
    """
    def __init__(self, in_channels) -> None:
        super(NonLocalBlock, self).__init__()
        self.channels = in_channels
        
        self.gn = nn.GroupNorm(num_groups=32, num_channels=self.channels, eps=1e-6, affine=True)
        
        self.theta = nn.Conv2d(self.channels, self.channels, 1)
        self.phi = nn.Conv2d(self.channels, self.channels, 1)
        self.g = nn.Conv2d(self.channels, self.channels, 1)
        self.output_conv = nn.Conv2d(self.channels, self.channels, 1)
        
        
    def forward(self, x):
        
        batch_size, C, H, W = x.shape
        g_norm = self.g(x)
        theta = self.theta(g_norm).reshape(batch_size, C, H*W).permute(0,2,1) # Shape: (B, HW, C)
        phi = self.phi(g_norm).reshape(batch_size, C, H*W)                     # Shape: (B, C, HW)
        g = self.g(g_norm).reshape(batch_size, C, H*W)                         # Shape: (B, C, HW)
        
        scores = torch.matmul(theta, phi)                                      # Shape: (B, HW, HW)
        scores = scores * (int(C)**(-0.5))
        wi = F.softmax(scores, dim=2)
        
        z = torch.matmul(g, wi.permute(0, 2, 1)).reshape(batch_size, C, H, W)  # Shape: (B, C, HW) --> Shape: (B, C, H, W)
        return x + self.output_conv(z)