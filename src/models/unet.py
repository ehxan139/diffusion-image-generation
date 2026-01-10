"""
U-Net architecture for DDPM denoising.

Simplified, production-ready implementation.
"""

import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings.
        
        Args:
            t: Timesteps [B]
            
        Returns:
            Embeddings [B, dim]
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """
    U-Net for DDPM denoising.
    
    Simplified architecture suitable for 64x64 images.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 256
    ):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_channels, time_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_channels * 4, base_channels * 2, time_dim)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_channels * 2, base_channels, time_dim)
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy images [B, C, H, W]
            t: Timesteps [B]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        
        # Bottleneck
        b = self.bottleneck(e3, t_emb)
        
        # Decoder with skip connections
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1), t_emb)
        
        return self.out(d2)
