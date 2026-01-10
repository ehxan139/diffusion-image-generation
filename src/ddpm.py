"""
Denoising Diffusion Probabilistic Model (DDPM)

Production-ready implementation for image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class DDPM:
    """
    Denoising Diffusion Probabilistic Model for image generation.
    
    Implements the forward diffusion process (adding noise) and reverse process
    (denoising with a trained model) as described in "Denoising Diffusion 
    Probabilistic Models" (Ho et al., 2020).
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize DDPM.
        
        Args:
            model: Denoising model (e.g., U-Net)
            timesteps: Number of diffusion steps
            beta_start: Starting variance schedule value
            beta_end: Ending variance schedule value
            device: Device to run on
        """
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Variance schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute values for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to clean images.
        
        Args:
            x0: Clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-generated noise
            
        Returns:
            Tuple of (noisy_images, noise_used)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        
        return noisy, noise
    
    @torch.no_grad()
    def reverse_diffusion(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise to generate images.
        
        Args:
            x: Starting noise [B, C, H, W]
            return_intermediate: Whether to return intermediate steps
            
        Returns:
            Generated images (or list of intermediate images)
        """
        self.model.eval()
        batch_size = x.shape[0]
        
        intermediates = [] if return_intermediate else None
        
        for t in reversed(range(self.timesteps)):
            # Timestep tensor
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_tensor)
            
            # Compute mean of posterior
            sqrt_recip_alpha = self.sqrt_recip_alphas[t]
            beta = self.betas[t]
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
            
            mean = sqrt_recip_alpha * (
                x - beta * noise_pred / sqrt_one_minus_alpha_cumprod
            )
            
            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
            
            if return_intermediate:
                intermediates.append(x.cpu())
        
        return intermediates if return_intermediate else x
    
    def train_step(
        self,
        x0: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single training step.
        
        Args:
            x0: Clean images [B, C, H, W]
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        self.model.train()
        batch_size = x0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(x0)
        x_noisy, _ = self.forward_diffusion(x0, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_noisy, t)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        image_size: Tuple[int, int],
        channels: int = 3
    ) -> torch.Tensor:
        """
        Generate samples from noise.
        
        Args:
            num_samples: Number of images to generate
            image_size: (height, width)
            channels: Number of channels
            
        Returns:
            Generated images [N, C, H, W]
        """
        # Start from pure noise
        x = torch.randn(
            num_samples,
            channels,
            image_size[0],
            image_size[1],
            device=self.device
        )
        
        # Denoise
        return self.reverse_diffusion(x)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent singularities
        
    Returns:
        Beta schedule tensor
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
