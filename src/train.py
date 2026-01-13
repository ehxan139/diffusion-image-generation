"""
Training utilities for diffusion models.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import os


class Trainer:
    """Training loop for DDPM."""

    def __init__(
        self,
        ddpm,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.ddpm = ddpm
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        total_loss = 0
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            loss = self.ddpm.train_step(x, self.optimizer)
            total_loss += loss

            pbar.set_postfix({'loss': f'{loss:.4f}'})

        if self.scheduler:
            self.scheduler.step()

        return total_loss / len(self.dataloader)

    def train(self, epochs: int, save_every: int = 10, save_dir: str = "checkpoints"):
        """Train for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, save_dir)

    def save_checkpoint(self, epoch: int, save_dir: str):
        """Save model checkpoint."""
        path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.ddpm.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")
