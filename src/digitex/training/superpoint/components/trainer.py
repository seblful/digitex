import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # Added for progress bars


class HeatmapKeypointTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_keypoints: int,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        heatmap_sigma: float = 2.0,
        use_tensorboard: bool = True,
    ) -> None:
        """
        Trainer for heatmap-based keypoint prediction

        Args:
            model: Keypoint prediction model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            max_keypoints: Maximum number of keypoints
            device: Device for training (cuda or cpu)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory to save model checkpoints
            heatmap_sigma: Sigma for Gaussian heatmap generation
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_keypoints = max_keypoints
        self.heatmap_sigma = heatmap_sigma

        # Loss function - MSE with masking for absent keypoints
        self.criterion = nn.MSELoss(reduction="none")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Setup directories
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir) if use_tensorboard else None
        self.global_step = 0

    def masked_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate masked MSE loss

        Args:
            predictions: Predicted heatmaps (B, K, H, W)
            targets: Ground truth heatmaps (B, K, H, W)
            masks: Keypoint presence masks (B, K)

        Returns:
            Masked loss value
        """
        # Expand mask to match heatmap dimensions
        mask_expanded = masks.unsqueeze(-1).unsqueeze(-1)

        # Calculate element-wise loss
        loss_per_element = self.criterion(predictions, targets)

        # Apply mask and average
        masked_loss = loss_per_element * mask_expanded
        return masked_loss.sum() / (mask_expanded.sum() + 1e-6)

    def train_epoch(self, epoch: int, num_epochs: int) -> float:
        """Run one training epoch, return average loss"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        desc = f"Epoch [{epoch}/{num_epochs}] Training"
        with tqdm(total=num_batches, desc=desc, leave=True, unit="step") as pbar:
            for batch_idx, (images, heatmaps, masks) in enumerate(self.train_loader):
                # Move data to device
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Calculate loss
                loss = self.masked_loss(outputs, heatmaps, masks)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()

                # Log to TensorBoard
                if self.writer and batch_idx % 10 == 0:
                    self.writer.add_scalar(
                        "train/loss_batch", loss.item(), self.global_step
                    )
                    self.global_step += 1

                # Update tqdm progress bar
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                pbar.update(1)

        return epoch_loss / num_batches

    @torch.no_grad()
    def validate(self, epoch: int = None, num_epochs: int = None) -> float:
        """Run validation, return average loss"""
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)

        desc = f"Epoch [{epoch}/{num_epochs}] Validation"
        if epoch is not None and num_epochs is not None:
            desc += f" [{epoch}/{num_epochs}]"

        with tqdm(total=num_batches, desc=desc, leave=True, unit="step") as pbar:
            for batch_idx, (images, heatmaps, masks) in enumerate(self.val_loader):
                # Move data to device
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                loss = self.masked_loss(outputs, heatmaps, masks)
                val_loss += loss.item()

                # Update tqdm progress bar
                avg_loss = val_loss / (batch_idx + 1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                pbar.update(1)

        return val_loss / num_batches

    def train(self, num_epochs: int, save_every: int = 5, early_stopping: int = 15):
        """
        Full training pipeline

        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            early_stopping: Stop training if validation loss doesn't improve for N epochs
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0

        print(f"Start training on {self.device}.")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, num_epochs)
            val_loss = self.validate(epoch, num_epochs)
            self.scheduler.step(val_loss)

            print(f"Val Loss: {val_loss:.6f}")

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    print(f"Early stopping after {epoch} epochs")
                    break

        print(f"Best validation loss: {best_val_loss:.6f}")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "max_keypoints": self.max_keypoints,
        }

        if is_best:
            filename = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, filename)
            print(f"Saved best model with val loss {val_loss:.6f}")
        else:
            filename = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(state, filename)
            print(f"Saved checkpoint for epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
            f"val loss {checkpoint['val_loss']:.6f}"
        )
