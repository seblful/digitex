import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class HeatmapKeypointTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_keypoints: int,
        device: torch.device,
        lr: float,
        weight_decay: float,
        log_dir: str,
        checkpoint_dir: str,
        checkpoint_path: str | None = None,
        use_tensorboard: bool = True,
        visibility_loss_weight: float = 1.0,
    ) -> None:
        """
        Trainer for heatmap-based keypoint prediction with visibility mask prediction

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
            checkpoint_path: Path to a model checkpoint to load (optional)
            use_tensorboard: Whether to use TensorBoard logging
            visibility_loss_weight: Weight for visibility loss in combined loss
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_keypoints = max_keypoints
        self.visibility_loss_weight = visibility_loss_weight

        # Loss functions
        self.heatmap_criterion = nn.HuberLoss(reduction="none")
        self.visibility_criterion = nn.BCELoss(reduction="mean")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Setup directories
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir) if use_tensorboard else None
        self.global_step = 0

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
            f"val loss {checkpoint['val_loss']:.6f}."
        )

    def combined_loss(
        self,
        heatmap_predictions: torch.Tensor,
        visibility_predictions: torch.Tensor,
        heatmap_targets: torch.Tensor,
        visibility_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss for heatmaps and visibility masks

        Args:
            heatmap_predictions: Predicted heatmaps (B, K, H, W)
            visibility_predictions: Predicted visibility masks (B, K)
            heatmap_targets: Ground truth heatmaps (B, K, H, W)
            visibility_targets: Ground truth visibility masks (B, K)

        Returns:
            Tuple of (total_loss, heatmap_loss, visibility_loss)
        """
        # Convert visibility targets to float for loss calculation
        visibility_targets = visibility_targets.float()

        # Calculate visibility loss (binary cross-entropy)
        visibility_loss = self.visibility_criterion(
            visibility_predictions, visibility_targets
        )

        # Calculate visibility-aware heatmap loss
        # Only compute heatmap loss for visible keypoints
        mask_expanded = (
            visibility_targets.unsqueeze(-1)
            .unsqueeze(-1)
            .expand_as(heatmap_predictions)
        )
        heatmap_loss_per_element = self.heatmap_criterion(
            heatmap_predictions, heatmap_targets
        )
        masked_heatmap_loss = heatmap_loss_per_element * mask_expanded
        heatmap_loss = masked_heatmap_loss.sum() / (mask_expanded.sum() + 1e-6)

        # Combine losses
        total_loss = heatmap_loss + self.visibility_loss_weight * visibility_loss

        return total_loss, heatmap_loss, visibility_loss

    def train_epoch(self, epoch: int, num_epochs: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        epoch_heatmap_loss = 0.0
        epoch_visibility_loss = 0.0
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
                heatmap_outputs, visibility_outputs = self.model(images)

                # Calculate combined loss
                total_loss, heatmap_loss, visibility_loss = self.combined_loss(
                    heatmap_outputs, visibility_outputs, heatmaps, masks
                )

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Update metrics
                epoch_loss += total_loss.item()
                epoch_heatmap_loss += heatmap_loss.item()
                epoch_visibility_loss += visibility_loss.item()

                # Log to TensorBoard
                if self.writer and batch_idx % 10 == 0:
                    self.writer.add_scalar(
                        "train/total_loss_batch", total_loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/heatmap_loss_batch",
                        heatmap_loss.item(),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "train/visibility_loss_batch",
                        visibility_loss.item(),
                        self.global_step,
                    )
                    self.global_step += 1

                # Update tqdm progress bar
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_heatmap_loss = epoch_heatmap_loss / (batch_idx + 1)
                avg_visibility_loss = epoch_visibility_loss / (batch_idx + 1)
                pbar.set_postfix(
                    total=f"{avg_loss:.4f}",
                    heatmap=f"{avg_heatmap_loss:.4f}",
                    visibility=f"{avg_visibility_loss:.4f}",
                )
                pbar.update(1)

        return epoch_loss / num_batches

    @torch.no_grad()
    def validate(self, epoch: int = None, num_epochs: int = None) -> float:
        self.model.eval()
        val_loss = 0.0
        val_heatmap_loss = 0.0
        val_visibility_loss = 0.0
        num_batches = len(self.val_loader)

        desc = f"Epoch [{epoch}/{num_epochs}] Validation"

        with tqdm(total=num_batches, desc=desc, leave=True, unit="step") as pbar:
            for batch_idx, (images, heatmaps, masks) in enumerate(self.val_loader):
                # Move data to device
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                heatmap_outputs, visibility_outputs = self.model(images)

                # Calculate combined loss
                total_loss, heatmap_loss, visibility_loss = self.combined_loss(
                    heatmap_outputs, visibility_outputs, heatmaps, masks
                )

                val_loss += total_loss.item()
                val_heatmap_loss += heatmap_loss.item()
                val_visibility_loss += visibility_loss.item()

                # Update tqdm progress bar
                avg_loss = val_loss / (batch_idx + 1)
                avg_heatmap_loss = val_heatmap_loss / (batch_idx + 1)
                avg_visibility_loss = val_visibility_loss / (batch_idx + 1)
                pbar.set_postfix(
                    total=f"{avg_loss:.4f}",
                    heatmap=f"{avg_heatmap_loss:.4f}",
                    visibility=f"{avg_visibility_loss:.4f}",
                )
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

        # Start training
        print(f"Start training on {self.device}.")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, num_epochs)
            val_loss = self.validate(epoch, num_epochs)
            self.scheduler.step(val_loss)

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
                    print(f"Early stopping after {epoch} epochs.")
                    break

        print(f"Best validation loss: {best_val_loss:.6f}")

        if self.writer:
            self.writer.close()

    def save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False
    ) -> None:
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
            print(f"Saved best model with val loss {val_loss:.6f}.")
        else:
            filename = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(state, filename)
            print(f"Saved checkpoint for epoch {epoch}.")
