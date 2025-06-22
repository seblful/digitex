import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class IoUMetric:
    """
    Intersection over Union (IoU) metric for binary segmentation.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6) -> None:
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate IoU metric.

        Args:
            predictions: Predicted masks (B, 1, H, W) - logits or probabilities
            targets: Ground truth masks (B, 1, H, W) - binary values [0, 1]

        Returns:
            IoU score
        """
        # Apply sigmoid and threshold to predictions
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > self.threshold).float()

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection

        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.item()


class MaskSegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float,
        weight_decay: float,
        log_dir: str,
        checkpoint_dir: str,
        checkpoint_path: str | None = None,
        use_tensorboard: bool = True,
        iou_threshold: float = 0.5,
    ) -> None:
        """
        Trainer for segmentation mask prediction using SimpleUNet

        Args:
            model: Segmentation model (SimpleUNet)
            train_loader: DataLoader for training data (images and masks)
            val_loader: DataLoader for validation data (images and masks)
            device: Device for training (cuda or cpu)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory to save model checkpoints
            checkpoint_path: Path to a model checkpoint to load (optional)
            use_tensorboard: Whether to use TensorBoard logging
            iou_threshold: Threshold for IoU metric calculation
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function and metrics
        self.criterion = nn.BCEWithLogitsLoss()

        self.iou_metric = IoUMetric(threshold=iou_threshold)

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

    def calculate_loss_and_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Calculate loss and metrics for segmentation masks

        Args:
            predictions: Predicted masks (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary values [0, 1]

        Returns:
            Tuple of (loss, iou_score)
        """
        # Calculate loss
        loss = self.criterion(predictions, targets)

        # Calculate IoU metric
        iou_score = self.iou_metric(predictions, targets)

        return loss, iou_score

    def train_epoch(self, epoch: int, num_epochs: int) -> tuple[float, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        num_batches = len(self.train_loader)

        desc = f"Epoch [{epoch}/{num_epochs}] Training"
        with tqdm(total=num_batches, desc=desc, leave=True, unit="step") as pbar:
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(images)

                # Calculate loss and metrics
                loss, iou_score = self.calculate_loss_and_metrics(predictions, masks)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                epoch_iou += iou_score

                # Log to TensorBoard
                if self.writer and batch_idx % 10 == 0:
                    self.writer.add_scalar(
                        "train/loss_batch", loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/iou_batch", iou_score, self.global_step
                    )
                    self.global_step += 1

                # Update tqdm progress bar
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_iou = epoch_iou / (batch_idx + 1)
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    iou=f"{avg_iou:.4f}",
                )
                pbar.update(1)

        return epoch_loss / num_batches, epoch_iou / num_batches

    @torch.no_grad()
    def validate(
        self, epoch: int = None, num_epochs: int = None
    ) -> tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        val_iou = 0.0
        num_batches = len(self.val_loader)

        desc = f"Epoch [{epoch}/{num_epochs}] Validation"

        with tqdm(total=num_batches, desc=desc, leave=True, unit="step") as pbar:
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                predictions = self.model(images)

                # Calculate loss and metrics
                loss, iou_score = self.calculate_loss_and_metrics(predictions, masks)

                val_loss += loss.item()
                val_iou += iou_score

                # Update tqdm progress bar
                avg_loss = val_loss / (batch_idx + 1)
                avg_iou = val_iou / (batch_idx + 1)
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    iou=f"{avg_iou:.4f}",
                )
                pbar.update(1)

        return val_loss / num_batches, val_iou / num_batches

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
            train_loss, train_iou = self.train_epoch(epoch, num_epochs)
            val_loss, val_iou = self.validate(epoch, num_epochs)
            self.scheduler.step(val_loss)

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                self.writer.add_scalar("train/iou_epoch", train_iou, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("val/iou", val_iou, epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Print epoch summary
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
            )

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_loss, val_iou)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                self.save_checkpoint(epoch, val_loss, val_iou, is_best=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    print(f"Early stopping after {epoch} epochs.")
                    break

        print(f"Best validation loss: {best_val_loss:.6f}")

        if self.writer:
            self.writer.close()

    def save_checkpoint(
        self, epoch: int, val_loss: float, val_iou: float, is_best: bool = False
    ) -> None:
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_iou": val_iou,
        }

        if is_best:
            filename = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, filename)
            print(
                f"Saved best model with val loss {val_loss:.6f}, val IoU {val_iou:.4f}."
            )
        else:
            filename = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(state, filename)
            print(f"Saved checkpoint for epoch {epoch}.")
