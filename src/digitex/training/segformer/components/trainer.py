import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation
import evaluate

from .losses import CombinedLoss, FocalLoss


class SegFormerTrainer:
    def __init__(
        self,
        model: SegformerForSemanticSegmentation,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_classes: int,
        lr: float,
        weight_decay: float,
        log_dir: str,
        checkpoint_dir: str,
        checkpoint_path: Optional[str] = None,
        use_tensorboard: bool = True,
        use_custom_loss: bool = True,
        loss_type: str = "combined",  # "combined", "focal", "ce"
    ) -> None:
        """
        Initialize SegFormer trainer.

        Args:
            model: SegFormer model for semantic segmentation
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device for training (cuda or cpu)
            num_classes: Number of segmentation classes
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory to save model checkpoints
            checkpoint_path: Path to a model checkpoint to load (optional)
            use_tensorboard: Whether to use TensorBoard logging
            use_custom_loss: Whether to use custom loss function instead of model's default
            loss_type: Type of loss function ("combined", "focal", "ce")
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.use_custom_loss = use_custom_loss

        # Setup custom loss function if enabled
        if use_custom_loss:
            if loss_type == "combined":
                self.criterion = CombinedLoss(
                    dice_weight=0.5,
                    focal_weight=0.5,
                    focal_alpha=0.25,  # Lower alpha for background class
                    focal_gamma=2.0,
                    ignore_index=255,
                )
            elif loss_type == "focal":
                self.criterion = FocalLoss(
                    alpha=0.25,  # Lower alpha for background class
                    gamma=2.0,
                    ignore_index=255,
                )
            else:  # cross-entropy with class weights
                # Calculate class weights based on data analysis
                # Background: 99.71%, Keypoint: 0.29% -> weights: [0.0029, 0.9971]
                # Scale down to prevent gradient explosion
                class_weights = torch.tensor(
                    [0.1, 50.0], device=device
                )  # [background, keypoint] - scaled down from analysis
                self.criterion = nn.CrossEntropyLoss(
                    weight=class_weights, ignore_index=255
                )
        else:
            self.criterion = None

        # Setup IoU metric from evaluate library
        self.train_metric = evaluate.load("mean_iou")
        self.val_metric = evaluate.load("mean_iou")

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Use validation IoU for scheduling instead of loss
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=7
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

        # Best validation metrics for model saving
        self.best_val_loss = float("inf")
        self.best_val_iou = 0.0

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def _save_best_checkpoint(
        self, epoch: int, val_loss: float, val_iou: float
    ) -> None:
        # Save best model based on validation IoU
        if val_iou > self.best_val_iou:
            self.best_val_iou = val_iou
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with IoU: {val_iou:.4f}")

    def _save_latest_checkpoint(
        self, epoch: int, val_loss: float, val_iou: float
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_iou": val_iou,
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest_model.pth")
        torch.save(checkpoint, latest_path)
        print(f"Saved checkpoint with IoU: {val_iou:.4f}")

    def _add_batch_to_metric(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        metric: evaluate.EvaluationModule,
    ) -> None:
        """
        Add a batch to the IoU metric accumulator.

        Args:
            logits: Model predictions (B, C, H, W) - logits
            labels: Ground truth labels (B, H, W) - class indices
            metric: The evaluate metric instance to use
        """
        # Upsample logits to match label size if needed
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)

        # Convert to numpy with explicit int32 dtype to avoid Pillow warnings
        predictions_np = predicted.detach().cpu().numpy().astype("int32")
        references_np = labels.detach().cpu().numpy().astype("int32")

        # Add batch to metric accumulator
        metric.add_batch(
            predictions=predictions_np,
            references=references_np,
        )

    def train_epoch(self, epoch: int, num_epochs: int) -> tuple[float, dict]:
        self.model.train()

        # Reset training metric for new epoch
        self.train_metric = evaluate.load("mean_iou")

        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        desc = f"Epoch [{epoch}/{num_epochs}] Training"
        with tqdm(total=num_batches, desc=desc, leave=True, unit="step") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                if self.use_custom_loss:
                    # Use custom loss function
                    outputs = self.model(pixel_values=pixel_values)
                    loss = self.criterion(outputs.logits, labels)
                else:
                    # Use model's default loss
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                # Add batch to IoU metric accumulator
                self._add_batch_to_metric(outputs.logits, labels, self.train_metric)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()

                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar(
                        "train/loss_step", loss.item(), self.global_step
                    )

                self.global_step += 1

                # Update progress bar
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                )
                pbar.update(1)

        # Compute final IoU for the epoch
        train_iou_results = self.train_metric.compute(
            num_labels=self.num_classes, ignore_index=255, reduce_labels=False
        )
        final_iou = train_iou_results["mean_iou"]

        # Also compute per-class IoU for debugging
        per_class_iou = train_iou_results.get("per_category_iou", [])
        if len(per_class_iou) >= 2:
            background_iou = per_class_iou[0] if per_class_iou[0] is not None else 0.0
            keypoint_iou = per_class_iou[1] if per_class_iou[1] is not None else 0.0
            print(
                f"  Train - Background IoU: {background_iou:.4f}, Keypoint IoU: {keypoint_iou:.4f}"
            )

        # Calculate final epoch metrics
        final_loss = epoch_loss / num_batches
        final_metrics = {"iou": final_iou}

        return final_loss, final_metrics

    @torch.no_grad()
    def validate(self, epoch: int = None, num_epochs: int = None) -> tuple[float, dict]:
        self.model.eval()

        # Reset validation metric for new epoch
        self.val_metric = evaluate.load("mean_iou")

        val_loss = 0.0
        num_batches = len(self.val_loader)

        desc = f"Epoch [{epoch}/{num_epochs}] Validation" if epoch else "Validation"
        with tqdm(total=num_batches, desc=desc, leave=True, unit="step") as pbar:
            for batch_idx, batch in enumerate(self.val_loader):
                # Move data to device
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                if self.use_custom_loss:
                    # Use custom loss function
                    outputs = self.model(pixel_values=pixel_values)
                    loss = self.criterion(outputs.logits, labels)
                else:
                    # Use model's default loss
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                # Add batch to IoU metric accumulator
                self._add_batch_to_metric(outputs.logits, labels, self.val_metric)

                # Update metrics
                val_loss += loss.item()

                # Update progress bar
                avg_loss = val_loss / (batch_idx + 1)
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                )
                pbar.update(1)

        # Compute final IoU for the epoch
        val_iou_results = self.val_metric.compute(
            num_labels=self.num_classes, ignore_index=255, reduce_labels=False
        )
        final_iou = val_iou_results["mean_iou"]

        # Also compute per-class IoU for debugging
        per_class_iou = val_iou_results.get("per_category_iou", [])
        if len(per_class_iou) >= 2:
            background_iou = per_class_iou[0] if per_class_iou[0] is not None else 0.0
            keypoint_iou = per_class_iou[1] if per_class_iou[1] is not None else 0.0
            print(
                f"  Val - Background IoU: {background_iou:.4f}, Keypoint IoU: {keypoint_iou:.4f}"
            )

        # Calculate final validation metrics
        final_loss = val_loss / num_batches
        final_metrics = {"iou": final_iou}

        return final_loss, final_metrics

    def train(
        self, num_epochs: int, save_every: int = 10, early_stopping: int = None
    ) -> None:
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            early_stopping: Stop training if no improvement for N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch, num_epochs)

            # Validate
            val_loss, val_metrics = self.validate(epoch, num_epochs)

            # Update learning rate scheduler based on validation IoU
            self.scheduler.step(val_metrics["iou"])

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                self.writer.add_scalar("val/loss_epoch", val_loss, epoch)
                self.writer.add_scalar("train/iou_epoch", train_metrics["iou"], epoch)
                self.writer.add_scalar("val/iou_epoch", val_metrics["iou"], epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Print epoch summary
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['iou']:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['iou']:.4f}"
            )

            # Check for best model and save if improved (every epoch)
            self._save_best_checkpoint(epoch, val_loss, val_metrics["iou"])

            # Save regular checkpoint
            if epoch % save_every == 0:
                self._save_latest_checkpoint(epoch, val_loss, val_metrics["iou"])

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        # Save final checkpoint
        self._save_latest_checkpoint(num_epochs, val_loss, val_metrics["iou"])

        if self.writer:
            self.writer.close()

        print(f"Best validation loss: {best_val_loss:.6f}")
