import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.

    The Dice coefficient is a measure of overlap between two samples.
    Dice Loss = 1 - Dice Coefficient
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100) -> None:
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss.

        Args:
            predictions: Model predictions (B, C, H, W) - logits
            targets: Ground truth labels (B, H, W) - class indices

        Returns:
            Dice loss value
        """
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)

        # Get number of classes
        num_classes = predictions.shape[1]

        # Upsample predictions to match target size if needed
        if predictions.shape[2:] != targets.shape[1:]:
            predictions = F.interpolate(
                predictions,
                size=targets.shape[1:],
                mode="bilinear",
                align_corners=False,
            )

        # Create mask for valid pixels (ignore ignore_index)
        if self.ignore_index >= 0:
            valid_mask = targets != self.ignore_index
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        total_loss = 0.0

        for class_idx in range(num_classes):
            # Get predictions and targets for current class
            pred_class = predictions[:, class_idx, :, :]  # (B, H, W)
            target_class = (targets == class_idx).float()  # (B, H, W)

            # Apply valid mask
            pred_class = pred_class * valid_mask.float()
            target_class = target_class * valid_mask.float()

            # Flatten tensors
            pred_flat = pred_class.view(-1)
            target_flat = target_class.view(-1)

            # Calculate Dice coefficient
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()

            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - dice_coeff

            total_loss += dice_loss

        return total_loss / num_classes


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in semantic segmentation.

    Focal Loss = -α(1-p_t)^γ * log(p_t)
    where p_t is the model's estimated probability for the true class.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            ignore_index: Index to ignore in loss calculation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss.

        Args:
            predictions: Model predictions (B, C, H, W) - logits
            targets: Ground truth labels (B, H, W) - class indices

        Returns:
            Focal loss value
        """
        # Upsample predictions to match target size if needed
        if predictions.shape[2:] != targets.shape[1:]:
            predictions = F.interpolate(
                predictions,
                size=targets.shape[1:],
                mode="bilinear",
                align_corners=False,
            )

        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(
            predictions, targets, ignore_index=self.ignore_index, reduction="none"
        )

        # Calculate probabilities
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function using both Dice Loss and Focal Loss.

    This combines the benefits of both losses:
    - Dice Loss: Good for handling class imbalance and optimizing overlap
    - Focal Loss: Good for handling hard examples and class imbalance
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        dice_smooth: float = 1e-6,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        ignore_index: int = -100,
    ) -> None:
        """
        Initialize Combined Loss.

        Args:
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            dice_smooth: Smoothing factor for Dice loss
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            predictions: Model predictions (B, C, H, W) - logits
            targets: Ground truth labels (B, H, W) - class indices

        Returns:
            Combined loss value
        """
        dice_loss = self.dice_loss(predictions, targets)
        focal_loss = self.focal_loss(predictions, targets)

        combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return combined_loss
