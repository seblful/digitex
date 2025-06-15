import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
import torch.nn.functional as F


class HeatmapKeypointDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        image_size: tuple[int, int] = None,
        heatmap_size: tuple[int, int] = None,
    ) -> None:
        """
        Args:
            dataset_dir: Path to the dataset directory containing images.
            image_size: Target size for images (height, width). If None, keeps original size.
            heatmap_size: Target size for heatmaps (height, width). If None, keeps original size.
        """

        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.heatmaps_dir = os.path.join(dataset_dir, "heatmaps")
        self.mask_dir = os.path.join(dataset_dir, "masks")

        self.image_size = image_size
        self.heatmap_size = heatmap_size

        # List all image files in the images directory
        self.image_relpaths = sorted(
            [
                fname
                for fname in os.listdir(self.images_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        # Setup transforms
        self._setup_transforms()

    @staticmethod
    def _normalize_to_float(x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to float and normalize to [0, 1]"""
        return x.float() / 255.0

    def _setup_transforms(self) -> None:
        """Setup image transforms based on configuration."""
        transforms = []

        # Convert to float and normalize to [0, 1]
        transforms.append(self._normalize_to_float)

        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        self.image_transforms = T.Compose(transforms)

        # Setup resize transform if image_size is specified
        if self.image_size is not None:
            self.image_resize = T.Resize(self.image_size, antialias=True)
        else:
            self.image_resize = None

    def __len__(self) -> int:
        return len(self.image_relpaths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relpath = self.image_relpaths[idx]
        image_path = os.path.join(self.images_dir, relpath)

        # Load and transform image
        img = read_image(image_path)
        img = self._transform_image(img)

        # Load and transform heatmaps
        heatmaps, masks = self._load_and_transform_heatmaps(relpath)

        return img, heatmaps, masks

    def _transform_image(self, img: torch.Tensor) -> torch.Tensor:
        """Apply transforms to image."""
        # Resize if specified
        if self.image_resize is not None:
            img = self.image_resize(img)

        # Apply other transforms (normalization, etc.)
        img = self.image_transforms(img)

        return img

    def _load_and_transform_heatmaps(
        self, relpath: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load heatmaps and masks, applying resizing if specified."""
        base = os.path.splitext(os.path.basename(relpath))[0]
        heatmap_path = os.path.join(self.heatmaps_dir, f"{base}.pt")
        mask_path = os.path.join(self.mask_dir, f"{base}.pt")

        heatmaps = torch.load(heatmap_path)  # Shape: (num_keypoints, H, W)
        masks = torch.load(mask_path)  # Shape: (num_keypoints,)

        # Resize heatmaps if specified
        if self.heatmap_size is not None:
            # Get current heatmap size
            current_size = heatmaps.shape[-2:]  # (H, W)
            target_size = self.heatmap_size  # (H, W)

            if current_size != target_size:
                # Resize heatmaps using bilinear interpolation
                heatmaps = F.interpolate(
                    heatmaps.unsqueeze(
                        0
                    ),  # Add batch dimension: (1, num_keypoints, H, W)
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # Remove batch dimension: (num_keypoints, H, W)

        return heatmaps, masks
