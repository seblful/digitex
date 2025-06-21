import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np


class MaskDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        image_size: tuple[int, int] = None,
        mask_size: tuple[int, int] = None,
    ) -> None:
        """
        Dataset for loading images and corresponding segmentation masks.

        Args:
            dataset_dir: Path to the dataset directory containing images and masks.
            image_size: Target size for images (height, width). If None, keeps original size.
            mask_size: Target size for masks (height, width). If None, keeps original size.
        """

        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.masks_dir = os.path.join(dataset_dir, "masks")

        self.image_size = image_size
        self.mask_size = mask_size

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
        return x.float() / 255.0

    def _setup_transforms(self) -> None:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        relpath = self.image_relpaths[idx]
        image_path = os.path.join(self.images_dir, relpath)

        # Load and transform image
        img = self._load_and_transform_image(image_path)

        # Load and transform mask
        mask = self._load_and_transform_mask(relpath)

        return img, mask

    def _load_and_transform_image(self, image_path: str) -> torch.Tensor:
        img = read_image(image_path)

        # Resize if specified
        if self.image_resize is not None:
            img = self.image_resize(img)

        # Apply other transforms (normalization, etc.)
        img = self.image_transforms(img)

        return img

    def _load_and_transform_mask(self, relpath: str) -> torch.Tensor:
        base = os.path.splitext(os.path.basename(relpath))[0]
        mask_path = os.path.join(self.masks_dir, f"{base}.png")

        # Load mask using PIL and convert to tensor
        mask_pil = Image.open(mask_path).convert("L")
        mask_array = np.array(mask_pil, dtype=np.float32)

        # Normalize to [0, 1] range
        mask_array = mask_array / 255.0

        # Convert to tensor and add channel dimension: (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask_array).unsqueeze(0)

        # Resize mask if specified
        if self.mask_size is not None:
            # Get current mask size
            current_size = mask.shape[-2:]  # (H, W)
            target_size = self.mask_size  # (H, W)

            if current_size != target_size:
                # Resize mask using nearest neighbor interpolation to preserve binary nature
                mask = F.interpolate(
                    mask.unsqueeze(0),  # Add batch dimension: (1, 1, H, W)
                    size=target_size,
                    mode="nearest",
                ).squeeze(0)  # Remove batch dimension: (1, H, W)

        return mask
