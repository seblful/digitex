import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T


class HeatmapKeypointDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        """
        Args:
            dataset_dir: Path to the dataset directory containing images.
            heatmap_dir: Directory containing precomputed heatmaps/masks (required).
        """

        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.heatmaps_dir = os.path.join(dataset_dir, "heatmaps")
        self.mask_dir = os.path.join(dataset_dir, "masks")

        # List all image files in the images directory
        self.image_relpaths = sorted(
            [
                fname
                for fname in os.listdir(self.images_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.image_relpaths)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        relpath = self.image_relpaths[idx]
        image_path = os.path.join(self.images_dir, relpath)
        img = read_image(image_path)
        img = self.transform(img)

        heatmaps, masks = self._load_heatmaps(relpath)
        return img, heatmaps, masks

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        img = img.float() / 255.0
        img = self._normalize(img)

        return img

    def _load_heatmaps(self, relpath: str) -> tuple[torch.Tensor, torch.Tensor]:
        base = os.path.splitext(os.path.basename(relpath))[0]
        heatmap_path = os.path.join(self.heatmaps_dir, f"{base}.pt")
        mask_path = os.path.join(self.mask_dir, f"{base}.pt")
        heatmaps = torch.load(heatmap_path)
        masks = torch.load(mask_path)
        return heatmaps, masks
