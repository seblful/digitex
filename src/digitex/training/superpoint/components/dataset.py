import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T

from digitex.core.processors.file import FileProcessor

from .augmenter import KeypointAugmenter


class KeypointDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        image_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        max_keypoints: int,
        load_heatmaps: bool = False,
        heatmap_dir: Optional[str] = None,
        sigma: float = 2.0,
    ) -> None:
        """
        Args:
            dataset_dir: Path to the dataset directory containing images.
            image_size: Tuple (H, W) to resize images to.
            heatmap_size: Tuple (H, W) to resize heatmaps to.
            max_keypoints: Maximum number of keypoints to keep.
            load_heatmaps: If True, load precomputed heatmaps/masks from heatmap_dir.
            heatmap_dir: Directory containing precomputed heatmaps/masks (if used).
            sigma: Standard deviation for Gaussian kernel in heatmap generation.
        """
        self.dataset_dir = dataset_dir
        self.labels_json_path = os.path.join(dataset_dir, "labels.json")

        self.load_heatmaps = load_heatmaps
        self.heatmap_dir = heatmap_dir

        self.max_keypoints = max_keypoints
        self.sigma = sigma  # Store sigma

        self.image_size = image_size
        self.heatmap_size = heatmap_size

        self.labels = FileProcessor.read_json(self.labels_json_path)
        self.image_relpaths = list(self.labels.keys())

        # Compose resize transform for images
        self._resize_image = T.Resize(self.image_size, antialias=True)

    def __len__(self) -> int:
        return len(self.image_relpaths)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        relpath = self.image_relpaths[idx]
        label = self.labels[relpath]

        # Get an image
        image_path = os.path.join(self.dataset_dir, relpath)
        img = read_image(image_path)
        orig_img_height, orig_img_width = img.shape[1], img.shape[2]
        img = self.transform(img)

        # Get target
        abs_kps_obj = KeypointAugmenter.create_abs_kps_obj_from_label(label, clip=False)
        vis_coords = abs_kps_obj.get_vis_coords()
        vis_coords_resized = self._resize_coords(
            vis_coords, (orig_img_height, orig_img_width)
        )
        target = self._get_target(relpath, vis_coords_resized)

        return img, target

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        img = self._resize_image(img)
        img = img.float() / 255.0  # shape: (C, H, W), normalized

        return img

    def _resize_coords(
        self, vis_coords, orig_image_size: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Scale keypoints from original image space to resized image space."""
        if not vis_coords:
            return []

        scale_x = self.image_size[1] / orig_image_size[1]
        scale_y = self.image_size[0] / orig_image_size[0]
        return [(x * scale_x, y * scale_y) for x, y in vis_coords]

    def _get_target(
        self,
        relpath: str,
        vis_coords,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.load_heatmaps and self.heatmap_dir is not None:
            return self._load_precomputed_heatmaps(relpath)
        else:
            return self._generate_heatmaps_on_the_fly(vis_coords)

    def _load_precomputed_heatmaps(
        self, relpath: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = os.path.splitext(os.path.basename(relpath))[0]
        heatmap_path = os.path.join(self.heatmap_dir, f"{base}_heatmap.pt")
        mask_path = os.path.join(self.heatmap_dir, f"{base}_mask.pt")
        heatmaps = torch.load(heatmap_path)
        mask = torch.load(mask_path)
        return (heatmaps, mask)

    def _generate_heatmaps_on_the_fly(
        self,
        vis_coords,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_keypoints = self.max_keypoints  # Use fixed max_keypoints

        if not vis_coords:
            heatmaps = torch.zeros(
                max_keypoints, self.heatmap_size[0], self.heatmap_size[1]
            )
            mask = torch.zeros(max_keypoints, dtype=torch.float32)
            return (heatmaps, mask)
        else:
            # Pad or truncate keypoints to max_keypoints
            keypoints = [
                torch.tensor([kp[0], kp[1]], dtype=torch.float32) for kp in vis_coords
            ]
            if len(keypoints) < max_keypoints:
                keypoints += [torch.full([2], float("nan"))] * (
                    max_keypoints - len(keypoints)
                )
            else:
                keypoints = keypoints[:max_keypoints]
            heatmaps, mask = self.generate_heatmaps(
                keypoints=keypoints,
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                max_keypoints=max_keypoints,
                sigma=self.sigma,
            )
            return (heatmaps, mask)

    @staticmethod
    def generate_heatmaps(
        keypoints: list[Optional[torch.Tensor]],
        image_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        max_keypoints: int,
        sigma: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate heatmaps and presence mask from keypoints

        Args:
            keypoints: List of keypoints per image, each tensor of shape (N, 2)
                       (x, y coordinates in original image space)
            image_size: Original image dimensions (height, width)
            heatmap_size: Output heatmap dimensions (height, width)
            max_keypoints: Maximum number of keypoints to support
            sigma: Standard deviation for Gaussian kernel

        Returns:
            heatmaps: Tensor of shape (max_keypoints, output_h, output_w)
            mask: Binary presence mask of shape (max_keypoints,)
        """
        heatmaps = torch.zeros(max_keypoints, heatmap_size[0], heatmap_size[1])
        mask = torch.zeros(max_keypoints, dtype=torch.float32)

        # Scale factors for coordinate transformation
        h_scale = heatmap_size[0] / image_size[0]
        w_scale = heatmap_size[1] / image_size[1]

        # Create grid for heatmap generation
        y_grid, x_grid = torch.meshgrid(
            torch.arange(heatmap_size[0], dtype=torch.float32),
            torch.arange(heatmap_size[1], dtype=torch.float32),
            indexing="ij",
        )

        for kp_idx, kp in enumerate(keypoints[:max_keypoints]):
            if kp is None or torch.isnan(kp).any():
                continue

            mask[kp_idx] = 1.0

            # Scale keypoint coordinates to heatmap space
            x = kp[0] * w_scale
            y = kp[1] * h_scale

            # Create Gaussian heatmap
            dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
            exponent = dist_sq / (2 * sigma**2)
            heatmap = torch.exp(-exponent)

            # Normalize to 0-1 range
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            heatmaps[kp_idx] = heatmap

        return heatmaps, mask
