import os
import random
import shutil
from PIL import Image

import numpy as np

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor

from .annotation import AnnotationCreator


class MaskGenerator:
    def __init__(self, mask_radius_ratio: float = 0.02) -> None:
        self.mask_radius_ratio = mask_radius_ratio

    def calculate_mask_radius(self, image_dimensions: tuple[int, int]) -> int:
        height, width = image_dimensions
        min_dimension = min(height, width)
        radius = int(min_dimension * self.mask_radius_ratio)
        return max(1, radius)

    def generate_mask_from_keypoints(
        self,
        keypoint_label: list[tuple[int, int, int]],
        image_dimensions: tuple[int, int],
    ) -> np.ndarray:
        height, width = image_dimensions
        mask = np.zeros((height, width), dtype=np.uint8)

        if not keypoint_label:
            return mask

        # Calculate mask radius for this image size
        mask_radius = self.calculate_mask_radius(image_dimensions)

        # Create coordinate grids for efficient mask generation
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        for keypoint in keypoint_label:
            if not keypoint or len(keypoint) < 3:
                continue

            x, y, visibility = keypoint

            # Only process visible keypoints
            if visibility != 1:
                continue

            # Create circular mask around keypoint
            distance_squared = (x_coords - x) ** 2 + (y_coords - y) ** 2
            circle_mask = distance_squared <= mask_radius**2

            # Combine with existing mask using logical OR
            mask = np.logical_or(mask, circle_mask).astype(np.uint8)

        # Convert to 0-255 range for image saving
        return mask * 255
