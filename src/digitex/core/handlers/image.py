"""Image handling utilities."""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CROP_OFFSET = 0.025


class ImageHandler:
    """Handler for image operations including cropping and processing."""

    @staticmethod
    def crop_image(
        image: Image.Image,
        points: list[float],
        offset: float = DEFAULT_CROP_OFFSET,
    ) -> Image.Image:
        """Crop an image using a polygon and add white border.

        Args:
            image: Input PIL Image.
            points: Normalized polygon points [x1, y1, x2, y2, ...] (0-1).
            offset: Border size as fraction of image height.

        Returns:
            Cropped PIL Image with white border.

        Raises:
            ValueError: If points list is invalid.
        """
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]

        pts = np.array(points, dtype=np.float32)
        if len(pts) % 2 != 0:
            raise ValueError("Points list must contain an even number of values")

        pts = pts.reshape(-1, 2)

        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        img = img[y : y + h, x : x + w].copy()

        pts = pts - pts.min(axis=0)
        pts_int = pts.astype(np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts_int], -1, (255, 255, 255), -1, cv2.LINE_AA)

        result = cv2.bitwise_and(img, img, mask=mask)
        bg = np.ones_like(img, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        result = bg + result

        border = int(height * offset)
        result = cv2.copyMakeBorder(
            result, border, border, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    @staticmethod
    def get_random_image(
        images_listdir: list[str],
        images_dir: str | Path,
    ) -> tuple[Image.Image, str]:
        """Get a random image from a directory.

        Args:
            images_listdir: List of image filenames.
            images_dir: Directory containing the images.

        Returns:
            Tuple of (image, image_name).

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            IOError: If the image cannot be read.
        """
        import random

        images_dir = Path(images_dir)

        rand_image_name = random.choice(images_listdir)
        rand_image_path = images_dir / rand_image_name

        if not rand_image_path.exists():
            raise FileNotFoundError(f"Image file not found: {rand_image_path}")

        rand_image = Image.open(rand_image_path)

        return rand_image, rand_image_name
