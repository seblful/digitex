"""Image processing utilities."""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BG_THRESHOLD = 200
DEFAULT_LOWER_BLUE = np.array([70, 30, 30])
DEFAULT_UPPER_BLUE = np.array([130, 255, 255])
DEFAULT_BORDER_MULTIPLIER = 5


def resize_img(
    img: np.ndarray,
    max_height: int,
) -> np.ndarray:
    """Resize numpy image to fit within maximum height while maintaining aspect ratio.

    Args:
        img: Input numpy image.
        max_height: Maximum allowed height in pixels.

    Returns:
        Resized image if height exceeds max_height, otherwise original image.
    """
    height, width = img.shape[:2]

    if height > max_height:
        aspect_ratio = width / height
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return img


def resize_image(image: Image.Image, max_height: int) -> Image.Image:
    """Resize PIL image to fit within maximum height.

    Args:
        image: Input PIL Image.
        max_height: Maximum allowed height. If 0, no resizing.

    Returns:
        Resized PIL Image.
    """
    if max_height <= 0:
        return image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = resize_img(img, max_height)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


import cv2
import numpy as np
from PIL import Image

# Dummy constants (Replace with your actual defaults)
DEFAULT_LOWER_BLUE = np.array([100, 150, 0])
DEFAULT_UPPER_BLUE = np.array([140, 255, 255])
DEFAULT_BORDER_MULTIPLIER = 5
DEFAULT_BG_THRESHOLD = 240


class ImageCropper:
    """Processor for image cropping operations using perspective transformations."""

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    @classmethod
    def _polygon_to_quadrilateral(
        cls, polygon: list[tuple[int, int]], max_angle: float = 4.0
    ) -> np.ndarray:
        pts = np.array(polygon, dtype=np.int32)
        rect = cv2.minAreaRect(pts)

        if abs(min(rect[2], 90 - rect[2])) > max_angle:
            x, y, w, h = cv2.boundingRect(pts)
            bbox = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )
        else:
            bbox = cv2.boxPoints(rect)

        return cls._order_points(bbox)

    @staticmethod
    def _get_transform_params(pts: np.ndarray) -> tuple[int, int, np.ndarray]:
        w = max(
            int(np.linalg.norm(pts[0] - pts[1])), int(np.linalg.norm(pts[2] - pts[3]))
        )
        h = max(
            int(np.linalg.norm(pts[1] - pts[2])), int(np.linalg.norm(pts[3] - pts[0]))
        )

        dst_points = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        return w, h, cv2.getPerspectiveTransform(pts, dst_points)

    @classmethod
    def crop_image_by_polygon(
        cls, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img_np = np.array(image.convert("RGB"))
        pts = (
            np.array(polygon, dtype=np.float32)
            if len(polygon) == 4
            else cls._polygon_to_quadrilateral(polygon)
        )

        w, h, persp_M = cls._get_transform_params(pts)
        return Image.fromarray(cv2.warpPerspective(img_np, persp_M, (w, h)))

    @classmethod
    def cut_out_image_by_polygon(
        cls, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img_np = np.array(image.convert("RGBA"))
        pts = cls._polygon_to_quadrilateral(polygon)
        w, h, persp_M = cls._get_transform_params(pts)

        warped_img = cv2.warpPerspective(img_np, persp_M, (w, h))

        poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        tr_pts = cv2.perspectiveTransform(poly_np, persp_M).astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [tr_pts], 255)

        # Apply mask directly to Alpha channel
        warped_img[:, :, 3] = cv2.bitwise_and(warped_img[:, :, 3], mask)
        return Image.fromarray(warped_img, mode="RGBA")


class SegmentProcessor:
    """Processor for image segment background removal."""

    @classmethod
    def remove_color(cls, image: Image.Image) -> Image.Image:
        """Remove blue color regions from an image using inpainting."""
        img_np = np.array(image)
        has_alpha = img_np.shape[-1] == 4

        # cv2.inpaint requires a 3-channel image, so we separate RGB from Alpha
        rgb_img = img_np[:, :, :3] if has_alpha else img_np

        # Convert RGB to HSV (bounds are the same as BGR, just mapping differently)
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, DEFAULT_LOWER_BLUE, DEFAULT_UPPER_BLUE)

        kernel = np.ones(
            (DEFAULT_BORDER_MULTIPLIER, DEFAULT_BORDER_MULTIPLIER), np.uint8
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint the RGB channels
        inpainted_rgb = cv2.inpaint(rgb_img, mask, 3, cv2.INPAINT_TELEA)

        # Recombine with alpha if it existed
        if has_alpha:
            img_np[:, :, :3] = inpainted_rgb
            return Image.fromarray(img_np, mode="RGBA")

        return Image.fromarray(inpainted_rgb, mode="RGB")

    @classmethod
    def remove_bg_threshold(
        cls, image: Image.Image, threshold: int = DEFAULT_BG_THRESHOLD
    ) -> Image.Image:
        """Remove background using white-pixel threshold."""
        if not (0 <= threshold <= 255):
            raise ValueError(
                f"Threshold must be an integer in range 0-255, got {threshold}"
            )

        # Forcing RGBA handles shape, dtype, and guarantees an alpha channel exists
        img_np = np.array(image.convert("RGBA"))

        # Calculate brightness mask based on the RGB channels
        gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        _, new_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Combine new mask with existing alpha channel (so we don't overwrite previous cropping)
        img_np[:, :, 3] = cv2.bitwise_and(img_np[:, :, 3], new_mask)

        return Image.fromarray(img_np, mode="RGBA")
