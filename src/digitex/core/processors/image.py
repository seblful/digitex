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


class ImageCropper:
    """Processor for image cropping operations using perspective transformations."""

    @staticmethod
    def _get_perspective_matrix(
        pts: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(pts, dst_points)

    @staticmethod
    def _perspective_transform(
        polygon: list[tuple[int, int]],
        persp_M: np.ndarray,
    ) -> np.ndarray:
        poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        warped_poly = cv2.perspectiveTransform(poly_np, persp_M).astype(np.int32)
        return warped_poly

    @staticmethod
    def _warp_perspective(
        img: np.ndarray,
        persp_M: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        return cv2.warpPerspective(img, persp_M, (width, height))

    @staticmethod
    def _get_quadrilateral_size(pts: np.ndarray) -> tuple[int, int]:
        width_a = np.linalg.norm(pts[0] - pts[1])
        width_b = np.linalg.norm(pts[2] - pts[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(pts[1] - pts[2])
        height_b = np.linalg.norm(pts[3] - pts[0])
        max_height = max(int(height_a), int(height_b))

        return max_width, max_height

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _polygon_to_quadrilateral(
        polygon: list[tuple[int, int]],
        max_angle: float = 4.0,
    ) -> np.ndarray:
        pts = np.array(polygon, dtype=np.int32)

        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        angle_delta = abs(min(angle - 0, 90 - angle))

        if angle_delta > max_angle:
            x, y, w, h = cv2.boundingRect(pts)
            bbox = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.float32,
            )
        else:
            bbox = cv2.boxPoints(rect)

        return ImageCropper._order_points(bbox)

    @staticmethod
    def _paste_on_white_background(
        img: np.ndarray,
        pts: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        result = cv2.bitwise_and(img, img, mask=mask)

        white_bg = np.ones_like(result) * 255
        mask_inv = cv2.bitwise_not(mask)
        white_masked = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
        return cv2.add(result, white_masked)

    def crop_image_by_polygon(
        self,
        image: Image.Image,
        polygon: list[tuple[int, int]],
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if len(polygon) == 4:
            pts = np.array(polygon, dtype=np.float32)
        else:
            pts = self._polygon_to_quadrilateral(polygon)

        width, height = self._get_quadrilateral_size(pts)
        persp_M = self._get_perspective_matrix(pts, width, height)
        cropped_img = self._warp_perspective(img, persp_M, int(width), int(height))

        return Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    def cut_out_image_by_polygon(
        self,
        image: Image.Image,
        polygon: list[tuple[int, int]],
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        pts = self._polygon_to_quadrilateral(polygon)
        width, height = self._get_quadrilateral_size(pts)
        persp_M = self._get_perspective_matrix(pts, width, height)
        warped_img = self._warp_perspective(img, persp_M, width, height)

        tr_pts = self._perspective_transform(polygon, persp_M)
        bg_img = self._paste_on_white_background(warped_img, tr_pts, width, height)

        return Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))


class SegmentProcessor:
    """Processor for image segment background removal."""

    def remove_color(self, img: np.ndarray) -> np.ndarray:
        """Remove blue color regions from an image using inpainting.

        Args:
            img: Input image in BGR format.

        Returns:
            Image with blue color regions removed.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, DEFAULT_LOWER_BLUE, DEFAULT_UPPER_BLUE)
        kernel = np.ones((DEFAULT_BORDER_MULTIPLIER, DEFAULT_BORDER_MULTIPLIER), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    def remove_bg_threshold(self, segment_bgr: np.ndarray, threshold: int = DEFAULT_BG_THRESHOLD) -> np.ndarray:
        """Remove background using white-pixel threshold.

        Args:
            segment_bgr: Input segment in BGR format.
            threshold: Brightness threshold (0-255). Pixels brighter than threshold become transparent.

        Returns:
            4-channel BGRA image with transparent background.
        """
        if segment_bgr.size == 0:
            raise ValueError("Image is empty")
        if segment_bgr.dtype != np.uint8:
            raise ValueError(f"Image must have dtype uint8, got {segment_bgr.dtype}")
        if len(segment_bgr.shape) != 3 or segment_bgr.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels, got shape {segment_bgr.shape}")
        if not isinstance(threshold, int) or not (0 <= threshold <= 255):
            raise ValueError(f"Threshold must be an integer in range 0-255, got {threshold}")

        gray = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(segment_bgr)
        return cv2.merge([b, g, r, binary_mask])


SegmentHandler = SegmentProcessor
