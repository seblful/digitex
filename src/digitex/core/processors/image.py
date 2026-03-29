"""Image processing utilities."""

import logging
import math

import cv2
import numpy as np
from deskew import determine_skew
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BG_THRESHOLD = 200
DEFAULT_SATURATION_THRESHOLD = 80
DEFAULT_DILATE_ITERATIONS = 2
DEFAULT_GAMMA = 0.6
DEFAULT_SKEW_MAX_DIM = 200


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
    def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
        old_height, old_width = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(
            np.cos(angle_radian) * old_width
        )
        height = abs(np.sin(angle_radian) * old_width) + abs(
            np.cos(angle_radian) * old_height
        )

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[0, 2] += (width - old_width) / 2
        rot_mat[1, 2] += (height - old_height) / 2
        return cv2.warpAffine(
            image,
            rot_mat,
            (int(round(width)), int(round(height))),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

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

        img = np.array(image.convert("RGB"))
        pts = (
            np.array(polygon, dtype=np.float32)
            if len(polygon) == 4
            else cls._polygon_to_quadrilateral(polygon)
        )

        w, h, persp_M = cls._get_transform_params(pts)
        return Image.fromarray(cv2.warpPerspective(img, persp_M, (w, h)))

    @staticmethod
    def _detect_skew_angle(img: np.ndarray) -> float | None:
        grayscale = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        alpha = img[:, :, 3]
        if not np.all(alpha == 255):
            mask = alpha.astype(np.float32) / 255.0
            white_bg = np.full_like(grayscale, 255, dtype=np.float32)
            grayscale = (
                grayscale.astype(np.float32) * mask + white_bg * (1.0 - mask)
            ).astype(np.uint8)

        h, w = grayscale.shape
        if max(h, w) > DEFAULT_SKEW_MAX_DIM:
            scale = DEFAULT_SKEW_MAX_DIM / max(h, w)
            grayscale = cv2.resize(
                grayscale,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        _, thresh = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        return determine_skew(thresh, sigma=0.0, num_peaks=20, min_deviation=0.01)

    @classmethod
    def _deskew(cls, img: np.ndarray) -> np.ndarray:
        angle = cls._detect_skew_angle(img)
        if angle is not None and angle != 0.0:
            logger.debug(f"Detected skew angle: {angle:.2f} degrees, applying rotation")
            return cls._rotate(img, angle)
        return img

    @classmethod
    def cut_out_image_by_polygon(
        cls, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img = np.array(image.convert("RGBA"))
        pts = cls._polygon_to_quadrilateral(polygon)
        w, h, persp_M = cls._get_transform_params(pts)

        warped_img = cv2.warpPerspective(img, persp_M, (w, h))

        poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        tr_pts = cv2.perspectiveTransform(poly_np, persp_M).astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [tr_pts], 255)
        warped_img[:, :, 3] = cv2.bitwise_and(warped_img[:, :, 3], mask)

        warped_img = cls._deskew(warped_img)

        return Image.fromarray(warped_img, mode="RGBA")


class SegmentProcessor:
    """Processor for image segment background removal."""

    @staticmethod
    def remove_color(
        img: np.ndarray,
        saturation_threshold: int = DEFAULT_SATURATION_THRESHOLD,
        dilate_iterations: int = DEFAULT_DILATE_ITERATIONS,
    ) -> np.ndarray:
        """Remove all colored pixels, keeping only grayscale (gray/black/white).

        Args:
            img: Input RGBA numpy array.
            saturation_threshold: Maximum saturation value to consider grayscale (0-255).
            dilate_iterations: Number of dilation iterations for color mask.

        Returns:
            RGBA numpy array with colored pixels made transparent.
        """
        hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
        color_mask = hsv[:, :, 1] > saturation_threshold

        if dilate_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), kernel, iterations=dilate_iterations
            ).astype(bool)

        img = img.copy()
        img[color_mask, 3] = 0
        return img

    @staticmethod
    def remove_bg(img: np.ndarray, threshold: int = DEFAULT_BG_THRESHOLD) -> np.ndarray:
        """Remove background using white-pixel threshold.

        Args:
            img: Input RGBA numpy array.
            threshold: Brightness threshold (0-255).

        Returns:
            RGBA numpy array with bright pixels made transparent.
        """
        if not (0 <= threshold <= 255):
            raise ValueError(
                f"Threshold must be an integer in range 0-255, got {threshold}"
            )

        img = img.copy()
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        _, new_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        img[:, :, 3] = cv2.bitwise_and(img[:, :, 3], new_mask)
        return img

    @staticmethod
    def increase_darkness(img: np.ndarray, gamma: float = DEFAULT_GAMMA) -> np.ndarray:
        """Apply gamma correction to darken mid-tones and increase contrast.

        Args:
            img: Input RGBA numpy array.
            gamma: Gamma value. Values < 1.0 darken, > 1.0 lighten.

        Returns:
            RGBA numpy array with gamma correction applied.

        Raises:
            ValueError: If gamma is not positive.
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        img = img.copy()
        rgb = img[:, :, :3].astype(np.float32) / 255.0
        corrected = np.power(rgb, 1.0 / gamma) * 255.0
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        img[:, :, :3] = corrected
        return img

    @staticmethod
    def add_white_background(img: np.ndarray) -> np.ndarray:
        """Composite RGBA image onto white background.

        Args:
            img: Input RGBA numpy array.

        Returns:
            RGB numpy array with white background replacing transparency.
        """
        alpha = img[:, :, 3:4] / 255.0
        white_bg = np.ones_like(img[:, :, :3]) * 255
        rgb = img[:, :, :3] * alpha + white_bg * (1 - alpha)
        return rgb.astype(np.uint8)

    @staticmethod
    def process(
        image: Image.Image,
        saturation_threshold: int = DEFAULT_SATURATION_THRESHOLD,
        bg_threshold: int = DEFAULT_BG_THRESHOLD,
        gamma: float = DEFAULT_GAMMA,
    ) -> Image.Image:
        """Apply color removal, background removal, darkness increase, and white background.

        Args:
            image: Input PIL Image.
            saturation_threshold: Max saturation to keep (higher = removes more colors).
            bg_threshold: Brightness threshold for background removal (higher = keeps more).
            gamma: Gamma for darkness adjustment. < 1.0 darkens, 1.0 = no change.

        Returns:
            RGB image suitable for JPG format.
        """
        img = np.array(image.convert("RGBA"))
        img = SegmentProcessor.remove_color(img, saturation_threshold)
        img = SegmentProcessor.remove_bg(img, bg_threshold)
        img = SegmentProcessor.increase_darkness(img, gamma)
        rgb = SegmentProcessor.add_white_background(img)
        return Image.fromarray(rgb, mode="RGB")
