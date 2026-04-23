"""Image processing utilities."""

import math

import cv2
import numpy as np
import structlog
from deskew import determine_skew
from PIL import Image, ImageOps

logger = structlog.get_logger()

DEFAULT_BG_THRESHOLD = 200
DEFAULT_SATURATION_THRESHOLD = 100
DEFAULT_DILATE_ITERATIONS = 2
DEFAULT_GAMMA = 0.6
DEFAULT_SKEW_MAX_DIM = 400


def resize_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    return ImageOps.contain(
        image, (max_width, max_height), method=Image.Resampling.BILINEAR
    )


# --- segment processing ---

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


# --- image cropping helpers ---

def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    rad = math.radians(angle)
    sin_a, cos_a = math.sin(rad), math.cos(rad)
    new_w = int(round(abs(sin_a) * h + abs(cos_a) * w))
    new_h = int(round(abs(sin_a) * w + abs(cos_a) * h))

    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    mat[0, 2] += (new_w - w) / 2
    mat[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(
        img,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.empty((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _polygon_to_quad(
    polygon: list[tuple[int, int]], max_angle: float = 4.0
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

    return _order_quad_points(bbox)


def _perspective_transform(pts: np.ndarray) -> tuple[int, int, np.ndarray]:
    w = max(
        int(np.linalg.norm(pts[0] - pts[1])),
        int(np.linalg.norm(pts[2] - pts[3])),
    )
    h = max(
        int(np.linalg.norm(pts[1] - pts[2])),
        int(np.linalg.norm(pts[3] - pts[0])),
    )
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    return w, h, cv2.getPerspectiveTransform(pts, dst)


def _prepare_for_skew_detection(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    alpha = img[:, :, 3]
    if not np.all(alpha == 255):
        a = alpha.astype(np.float32) / 255.0
        gray = (gray.astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)

    h, w = gray.shape
    if max(h, w) > DEFAULT_SKEW_MAX_DIM:
        scale = DEFAULT_SKEW_MAX_DIM / max(h, w)
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


# --- public classes ---

class ImageCropper:
    """Processor for image cropping operations using perspective transformations."""

    @staticmethod
    def cut_out_image_by_polygon(
        image: Image.Image, polygon: list[tuple[int, int]]
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img = np.array(image.convert("RGBA"))
        pts = _polygon_to_quad(polygon)
        w, h, M = _perspective_transform(pts)

        warped = cv2.warpPerspective(img, M, (w, h))

        poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        tr_pts = cv2.perspectiveTransform(poly_np, M).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [tr_pts], 255)
        warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], mask)

        thresh = _prepare_for_skew_detection(warped)
        angle = determine_skew(thresh, sigma=0.0, num_peaks=20, min_deviation=0.01)
        if angle is not None and angle != 0.0:
            logger.debug("Detected skew angle, applying rotation", angle=angle)
            warped = _rotate(warped, angle)

        return Image.fromarray(warped, mode="RGBA")


class SegmentProcessor:
    """Processor for image segment background removal."""

    # Expose module-level functions as static methods for backward compatibility
    remove_color = staticmethod(remove_color)
    remove_bg = staticmethod(remove_bg)
    increase_darkness = staticmethod(increase_darkness)
    add_white_background = staticmethod(add_white_background)

    def process(
        self,
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
        img = remove_color(img, saturation_threshold)
        img = remove_bg(img, bg_threshold)
        img = increase_darkness(img, gamma)
        rgb = add_white_background(img)
        return Image.fromarray(rgb, mode="RGB")
