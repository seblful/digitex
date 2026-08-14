"""Image processing utilities."""

import math

import cv2
import numpy as np
from PIL import Image, ImageOps

from digitex.domain.entities import PixelPolygon


def resize_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    return ImageOps.contain(
        image, (max_width, max_height), method=Image.Resampling.BILINEAR
    )


# --- background flatten ---


def add_white_background(image: Image.Image) -> Image.Image:
    """Composite an image onto a white background.

    A crop's polygon mask leaves everything outside the region transparent, and
    JPEG has no alpha channel — the transparency must be flattened onto white
    before saving.

    Args:
        image: Input PIL Image.

    Returns:
        RGB image suitable for JPG format.
    """
    img = np.array(image.convert("RGBA"))
    alpha = img[:, :, 3:4] / 255.0
    white_bg = np.ones_like(img[:, :, :3]) * 255
    rgb = img[:, :, :3] * alpha + white_bg * (1 - alpha)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


# --- image cropping helpers ---


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotate by *angle* degrees counterclockwise, growing the canvas to fit."""
    return Image.fromarray(_rotate(np.array(image), angle))


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    rad = math.radians(angle)
    sin_a, cos_a = math.sin(rad), math.cos(rad)
    new_w = round(abs(sin_a) * h + abs(cos_a) * w)
    new_h = round(abs(sin_a) * w + abs(cos_a) * h)

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


def _polygon_to_quad(polygon: PixelPolygon, max_angle: float = 4.0) -> np.ndarray:
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
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return w, h, cv2.getPerspectiveTransform(pts, dst)


def cut_out_image_by_polygon(image: Image.Image, polygon: PixelPolygon) -> Image.Image:
    """Cut *polygon* out of *image*, deskewed by a perspective transform.

    Everything outside the polygon is left transparent in the returned RGBA
    crop — flatten it (:func:`add_white_background`) before saving to a
    format with no alpha channel.
    """
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

    return Image.fromarray(warped, mode="RGBA")
