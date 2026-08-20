"""Lifting one region off a page as its own straightened picture.

The detector hands back a polygon, not a rectangle: the page sat crooked on the
glass, and the outline was traced around content that is not rectangular
either. Both are undone here — the polygon is squared onto a quadrilateral and
unwarped onto it, and whatever the outline did not cover is left transparent
rather than filled, so the caller decides what the surroundings become.

Free rotation lives here too, being the same kind of work: resampling pixels
through a matrix onto a canvas sized to hold the result.
"""

import math

import cv2
import numpy as np
from PIL import Image

from digitex.domain.entities import PixelPolygon


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotate by *angle* degrees counterclockwise, growing the canvas to fit."""
    pixels = np.array(image)
    height, width = pixels.shape[:2]

    # A rotated rectangle needs more room than it started with. Keeping the
    # source's dimensions would slice the corners off the very crop the
    # rotation was called to straighten.
    radians = math.radians(angle)
    sin, cos = abs(math.sin(radians)), abs(math.cos(radians))
    grown_width = round(sin * height + cos * width)
    grown_height = round(sin * width + cos * height)

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    # The matrix turns about the source's centre, which is off-centre on the
    # grown canvas — this puts it back in the middle.
    matrix[0, 2] += (grown_width - width) / 2
    matrix[1, 2] += (grown_height - height) / 2

    rotated = cv2.warpAffine(
        pixels,
        matrix,
        (grown_width, grown_height),
        flags=cv2.INTER_LINEAR,
        # Replicating the edge rather than filling black keeps the paper's own
        # tone in the wedges the rotation opens along the sides.
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated)


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Four corners as top-left, top-right, bottom-right, bottom-left.

    A perspective transform pairs corner with corner, so source and
    destination have to agree on which corner is which. x + y is least at the
    top-left and greatest at the bottom-right; y - x tells the remaining two
    apart, being least at the top-right.
    """
    diagonal = pts.sum(axis=1)
    antidiagonal = pts[:, 1] - pts[:, 0]
    # float32 because getPerspectiveTransform accepts nothing else.
    return np.array(
        [
            pts[np.argmin(diagonal)],
            pts[np.argmin(antidiagonal)],
            pts[np.argmax(diagonal)],
            pts[np.argmax(antidiagonal)],
        ],
        dtype=np.float32,
    )


def _polygon_to_quad(polygon: PixelPolygon, max_angle: float = 4.0) -> np.ndarray:
    """The four corners *polygon* is unwarped onto, in corner order.

    The tightest rotated rectangle follows the region's own edges, so
    unwarping onto it is what takes a crooked scan's tilt out. That is only
    worth doing for a small tilt: a large one means the outline was never a
    rotated rectangle — a loosely traced figure, a region straddling a column
    break — and forcing one onto it shears the content. Those keep the
    axis-aligned bounding box, which crops without resampling anything.
    """
    points = np.array(polygon, dtype=np.int32)
    tightest = cv2.minAreaRect(points)

    if abs(min(tightest[2], 90 - tightest[2])) > max_angle:
        x, y, width, height = cv2.boundingRect(points)
        corners = np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
    else:
        corners = cv2.boxPoints(tightest)

    return _order_quad_points(corners)


def _perspective_transform(pts: np.ndarray) -> tuple[int, int, np.ndarray]:
    """The size an ordered quad unwarps to, and the matrix that takes it there.

    Opposite sides of the quad disagree — that disagreement *is* the
    perspective. Each output dimension takes the longer of its pair, so the
    unwarp stretches the short side up to the long one rather than squeezing
    the long one down, which would throw detail the scan captured away.
    """
    top = int(np.linalg.norm(pts[0] - pts[1]))
    right = int(np.linalg.norm(pts[1] - pts[2]))
    bottom = int(np.linalg.norm(pts[2] - pts[3]))
    left = int(np.linalg.norm(pts[3] - pts[0]))
    width, height = max(top, bottom), max(right, left)

    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return width, height, cv2.getPerspectiveTransform(pts, corners)


def _polygon_mask(
    polygon: PixelPolygon, transform: np.ndarray, width: int, height: int
) -> np.ndarray:
    """*polygon* carried through *transform*, filled in as a 0/255 stencil."""
    corners = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    unwarped = cv2.perspectiveTransform(corners, transform).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [unwarped], 255)
    return mask


def cut_out_image_by_polygon(image: Image.Image, polygon: PixelPolygon) -> Image.Image:
    """Cut *polygon* out of *image*, deskewed by a perspective transform.

    Everything outside the polygon is left transparent in the returned RGBA
    crop — flatten it (:func:`add_white_background`) before saving to a
    format with no alpha channel.

    Raises:
        ValueError: If *polygon* has fewer than four points.
    """
    if len(polygon) < 4:
        raise ValueError("Polygon must have 4 or more points")

    quad = _polygon_to_quad(polygon)
    width, height, transform = _perspective_transform(quad)
    warped = cv2.warpPerspective(
        np.array(image.convert("RGBA")), transform, (width, height)
    )

    # The quad is only the outline's hull, so the warp hands back whatever sits
    # in the outline's concavities as well. Punching the polygon out of the
    # alpha channel is what makes the crop follow the traced shape rather than
    # the box around it.
    mask = _polygon_mask(polygon, transform, width, height)
    warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], mask)
    return Image.fromarray(warped, mode="RGBA")
