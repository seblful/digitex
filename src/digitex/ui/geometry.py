"""Canvas arithmetic for the review window.

Pure: numbers in, numbers out. The rules that decide where a click lands, how
far a zoom moves the page under the cursor and which edge a new point joins
live here rather than in the window, so they can be exercised without a
display — a Tk widget cannot be asked what it thinks a coordinate means.
"""

from __future__ import annotations

Point = tuple[int, int]


def fit_scale(image: tuple[int, int], canvas: tuple[int, int]) -> float:
    """The largest scale at which *image* still fits inside *canvas*."""
    width, height = canvas
    return min(width / max(image[0], 1), height / max(image[1], 1))


def clamp_scale(scale: float, min_scale: float, max_scale: float) -> float:
    """Hold *scale* between a page still readable and a pixel still meaningful."""
    return max(min_scale, min(scale, max_scale))


def visible_box(
    origin: tuple[float, float],
    view: tuple[int, int],
    page: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """The part of the rendered page inside the viewport, in canvas pixels.

    *origin* is the view's top-left in canvas coordinates, *view* its size,
    *page* the image's size at the current scale. None when the two do not
    overlap at all, which happens mid-scroll on a page smaller than its canvas.
    """
    left = max(0.0, origin[0])
    top = max(0.0, origin[1])
    right = min(page[0], origin[0] + view[0])
    bottom = min(page[1], origin[1] + view[1])
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def anchored_origin(origin: float, pointer: float, ratio: float) -> float:
    """Where the view must start after a zoom to hold a point under the cursor.

    All three arguments are in canvas pixels at the *old* scale: *origin* is
    the view's left (or top) edge, *pointer* the cursor. Scaling by *ratio*
    moves the pixel under the cursor to ``pointer * ratio``; keeping it on the
    same spot of the screen means keeping its distance from the edge.
    """
    return pointer * ratio - (pointer - origin)


def scroll_fraction(origin: float, extent: float) -> float:
    """*origin* as the 0-1 fraction ``xview_moveto`` wants, clamped to the page."""
    if extent <= 0:
        return 0.0
    return max(0.0, min(1.0, origin / extent))


def distance_to_segment(point: Point, start: Point, end: Point) -> float:
    """Squared distance from *point* to the segment *start*-*end*."""
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    if length == 0:
        return (px - x0) ** 2 + (py - y0) ** 2
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length))
    return (px - x0 - t * dx) ** 2 + (py - y0 - t * dy) ** 2


def nearest_edge(polygon: list[Point], point: Point) -> int:
    """Index of the vertex *point* should be inserted after.

    Raises:
        ValueError: If *polygon* has no points.
    """
    if not polygon:
        raise ValueError("An empty polygon has no edges")
    return min(
        range(len(polygon)),
        key=lambda i: distance_to_segment(
            point, polygon[i], polygon[(i + 1) % len(polygon)]
        ),
    )


def moved(polygon: list[Point], dx: int, dy: int) -> list[Point]:
    """*polygon* shifted by (dx, dy)."""
    return [(x + dx, y + dy) for x, y in polygon]


def bounds(polygon: list[Point]) -> tuple[int, int, int, int]:
    """The (left, top, right, bottom) box around *polygon*.

    Raises:
        ValueError: If *polygon* has no points.
    """
    if not polygon:
        raise ValueError("An empty polygon has no bounds")
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def top_left(polygon: list[Point]) -> Point:
    """The point a caption hangs off: topmost, and leftmost among those."""
    return min(polygon, key=lambda p: (p[1], p[0]))
