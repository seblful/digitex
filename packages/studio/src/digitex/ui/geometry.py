"""Canvas arithmetic for the review window.

Numbers in, numbers out. These are the rules a mouse gesture is read by: where
a click lands, how far the view has to move so a zoom magnifies the pixel under
the cursor instead of scrolling away from it, which edge a new vertex belongs
on. They sit outside the window because a Tk canvas cannot be asked what it
thinks a coordinate means — asserting any of this through a widget needs a
display, and asserting it here needs nothing.

Only arithmetic: no widget, no image. Cutting a polygon out of a page is
:mod:`digitex.imaging`'s job, and what a polygon *means* is
:class:`~digitex.ui.edits.PageEdits`'.
"""

from __future__ import annotations

Point = tuple[int, int]


def fit_scale(image: tuple[int, int], canvas: tuple[int, int]) -> float:
    """The largest scale at which *image* still fits inside *canvas*.

    Whichever side runs out first decides. The floors of 1 keep a zero-sized
    image — a page not yet loaded — from dividing by nothing.
    """
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

    *origin* is the view's top-left in canvas coordinates, *view* its size and
    *page* the image's size at the current scale. The result is the overlap of
    the two rectangles, or None when there is none — which happens mid-scroll
    over a page smaller than the canvas it sits on.
    """
    left = max(origin[0], 0.0)
    top = max(origin[1], 0.0)
    right = min(origin[0] + view[0], page[0])
    bottom = min(origin[1] + view[1], page[1])
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def anchored_origin(origin: float, pointer: float, ratio: float) -> float:
    """Where the view must start after a zoom to hold a point under the cursor.

    All three arguments are in canvas pixels at the *old* scale: *origin* is the
    view's left (or top) edge and *pointer* the cursor. Scaling by *ratio* moves
    the pixel under the cursor to ``pointer * ratio``; leaving it on the same
    spot of the screen means leaving its distance from the edge alone.
    """
    return pointer * ratio - (pointer - origin)


def scroll_fraction(origin: float, extent: float) -> float:
    """*origin* as the 0-1 fraction ``xview_moveto`` wants, clamped to the page."""
    if extent <= 0:
        return 0.0
    return min(max(origin / extent, 0.0), 1.0)


def distance_to_segment(point: Point, start: Point, end: Point) -> float:
    """Squared distance from *point* to the segment *start*-*end*.

    Squared, because every caller only ever compares one against another and a
    square root would change no ordering.
    """
    px, py = point
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    span = dx * dx + dy * dy
    if span == 0:
        # The two ends coincide — a polygon can carry a doubled vertex.
        return (px - x0) ** 2 + (py - y0) ** 2
    # How far along the segment the perpendicular foot falls, held inside it so
    # a point beyond either end measures against that end.
    along = min(max(((px - x0) * dx + (py - y0) * dy) / span, 0.0), 1.0)
    return (px - x0 - along * dx) ** 2 + (py - y0 - along * dy) ** 2


def nearest_edge(polygon: list[Point], point: Point) -> int:
    """Index of the vertex *point* should be inserted after.

    The closing edge back to the first vertex counts like any other, so a click
    below the last corner of a box lands where it looks like it should.

    Raises:
        ValueError: If *polygon* has no points.
    """
    if not polygon:
        raise ValueError("An empty polygon has no edges")
    last = len(polygon)
    return min(
        range(last),
        key=lambda at: distance_to_segment(
            point, polygon[at], polygon[(at + 1) % last]
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
    return min(polygon, key=lambda point: (point[1], point[0]))
