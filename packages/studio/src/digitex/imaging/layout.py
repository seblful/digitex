"""Putting the pieces of one question back together as a single picture.

A question printed across a page break arrives as two crops off two pages.
Where each piece lands is worked out by :func:`stacked_layout` and drawn by
:func:`stack_vertically` — kept apart so the join editor can show a reviewer
the layout the saved file will have without rendering it first: same walk,
same offsets, same white band between pieces.
"""

from collections.abc import Sequence

from PIL import Image


def stacked_layout(
    sizes: Sequence[tuple[int, int]],
    gap: int = 0,
    offsets: Sequence[tuple[int, int]] = (),
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Where each piece of a stack sits, and how big the stack comes out.

    Pieces run top to bottom with their left edges aligned and *gap* pixels
    between them, each carrying its own nudge from *offsets* — right and down,
    negative to go left or to close the seam. That nudge is how the two halves
    of a question printed across a page break are lined up. It carries the
    pieces below it along, so lining up one seam leaves the seams above it
    where they were.

    ``offsets[0]`` is ignored and a missing offset counts as no nudge: the
    first piece anchors the stack and has nothing to sit against.

    Args:
        sizes: Each piece's (width, height), in reading order.
        gap: White band between two pieces, in pixels.
        offsets: Each piece's (dx, dy) against the piece above it.

    Returns:
        The (width, height) the stack needs, and each piece's top-left in it —
        normalized, so a piece nudged left of the first still lands at x >= 0.
    """
    x = y = 0
    placements: list[tuple[int, int]] = []
    for index, (_, height) in enumerate(sizes):
        if index:
            dx, dy = offsets[index] if index < len(offsets) else (0, 0)
            x += dx
            y += gap + dy
        placements.append((x, y))
        # Clearing this piece is what puts the next one below it — folded into
        # the walk so a nudge lands on top of the accumulated height rather
        # than replacing it.
        y += height

    # A leftward or upward nudge can put a piece outside the first one's
    # quadrant, where a paste would silently drop it. Shifting everything back
    # into positive territory keeps the whole stack on the canvas.
    left = min((x for x, _ in placements), default=0)
    top = min((y for _, y in placements), default=0)
    positions = [(x - left, y - top) for x, y in placements]

    far_corners = [
        (x + width, y + height)
        for (x, y), (width, height) in zip(positions, sizes, strict=True)
    ]
    return (
        max((right for right, _ in far_corners), default=0),
        max((bottom for _, bottom in far_corners), default=0),
    ), positions


def stack_vertically(
    images: Sequence[Image.Image],
    gap: int = 0,
    offsets: Sequence[tuple[int, int]] = (),
) -> Image.Image:
    """Stack *images* top to bottom on white, laid out by :func:`stacked_layout`.

    How a question split across a page break is made whole again. Its pieces
    were cut from different pages and come back at different widths, so the
    canvas grows to hold the widest rather than any piece being scaled — the
    text keeps the size it was scanned at, and nothing in the result says which
    piece came from where.

    Args:
        images: The pieces, in reading order. At least one.
        gap: White band between two pieces, in pixels.
        offsets: Each piece's nudge against the piece above it.

    Returns:
        RGB image holding every piece. The single piece itself when there is
        only one — nothing to stack it against, and nothing to pad it to.

    Raises:
        ValueError: If *images* is empty.
    """
    if not images:
        raise ValueError("Nothing to stack")
    if len(images) == 1:
        return images[0]

    size, positions = stacked_layout([image.size for image in images], gap, offsets)
    stacked = Image.new("RGB", size, "white")
    for image, position in zip(images, positions, strict=True):
        stacked.paste(image.convert("RGB"), position)
    return stacked
