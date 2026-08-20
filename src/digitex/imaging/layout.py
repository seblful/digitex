"""Stacking the pieces of one question into the image it is saved as.

A question printed across a page break is two pictures of one question.
The layout is computed separately from the drawing so the join editor can
show a reviewer exactly what the saved file will look like — same walk,
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

    Each piece lands below the one before it with their left edges together and
    *gap* pixels between — plus its own nudge from *offsets*, which is how the
    two halves of a question printed across a page break are lined up: right
    and down, negative to go left or to close the seam. A nudge carries the
    pieces below it along, so lining up one seam never disturbs another.

    ``offsets[0]`` is ignored, and a missing offset is no nudge: the first piece
    has nothing to sit against.

    Args:
        sizes: Each piece's (width, height), in reading order.
        gap: White band between two pieces, in pixels.
        offsets: Each piece's (dx, dy) against the piece above it.

    Returns:
        The (width, height) the stack needs, and each piece's top-left in it —
        normalized, so a piece nudged left of the first still lands at x >= 0.
    """
    x = y = 0
    positions: list[tuple[int, int]] = []
    for index, (_, _) in enumerate(sizes):
        if index:
            dx, dy = offsets[index] if index < len(offsets) else (0, 0)
            x += dx
            y += sizes[index - 1][1] + gap + dy
        positions.append((x, y))

    left = min((x for x, _ in positions), default=0)
    top = min((y for _, y in positions), default=0)
    positions = [(x - left, y - top) for x, y in positions]
    corners = [
        (x + width, y + height)
        for (x, y), (width, height) in zip(positions, sizes, strict=True)
    ]
    return (
        max((right for right, _ in corners), default=0),
        max((bottom for _, bottom in corners), default=0),
    ), positions


def stack_vertically(
    images: Sequence[Image.Image],
    gap: int = 0,
    offsets: Sequence[tuple[int, int]] = (),
) -> Image.Image:
    """Stack *images* top to bottom on white, laid out by :func:`stacked_layout`.

    How a question split across a page break is put back together: its pieces
    are cut from different pages and come back at different widths, so the
    canvas grows to hold them rather than any of them being scaled — the text
    keeps the size it was scanned at, and a reader cannot tell which piece came
    from where.

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
