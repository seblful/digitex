"""Averaging scanner grain out of a page without touching its strokes.

A bilateral filter. Every pixel is averaged with the square of neighbours
around it, but each neighbour's vote is scaled by how far its tone sits from
the centre's — so grain inside the paper and grain inside a letter both
average away, while the step between them is a tonal distance the weighting
refuses to cross. What a plain blur would smear, this leaves as an edge.

The weights are NAPS2's, computed on the fly rather than tabulated.
"""

import math

import numpy as np
from PIL import Image

# The side of the square of neighbours each pixel is averaged against.
_FILTER_SIZE = 15

_FILTER_HALF = _FILTER_SIZE // 2

# The gray distance at which a neighbour's vote reaches zero. Two tones this
# far apart are held to be different things, not one thing plus noise.
_COLOR_DIST_MAX = 32

# Vectorising the filter means sweeping the image once per neighbour, so a
# whole page would cross the memory bus 225 times. Working a tile at a time
# keeps the sweep inside cache instead. Purely a matter of speed — the output
# does not depend on it.
_TILE_SIZE = 256


def _spatial_weights() -> np.ndarray:
    """Neighbour weight by distance: full at the centre, nothing at a corner."""
    offsets = np.arange(_FILTER_SIZE) - _FILTER_HALF
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    furthest = math.sqrt(2 * _FILTER_HALF**2)
    return ((1 - np.hypot(dx, dy) / furthest) * 256).astype(np.int32)


def _filter_block(
    padded: np.ndarray,
    block: np.ndarray,
    top: int,
    left: int,
    spatial: np.ndarray,
) -> np.ndarray:
    """Weigh every pixel of *block* against its square of neighbours.

    *padded* holds the whole page with a half-window margin, so *top* and
    *left* — the block's origin on the page — index straight into it.

    The centre tap is always one of the 225, and it always weighs the full
    ``_COLOR_DIST_MAX * 256`` against itself, so the divisor below cannot
    reach zero.
    """
    height, width = block.shape
    centre = block.astype(np.int16)
    weight_sum = np.zeros(block.shape, dtype=np.int32)
    tone_sum = np.zeros(block.shape, dtype=np.int32)

    for row in range(_FILTER_SIZE):
        for column in range(_FILTER_SIZE):
            nearness = int(spatial[row, column])
            # The four corners of the square fall exactly on the radius where
            # the distance weight hits zero.
            if not nearness:
                continue
            neighbour = padded[
                top + row : top + row + height, left + column : left + column + width
            ]
            # NAPS2's tonal weight, worked out rather than looked up: a
            # neighbour counts for less the further off its tone is, and for
            # nothing at all past _COLOR_DIST_MAX. Arithmetic beats a lookup
            # here because the table is too big to stay in cache — and it is
            # written in place, since a fresh temporary on each of 225 taps
            # would undo what the tiling bought.
            likeness = np.abs(centre - neighbour)
            np.subtract(_COLOR_DIST_MAX, likeness, out=likeness)
            np.maximum(likeness, 0, out=likeness)

            weight = likeness.astype(np.int32) * nearness
            weight_sum += weight
            tone_sum += weight * neighbour

    return (tone_sum // weight_sum).astype(np.uint8)


def _filterable(gray: np.ndarray) -> np.ndarray:
    """Which pixels the filter is allowed to touch.

    Two exclusions. A pixel nearer an edge than the window's reach has no full
    square of neighbours to average against, so it keeps the value it came
    with. And a white pixel with white on either side is an untouched run of
    paper — averaging it against grain that is still short of the white point
    would only pull it back off 255.
    """
    height, width = gray.shape
    reach = _FILTER_HALF + 1
    allowed = np.zeros(gray.shape, dtype=bool)
    allowed[reach : height - _FILTER_HALF, reach : width - _FILTER_HALF] = True

    white = gray == 255
    allowed[:, 1:-1] &= ~(white[:, 1:-1] & white[:, :-2] & white[:, 2:])
    return allowed


def denoise_scan(image: Image.Image) -> Image.Image:
    """Average scanner grain out of a page without softening the text.

    Runs after :func:`whiten_scan`, for the reason NAPS2 gives: the filter
    reads tonal distance in raw gray levels, so on a page whose range is still
    compressed a real edge reads as a small difference and averages away with
    the grain. Whitening first is also what makes this pass cheap — the runs
    of pure white it leaves behind are skipped outright.

    Args:
        image: Input page, color or grayscale.

    Returns:
        Grayscale ("L") page, same size as the input.
    """
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    padded = np.pad(gray, _FILTER_HALF, mode="edge")
    spatial = _spatial_weights()
    allowed = _filterable(gray)

    smoothed = gray.copy()
    for top in range(0, height, _TILE_SIZE):
        for left in range(0, width, _TILE_SIZE):
            rows = slice(top, min(top + _TILE_SIZE, height))
            columns = slice(left, min(left + _TILE_SIZE, width))
            keep = allowed[rows, columns]
            # Nothing but margin and paper in this tile — the 225-tap sweep
            # would be thrown away by the mask.
            if not keep.any():
                continue
            filtered = _filter_block(padded, gray[rows, columns], top, left, spatial)
            np.copyto(smoothed[rows, columns], filtered, where=keep)
    return Image.fromarray(smoothed, mode="L")
