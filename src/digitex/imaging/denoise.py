"""Speckle removal that leaves strokes alone.

A bilateral-style filter: pixels average only with neighbours of a
similar colour, so scanner grain in the paper is smoothed away while the
edge of a letter — a large colour distance — is not crossed.
"""

import math

import numpy as np
from PIL import Image

# The filter's square around each pixel, and the gray distance at which a
# neighbour's weight reaches zero.
_FILTER_SIZE = 15

_FILTER_HALF = _FILTER_SIZE // 2

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
    """
    height, width = block.shape
    centre = block.astype(np.int16)
    weights = np.zeros(block.shape, dtype=np.int32)
    totals = np.zeros(block.shape, dtype=np.int32)

    for row in range(_FILTER_SIZE):
        for column in range(_FILTER_SIZE):
            nearness = int(spatial[row, column])
            if not nearness:
                continue
            neighbour = padded[
                top + row : top + row + height, left + column : left + column + width
            ]
            # NAPS2's weight table, worked out rather than looked up: a
            # neighbour counts for less the further off its tone is, and for
            # nothing at all past _COLOR_DIST_MAX. Arithmetic beats a lookup
            # here because the table is too big to stay in cache.
            tone = np.abs(centre - neighbour)
            np.subtract(_COLOR_DIST_MAX, tone, out=tone)
            np.maximum(tone, 0, out=tone)

            weight = tone.astype(np.int32) * nearness
            weights += weight
            totals += weight * neighbour

    return (totals // weights).astype(np.uint8)


def denoise_scan(image: Image.Image) -> Image.Image:
    """Average scanner grain out of a page without softening the text.

    A bilateral filter weighs a neighbour by how far away it is *and* how
    different it is, so grain within the paper and within a stroke averages
    away while the edge between them survives.

    Runs after :func:`whiten_scan`, for the reason NAPS2 gives: the filter
    reads distance in raw gray levels, so a page whose range is still
    compressed loses fine detail to it — and the pass is cheaper once the
    paper is uniform, which is what the untouched runs of white buy.

    Args:
        image: Input page, color or grayscale.

    Returns:
        Grayscale ("L") page, same size as the input.
    """
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    padded = np.pad(gray, _FILTER_HALF, mode="edge")
    spatial = _spatial_weights()

    # Pixels within the filter's reach of an edge are left alone, as is any
    # white pixel between two others — an untouched run of paper.
    edge = _FILTER_HALF + 1
    inside = np.zeros(gray.shape, dtype=bool)
    inside[edge : height - _FILTER_HALF, edge : width - _FILTER_HALF] = True
    white = gray == 255
    inside[:, 1:-1] &= ~(white[:, 1:-1] & white[:, :-2] & white[:, 2:])

    smoothed = gray.copy()
    for top in range(0, height, _TILE_SIZE):
        for left in range(0, width, _TILE_SIZE):
            rows = slice(top, min(top + _TILE_SIZE, height))
            columns = slice(left, min(left + _TILE_SIZE, width))
            keep = inside[rows, columns]
            if not keep.any():
                continue
            block = _filter_block(padded, gray[rows, columns], top, left, spatial)
            np.copyto(smoothed[rows, columns], block, where=keep)
    return Image.fromarray(smoothed, mode="L")
