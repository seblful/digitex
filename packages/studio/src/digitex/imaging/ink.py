"""Where a scan's print is, and which way its lines run.

:mod:`digitex.imaging.levels` asks the histogram where the page's ink and paper
sit as *tones*, to correct them. This asks a different question of the same
pixels: which of them are print, and at what angle. Nothing here corrects
anything — the answers are what :mod:`digitex.imaging.outlines` measures a
region's margin against.

The threshold is local, not global. A scan's lamp falls off toward the spine,
the paper is grey, and the dark edge of the book is a large dark *background*
rather than ink; subtracting the page's own blur is what tells those apart and
what a single cutoff cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

# The background estimate's blur radius, as a fraction of the page diagonal.
# Wider than any stroke, narrower than the shadow it has to follow.
_BG_SIGMA = 0.006

# Nothing dimmer than this counts as print. Otsu on a page that is mostly paper
# can settle on the scanner's own noise floor, and this is the floor under it.
_INK_FLOOR = 14

# The smallest blob that may hold an outline out, as a fraction of the page
# diagonal. Squared, that is some 11 px on a 2400 px page: dust goes, a comma
# stays.
_SPECK = 0.0012

# How far off axis a page is read as tilted. Past this the search has found a
# figure's edge rather than the text, and the page keeps its own axes. The
# archive's worst page sits at 5.4 degrees.
MAX_SKEW = 6.0
_SKEW_STEP = 0.1

# The skew search runs on a page this many times smaller. A tilt is a whole-page
# property and resolving it costs no detail worth having.
_SKEW_SCALE = 0.25


@dataclass(frozen=True)
class PageInk:
    """One page's print, and the angle its lines run at."""

    mask: np.ndarray
    """0/255, specks removed, in the orientation the scan arrived in."""

    skew: float
    """Degrees. Turning the page by this much is what levels its text."""

    @property
    def shape(self) -> tuple[int, int]:
        """The page's (height, width) in pixels."""
        height, width = self.mask.shape[:2]
        return height, width


def _thresholded(gray: np.ndarray) -> np.ndarray:
    """Where the page is darker than its own local paper tone."""
    sigma = _BG_SIGMA * math.hypot(*gray.shape)
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    contrast = cv2.subtract(background, gray)
    level, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return ((contrast > max(level, _INK_FLOOR)) * 255).astype(np.uint8)


def despeckled(mask: np.ndarray, min_area: int) -> np.ndarray:
    """*mask* without the blobs too small to be print."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # OpenCV's stubs leave the label map's dtype open; it is an integer index
    # into ``keep``, and saying so is what lets it be used as one.
    index = labels.astype(np.intp)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[index] * 255).astype(np.uint8)


def rotation(shape: tuple[int, int], angle: float) -> np.ndarray:
    """The 2x3 matrix that turns an image of *shape* by *angle* about its centre."""
    height, width = shape
    return cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)


def turn(mask: np.ndarray, angle: float) -> np.ndarray:
    """*mask* rotated by *angle*, on a canvas of its own size.

    The same size on purpose: a page's tilt is a few degrees, so nothing that
    matters leaves the frame, and keeping the canvas means one matrix carries
    coordinates both ways with no offset to track alongside it.
    """
    height, width = mask.shape[:2]
    return cv2.warpAffine(
        mask, rotation((height, width), angle), (width, height), flags=cv2.INTER_NEAREST
    )


def _skew(mask: np.ndarray) -> float:
    """The angle that has to come out for *mask*'s text to run level.

    The projection-profile criterion: rows through level text are either full of
    ink or empty of it, and that is the row profile with the widest spread.
    Turning the page a tenth of a degree at a time and keeping the sharpest
    profile finds the angle without having to identify a single text line first,
    which matters on a page whose lines interleave with an answer column.
    """
    small = cv2.resize(
        mask, None, fx=_SKEW_SCALE, fy=_SKEW_SCALE, interpolation=cv2.INTER_AREA
    )
    angles = np.arange(-MAX_SKEW, MAX_SKEW + _SKEW_STEP / 2, _SKEW_STEP)
    sharpness = [
        float((turn(small, angle).sum(axis=1, dtype=np.float64) ** 2).sum())
        for angle in angles
    ]
    return float(angles[int(np.argmax(sharpness))])


def read_ink(image: Image.Image) -> PageInk:
    """Find *image*'s print and the angle it runs at.

    Args:
        image: The page, in any mode Pillow can convert to greyscale.

    Returns:
        The print as a 0/255 mask, and the page's tilt in degrees.
    """
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    speck = max(4, round((_SPECK * math.hypot(width, height)) ** 2))
    mask = despeckled(_thresholded(gray), speck)
    return PageInk(mask=mask, skew=_skew(mask))


def row_runs(mask: np.ndarray, min_run: int = 1) -> list[tuple[int, int]]:
    """Every run of rows of *mask* holding ink, as (top, bottom) pairs.

    A run of inked rows is one line of print however many pieces its letters
    come in, which is why the callers here read the row profile rather than
    chase letters around.
    """
    inked = np.flatnonzero(mask.any(axis=1))
    if not inked.size:
        return []
    breaks = np.flatnonzero(np.diff(inked) > 1)
    starts = np.concatenate(([inked[0]], inked[breaks + 1]))
    ends = np.concatenate((inked[breaks], [inked[-1]]))
    return [
        (int(start), int(end) + 1)
        for start, end in zip(starts, ends, strict=True)
        if end + 1 - start >= min_run
    ]
