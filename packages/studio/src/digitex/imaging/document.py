"""The whole-page correction a scan goes through before extraction.

Flatten the lighting, find the levels, whiten, denoise — in that order, each
step reading statistics the one before it changed. The segmentation model is
trained on pages that have come through here, so what changes in this file
changes what the model sees.
"""

import numpy as np
from PIL import Image

from digitex.imaging.denoise import denoise_scan
from digitex.imaging.levels import (
    content_box,
    flatten_scan,
    scan_levels,
    whiten_scan,
)

# How far into a flattened page's white peak the white point goes. Flattening
# pulls the paper into a peak so tight that NAPS2's fifth (_PERCENTILE) lands
# barely under it and a gray film survives the stretch; a fiftieth reaches
# under the whole peak. Calibrated against the archive's figure pages, whose
# halftone tones come through it intact.
_FLAT_WHITE_PERCENTILE = 0.02
# Everything below this is ink or figure, not paper — the mass the black
# point is set from.
_INK_CEILING = 128
# How far into that dark mass the black point goes. At 300 dpi a stroke is
# mostly anti-aliasing — its core peak holds little of its mass — so the peak
# shoulder alone leaves text gray. A tenth of the way in is enough to turn the
# cores solid while the halo keeps its ramp; calibrated against the archive's
# halftone figures, which lose visible tone by a quarter of the way in.
_FLAT_BLACK_PERCENTILE = 10


def _stroke_black(gray: np.ndarray) -> int:
    """A black point deep enough that text comes out ink-black.

    The peak search can collapse onto a spike of already-black pixels and
    hand back zero, and even the true core peak holds little of a stroke's
    mass — so the black point is read off the dark mass itself instead.
    """
    dark = gray[gray < _INK_CEILING]
    if not dark.size:
        return 0
    return int(np.percentile(dark, _FLAT_BLACK_PERCENTILE))


def _flat_levels(flat: Image.Image) -> tuple[int, int] | None:
    """The (black, white) points for an already-flattened page.

    None when the histogram offers no usable pair, the same answer
    :func:`scan_levels` gives for a page better left alone.
    """
    levels = scan_levels(flat, _FLAT_WHITE_PERCENTILE)
    if levels is None:
        return None
    # The deeper claim wins: the peak shoulder when the histogram offers a
    # real ink peak, the stroke mass when the search collapses to a spike.
    return max(levels[0], _stroke_black(np.array(flat))), levels[1]


def _whitened(image: Image.Image, flatten: bool) -> Image.Image:
    """The levels-corrected page, flattened first unless told otherwise."""
    if not flatten:
        return whiten_scan(image)
    flat = flatten_scan(image)
    levels = _flat_levels(flat)
    return flat if levels is None else whiten_scan(flat, levels)


def correct_document(
    image: Image.Image, crop_margin: bool = True, flatten: bool = True
) -> Image.Image:
    """Clean a scanned page: flatten its shadows, fix its levels, denoise.

    NAPS2's document correction end to end, led by the illumination flatten
    NAPS2 has no answer to, plus the border removal it leaves as a TODO.

    Args:
        image: Input page, color or grayscale.
        crop_margin: Whether to cut the scanner's white canvas off the
            result. Pass False to keep the scan's own dimensions.
        flatten: Whether to divide the illumination out before the levels
            run. Pass False for a document whose light print should come
            through as it is — an answer sheet's shaded table rows would
            bleach to white wherever a shadow crossed them.

    Returns:
        Grayscale ("L") page — the size of the page within the scan, or of
        the whole scan when *crop_margin* is off.
    """
    corrected = denoise_scan(_whitened(image, flatten))
    if not crop_margin:
        return corrected

    # Measured on the untouched scan, cut off the finished one: correcting the
    # page turns its paper the same pure white as the canvas around it, and
    # after that there is no edge left to find.
    rows, columns = content_box(np.array(image.convert("L")))
    return corrected.crop((columns.start, rows.start, columns.stop, rows.stop))
