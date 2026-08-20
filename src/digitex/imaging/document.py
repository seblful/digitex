"""The whole-page correction a scan goes through before extraction.

Flatten the lighting, find the levels, whiten, denoise — in that order,
because each step reads statistics the one before it changed. The
segmentation model is trained on pages that have been through this.
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

# The white shoulder for a flattened page. Flattening pulls the paper into a
# peak so tight that NAPS2's fifth (_PERCENTILE) lands barely under it and a
# gray film survives the stretch; a fiftieth reaches under the whole peak.
# Calibrated against the archive's figure pages — halftone tones survive it.
_FLAT_WHITE_PERCENTILE = 0.02
# Everything below this is ink or figure, not paper — the mass the black
# point is set from.
_INK_CEILING = 128
# The black point sits this far into the page's dark mass, as a percentile.
# At 300 dpi a stroke is mostly anti-aliasing — its core peak holds little of
# its mass — so the peak shoulder alone leaves text gray. A tenth of the way
# in is enough to turn the cores solid while the halo keeps its ramp;
# calibrated against the archive's halftone figures, which lose visible
# tone by a quarter of the way in.
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


def _whitened(image: Image.Image, flatten: bool) -> Image.Image:
    """The levels-corrected page, flattened first unless told otherwise."""
    if not flatten:
        return whiten_scan(image)
    flat = flatten_scan(image)
    levels = scan_levels(flat, _FLAT_WHITE_PERCENTILE)
    if levels is None:
        return flat
    # The deeper claim wins: the peak shoulder when the histogram offers a
    # real ink peak, the stroke mass when the search collapses to a spike.
    black = max(levels[0], _stroke_black(np.array(flat)))
    return whiten_scan(flat, (black, levels[1]))


def correct_document(
    image: Image.Image, crop_margin: bool = True, flatten: bool = True
) -> Image.Image:
    """Clean a scanned page: flatten its shadows, fix its levels, denoise.

    NAPS2's document correction end to end, led by the illumination flatten
    NAPS2 has no answer to, plus the border removal it leaves as a TODO. The
    margin is measured on the way in and cut off on the way out, because
    correcting the page turns its paper the same pure white as the canvas
    around it — after the fact there is no edge left to find.

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
    if not crop_margin:
        return denoise_scan(_whitened(image, flatten))
    rows, columns = content_box(np.array(image.convert("L")))
    corrected = denoise_scan(_whitened(image, flatten))
    return corrected.crop((columns.start, rows.start, columns.stop, rows.stop))
