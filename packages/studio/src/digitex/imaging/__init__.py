"""Pixel work: cropping, deskewing, resizing, and OCR.

Every module here reaches for a heavy third-party stack — OpenCV, NumPy,
Pillow, Tesseract — which is what puts the package off-limits to the bot and
the database layer. The production image installs none of it, so an import
from there is an ImportError on the VPS rather than a slow start.

``ocr`` is the exception to the re-exports below. It is reached by module
(``from digitex.imaging.ocr import TextExtractor``) because pytesseract is the
one dependency that also wants a binary on PATH, and only the extraction
pipeline asks for it.

The rest is split by subject rather than by pipeline stage — ``scale`` fits an
image to a box, ``layout`` stacks the pieces of one question, ``levels`` reads
a scan's black and white points and corrects to them, ``denoise`` averages
grain out without crossing a stroke, ``document`` runs the whole-page
correction end to end, and ``crop`` lifts a polygon off a page straightened.
The names come back out here, so a caller asks for the operation rather than
for the file it happens to live in.
"""

from .crop import cut_out_image_by_polygon, rotate_image
from .denoise import denoise_scan
from .document import correct_document
from .layout import stack_vertically, stacked_layout
from .levels import flatten_scan, scan_levels, whiten_scan
from .scale import add_white_background, resize_image

__all__ = [
    "add_white_background",
    "correct_document",
    "cut_out_image_by_polygon",
    "denoise_scan",
    "flatten_scan",
    "resize_image",
    "rotate_image",
    "scan_levels",
    "stack_vertically",
    "stacked_layout",
    "whiten_scan",
]
