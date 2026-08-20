"""Pixel work: cropping, deskewing, resizing, and OCR.

Everything here pulls in a heavy third-party stack — OpenCV, NumPy, Pillow,
Tesseract — so this package is off-limits to the bot and the database layer.
The production image installs none of it.

``ocr`` is imported directly (``from digitex.imaging.ocr import
TextExtractor``) rather than re-exported here, because pytesseract is the one
dependency that needs a binary on PATH and only the extraction pipeline wants
it.

The pixel work is split by subject — ``scale`` fits an image to a box,
``layout`` stacks the pieces of one question, ``levels`` reads a scan's black
and white points, ``denoise`` removes speckle without crossing a stroke,
``document`` is the whole-page correction pipeline, and ``crop`` cuts a polygon
out straightened. All re-exported here, so a caller names what it wants rather
than the file it happens to live in.
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
