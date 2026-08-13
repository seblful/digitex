"""Pixel work: cropping, deskewing, resizing, and OCR.

Everything here pulls in a heavy third-party stack — OpenCV, NumPy, Pillow,
Tesseract — so this package is off-limits to the bot and the database layer.
The production image installs none of it.

``ocr`` is imported directly (``from digitex.imaging.ocr import
TextExtractor``) rather than re-exported here, because pytesseract is the one
dependency that needs a binary on PATH and only the extraction pipeline wants
it.
"""

from .image import (
    ImageCropper,
    add_white_background,
    resize_image,
    rotate_image,
)

__all__ = [
    "ImageCropper",
    "add_white_background",
    "resize_image",
    "rotate_image",
]
