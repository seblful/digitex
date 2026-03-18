"""Digitex - Document digitization toolkit."""

from .config import get_settings
from .core import PageDataCreator
from .core.handlers import ImageHandler, LabelHandler, PDFHandler
from .core.processors import FileProcessor, ImageProcessor
from .ml import Predictor, Trainer

__all__ = [
    "get_settings",
    "PageDataCreator",
    "ImageHandler",
    "LabelHandler",
    "PDFHandler",
    "FileProcessor",
    "ImageProcessor",
    "Predictor",
    "Trainer",
]
