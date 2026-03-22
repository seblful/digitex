"""Image extraction module."""

from .book_extractor import BookExtractor
from .page_extractor import PageExtractor
from .tests_extractor import TestsExtractor

__all__ = ["BookExtractor", "PageExtractor", "TestsExtractor"]
