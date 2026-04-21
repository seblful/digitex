"""Image extraction module."""

from .answers_extractor import AnswersExtractor
from .book_extractor import BookExtractor
from .page_extractor import PageExtractor
from .tests_extractor import TestsExtractor

__all__ = ["AnswersExtractor", "BookExtractor", "PageExtractor", "TestsExtractor"]
