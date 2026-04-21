"""Digitex - Document digitization toolkit."""

from .core.handlers import LabelHandler
from .core.processors import FileProcessor
from .creators import PageDataCreator
from .extractors import (
    AnswersExtractor,
    BaseExtractor,
    BookExtractor,
    ExtractionResult,
    ExtractorFactory,
    ManualExtractor,
    PageExtractor,
    TestsExtractor,
)
from .label_studio import LabelStudioClient, TaskPredictor
from .ml import Predictor, Trainer

__all__ = [
    # Extractors
    "AnswersExtractor",
    "BaseExtractor",
    "BookExtractor",
    "ManualExtractor",
    "PageExtractor",
    "TestsExtractor",
    # Factory
    "ExtractorFactory",
    # Results
    "ExtractionResult",
    # Other
    "PageDataCreator",
    "LabelHandler",
    "FileProcessor",
    "LabelStudioClient",
    "TaskPredictor",
    "Predictor",
    "Trainer",
]
