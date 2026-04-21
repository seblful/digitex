"""Image extraction module."""

from .answers_extractor import AnswersExtractor
from .base import BaseExtractor, ExtractionResult, ExtractorProtocol
from .book_extractor import BookExtractor
from .exceptions import (
    APIError,
    ConflictResolutionError,
    DirectoryNotFoundError,
    ExtractionError,
    ExtractionValidationError,
    InvalidFilenameError,
    ModelNotFoundError,
)
from .factory import ExtractorFactory
from .manual_extractor import ManualExtractor
from .page_extractor import PageExtractor
from .progress import JSONProgressTracker, ProgressTracker
from .tests_extractor import TestsExtractor
from .utils import (
    IMAGE_EXTENSIONS,
    count_images_by_hierarchy,
    count_total_images,
    find_image_files,
    get_mode_values,
    renumber_directory_tree,
    renumber_folder_sequentially,
)

__all__ = [
    # Base classes
    "BaseExtractor",
    "ExtractorProtocol",
    "ExtractionResult",
    # Extractors
    "AnswersExtractor",
    "BookExtractor",
    "ManualExtractor",
    "PageExtractor",
    "TestsExtractor",
    # Factory
    "ExtractorFactory",
    # Progress tracking
    "ProgressTracker",
    "JSONProgressTracker",
    # Exceptions
    "ExtractionError",
    "DirectoryNotFoundError",
    "InvalidFilenameError",
    "ConflictResolutionError",
    "ExtractionValidationError",
    "ModelNotFoundError",
    "APIError",
    # Utilities
    "IMAGE_EXTENSIONS",
    "find_image_files",
    "count_images_by_hierarchy",
    "count_total_images",
    "get_mode_values",
    "renumber_folder_sequentially",
    "renumber_directory_tree",
]
