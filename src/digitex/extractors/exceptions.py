"""Custom exceptions for the extraction module.

Each subclass also inherits the closest builtin, so a caller that only knows
about ``FileNotFoundError`` or ``ValueError`` still catches the right thing.
"""

from pathlib import Path


class ExtractionError(Exception):
    """Base exception for all extraction-related errors."""


class DirectoryNotFoundError(ExtractionError, FileNotFoundError):
    """Raised when a required directory does not exist."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(f"Directory not found: {path}")
        self.path = Path(path)


class InvalidFilenameError(ExtractionError, ValueError):
    """Raised when a filename doesn't match the expected pattern."""

    def __init__(self, filename: str, expected_format: str) -> None:
        super().__init__(
            f"Invalid filename format: {filename}. Expected format: {expected_format}"
        )
        self.filename = filename


class ModelNotFoundError(ExtractionError, FileNotFoundError):
    """Raised when a required ML model file is not found."""

    def __init__(self, model_path: Path | str) -> None:
        super().__init__(f"Model file not found: {model_path}")
        self.model_path = Path(model_path)


class APIError(ExtractionError):
    """Raised when an external API call fails."""

    def __init__(self, service: str, message: str) -> None:
        super().__init__(f"{service} API error: {message}")
        self.service = service
