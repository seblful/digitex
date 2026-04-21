"""Custom exceptions for the extraction module."""

from pathlib import Path


class ExtractionError(Exception):
    """Base exception for all extraction-related errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class DirectoryNotFoundError(ExtractionError, FileNotFoundError):
    """Raised when a required directory does not exist."""

    def __init__(self, path: Path | str, context: dict | None = None) -> None:
        message = f"Directory not found: {path}"
        super().__init__(message, context={**(context or {}), "path": str(path)})


class InvalidFilenameError(ExtractionError, ValueError):
    """Raised when a filename doesn't match the expected pattern."""

    def __init__(
        self, filename: str, expected_format: str, context: dict | None = None
    ) -> None:
        message = (
            f"Invalid filename format: {filename}. Expected format: {expected_format}"
        )
        super().__init__(message, context={**(context or {}), "filename": filename})


class ConflictResolutionError(ExtractionError):
    """Raised when a file conflict cannot be resolved."""

    def __init__(
        self,
        file_path: Path | str,
        reason: str,
        context: dict | None = None,
    ) -> None:
        message = f"Cannot resolve conflict for {file_path}: {reason}"
        super().__init__(message, context={**(context or {}), "file_path": str(file_path)})


class ExtractionValidationError(ExtractionError):
    """Raised when extraction validation fails."""

    pass


class ModelNotFoundError(ExtractionError, FileNotFoundError):
    """Raised when a required ML model file is not found."""

    def __init__(self, model_path: Path | str, context: dict | None = None) -> None:
        message = f"Model file not found: {model_path}"
        super().__init__(message, context={**(context or {}), "model_path": str(model_path)})


class APIError(ExtractionError):
    """Raised when an external API call fails."""

    def __init__(
        self, service: str, message: str, context: dict | None = None
    ) -> None:
        full_message = f"{service} API error: {message}"
        super().__init__(full_message, context={**(context or {}), "service": service})
