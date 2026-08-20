"""The failures extraction raises, each also wearing a builtin's face.

Every class inherits :class:`ExtractionError` *and* the closest builtin, so a
caller that never heard of this module still catches the right thing: a missing
book directory answers to ``FileNotFoundError``, an unparseable page name to
``ValueError``. That pairing is the reason these exist at all rather than bare
builtins carrying a formatted message.

Each one keeps what it was raised about as an attribute. The message is for a
human reading a terminal; the attribute is for the caller that has to name the
offending file.
"""

from __future__ import annotations

from pathlib import Path


class ExtractionError(Exception):
    """Base for everything the extraction pipeline raises."""


class DirectoryNotFoundError(ExtractionError, FileNotFoundError):
    """A directory the run needs — an archive, a subject, a year — is not there."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(f"Directory not found: {path}")
        self.path = Path(path)


class InvalidFilenameError(ExtractionError, ValueError):
    """A filename the corpus layout cannot read, and the shape it wanted."""

    def __init__(self, filename: str, expected_format: str) -> None:
        super().__init__(
            f"Invalid filename format: {filename}. Expected format: {expected_format}"
        )
        self.filename = filename


class ModelNotFoundError(ExtractionError, FileNotFoundError):
    """The segmentation checkpoint is missing.

    Raised where the model is built rather than where it is first used, so the
    command fails before it has walked an archive.
    """

    def __init__(self, model_path: Path | str) -> None:
        super().__init__(f"Model file not found: {model_path}")
        self.model_path = Path(model_path)


class APIError(ExtractionError):
    """An external service failed. Carries which one, so the message says so."""

    def __init__(self, service: str, message: str) -> None:
        super().__init__(f"{service} API error: {message}")
        self.service = service


class ReviewAborted(ExtractionError):
    """A page reviewer walked away: stop the whole run, not just this page.

    Deliberately outside the per-page error handling in ``BookExtractor``.
    Counted as one page's failure, the book would still finish — and a year
    recorded as finished is never reopened, so every page the reviewer never
    saw would be skipped forever.
    """

    def __init__(self, page_name: str = "") -> None:
        at = f" at {page_name}" if page_name else ""
        super().__init__(f"Review aborted{at}")
        self.page_name = page_name
