"""Progress tracking abstraction for extractors."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ProgressTracker(ABC):
    """Abstract base class for progress tracking."""

    @abstractmethod
    def is_completed(self, subject: str, identifier: str) -> bool:
        """Check if an extraction unit is completed.

        Args:
            subject: Subject name (e.g., 'math', 'biology').
            identifier: Unique identifier (e.g., year '2020').

        Returns:
            True if completed, False otherwise.
        """
        pass

    @abstractmethod
    def mark_completed(self, subject: str, identifier: str) -> None:
        """Mark an extraction unit as completed.

        Args:
            subject: Subject name.
            identifier: Unique identifier.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist progress to storage."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load progress from storage."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all progress."""
        pass


class JSONProgressTracker(ProgressTracker):
    """JSON file-based progress tracker."""

    def __init__(self, path: Path) -> None:
        """Initialize the tracker.

        Args:
            path: Path to the JSON progress file.
        """
        self._path = path
        self._completed: dict[str, set[str]] = {}
        self.load()

    def is_completed(self, subject: str, identifier: str) -> bool:
        """Check if extraction is completed for a subject/identifier pair."""
        return identifier in self._completed.get(subject, set())

    def mark_completed(self, subject: str, identifier: str) -> None:
        """Mark extraction as completed."""
        if subject not in self._completed:
            self._completed[subject] = set()
        self._completed[subject].add(identifier)

    def save(self) -> None:
        """Save progress to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: sorted(v) for k, v in self._completed.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved progress", path=str(self._path))

    def load(self) -> None:
        """Load progress from JSON file."""
        if not self._path.exists():
            self._completed = {}
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._completed = {k: set(v) for k, v in data.items()}
            logger.debug("Loaded progress", path=str(self._path), subjects=len(data))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load progress file, starting fresh",
                path=str(self._path),
                error=str(e),
            )
            self._completed = {}

    def get_completed_subjects(self) -> set[str]:
        """Get all subjects with completed extractions."""
        return set(self._completed.keys())

    def get_completed_identifiers(self, subject: str) -> set[str]:
        """Get all completed identifiers for a subject."""
        return self._completed.get(subject, set())

    def clear(self) -> None:
        """Clear all progress."""
        self._completed = {}
        if self._path.exists():
            self._path.unlink()
        logger.info("Cleared all progress", path=str(self._path))
