"""Base classes and protocols for extractors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ExtractionResult:
    """Result of an extraction operation."""

    success: bool
    processed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        processed: int = 0,
        skipped: int = 0,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ExtractionResult":
        """Create a successful extraction result."""
        return cls(
            success=True,
            processed=processed,
            skipped=skipped,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        errors: list[str],
        processed: int = 0,
        warnings: list[str] | None = None,
    ) -> "ExtractionResult":
        """Create a failed extraction result."""
        return cls(
            success=False,
            processed=processed,
            errors=errors,
            warnings=warnings or [],
        )

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        """Merge two extraction results."""
        return ExtractionResult(
            success=self.success and other.success,
            processed=self.processed + other.processed,
            skipped=self.skipped + other.skipped,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            metadata={**self.metadata, **other.metadata},
        )


class ExtractorProtocol(Protocol):
    """Protocol defining the interface for all extractors."""

    def extract(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Run extraction and return results."""
        ...

    def validate(self) -> bool:
        """Validate prerequisites for extraction.

        Returns:
            True if all prerequisites are met, False otherwise.
        """
        ...


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    @abstractmethod
    def extract(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Run extraction and return results.

        Returns:
            ExtractionResult with success status and statistics.
        """
        pass

    def validate(self) -> bool:
        """Validate prerequisites for extraction.

        Returns:
            True if all prerequisites are met, False otherwise.
        """
        try:
            self._validate_prerequisites()
            return True
        except Exception:
            return False

    @abstractmethod
    def _validate_prerequisites(self) -> None:
        """Validate that all prerequisites are met.

        Raises:
            Exception: If any prerequisite is not met.
        """
        pass
