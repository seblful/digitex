"""Shared extraction types — the run's configuration and its result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExtractionConfig:
    """How a question image is detected and written.

    Resolved once at the CLI boundary and passed down whole, so the page,
    book, and subject levels no longer restate the same three arguments. The
    default values live on ``ExtractionSettings`` — the one place the CLI
    reads them from — so they are not respelled here.

    Carries no model path: the CLI builds the detector and hands it over, so
    nothing below it knows a checkpoint exists.
    """

    image_format: str
    question_max_width: int
    question_max_height: int


@dataclass
class ExtractionResult:
    """Result of an extraction operation.

    ``success`` means the run itself completed, not that every item in it did:
    a book whose pages partly failed reports ``success=True`` with ``failed``
    and ``errors`` populated. Renderers must show those in both branches.
    """

    success: bool
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        processed: int = 0,
        skipped: int = 0,
        failed: int = 0,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Create a successful extraction result."""
        return cls(
            success=True,
            processed=processed,
            skipped=skipped,
            failed=failed,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        errors: list[str],
        processed: int = 0,
        warnings: list[str] | None = None,
    ) -> ExtractionResult:
        """Create a failed extraction result."""
        return cls(
            success=False,
            processed=processed,
            errors=errors,
            warnings=warnings or [],
        )

    def merge(self, other: ExtractionResult) -> ExtractionResult:
        """Merge two extraction results, summing their counts."""
        return ExtractionResult(
            success=self.success and other.success,
            processed=self.processed + other.processed,
            skipped=self.skipped + other.skipped,
            failed=self.failed + other.failed,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            metadata={**self.metadata, **other.metadata},
        )
