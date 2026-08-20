"""How a question image is detected and written.

What a run *produced* is in :mod:`digitex.pipeline.outcome`.
"""

from __future__ import annotations

from dataclasses import dataclass


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
