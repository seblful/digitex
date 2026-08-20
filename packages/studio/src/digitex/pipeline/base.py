"""How a question image is detected and written.

What a run *produced* is in :mod:`digitex.pipeline.outcome`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionConfig:
    """How a question image is detected and written.

    Resolved once at the CLI boundary and passed down whole, so the page, book
    and subject levels no longer restate the same three arguments. The defaults
    belong to ``ExtractionSettings`` — the one place the CLI reads them from —
    and are deliberately not respelled here.

    No model path: the CLI builds the detector and hands it over, so nothing
    below that knows a checkpoint exists.
    """

    image_format: str
    question_max_width: int
    question_max_height: int
