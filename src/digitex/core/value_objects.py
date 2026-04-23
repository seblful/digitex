"""Value objects for the domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuestionKey:
    """Identifies a question within an option by part and number.

    Corresponds to keys in answers.json (e.g. "A1", "B12") and the
    filesystem path segment {part}/{number}.jpg.
    """

    part: Literal["A", "B"]
    number: int

    @classmethod
    def parse(cls, raw: str) -> QuestionKey:
        raw = raw.strip().upper()
        if len(raw) < 2 or raw[0] not in ("A", "B") or not raw[1:].isdigit():
            raise ValueError(f"Invalid question key: {raw!r}")
        return cls(part=raw[0], number=int(raw[1:]))  # type: ignore[arg-type]

    def __str__(self) -> str:
        return f"{self.part}{self.number}"
