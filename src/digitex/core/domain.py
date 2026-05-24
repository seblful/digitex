"""Domain types — the single home for entities and value objects.

`ExamType` and `QuestionKey` are value objects (immutable, no identity).
`Question`, `Session`, `Student`, `TestResult`, `AuthorizedUser` are repository
return-shapes (Pydantic). Everything that crosses a module boundary should
import from this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 — Pydantic needs runtime type
from typing import Literal

from pydantic import BaseModel, Field

ExamType = Literal["CE", "CT"]
Part = Literal["A", "B"]


@dataclass(frozen=True)
class QuestionKey:
    """Identifies a question within an option by part and number.

    Corresponds to keys in answers.json (e.g. "A1", "B12") and the
    filesystem path segment {part}/{number}.jpg.
    """

    part: Part
    number: int

    @classmethod
    def parse(cls, raw: str) -> QuestionKey:
        raw = raw.strip().upper()
        if len(raw) < 2 or raw[0] not in ("A", "B") or not raw[1:].isdigit():
            raise ValueError(f"Invalid question key: {raw!r}")
        part: Part = "A" if raw[0] == "A" else "B"
        return cls(part=part, number=int(raw[1:]))

    def __str__(self) -> str:
        return f"{self.part}{self.number}"


class Student(BaseModel):
    student_id: int
    telegram_id: int
    name: str
    username: str | None = None


class Question(BaseModel):
    question_id: int
    part: Part
    question_number: int
    # Empty by default — repositories return metadata only. Bytes are fetched
    # on demand via QuestionRepository.get_image when no telegram_file_id is
    # cached (renderers must check before passing the question to send).
    image_data: bytes = b""
    telegram_file_id: str | None = None
    num_options: int = 5


class Session(BaseModel):
    session_id: int
    student_id: int
    option_id: int
    started_at: datetime
    completed_at: datetime | None = None


class TestResult(BaseModel):
    session_id: int
    exam_type: ExamType = "CT"
    part_a_score: int
    part_b_score: int
    total_score: int
    max_score: int
    time_spent: float = Field(description="Total time in seconds")
    completed_at: datetime


class AuthorizedUser(BaseModel):
    """A user's registration/authorization record."""

    telegram_id: int
    full_name: str
    telegram_username: str | None = None
    status: Literal["pending", "approved", "rejected"]
    created_at: datetime
    handled_at: datetime | None = None
    handled_by: int | None = None


__all__ = [
    "AuthorizedUser",
    "ExamType",
    "Part",
    "Question",
    "QuestionKey",
    "Session",
    "Student",
    "TestResult",
]
