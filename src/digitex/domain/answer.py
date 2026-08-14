"""The answer key to a question, and the matching rules it carries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digitex.domain.entities import Part


@dataclass(frozen=True)
class AnswerKey:
    """The correct answer to a question, or the recorded absence of one.

    Built where the row is read, so the part travels with the value and no
    later seam has to be told which matching rules apply. Part A keys are
    integer option indices; Part B keys are free text, with alternative
    correct values separated by "/" (e.g. "ANS1/ANS2").

    A question with no stored key — ``value`` None — matches nothing, in
    either part. ``populate_db`` loads such a Question so that its image is
    servable, and this is what keeps it from ever being scored right.
    """

    part: Part
    value: int | str | None

    def matches(self, student_answer: str) -> bool:
        """True when the student's reply is correct under this key's rules."""
        if self.value is None:
            return False
        if self.part == "A":
            return int(student_answer.strip()) == int(self.value)
        alternatives = [
            opt.strip() for opt in str(self.value).split("/") if opt.strip()
        ]
        return bool(alternatives) and student_answer.strip() in alternatives

    @property
    def stored(self) -> str | None:
        """The key as ``session_answers`` stores it and the bot displays it."""
        return None if self.value is None else str(self.value)
