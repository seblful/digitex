"""The answer key to a question, and the matching rules it carries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digitex.domain.entities import Part

# Part B keys list their acceptable spellings separated by this, e.g.
# "ANS1/ANS2". Part A keys are option indices and never carry one.
_ALTERNATIVE_SEPARATOR = "/"


@dataclass(frozen=True)
class AnswerKey:
    """The correct answer to a question, or the recorded absence of one.

    Built where the row is read, so the part travels with the value and no later
    seam has to be told which matching rules apply. Part A keys are integer
    option indices; Part B keys are free text, with alternative correct
    spellings separated by ``/``.

    A question with no stored key — ``value`` of None — matches nothing, in
    either part. Seeding loads such a question anyway so its image stays
    servable, and this is what stops it from ever being scored correct.
    """

    part: Part
    value: int | str | None

    def matches(self, student_answer: str) -> bool:
        """True when the student's reply is correct under this key's rules."""
        if self.value is None:
            return False
        if self.part == "A":
            return int(student_answer.strip()) == int(self.value)
        return student_answer.strip() in self._alternatives()

    def _alternatives(self) -> list[str]:
        """Every spelling this Part B key accepts, blanks discarded.

        A key of ``"/"`` or ``" "`` yields nothing, so it matches nothing —
        the same outcome as a missing key, which is what a blank key means.
        """
        return [
            spelling
            for raw in str(self.value).split(_ALTERNATIVE_SEPARATOR)
            if (spelling := raw.strip())
        ]

    @property
    def stored(self) -> str | None:
        """The key as ``session_answers`` stores it and the bot displays it."""
        return None if self.value is None else str(self.value)
