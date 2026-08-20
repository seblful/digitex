"""Check that each year's ``answers.json`` lines up with its question images.

The rules live here, away from any front end, so they can be tested without
spinning up the review window that shows them.

Every file this reads is generated and then hand-corrected, which sets the tone
for the whole module: a broken file is the thing the check exists to *report*,
never a reason to abort and leave every other year unchecked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeGuard

from digitex.domain.corpus import walk_question_images

if TYPE_CHECKING:
    from pathlib import Path

PartBCoverage = Literal["none", "partial", "all"]

AnswerMap = dict[str, dict[str, str]]
"""``{option: {label: answer}}`` — the shape answers.json is indexed as."""


def _is_answer_map(data: object) -> TypeGuard[AnswerMap]:
    """True when the parsed file really has the ``{option: {label: answer}}`` shape.

    One hand-edit turns the top level into a list, or an option's value into
    one, or an answer into a bare number — and every one of those would
    otherwise blow up deep inside the comparison below.
    """
    return isinstance(data, dict) and all(
        isinstance(option, dict)
        and all(isinstance(answer, str) for answer in option.values())
        for option in data.values()
    )


def _read_answer_map(path: Path) -> AnswerMap | None:
    """The answers at *path*, or None when they cannot be read as a map of maps.

    Valid JSON is not enough, so the shape check is folded in here: the two
    failures are indistinguishable to a caller, which only reports that the
    file needs a human.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return None
    return data if _is_answer_map(data) else None


def _image_questions(year_dir: Path) -> set[str]:
    """The ``{"A1", "B2", …}`` set the year's image filenames spell out."""
    return {
        f"{image.part.upper()}{image.number}"
        for image in walk_question_images(year_dir)
    }


def _options_carrying_part_b(answers: AnswerMap) -> tuple[int, int]:
    """How many Options carry a Part Б answer key, and how many there are.

    An Option counts when at least one of its B answers mentions Б — the letter
    a Part Б key is written with. Options are counted as they are written, not
    parsed as numbers: ``int(key)`` on a hand-edited option key used to abort
    the whole run.
    """
    with_b = 0
    for questions in answers.values():
        part_b = (value for label, value in questions.items() if label.startswith("B"))
        if any("Б" in value for value in part_b):
            with_b += 1
    return with_b, len(answers)


def _options_off_the_first(answers: AnswerMap) -> list[str]:
    """Options whose question set differs from the year's first Option's.

    Compared against the first Option *as written*, not against ``"1"``: a year
    whose sheets only produced Options 6-10 has no Option 1, and defaulting to
    an empty set would mark every Option as differing.
    """
    first = next(iter(answers), None)
    reference = set(answers[first]) if first is not None else set()
    return [option for option in answers if set(answers[option]) != reference]


@dataclass
class YearReport:
    """Validation outcome for a single year's worth of answers."""

    year: str
    answers_file_present: bool
    answers_file_valid: bool = True
    a_count: int = 0
    b_count: int = 0
    image_question_count: int = 0
    answer_question_count: int = 0
    missing_in_answers: list[str] = field(default_factory=list)
    missing_in_images: list[str] = field(default_factory=list)
    options_with_differing_questions: list[str] = field(default_factory=list)
    options_with_b: int = 0
    total_options: int = 0

    @property
    def has_mismatch(self) -> bool:
        return bool(self.missing_in_answers or self.missing_in_images)

    @property
    def options_differ(self) -> bool:
        return bool(self.options_with_differing_questions)

    @property
    def part_b_coverage(self) -> PartBCoverage:
        """How many of the year's Options carry a Part Б answer key.

        Part Б is hand-written on the answer sheets and the vision model misses
        it more often than anything else, so "none" and "partial" mean different
        things: the first says the whole year's Б keys are absent, the second
        that some sheets read and some did not.
        """
        if self.options_with_b == 0:
            return "none"
        if self.options_with_b < self.total_options:
            return "partial"
        return "all"

    @property
    def is_clean(self) -> bool:
        return (
            self.answers_file_present
            and self.answers_file_valid
            and not self.has_mismatch
            and not self.options_differ
            and self.options_with_b == self.total_options
        )


@dataclass
class ValidationReport:
    """Aggregate validation outcome across all years for one subject."""

    subject: str
    years: list[YearReport] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return sum(1 for year in self.years if not year.is_clean)


class AnswerValidator:
    """Validate the answers.json / image-tree alignment for a subject."""

    def __init__(self, extraction_output_dir: Path) -> None:
        self._extraction_output_dir = extraction_output_dir

    def validate(self, subject: str) -> ValidationReport:
        """Run the full validation pass for one subject.

        Raises:
            FileNotFoundError: if the subject's output directory does not exist.
        """
        output_dir = self._extraction_output_dir / subject
        if not output_dir.exists():
            raise FileNotFoundError(output_dir)

        years = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
        return ValidationReport(
            subject=subject,
            years=[self._validate_year(output_dir / year, year) for year in years],
        )

    def _validate_year(self, year_dir: Path, year: str) -> YearReport:
        answers_file = year_dir / "answers.json"
        if not answers_file.exists():
            return YearReport(year=year, answers_file_present=False)

        answers = _read_answer_map(answers_file)
        if answers is None:
            return YearReport(
                year=year, answers_file_present=True, answers_file_valid=False
            )

        answered = {label for questions in answers.values() for label in questions}
        pictured = _image_questions(year_dir)
        options_with_b, total_options = _options_carrying_part_b(answers)

        return YearReport(
            year=year,
            answers_file_present=True,
            a_count=sum(1 for label in answered if label.startswith("A")),
            b_count=sum(1 for label in answered if label.startswith("B")),
            image_question_count=len(pictured),
            answer_question_count=len(answered),
            missing_in_answers=sorted(pictured - answered),
            missing_in_images=sorted(answered - pictured),
            options_with_differing_questions=_options_off_the_first(answers),
            options_with_b=options_with_b,
            total_options=total_options,
        )


__all__ = ["AnswerValidator", "PartBCoverage", "ValidationReport", "YearReport"]
