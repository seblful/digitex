"""Validate that extracted answers.json files line up with question images.

Carved out of ``cli.extraction.check_answers`` so the rules are testable
without spinning up a Typer app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from digitex.core.corpus import walk_question_images

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class YearReport:
    """Validation outcome for a single year's worth of answers."""

    year: str
    answers_file_present: bool
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
    def is_clean(self) -> bool:
        return (
            self.answers_file_present
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
        return sum(1 for y in self.years if not y.is_clean)


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

        years = sorted(d.name for d in output_dir.iterdir() if d.is_dir())
        report = ValidationReport(subject=subject)
        for year in years:
            report.years.append(self._validate_year(output_dir / year, year))
        return report

    def _validate_year(self, year_dir: Path, year: str) -> YearReport:
        answers_file = year_dir / "answers.json"
        if not answers_file.exists():
            return YearReport(year=year, answers_file_present=False)

        with answers_file.open(encoding="utf-8") as f:
            answers_data = json.load(f)

        answer_questions: set[str] = set()
        for option_data in answers_data.values():
            answer_questions.update(option_data.keys())

        image_questions = self._scan_image_questions(year_dir)

        first_option_questions = set(answers_data.get("1", {}).keys())
        differing_options = [
            opt
            for opt in answers_data
            if set(answers_data[opt].keys()) != first_option_questions
        ]

        options_with_b, total_options = self._count_options_with_b(answers_data)

        return YearReport(
            year=year,
            answers_file_present=True,
            a_count=sum(1 for k in answer_questions if k.startswith("A")),
            b_count=sum(1 for k in answer_questions if k.startswith("B")),
            image_question_count=len(image_questions),
            answer_question_count=len(answer_questions),
            missing_in_answers=sorted(image_questions - answer_questions),
            missing_in_images=sorted(answer_questions - image_questions),
            options_with_differing_questions=differing_options,
            options_with_b=options_with_b,
            total_options=total_options,
        )

    @staticmethod
    def _scan_image_questions(year_dir: Path) -> set[str]:
        """Build the ``{"A1", "B2", …}`` set from on-disk image filenames."""
        return {
            f"{img.part.upper()}{img.number}" for img in walk_question_images(year_dir)
        }

    @staticmethod
    def _count_options_with_b(
        answers_data: dict[str, dict[str, str]],
    ) -> tuple[int, int]:
        """Count how many options have at least one Part B answer containing 'Б'."""
        options_with_b = 0
        total = 0
        for opt in sorted(answers_data, key=lambda k: int(k)):  # noqa: PLW0108
            part_b = {k: v for k, v in answers_data[opt].items() if k.startswith("B")}
            if any("Б" in v for v in part_b.values()):
                options_with_b += 1
            total += 1
        return options_with_b, total


__all__ = ["AnswerValidator", "ValidationReport", "YearReport"]
