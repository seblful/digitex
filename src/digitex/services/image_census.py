"""Count a subject's extracted question images and judge them complete or not.

Carved out of ``cli.extraction.count_questions``, which computed the verdict and
spent it on a terminal colour. The rules live here as values so they can be
asserted on, and so ``check-answers`` could reuse them.

The signal is modal: within one year, every Option should hold the same number
of Part A images as its neighbours, and likewise for Part B. A Part whose count
is off its year's mode means a page was missed or double-extracted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from digitex.core.corpus import walk_question_images
from digitex.core.domain import OPTIONS_PER_BOOK

if TYPE_CHECKING:
    from pathlib import Path


def _modes(values: list[int]) -> set[int]:
    """The most frequent value(s) in *values*, or empty for no values.

    Ties are all returned: two Options with 20 images and two with 21 leaves
    both counts modal, and neither is evidence of a missed page.
    """
    if not values:
        return set()
    counter = Counter(values)
    max_count = counter.most_common(1)[0][1]
    return {value for value, count in counter.items() if count == max_count}


@dataclass(frozen=True)
class PartCount:
    """How many question images one Option's Part holds."""

    option: str
    part: str
    images: int
    off_mode: bool


@dataclass(frozen=True)
class YearCensus:
    """One year's Options, and whether its counts hang together."""

    year: str
    parts: list[PartCount] = field(default_factory=list)

    @property
    def options(self) -> int:
        return len({part.option for part in self.parts})

    @property
    def missing_options(self) -> bool:
        """Fewer Options than a Book carries — pages are missing outright."""
        return self.options < OPTIONS_PER_BOOK

    @property
    def images(self) -> int:
        return sum(part.images for part in self.parts)

    @property
    def is_complete(self) -> bool:
        return not self.missing_options and not any(p.off_mode for p in self.parts)


@dataclass(frozen=True)
class SubjectCensus:
    """Every year of one subject's extraction output."""

    subject: str
    years: list[YearCensus] = field(default_factory=list)

    @property
    def images(self) -> int:
        return sum(year.images for year in self.years)

    @property
    def folders(self) -> int:
        """Option/Part folders holding at least one question image."""
        return sum(len(year.parts) for year in self.years)

    @property
    def is_empty(self) -> bool:
        return not self.years

    @property
    def is_complete(self) -> bool:
        return all(year.is_complete for year in self.years)


class ImageCensus:
    """Count the question images under one subject's extraction output."""

    def __init__(self, extraction_output_dir: Path) -> None:
        self._extraction_output_dir = extraction_output_dir

    def take(self, subject: str) -> SubjectCensus:
        """Count every year of *subject*, newest-numbered year last.

        Raises:
            FileNotFoundError: If the subject has no extraction output folder.
        """
        subject_dir = self._extraction_output_dir / subject
        if not subject_dir.is_dir():
            raise FileNotFoundError(subject_dir)

        years = [
            self._year_census(year_dir)
            for year_dir in sorted(subject_dir.iterdir(), key=_year_key)
            if year_dir.is_dir()
        ]
        return SubjectCensus(
            subject=subject, years=[year for year in years if year.parts]
        )

    @staticmethod
    def _year_census(year_dir: Path) -> YearCensus:
        counts: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for image in walk_question_images(year_dir):
            per_part = counts[image.option]
            per_part[image.part] = per_part.get(image.part, 0) + 1

        by_part: defaultdict[str, list[int]] = defaultdict(list)
        for per_part in counts.values():
            for part, count in per_part.items():
                by_part[part].append(count)
        modes = {part: _modes(values) for part, values in by_part.items()}

        parts = [
            PartCount(
                option=option,
                part=part,
                images=count,
                off_mode=count not in modes[part],
            )
            for option in sorted(counts, key=_numeric_key)
            for part, count in sorted(counts[option].items())
        ]
        return YearCensus(year=year_dir.name, parts=parts)


def _year_key(path: Path) -> tuple[int, str]:
    return _numeric_key(path.name)


def _numeric_key(name: str) -> tuple[int, str]:
    """Order numeric names numerically, non-numeric ones after them by name."""
    return (int(name), "") if name.isdigit() else (1 << 31, name)
