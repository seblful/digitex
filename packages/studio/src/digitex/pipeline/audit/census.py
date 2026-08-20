"""Count a subject's extracted question images and judge them complete or not.

The verdict lives here as values, so any front end can render it and a test can
assert on it — today the front end is the review window's stats tab.

The signal is modal rather than absolute. Nothing knows how many questions a
given year's paper held, but within one year every Option should hold the same
number of Part A images as its neighbours, and likewise for Part B. A Part whose
count is off its year's mode is where a page was missed or extracted twice.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from digitex.domain.corpus import walk_question_images
from digitex.domain.entities import OPTIONS_PER_BOOK

if TYPE_CHECKING:
    from pathlib import Path


def _modes(values: list[int]) -> set[int]:
    """The most frequent value(s) in *values*, or empty for no values.

    Every tied value comes back: two Options with 20 images and two with 21
    leaves both counts modal, and neither is evidence of a missed page.
    """
    if not values:
        return set()
    counts = Counter(values)
    most = counts.most_common(1)[0][1]
    return {value for value, count in counts.items() if count == most}


def _numeric_key(name: str) -> tuple[int, str]:
    """Order numeric names numerically, non-numeric ones after them by name.

    Option and year directories are hand-editable, so one of them is eventually
    called ``1a``. ``int(name)`` on that used to abort the whole count.
    """
    return (int(name), "") if name.isdigit() else (1 << 31, name)


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
        return not self.missing_options and not any(
            part.off_mode for part in self.parts
        )


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


class ImageCensus:
    """Count the question images under one subject's extraction output."""

    def __init__(self, extraction_output_dir: Path) -> None:
        self._extraction_output_dir = extraction_output_dir

    def take(self, subject: str) -> SubjectCensus:
        """Count every year of *subject*, newest-numbered year last.

        A year that produced no image at all is left out rather than reported
        empty: it is a year nobody has extracted yet, not a year with a problem.

        Raises:
            FileNotFoundError: If the subject has no extraction output folder.
        """
        subject_dir = self._extraction_output_dir / subject
        if not subject_dir.is_dir():
            raise FileNotFoundError(subject_dir)

        years = [
            self.take_year(year_dir)
            for year_dir in sorted(
                subject_dir.iterdir(), key=lambda d: _numeric_key(d.name)
            )
            if year_dir.is_dir()
        ]
        return SubjectCensus(
            subject=subject, years=[year for year in years if year.parts]
        )

    @staticmethod
    def take_year(year_dir: Path) -> YearCensus:
        """Count one year's output tree. Empty when the directory does not exist."""
        if not year_dir.is_dir():
            return YearCensus(year=year_dir.name)

        # {option: {part: images}}, which is the shape both the modal
        # comparison and the reported rows are read out of.
        held: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for image in walk_question_images(year_dir):
            held[image.option][image.part] += 1

        # A Part is scored against its own Part across the year's Options: Part
        # B holding a third of Part A's images is normal, not a miss.
        per_part: defaultdict[str, list[int]] = defaultdict(list)
        for parts in held.values():
            for part, count in parts.items():
                per_part[part].append(count)
        modes = {part: _modes(counts) for part, counts in per_part.items()}

        return YearCensus(
            year=year_dir.name,
            parts=[
                PartCount(
                    option=option,
                    part=part,
                    images=count,
                    off_mode=count not in modes[part],
                )
                for option in sorted(held, key=_numeric_key)
                for part, count in sorted(held[option].items())
            ],
        )
