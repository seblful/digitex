"""Tests for the image census — ``take()`` is the test surface.

The verdict these assertions pin used to be a terminal colour inside
``count-questions``, so none of it could be checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from digitex.domain.entities import OPTIONS_PER_BOOK
from digitex.pipeline.audit.census import ImageCensus

if TYPE_CHECKING:
    from pathlib import Path


def _questions(part_dir: Path, count: int) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, count + 1):
        (part_dir / f"{number}.jpg").write_bytes(b"x")


def _year(
    root: Path, year: str, per_option: dict[int, dict[str, int]] | None = None
) -> None:
    """Seed ``{year}/{option}/{part}/`` with the given image counts."""
    counts = per_option or {
        option: {"A": 20, "B": 6} for option in range(1, OPTIONS_PER_BOOK + 1)
    }
    for option, parts in counts.items():
        for part, images in parts.items():
            _questions(root / year / str(option) / part, images)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


class TestImageCensus:
    def test_unknown_subject_raises(self, output_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ImageCensus(output_dir).take("biology")

    def test_subject_with_no_years_is_empty(self, output_dir: Path) -> None:
        (output_dir / "biology").mkdir(parents=True)

        census = ImageCensus(output_dir).take("biology")

        assert census.is_empty
        assert census.images == 0

    def test_a_missing_year_directory_counts_as_empty(self, output_dir: Path) -> None:
        """The documented contract — not a FileNotFoundError from the walk."""
        census = ImageCensus.take_year(output_dir / "biology" / "2099")

        assert census.year == "2099"
        assert census.parts == []

    def test_a_full_year_is_complete(self, output_dir: Path) -> None:
        _year(output_dir / "biology", "2020")

        census = ImageCensus(output_dir).take("biology")

        year = census.years[0]
        assert year.year == "2020"
        assert year.options == OPTIONS_PER_BOOK
        assert not year.missing_options
        assert year.is_complete

    def test_counts_and_folders_add_up(self, output_dir: Path) -> None:
        _year(output_dir / "biology", "2020", {1: {"A": 20, "B": 6}, 2: {"A": 20}})

        census = ImageCensus(output_dir).take("biology")

        assert census.images == 46
        assert census.folders == 3
        assert census.years[0].images == 46

    def test_too_few_options_is_incomplete(self, output_dir: Path) -> None:
        _year(output_dir / "biology", "2020", {1: {"A": 20}, 2: {"A": 20}})

        year = ImageCensus(output_dir).take("biology").years[0]

        assert year.missing_options
        assert not year.is_complete

    def test_an_option_off_the_modal_count_is_flagged(self, output_dir: Path) -> None:
        """The signal a page was missed: one Option short of its neighbours."""
        counts = {option: {"A": 20} for option in range(1, OPTIONS_PER_BOOK + 1)}
        counts[4] = {"A": 19}
        _year(output_dir / "biology", "2020", counts)

        year = ImageCensus(output_dir).take("biology").years[0]

        assert not year.missing_options
        assert not year.is_complete
        off = [part for part in year.parts if part.off_mode]
        assert [(p.option, p.part, p.images) for p in off] == [("4", "A", 19)]

    def test_a_tie_between_two_counts_flags_neither(self, output_dir: Path) -> None:
        """Half the Options at 20 and half at 21 is not evidence of a miss."""
        counts = {
            option: {"A": 20 if option <= OPTIONS_PER_BOOK // 2 else 21}
            for option in range(1, OPTIONS_PER_BOOK + 1)
        }
        _year(output_dir / "biology", "2020", counts)

        year = ImageCensus(output_dir).take("biology").years[0]

        assert not any(part.off_mode for part in year.parts)
        assert year.is_complete

    def test_parts_are_scored_against_their_own_mode(self, output_dir: Path) -> None:
        """Part B holding fewer images than Part A is normal, not a miss."""
        _year(output_dir / "biology", "2020")

        year = ImageCensus(output_dir).take("biology").years[0]

        assert not any(part.off_mode for part in year.parts)

    def test_years_come_back_in_numeric_order(self, output_dir: Path) -> None:
        for year in ("2021", "2009", "2020"):
            _year(output_dir / "biology", year, {1: {"A": 1}})

        census = ImageCensus(output_dir).take("biology")

        assert [year.year for year in census.years] == ["2009", "2020", "2021"]

    def test_options_come_back_in_numeric_order(self, output_dir: Path) -> None:
        _year(
            output_dir / "biology",
            "2020",
            {1: {"A": 1}, 2: {"A": 1}, 10: {"A": 1}},
        )

        year = ImageCensus(output_dir).take("biology").years[0]

        assert [part.option for part in year.parts] == ["1", "2", "10"]

    def test_non_question_files_are_not_counted(self, output_dir: Path) -> None:
        """Only numbered images are questions — a stray file is not one."""
        part_dir = output_dir / "biology" / "2020" / "1" / "A"
        _questions(part_dir, 3)
        (part_dir / "notes.jpg").write_bytes(b"x")
        (part_dir / "answers.json").write_bytes(b"{}")

        year = ImageCensus(output_dir).take("biology").years[0]

        assert year.images == 3
