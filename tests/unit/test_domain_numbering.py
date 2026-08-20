"""The rule that keeps the output tree in order without a renumbering pass.

`numbering_fault` is what replaced the renumbering command: rather than
repairing a tree that went out of order, both sides of the reviewer seam refuse
to write it that way. It reads the disk but needs no display, so it is testable
here; that the *two sides agree* is asserted in `tests/characterization`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.domain.entities import PixelPolygon
from digitex.domain.numbering import numbering_fault
from digitex.domain.placement import (
    PageRegion,
    PlacedQuestion,
    QuestionPlacement,
)

if TYPE_CHECKING:
    from pathlib import Path

POLYGON = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])


def _placed(option: int, part: str, *numbers: int) -> list[PlacedQuestion]:
    return [
        PlacedQuestion(
            regions=[PageRegion(label="question", polygon=POLYGON)],
            placement=QuestionPlacement(option=option, part=part, number=number),
        )
        for number in numbers
    ]


def _fill(root: Path, option: int, part: str, *numbers: int) -> None:
    folder = root / str(option) / part
    folder.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        (folder / f"{number}.jpg").write_bytes(b"x")


class TestNumberingFault:
    def test_a_fresh_folder_must_start_at_one(self, tmp_path: Path) -> None:
        assert numbering_fault(_placed(1, "A", 1, 2), tmp_path) is None

        fault = numbering_fault(_placed(1, "A", 2, 3), tmp_path)

        assert fault is not None
        assert (fault.free, fault.placement.number) == (1, 2)
        assert fault.collides is False  # a gap, not an overwrite

    def test_continuing_from_what_is_on_disk_is_clean(self, tmp_path: Path) -> None:
        _fill(tmp_path, 1, "A", 1, 2, 3, 4, 5)

        assert numbering_fault(_placed(1, "A", 6, 7, 8), tmp_path) is None

    def test_landing_on_an_existing_number_is_a_collision(self, tmp_path: Path) -> None:
        _fill(tmp_path, 1, "A", 1, 2, 3, 4, 5)

        fault = numbering_fault(_placed(1, "A", 3, 4), tmp_path)

        assert fault is not None
        assert fault.collides is True
        assert fault.free == 6
        assert fault.position == 0

    def test_landing_past_the_end_would_leave_a_gap(self, tmp_path: Path) -> None:
        """The hole renumbering used to repair, refused before it is written."""
        _fill(tmp_path, 1, "A", 1, 2, 3, 4, 5)

        fault = numbering_fault(_placed(1, "A", 9, 10), tmp_path)

        assert fault is not None
        assert fault.collides is False
        assert fault.free == 6

    def test_only_where_each_folder_starts_is_checked(self, tmp_path: Path) -> None:
        """The rest follow by construction, so they cannot be wrong on their own."""
        _fill(tmp_path, 1, "A", 1)

        assert numbering_fault(_placed(1, "A", 2, 3, 4), tmp_path) is None

    def test_a_reentered_folder_is_checked_like_the_first(self, tmp_path: Path) -> None:
        """A marker mid-page resets the counter, so a re-entered run restarts at 1.

        An option marker OCR missed plus a part marker it read produces exactly
        this shape — the re-entry must not hide behind the folder's first run.
        """
        _fill(tmp_path, 1, "A", 1, 2)

        placed = _placed(1, "A", 3) + _placed(1, "B", 1) + _placed(1, "A", 1)
        fault = numbering_fault(placed, tmp_path)

        assert fault is not None
        assert fault.position == 2
        assert fault.collides is True
        assert fault.free == 3

    def test_a_second_folder_is_checked_on_its_own_terms(self, tmp_path: Path) -> None:
        """A page spanning two parts: the first continues, the second collides."""
        _fill(tmp_path, 1, "A", 1, 2)
        _fill(tmp_path, 1, "B", 1, 2, 3)
        placed = _placed(1, "A", 3, 4) + _placed(1, "B", 2)

        fault = numbering_fault(placed, tmp_path)

        assert fault is not None
        assert fault.placement == QuestionPlacement(1, "B", 2)
        assert fault.position == 2
        assert fault.free == 4

    def test_a_page_with_no_questions_has_nothing_to_fault(
        self, tmp_path: Path
    ) -> None:
        assert numbering_fault([], tmp_path) is None
