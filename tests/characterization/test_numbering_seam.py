"""The numbering rule, asserted on both sides of the reviewer seam at once.

`docs/glossary.md` states it as one rule: *the review window refuses to approve a
fault, and the extractor replays every page through the same check before
writing — a gap refuses the page, a collision keeps the existing file.* It is
the most subtle rule in the project, and it is the reason a resumed year can
replay its own pages without destroying what it already wrote.

But it lives in two places. `PageEdits.numbering` decides what a reviewer may
approve; `PageExtractor.extract` decides what actually gets written. Every
existing test covers one side or the other. Nothing yet fails if the two drift
apart, which is exactly what a restructuring that moves either one can do —
and the damage would be silent and permanent: an overwritten question image,
or a hole in the output tree that no renumbering pass exists to close.

These tests are written against behaviour, not structure: a page, a folder,
and what each side says about them. They are meant to survive every phase of
the rewrite untouched. If a phase makes one of them hard to express, that is
the phase getting the rule wrong, not the test needing an edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

from digitex.domain.corpus import question_image_path
from digitex.domain.entities import Detection, PixelPolygon
from digitex.domain.placement import PageExtractionState, PageRegion
from digitex.pipeline.base import ExtractionConfig
from digitex.pipeline.page import PageExtractor
from digitex.ui.edits import PageEdits

if TYPE_CHECKING:
    from pathlib import Path

FIRST_QUESTION = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])
SECOND_QUESTION = PixelPolygon([(10, 90), (200, 90), (200, 130), (10, 130)])


class _FakePredictor:
    def predict(self, image: Image.Image) -> list[Detection]:
        return [
            Detection(label="question", polygon=FIRST_QUESTION, score=0.9),
            Detection(label="question", polygon=SECOND_QUESTION, score=0.9),
        ]


class _FakeTextExtractor:
    language = "rus"

    def extract_digits(self, image: Image.Image) -> list[int]:
        return []

    def extract_text(self, image: Image.Image) -> str:
        return ""

    def detect_skew(self, image: Image.Image) -> float:
        return 0.0


def _page() -> Image.Image:
    return Image.new("RGB", (300, 300), color="white")


def _regions() -> list[PageRegion]:
    """Two questions and no markers, so the entry state alone numbers them."""
    return [
        PageRegion(label="question", polygon=FIRST_QUESTION),
        PageRegion(label="question", polygon=SECOND_QUESTION),
    ]


def _entering_at(number: int) -> PageExtractionState:
    """A page that starts numbering at *number* in option 1, part A."""
    return PageExtractionState(option=1, part="A", question=number - 1)


def _already_extracted(output_dir: Path, *numbers: int) -> None:
    """Put question images on disk, the way an earlier page would have."""
    for number in numbers:
        path = question_image_path(output_dir, 1, "A", number, "jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), color="black").save(path)


def _reviewer_verdict(output_dir: Path, entering_at: int) -> str | None:
    """What the review window would say about the page — None when approvable."""
    edits = PageEdits()
    edits.load(_regions(), _entering_at(entering_at), output_dir)
    return edits.numbering().problem


def _extractor_verdict(output_dir: Path, entering_at: int) -> tuple[str, list[str]]:
    """What the extractor does with the same page.

    Returns the outcome — ``"wrote"``, ``"refused"`` or ``"kept"`` — and the
    files that exist afterwards, so a test can assert nothing was destroyed.
    """
    extractor = PageExtractor(
        ExtractionConfig(
            image_format="jpg",
            question_max_width=50,
            question_max_height=50,
        ),
        detector=_FakePredictor(),
        text_reader=_FakeTextExtractor(),
    )
    state = _entering_at(entering_at)
    try:
        collisions = extractor.extract(_page(), output_dir, state)
    except ValueError:
        outcome = "refused"
    else:
        outcome = "kept" if collisions else "wrote"
    written = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    return outcome, written


class TestAFreshFolderTakesThePage:
    def test_the_reviewer_approves_it(self, tmp_path: Path) -> None:
        assert _reviewer_verdict(tmp_path, entering_at=1) is None

    def test_the_extractor_writes_it(self, tmp_path: Path) -> None:
        outcome, written = _extractor_verdict(tmp_path, entering_at=1)

        assert outcome == "wrote"
        assert written == ["1/A/1.jpg", "1/A/2.jpg"]


class TestAGapIsRefusedByBothSides:
    """The unsurvivable fault: nothing would ever come back to fill the hole."""

    def test_the_reviewer_refuses_to_approve(self, tmp_path: Path) -> None:
        _already_extracted(tmp_path, 1)

        problem = _reviewer_verdict(tmp_path, entering_at=5)

        assert problem is not None
        assert "gap" in problem

    def test_the_extractor_refuses_the_page(self, tmp_path: Path) -> None:
        _already_extracted(tmp_path, 1)

        outcome, written = _extractor_verdict(tmp_path, entering_at=5)

        assert outcome == "refused"
        assert written == ["1/A/1.jpg"], "a refused page must write nothing"


class TestACollisionStopsTheReviewerButNotTheRun:
    """The asymmetry that lets a resumed year replay its own pages.

    The reviewer will not approve a page onto numbers that are already there,
    because a human seeing that has almost certainly set the entry state wrong.
    The extractor, replaying a year that was interrupted, meets its own earlier
    output constantly — so it keeps the existing file and reports the
    collision rather than failing the book.
    """

    def test_the_reviewer_refuses_to_approve(self, tmp_path: Path) -> None:
        _already_extracted(tmp_path, 1, 2, 3)

        problem = _reviewer_verdict(tmp_path, entering_at=1)

        assert problem is not None
        assert "already exists" in problem

    def test_the_extractor_keeps_what_is_already_there(self, tmp_path: Path) -> None:
        _already_extracted(tmp_path, 1, 2, 3)
        before = (tmp_path / "1" / "A" / "1.jpg").read_bytes()

        outcome, written = _extractor_verdict(tmp_path, entering_at=1)

        assert outcome == "kept"
        assert written == ["1/A/1.jpg", "1/A/2.jpg", "1/A/3.jpg"]
        assert (tmp_path / "1" / "A" / "1.jpg").read_bytes() == before

    def test_this_is_the_only_case_the_two_sides_differ_on(
        self, tmp_path: Path
    ) -> None:
        """Pins the asymmetry itself, so neither side can quietly adopt the other.

        Made symmetric in either direction, something breaks: a reviewer that
        allowed collisions would let a human overwrite an extracted question,
        and an extractor that refused them could never finish an interrupted
        year.
        """
        _already_extracted(tmp_path, 1, 2, 3)

        assert _reviewer_verdict(tmp_path, entering_at=1) is not None
        assert _extractor_verdict(tmp_path, entering_at=1)[0] != "refused"


class TestBothSidesReadTheSameFolder:
    @pytest.mark.parametrize("entering_at", [1, 2, 4, 7])
    def test_they_agree_on_whether_the_page_is_clean(
        self, tmp_path: Path, entering_at: int
    ) -> None:
        """Whatever is on disk, "approvable" and "writes cleanly" coincide.

        The collision case above is the documented exception; every other entry
        point has to give the same answer on both sides, or the review window
        is approving pages the extractor will reject and vice versa.
        """
        _already_extracted(tmp_path, 1, 2, 3)

        approvable = _reviewer_verdict(tmp_path, entering_at) is None
        outcome, _ = _extractor_verdict(tmp_path, entering_at)

        if outcome == "kept":
            pytest.skip("the documented collision asymmetry, pinned above")
        assert approvable == (outcome == "wrote")
