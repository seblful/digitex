"""Behaviour tests for a page review.

Most of these take a `ReviewController` and no window at all: what a review is
— which question each region lands in, what each row says, whether the page may
be approved, what verdict comes back — needs no display to check, and every one
of these used to skip on a machine without one.

Eight still take a real `_ReviewWindow`, and they are the ones that are
genuinely about the widget: turning canvas coordinates into page pixels,
arrow keys scrolling a canvas rather than nudging a region, a spinbox echoing
the model without echoing back. Those skip without a display, because a
display is the thing under test.

The window is built directly rather than through `TkPageReviewer.present`,
which blocks until a human answers.
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from digitex.domain.entities import PixelPolygon
from digitex.domain.placement import PageExtractionState, PageLabel, PageRegion
from digitex.imaging import stack_vertically
from digitex.pipeline.exceptions import ReviewAborted
from digitex.pipeline.pieces import HeldPiece
from digitex.pipeline.review import PageProposal
from digitex.ui.controller import ReviewController
from digitex.ui.edits import PageEdits
from digitex.ui.page_review import _ReviewWindow, resolve_verdict

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

PAGE_SIZE = (600, 900)


def _polygon(top: int) -> PixelPolygon:
    return PixelPolygon([(50, top), (550, top), (550, top + 60), (50, top + 60)])


def _region(label: PageLabel, top: int, reading: int | str | None = None) -> PageRegion:
    return PageRegion(label=label, polygon=_polygon(top), reading=reading)


def _bounds(polygon: PixelPolygon) -> tuple[int, int, int, int]:
    return (
        min(p[0] for p in polygon),
        min(p[1] for p in polygon),
        max(p[0] for p in polygon),
        max(p[1] for p in polygon),
    )


def _proposal(
    output_dir: Path,
    regions: list[PageRegion] | None = None,
    state: PageExtractionState | None = None,
    page_number: int = 0,
    page_count: int = 0,
    carried: list[HeldPiece] | None = None,
) -> PageProposal:
    """A page carrying one option marker, one part marker and two questions."""
    image = Image.new("RGB", PAGE_SIZE, "white")

    def crop_piece(polygon: PixelPolygon) -> Image.Image:
        return image.crop(_bounds(polygon))

    def crop(
        pieces: Sequence[PageRegion], carried_pieces: Sequence[HeldPiece]
    ) -> Image.Image:
        """Stands in for the extractor: the pieces of one question, stacked."""
        return stack_vertically(
            [piece.image for piece in carried_pieces]
            + [crop_piece(piece.polygon) for piece in pieces]
        )

    return PageProposal(
        image=image,
        regions=regions
        if regions is not None
        else [
            _region("option", 10, 1),
            _region("part", 90, "A"),
            _region("question", 200),
            _region("question", 300),
        ],
        state=state or PageExtractionState(),
        output_dir=output_dir,
        page_name="1.jpg",
        crop=crop,
        crop_piece=crop_piece,
        page_number=page_number,
        page_count=page_count,
        carried=carried or [],
    )


def _carried(page_name: str = "0.jpg") -> list[HeldPiece]:
    """One piece left unfinished by the page before, 40px tall."""
    return [HeldPiece(image=Image.new("RGB", (500, 40), "black"), page_name=page_name)]


class TestResolveVerdict:
    """The verdict translation is pure, so the seam's exits run headless.

    These are the three ways a run leaves the review window — the part of the
    adapter a display-gated suite would otherwise never assert on CI.
    """

    @staticmethod
    def _edits(output_dir: Path) -> PageEdits:
        edits = PageEdits()
        edits.load(
            [_region("question", 200)],
            PageExtractionState(option=1, part="A"),
            output_dir,
        )
        return edits

    def test_approve_hands_back_what_the_reviewer_edited(self, tmp_path: Path) -> None:
        edits = self._edits(tmp_path)

        reviewed = resolve_verdict("approve", edits, "1.jpg")

        assert reviewed is not None
        assert reviewed.regions is edits.regions
        assert reviewed.state is edits.state

    def test_skip_returns_none(self, tmp_path: Path) -> None:
        assert resolve_verdict("skip", self._edits(tmp_path), "1.jpg") is None

    def test_abort_raises_naming_the_page(self, tmp_path: Path) -> None:
        with pytest.raises(ReviewAborted, match=r"7\.jpg"):
            resolve_verdict("abort", self._edits(tmp_path), "7.jpg")


@pytest.fixture(scope="module")
def root() -> Iterator[tk.Tk]:
    """One interpreter for the module.

    Creating and tearing down a Tk root per test fails intermittently on
    Windows, which showed up as tests skipping themselves at random. One root
    and a window per test is both steadier and faster.
    """
    try:
        made = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    made.withdraw()
    yield made
    made.destroy()


@pytest.fixture
def control() -> ReviewController:
    """A review with no window around it — what most of these tests need."""
    return ReviewController()


@pytest.fixture
def window(root: tk.Tk) -> Iterator[_ReviewWindow]:
    """A real window, for the handful of tests that are about the widget."""
    made = _ReviewWindow(root)
    yield made
    made.top.destroy()


class TestLoadingAPage:
    def test_the_proposal_is_copied_not_borrowed(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        """Skipping must leave the extractor's own regions as they were."""
        proposal = _proposal(tmp_path)
        control.load(proposal)

        control.edits.regions[2].label = "part"

        assert proposal.regions[2].label == "question"

    def test_every_question_is_shown_where_it_would_be_saved(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))

        assert [str(p) for p in control.numbering.placements] == [
            "1/A/1",
            "1/A/2",
        ]
        assert control.numbering.problem is None
        assert control.approve_enabled

    def test_the_region_list_matches_the_page(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))

        rows = control.rows()
        values = [[row.label, row.reading, row.where] for row in rows]

        assert len(rows) == 4
        assert values[0][:2] == ["1. option", "1"]
        assert values[2] == ["3. question", "", "1/A/1"]

    def test_a_second_page_starts_with_a_clean_history(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)
        control.delete_selected(), control.refresh()
        assert control.edits.history.can_undo

        control.load(_proposal(tmp_path))

        assert control.edits.history.can_undo is False
        assert len(control.edits.regions) == 4


class TestEditing:
    def test_nudging_moves_the_selected_region_only(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        window._load(_proposal(tmp_path))
        window._select(2)
        before = list(window.control.edits.regions[3].polygon)

        window._on_arrow((1, 0), 10)

        assert next(iter(window.control.edits.regions[2].polygon)) == (60, 200)
        assert list(window.control.edits.regions[3].polygon) == before

    def test_undo_puts_an_edit_back(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        window._load(_proposal(tmp_path))
        window._select(2)
        window._on_arrow((0, 1), 10)

        window._undo()

        assert next(iter(window.control.edits.regions[2].polygon)) == (50, 200)
        assert window.control.edits.history.can_redo

    def test_redo_puts_it_back_again(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        window._load(_proposal(tmp_path))
        window._select(2)
        window._on_arrow((0, 1), 10)
        window._undo()

        window._redo()

        assert next(iter(window.control.edits.regions[2].polygon)) == (50, 210)

    def test_relabelling_a_question_renumbers_the_rest(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)

        control.relabel_selected("part")

        # The first question is now a marker, so the second takes its number.
        assert [str(p) for p in control.numbering.placements] == ["1/A/1"]

    def test_relabelling_drops_a_reading_that_belonged_to_the_old_kind(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(0)

        control.relabel_selected("part")

        assert control.edits.regions[0].reading is None

    def test_deleting_a_marker_is_caught_before_anything_is_written(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        """Without a marker a crop would land outside {option}/{part}/."""
        control.load(_proposal(tmp_path, state=PageExtractionState()))
        control.select(0)
        control.delete_selected()
        control.select(0)
        control.delete_selected()

        assert control.numbering.problem is not None
        assert "before any option/part marker" in control.numbering.problem
        assert not control.approve_enabled

    def test_drawing_adds_a_region_in_page_pixels(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        window._load(_proposal(tmp_path))
        window._set_scale(0.5)

        window._start_draw("question")
        window._draw_from = (50.0, 100.0)
        window._finish_draw(150.0, 200.0)

        assert len(window.control.edits.regions) == 5
        assert list(window.control.edits.regions[4].polygon) == [
            (100, 200),
            (300, 200),
            (300, 400),
            (100, 400),
        ]

    def test_a_stray_click_draws_nothing(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        window._load(_proposal(tmp_path))

        window._start_draw("question")
        window._draw_from = (50.0, 100.0)
        window._finish_draw(52.0, 102.0)

        assert len(window.control.edits.regions) == 4

    def test_sorting_puts_the_regions_in_reading_order(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        regions = [
            _region("question", 300),
            _region("option", 10, 1),
            _region("part", 90, "A"),
        ]
        control.load(_proposal(tmp_path, regions=regions))
        assert control.numbering.problem is not None  # question before its markers

        control.sort_by_reading_order(), control.refresh()

        assert [r.label for r in control.edits.regions] == [
            "option",
            "part",
            "question",
        ]
        assert control.numbering.problem is None


class TestNumbering:
    def test_a_number_already_on_disk_blocks_approval(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        taken = tmp_path / "1" / "A"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")

        control.load(_proposal(tmp_path))

        assert control.numbering.problem is not None
        assert "already exists" in control.numbering.problem
        assert not control.approve_enabled
        assert control.numbering.misnumbered == {2}

    def test_continue_from_disk_is_offered_only_when_it_would_help(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        """A marker leads this page, so the entry counter cannot move it."""
        taken = tmp_path / "1" / "A"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")

        control.load(_proposal(tmp_path))

        assert not control.continue_helps

    def test_continue_from_disk_picks_up_where_the_folder_left_off(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        taken = tmp_path / "2" / "B"
        taken.mkdir(parents=True)
        for number in (1, 2, 3):
            (taken / f"{number}.jpg").write_bytes(b"x")
        control.load(
            _proposal(
                tmp_path,
                regions=[_region("question", 200)],
                state=PageExtractionState(option=2, part="B"),
            )
        )
        assert control.continue_helps

        control.continue_from_disk()

        assert [str(p) for p in control.numbering.placements] == ["2/B/4"]
        assert control.numbering.problem is None

    def test_the_entry_state_can_be_typed_in(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        window._load(
            _proposal(
                tmp_path,
                regions=[_region("question", 200)],
                state=PageExtractionState(option=1, part="A"),
            )
        )

        window._question_var.set("5")

        assert window.control.edits.state.question == 5
        assert [str(p) for p in window.control.numbering.placements] == ["1/A/6"]

    def test_an_emptied_spinbox_does_not_lose_the_number(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        """Clearing the field to retype it must not be read as zero."""
        window._load(
            _proposal(
                tmp_path,
                regions=[_region("question", 200)],
                state=PageExtractionState(option=3, part="A"),
            )
        )

        window._option_var.set("")

        assert window.control.edits.state.option == 3


class TestCropPreview:
    def test_a_question_previews_the_file_that_would_be_written(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)

        control.preview()

        assert "saved as 1/A/1" in control.preview().caption
        assert control.preview().image is not None

    def test_a_marker_previews_what_ocr_was_pointed_at(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(0)

        control.preview()

        assert "option marker" in control.preview().caption

    def test_nothing_selected_previews_nothing(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)
        control.preview()

        control.select(None)
        control.preview()

        assert "select a region" in control.preview().caption

    def test_a_page_without_a_crop_callable_still_previews(
        self, window: _ReviewWindow, tmp_path: Path
    ) -> None:
        """`crop` is optional on the seam, so the window cannot depend on it."""
        image = Image.new("RGB", PAGE_SIZE, "white")
        window._load(
            PageProposal(
                image=image,
                regions=[_region("question", 200)],
                state=PageExtractionState(option=1, part="A"),
                output_dir=tmp_path,
            )
        )
        window._select(0)

        window.control.preview()

        assert window.control.preview().image is not None


class TestJoiningPieces:
    """Marking a question as printed in pieces, and lining the pieces up."""

    def test_the_checkbox_marks_the_selected_question(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)

        control.toggle_join()

        assert control.edits.regions[2].joins_next is True
        assert "piece 2 of 2" in control.rows()[3].where

    def test_a_marker_cannot_be_marked_as_a_piece(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(0)

        control.toggle_join()

        assert control.edits.regions[0].joins_next is False
        # The box is put back from the model rather than left lying.
        assert control.join_controls().joins_next is False
        assert not control.join_controls().can_toggle

    def test_the_last_question_shows_that_it_waits_for_the_next_page(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(3)
        control.toggle_join()

        assert control.rows()[3].where == "piece 1 → next page"
        assert "1 piece held" in control.status().counts
        # Holding a piece is not a fault: the next page finishes it.
        assert control.approve_enabled

    def test_the_controls_follow_the_selection(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        """A checkbox left showing the region before it would mark the wrong one."""
        control.load(_proposal(tmp_path))
        control.select(3)
        control.toggle_join()

        control.select(2)

        assert control.join_controls().joins_next is False
        assert control.join_controls().can_toggle

        control.select(0)

        assert not control.join_controls().can_toggle

    def test_lining_pieces_up_needs_more_than_one_piece(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)
        assert not control.join_controls().can_line_up

        control.toggle_join()

        assert control.join_controls().can_line_up

    def test_the_pieces_handed_to_the_editor_come_from_the_page(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)
        control.toggle_join(), control.refresh()

        pieces, origins = control.join_pieces(2)

        assert origins == [2, 3]
        assert [piece.movable for piece in pieces] == [True, True]

    def test_a_joined_question_previews_all_of_its_pieces(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))
        control.select(2)
        control.toggle_join()

        control.preview()

        assert "2 pieces joined" in control.preview().caption


class TestCarriedPieces:
    """A page finishing a question the page before it started."""

    def test_the_carried_piece_is_named_and_the_question_opens_selected(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path, carried=_carried("003.jpg")))

        assert "003.jpg" in (control.carried_summary() or "")
        assert control.carried_summary() is not None
        # The joined crop is what the reviewer has to check first.
        assert control.edits.selected == 2

    def test_a_page_handed_nothing_says_nothing(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))

        assert control.carried_summary() is None

    def test_the_first_question_previews_the_carried_piece_with_it(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path, carried=_carried()))
        control.select(2)

        control.preview()

        assert "2 pieces joined" in control.preview().caption
        row = [
            control.rows()[2].label,
            control.rows()[2].reading,
            control.rows()[2].where,
        ]
        assert row[2] == "1/A/1  piece 2 of 2"

    def test_the_editor_cannot_move_a_piece_from_an_earlier_page(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path, carried=_carried()))

        pieces, origins = control.join_pieces(2)

        assert origins == [None, 2]
        assert [piece.movable for piece in pieces] == [False, True]

    def test_discarding_the_carried_piece_drops_it_from_the_page(
        self, control: ReviewController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "digitex.ui.page_review.messagebox.askokcancel", lambda *a, **k: True
        )
        control.load(_proposal(tmp_path, carried=_carried()))

        control.discard_carried_pieces()

        assert control.discard_carried is True
        assert control.carried_summary() is None
        assert control.edits.takes_carried(2) is False

    def test_a_page_is_handed_the_pieces_unless_it_says_otherwise(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        """Only a discard drops them — a plain approval joins them."""
        control.load(_proposal(tmp_path, carried=_carried()))

        control.finish("approve")

        assert control.discard_carried is False


class TestStatusLine:
    def test_the_page_reports_its_place_in_the_run(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path, page_number=3, page_count=40))

        assert "page 3 of 40" in control.status().where
        assert "2 questions, 2 markers" in control.status().counts

    def test_a_page_extracted_outside_a_book_says_only_its_name(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))

        assert control.status().where.strip() == "1.jpg"


class TestVerdict:
    def test_approving_a_faulty_page_is_refused(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        taken = tmp_path / "1" / "A"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")
        control.load(_proposal(tmp_path))

        control.finish("approve")

        assert control.verdict == "abort"  # unchanged: the click did nothing

    def test_approving_a_clean_page_settles_it(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))

        control.finish("approve")

        assert control.verdict == "approve"

    def test_skipping_needs_no_confirmation(
        self, control: ReviewController, tmp_path: Path
    ) -> None:
        control.load(_proposal(tmp_path))

        control.finish("skip")

        assert control.verdict == "skip"
