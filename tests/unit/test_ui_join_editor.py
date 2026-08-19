"""Behaviour tests for the window a question's pieces are lined up in.

The arithmetic it lays the pieces out with is `stacked_layout`, tested in
`test_imaging_image`. What is left for here is the window's own half: which
pieces it will move, what a drag does to their offsets, and what it hands back.

Every test skips where there is no display, so a headless CI runs the rest of
the suite untouched.
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from digitex.ui.join_editor import JoinEditor, JoinPiece

if TYPE_CHECKING:
    from collections.abc import Iterator

TOP = (400, 120)
BOTTOM = (400, 200)


def _pieces() -> list[JoinPiece]:
    """A question in two pieces: one carried from an earlier page, one on this."""
    return [
        JoinPiece(
            image=Image.new("RGB", TOP, "white"),
            offset=(0, 0),
            caption="from 003.jpg",
            movable=False,
        ),
        JoinPiece(
            image=Image.new("RGB", BOTTOM, "white"),
            offset=(0, 0),
            caption="region 1, this page",
        ),
    ]


@pytest.fixture(scope="module")
def root() -> Iterator[tk.Tk]:
    try:
        made = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    made.withdraw()
    yield made
    made.destroy()


@pytest.fixture
def editor(root: tk.Tk) -> Iterator[JoinEditor]:
    made = JoinEditor(root, _pieces())
    yield made
    if made.top.winfo_exists():
        made.top.destroy()


class TestWhatCanMove:
    def test_the_first_piece_anchors_the_stack(self, editor: JoinEditor) -> None:
        """Only the seams below it mean anything — the top piece has none."""
        assert editor._movable(0) is False

    def test_a_piece_cut_from_this_page_can_be_moved(self, editor: JoinEditor) -> None:
        assert editor._movable(1) is True

    def test_the_movable_piece_starts_selected(self, editor: JoinEditor) -> None:
        assert editor._selected == 1

    def test_a_piece_carried_from_an_earlier_page_is_left_alone(
        self, root: tk.Tk
    ) -> None:
        """Its seam was settled while that page was being reviewed."""
        pieces = _pieces()
        pieces.append(
            JoinPiece(
                image=Image.new("RGB", TOP, "white"),
                offset=(0, 0),
                caption="from 004.jpg",
                movable=False,
            )
        )
        made = JoinEditor(root, pieces)

        assert [made._movable(at) for at in range(3)] == [False, True, False]

        made.top.destroy()


class TestMovingAPiece:
    def test_a_nudge_lands_on_the_offset(self, editor: JoinEditor) -> None:
        editor._move(1, 6, -4)

        assert editor._offsets[1] == (6, -4)

    def test_nudges_accumulate(self, editor: JoinEditor) -> None:
        editor._move(1, 6, 0)
        editor._move(1, -2, 3)

        assert editor._offsets[1] == (4, 3)

    def test_a_piece_cannot_be_thrown_off_the_canvas(self, editor: JoinEditor) -> None:
        """A stray drag stops at the piece's own size, where it is still visible."""
        editor._move(1, 10_000, -10_000)

        assert editor._offsets[1] == (BOTTOM[0], -BOTTOM[1])

    def test_the_piece_is_drawn_where_its_offset_puts_it(
        self, editor: JoinEditor
    ) -> None:
        before = editor._canvas.coords(editor._images[1])

        editor._move(1, 40, 0)

        after = editor._canvas.coords(editor._images[1])
        assert after[0] > before[0]

    def test_the_readout_names_the_piece_and_its_offset(
        self, editor: JoinEditor
    ) -> None:
        editor._move(1, 6, -4)

        assert editor._readout["text"] == "piece 2: x +6, y -4 px"


class TestWhatItHandsBack:
    def test_done_hands_back_every_offset(self, editor: JoinEditor) -> None:
        editor._move(1, 6, -4)

        editor._done()

        assert editor._result == [(0, 0), (6, -4)]

    def test_cancel_hands_back_nothing(self, editor: JoinEditor) -> None:
        editor._move(1, 6, -4)

        editor._cancel()

        assert editor._result is None

    def test_reset_puts_every_piece_back(self, editor: JoinEditor) -> None:
        editor._move(1, 6, -4)

        editor._reset()

        assert editor._offsets == [(0, 0), (0, 0)]
