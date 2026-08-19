"""Lining the pieces of one question up by hand.

A question printed across a page break comes back as two crops that have to be
stacked into one image, and no rule decides how: the two pages were laid on the
scanner separately, so the halves meet at an offset only an eye can settle. This
window shows them stacked as they would be saved and lets every piece below the
first be dragged into place. The offsets it hands back are what the crop is
written with.

Nothing here reads the page — the pieces arrive already cut. The stacking is
:func:`digitex.imaging.stacked_layout`, the same arithmetic that builds the
file, so what is dragged here is what lands on disk.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import TYPE_CHECKING

from PIL import Image, ImageTk

from digitex.imaging import stacked_layout
from digitex.pipeline.pieces import PIECE_GAP
from digitex.ui.display import scaled

if TYPE_CHECKING:
    from collections.abc import Sequence

# The review window's palette, as far as this window needs it.
ACCENT = "#2f6fed"
CANVAS_BG = "#2b2d31"
MUTED = "#4b5563"
FIXED = "#8a94a6"

# Written for a 100% display, then put through `scaled()`.
CANVAS_WIDTH = 620
CANVAS_HEIGHT = 700

# Room around the stack, as a share of the canvas, so a piece dragged clear of
# the one above it does not leave the view.
MARGIN = 0.25

# Arrow keys nudge by a source pixel, Shift by a stride worth seeing.
NUDGE = 1
NUDGE_FAST = 10


@dataclass(frozen=True)
class JoinPiece:
    """One piece of a question, as the join editor shows it.

    ``offset`` is where it sits against the piece above it. ``movable`` is False
    for a piece whose offset this page cannot change — one carried in from an
    earlier page, which was lined up while that page was being reviewed.
    """

    image: Image.Image
    offset: tuple[int, int]
    caption: str
    movable: bool = True


class JoinEditor:
    """Modal window over one question's pieces, handing back their offsets.

    Built and run in one go::

        offsets = JoinEditor(parent, pieces).run()

    ``run`` returns None when the reviewer cancels — the pieces stay as they
    were — and otherwise one offset per piece, in the order they were given.
    """

    def __init__(
        self,
        parent: tk.Misc,
        pieces: Sequence[JoinPiece],
        gap: int = PIECE_GAP,
        scale: float = 1.0,
    ) -> None:
        self._pieces = list(pieces)
        self._gap = gap
        self._dpi_scale = scale
        self._offsets = [piece.offset for piece in self._pieces]
        self._result: list[tuple[int, int]] | None = None
        self._drag: tuple[float, float] | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._images: list[int] = []
        self._frames: list[int] = []
        self._captions: list[tuple[int, int]] = []
        self._selected = next(
            (index for index in range(len(self._pieces)) if self._movable(index)), None
        )

        self.top = tk.Toplevel(parent)
        self.top.title("Line up the pieces")
        self.top.transient(parent.winfo_toplevel())
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)

        self._build()
        # A scale for the stack as it stands, held for the session: refitting
        # under a drag would move the piece the reviewer is aiming with.
        self._scale = self._fit_scale()
        self._render()
        self._show_selection()

    # --- construction ---

    def _px(self, value: int) -> int:
        return scaled(value, self._dpi_scale)

    def _build(self) -> None:
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(1, weight=1)

        ttk.Label(
            self.top,
            text="Drag a piece to line it up with the one above."
            "  Arrow keys nudge it a pixel, Shift ten.",
            foreground=MUTED,
            padding=(10, 8),
        ).grid(row=0, column=0, sticky="w")

        self._canvas = tk.Canvas(
            self.top,
            width=self._px(CANVAS_WIDTH),
            height=self._px(CANVAS_HEIGHT),
            background=CANVAS_BG,
            highlightthickness=0,
            takefocus=True,
        )
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=10)
        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))

        footer = ttk.Frame(self.top, padding=(10, 8))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        self._readout = ttk.Label(footer, text="", foreground=MUTED)
        self._readout.grid(row=0, column=0, sticky="w")

        ttk.Button(footer, text="Done", command=self._done).grid(row=0, column=3)
        ttk.Button(footer, text="Cancel", command=self._cancel).grid(
            row=0, column=2, padx=6
        )
        ttk.Button(footer, text="Reset", command=self._reset).grid(row=0, column=1)

        for key, delta in (
            ("<Up>", (0, -1)),
            ("<Down>", (0, 1)),
            ("<Left>", (-1, 0)),
            ("<Right>", (1, 0)),
        ):
            self.top.bind(key, lambda _e, d=delta: self._nudge(d, NUDGE))
            self.top.bind(
                f"<Shift-{key[1:-1]}>", lambda _e, d=delta: self._nudge(d, NUDGE_FAST)
            )
        self.top.bind("<Tab>", lambda _e: self._cycle())
        self.top.bind("<Return>", lambda _e: self._done())
        self.top.bind("<Escape>", lambda _e: self._cancel())

    # --- the seam ---

    def run(self) -> list[tuple[int, int]] | None:
        """Show the window and block until the reviewer is done with it."""
        self.top.update_idletasks()
        self.top.grab_set()
        self._canvas.focus_set()
        self.top.wait_window()
        return self._result

    def _done(self) -> None:
        self._result = list(self._offsets)
        self.top.destroy()

    def _cancel(self) -> None:
        self._result = None
        self.top.destroy()

    def _reset(self) -> None:
        """Put every piece back where a plain stack would have left it."""
        self._offsets = [(0, 0) for _ in self._pieces]
        self._place()

    # --- layout ---

    def _movable(self, index: int) -> bool:
        """True for a piece this window can move — never the first, which anchors."""
        return index > 0 and self._pieces[index].movable

    def _layout(self) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        return stacked_layout(
            [piece.image.size for piece in self._pieces], self._gap, self._offsets
        )

    def _fit_scale(self) -> float:
        (width, height), _ = self._layout()
        view_w = max(self._canvas.winfo_reqwidth(), 1)
        view_h = max(self._canvas.winfo_reqheight(), 1)
        return min(view_w / max(width, 1), view_h / max(height, 1), 1.0)

    def _render(self) -> None:
        """Draw every piece once. Moving them afterwards only moves the items."""
        self._canvas.delete("all")
        self._photos = []
        self._images = []
        self._frames = []
        self._captions = []

        for index, piece in enumerate(self._pieces):
            shown = piece.image.convert("RGB").resize(
                (
                    max(1, round(piece.image.width * self._scale)),
                    max(1, round(piece.image.height * self._scale)),
                ),
                Image.Resampling.BILINEAR,
            )
            photo = ImageTk.PhotoImage(shown)
            self._photos.append(photo)
            self._images.append(
                self._canvas.create_image(0, 0, image=photo, anchor="nw")
            )
            self._frames.append(
                self._canvas.create_rectangle(0, 0, 0, 0, outline=FIXED, width=2)
            )
            label = self._canvas.create_text(
                0,
                0,
                text=f"{index + 1}. {piece.caption}",
                anchor="nw",
                fill=MUTED,
                font=("TkDefaultFont", 9),
            )
            chip = self._canvas.create_rectangle(
                0, 0, 0, 0, fill="#ffffff", outline=FIXED
            )
            self._canvas.tag_lower(chip, label)
            self._captions.append((label, chip))

        self._place()

    def _place(self) -> None:
        """Move the pieces to where the current offsets put them."""
        (width, height), positions = self._layout()
        scale = self._scale
        margin_x = self._canvas.winfo_reqwidth() * MARGIN
        margin_y = self._canvas.winfo_reqheight() * MARGIN
        self._canvas.configure(
            scrollregion=(
                -margin_x,
                -margin_y,
                width * scale + margin_x,
                height * scale + margin_y,
            )
        )

        for index, ((x, y), piece) in enumerate(
            zip(positions, self._pieces, strict=True)
        ):
            left, top = x * scale, y * scale
            right = left + piece.image.width * scale
            bottom = top + piece.image.height * scale
            self._canvas.coords(self._images[index], left, top)
            self._canvas.coords(self._frames[index], left, top, right, bottom)
            self._canvas.itemconfigure(
                self._frames[index],
                outline=ACCENT if index == self._selected else FIXED,
                dash=() if self._movable(index) else (4, 3),
            )

            label, chip = self._captions[index]
            self._canvas.coords(label, left + 6, top + 4)
            box = self._canvas.bbox(label)
            if box:
                self._canvas.coords(
                    chip, box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2
                )
            self._canvas.tag_raise(label)

        self._show_selection()

    def _show_selection(self) -> None:
        if self._selected is None:
            self._readout.configure(text="nothing here can be moved")
            return
        dx, dy = self._offsets[self._selected]
        self._readout.configure(
            text=f"piece {self._selected + 1}: x {dx:+d}, y {dy:+d} px"
        )

    # --- moving a piece ---

    def _select(self, index: int) -> None:
        if index == self._selected:
            return
        self._selected = index
        self._place()

    def _cycle(self) -> str:
        movable = [index for index in range(len(self._pieces)) if self._movable(index)]
        if movable:
            at = movable.index(self._selected) if self._selected in movable else -1
            self._select(movable[(at + 1) % len(movable)])
        return "break"

    def _piece_at(self, x: float, y: float) -> int | None:
        """The topmost piece under a canvas point, movable or not."""
        _, positions = self._layout()
        for index in reversed(range(len(self._pieces))):
            left, top = (value * self._scale for value in positions[index])
            piece = self._pieces[index]
            if (
                left <= x <= left + piece.image.width * self._scale
                and top <= y <= top + piece.image.height * self._scale
            ):
                return index
        return None

    def _on_press(self, event: tk.Event) -> None:
        self._canvas.focus_set()
        point = (self._canvas.canvasx(event.x), self._canvas.canvasy(event.y))
        index = self._piece_at(*point)
        if index is None or not self._movable(index):
            return
        self._select(index)
        self._drag = point

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag is None or self._selected is None:
            return
        x, y = self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)
        dx = round((x - self._drag[0]) / self._scale)
        dy = round((y - self._drag[1]) / self._scale)
        if dx == 0 and dy == 0:
            return
        self._move(self._selected, dx, dy)
        self._drag = (x, y)

    def _nudge(self, delta: tuple[int, int], step: int) -> str:
        if self._selected is not None:
            self._move(self._selected, delta[0] * step, delta[1] * step)
        return "break"

    def _move(self, index: int, dx: int, dy: int) -> None:
        """Shift one piece against the one above it, and everything below with it."""
        piece = self._pieces[index]
        x, y = self._offsets[index]
        # A seam never moves further than the piece itself is wide or tall, so
        # a stray drag cannot throw a piece off the canvas.
        self._offsets[index] = (
            _clamped(x + dx, piece.image.width),
            _clamped(y + dy, piece.image.height),
        )
        self._place()


def _clamped(value: int, limit: int) -> int:
    return max(-limit, min(value, limit))
