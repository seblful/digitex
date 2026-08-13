"""Tkinter review window — check a page's regions before they are cropped.

One adapter over the `PageReviewer` seam. The window draws the page, its
polygons and the option/part/number each question would be saved as, and lets
all of it be corrected with the mouse: drag a vertex or a whole polygon, add
and delete points, relabel a region, draw a missing one, reorder them, fix a
misread marker, or move where the page starts numbering. Every edit is
undoable, and the pane under the region list shows the crop that would be
written for whichever region is selected.

Every placement it shows comes from :func:`place_questions`, and every crop it
previews from the extractor's own cropping pipeline, so neither can drift from
what lands on disk.

One window serves the whole run: each page is loaded into it rather than
opening a new one, which is what lets the zoom, the pan and the chosen tab
survive from page to page.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import TYPE_CHECKING, Literal

import structlog
from PIL import Image, ImageTk

from digitex.domain.corpus import highest_question_number
from digitex.domain.entities import PixelPolygon
from digitex.pipeline.exceptions import ReviewAborted
from digitex.pipeline.placement import (
    PageExtractionState,
    PageLabel,
    PageRegion,
    QuestionPlacement,
    place_questions,
    reading_order_key,
)
from digitex.pipeline.review import (
    NumberingFault,
    PageProposal,
    QuestionCrop,
    ReviewedPage,
    numbering_fault,
)
from digitex.ui import geometry
from digitex.ui.display import BASE_DPI, enable_dpi_awareness, scaled
from digitex.ui.history import EditHistory, EditSnapshot, copy_regions
from digitex.ui.stats_panel import StatsPanel

if TYPE_CHECKING:
    from collections.abc import Callable

    from digitex.pipeline.audit.census import ImageCensus
    from digitex.pipeline.audit.validator import AnswerValidator

logger = structlog.get_logger()

Verdict = Literal["approve", "skip", "abort"]

# One colour per label, used for the outline, the caption and the handles, so a
# region's kind is readable without selecting it.
COLORS: dict[PageLabel, str] = {
    "question": "#2f6fed",
    "option": "#1f9d55",
    "part": "#e07b00",
}

LABELS: tuple[PageLabel, ...] = ("question", "option", "part")

# Overrides a region's own colour when its number would collide with, or leave
# a gap after, what the output tree already holds.
MISNUMBERED = "#c00000"

ACCENT = "#2f6fed"
ACCENT_ACTIVE = "#255ac4"
DISABLED = "#a8b0bd"
CANVAS_BG = "#2b2d31"
# Dark enough to read as text rather than as something switched off. Secondary
# information earns a lighter ink than the primary sort, not a faded one.
MUTED = "#4b5563"

# Half-width of a vertex handle, in screen pixels.
HANDLE = 5

# A crop needs four points — ImageCropper refuses fewer.
MIN_POINTS = 4

MIN_SCALE = 0.05
MAX_SCALE = 8.0
ZOOM_STEP = 1.25

# Arrow keys nudge by a source pixel, Shift by a stride worth seeing.
NUDGE = 1
NUDGE_FAST = 10

# Written for a 100% display; every one of them is put through `scaled()`
# against the display's own factor before it reaches a widget.
PANEL_WIDTH = 400
PREVIEW_HEIGHT = 230
FOOTER_HEIGHT = 46
PROBLEM_WRAP = 700

# A drawn box smaller than this is a stray click, not a region.
MIN_DRAWN_SIZE = 5

# Cropping runs a skew detection pass, so it waits for the selection to settle.
PREVIEW_DELAY_MS = 160
# Typing in a spinbox should land in the undo timeline once, not per keystroke.
STATE_COMMIT_MS = 600


def _int_or(raw: str, fallback: int) -> int:
    """Read a spinbox that the user may have emptied or typed junk into."""
    try:
        return int(raw)
    except ValueError:
        return fallback


class TkPageReviewer:
    """Interactive `PageReviewer`: shows each page and waits for a verdict.

    One hidden root and one window are created on first use and reused for the
    whole run, so zoom, pan, window size and the selected tab carry from page
    to page.

    ``census`` and ``validator`` are what the window's second tab reports on.
    Both are optional: without them the window still reviews pages, it just has
    nothing to tally.
    """

    def __init__(
        self,
        subject: str = "",
        census: ImageCensus | None = None,
        validator: AnswerValidator | None = None,
    ) -> None:
        self._subject = subject
        self._census = census
        self._validator = validator
        self._root: tk.Tk | None = None
        self._window: _ReviewWindow | None = None

    def __call__(self, proposal: PageProposal) -> ReviewedPage | None:
        """Show *proposal* and block until the page is approved, skipped or aborted.

        Raises:
            ReviewAborted: If the reviewer stopped the run.
        """
        window = self._ensure_window()
        verdict = window.present(proposal)

        if verdict == "abort":
            self.close()
            raise ReviewAborted(proposal.page_name)
        if verdict == "skip":
            logger.info("Page skipped in review", page=proposal.page_name)
            return None

        logger.info(
            "Page approved in review",
            page=proposal.page_name,
            regions=len(window.regions),
        )
        return ReviewedPage(regions=window.regions, state=window.state)

    def close(self) -> None:
        """Tear the window down — the run is over, one way or another."""
        if self._root is not None:
            self._root.destroy()
        self._root = None
        self._window = None

    def _ensure_window(self) -> _ReviewWindow:
        if self._window is None:
            # Before the root exists: Tk measures the screen when its
            # interpreter starts, and cannot be told about it afterwards.
            scale = enable_dpi_awareness()
            root = tk.Tk()
            root.withdraw()
            self._root = root
            self._window = _ReviewWindow(
                root,
                subject=self._subject,
                census=self._census,
                validator=self._validator,
                scale=scale,
            )
        return self._window


class _ReviewWindow:
    """The window itself. Built once, then loaded with one page after another.

    It owns a working copy of the current page's regions and entry state:
    approving hands those back, skipping drops them, and the proposal's own
    objects are never touched.
    """

    def __init__(
        self,
        root: tk.Tk,
        subject: str = "",
        census: ImageCensus | None = None,
        validator: AnswerValidator | None = None,
        scale: float = 1.0,
    ) -> None:
        self.regions: list[PageRegion] = []
        self.state = PageExtractionState()
        self._subject = subject
        self._dpi_scale = scale

        self._image = Image.new("RGB", (1, 1), "white")
        self._output_dir = Path()
        self._crop: QuestionCrop | None = None
        self._page_name = ""
        self._page_number = 0
        self._page_count = 0

        self._selected: int | None = None
        self._hover: int | None = None
        self._placements: dict[int, QuestionPlacement] = {}
        self._misnumbered: set[int] = set()
        self._problem: str | None = None

        self._scale = 1.0
        self._region = (0.0, 0.0, 1.0, 1.0)
        self._fitted = False
        self._photo: ImageTk.PhotoImage | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._render_pending = False
        self._loading = False
        self._jobs: dict[str, str] = {}

        self._history = EditHistory()
        self._drag: tuple[str, int, int] | None = None
        self._drag_from: tuple[float, float] | None = None
        self._draw_label: PageLabel | None = None
        self._draw_from: tuple[float, float] | None = None

        self.verdict: Verdict = "abort"

        # Point sizes are converted with the display's own DPI, so a font asked
        # for in points comes out the size the rest of the desktop draws it.
        root.tk.call("tk", "scaling", BASE_DPI * scale / 72.0)

        self.top = tk.Toplevel(root)
        self.top.title("Review")
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)
        width = int(self.top.winfo_screenwidth() * 0.92)
        height = int(self.top.winfo_screenheight() * 0.88)
        self.top.geometry(f"{width}x{height}+20+20")
        self.top.minsize(scaled(900, scale), scaled(600, scale))

        self._done = tk.BooleanVar(master=self.top, value=False)
        self._build(census, validator)

        # No transient(): the reviewer's root is withdrawn, and a toplevel made
        # transient for an unmapped master never maps itself on Windows.
        self.top.withdraw()

    # --- the seam ---

    def present(self, proposal: PageProposal) -> Verdict:
        """Load *proposal* and block until the reviewer decides what to do."""
        # Mapped before loading, so the canvas has a real size to fit the page
        # to — a withdrawn window reports 1x1 and would pin it at minimum zoom.
        self.top.deiconify()
        self.top.lift()
        self.top.update_idletasks()

        self._load(proposal)

        self.top.focus_force()
        self._canvas.focus_set()
        self.top.grab_set()

        self.verdict = "abort"
        self._done.set(False)
        self.top.wait_variable(self._done)
        self.top.grab_release()
        return self.verdict

    def _load(self, proposal: PageProposal) -> None:
        """Take on a new page, keeping the view where the reviewer left it."""
        self._loading = True

        resized = self._image.size != proposal.image.size
        self._image = proposal.image.convert("RGB")
        self._output_dir = proposal.output_dir
        self._crop = proposal.crop
        self._page_name = proposal.page_name or "page"
        self._page_number = proposal.page_number
        self._page_count = proposal.page_count

        self.regions = copy_regions(proposal.regions)
        self.state = replace(proposal.state)
        self._selected = None
        self._hover = None
        self._history.reset(self.regions, self.state)
        self._cancel_draw()
        # Cleared here rather than left to the debounce: a crop from the page
        # before, sitting under the new page's caption, is a lie for as long as
        # it is on screen.
        self._clear_preview()

        self.top.title(f"Review — {self._page_name} — {self._subject or 'extraction'}")
        self._option_var.set(str(self.state.option))
        self._part_var.set(self.state.part)
        self._question_var.set(str(self.state.question))
        self._stats.follow(self._subject, proposal.output_dir.name)

        self._loading = False

        # A book's pages are all the same size, so holding the zoom is what
        # lets a reviewer settle on one and stay there. A page of a different
        # size means a new book, or a fold-out — start it from a fit.
        if resized or not self._fitted:
            self._fit_when_ready()
        else:
            self._set_scale(self._scale)

        self._refresh()
        self._refresh_stats_if_shown()

    def _fit_when_ready(self) -> None:
        """Fit the page once the canvas knows how big it is.

        Tk gives a widget its real size only after the geometry manager has
        run, and the first page can arrive before that.
        """
        if self._view_size == (1, 1):
            self.top.after(50, self._fit_when_ready)
            return
        self._fitted = True
        self._zoom_to_fit()

    # --- construction ---

    def _build(
        self, census: ImageCensus | None, validator: AnswerValidator | None
    ) -> None:
        self.top.rowconfigure(1, weight=1)
        self.top.columnconfigure(0, weight=1)
        # The canvas is the only cell that grows; without floors of their own
        # the side panel and the footer buttons are squeezed out of the window.
        self.top.columnconfigure(1, minsize=self._px(PANEL_WIDTH))
        self.top.rowconfigure(3, minsize=self._px(FOOTER_HEIGHT))

        self._build_toolbar()
        self._build_canvas()
        self._build_panel(census, validator)
        self._build_status()
        self._build_footer()
        self._bind_keys()

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.top, padding=(8, 6))
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Button(bar, text="-", width=3, command=self._zoom_out).pack(side="left")
        ttk.Button(bar, text="+", width=3, command=self._zoom_in).pack(side="left")
        ttk.Button(bar, text="Fit", width=5, command=self._zoom_to_fit).pack(
            side="left"
        )
        ttk.Button(bar, text="1:1", width=5, command=lambda: self._set_scale(1.0)).pack(
            side="left"
        )

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        self._undo_button = ttk.Button(bar, text="↶ Undo", width=9, command=self._undo)
        self._undo_button.pack(side="left")
        self._redo_button = ttk.Button(bar, text="↷ Redo", width=9, command=self._redo)
        self._redo_button.pack(side="left", padx=(2, 0))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(bar, text="Draw:").pack(side="left", padx=(0, 4))
        self._draw_buttons: dict[PageLabel, ttk.Button] = {}
        for label in LABELS:
            button = ttk.Button(
                bar,
                text=label,
                width=9,
                command=lambda lbl=label: self._start_draw(lbl),
            )
            button.pack(side="left", padx=1)
            self._draw_buttons[label] = button

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        self._hint = ttk.Label(bar, text="", foreground=MUTED)
        self._hint.pack(side="left")

    def _build_canvas(self) -> None:
        frame = ttk.Frame(self.top)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            frame, background=CANVAS_BG, highlightthickness=0, takefocus=True
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")

        vbar = ttk.Scrollbar(frame, orient="vertical", command=self._canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = ttk.Scrollbar(frame, orient="horizontal", command=self._canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        # The scroll callbacks double as the window's "the view moved" signal:
        # the canvas calls them however the view moved — scrollbar, wheel, pan
        # or keyboard — which is exactly when the page bitmap has to be redrawn.
        self._canvas.configure(
            yscrollcommand=lambda *a: self._on_scrolled(vbar, *a),
            xscrollcommand=lambda *a: self._on_scrolled(hbar, *a),
        )

        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>", self._on_hover)
        self._canvas.bind("<Leave>", lambda _e: self._set_hover(None))
        self._canvas.bind("<Button-3>", self._on_context)
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        # Middle-drag pans, the way every image viewer does.
        self._canvas.bind("<Button-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan)

    def _build_panel(
        self, census: ImageCensus | None, validator: AnswerValidator | None
    ) -> None:
        notebook = self._notebook = ttk.Notebook(self.top)
        notebook.grid(row=1, column=1, sticky="nsew", padx=(6, 6), pady=(0, 2))

        page_tab = ttk.Frame(notebook, padding=8)
        notebook.add(page_tab, text="  This page  ")

        self._stats = StatsPanel(
            notebook, subject=self._subject, census=census, validator=validator
        )
        notebook.add(self._stats, text="  Extracted so far  ")
        notebook.bind(
            "<<NotebookTabChanged>>", lambda _e: self._refresh_stats_if_shown()
        )

        self._build_page_tab(page_tab)

    def _build_page_tab(self, panel: ttk.Frame) -> None:
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        self._build_entry_state(panel)

        tree_frame = ttk.Frame(panel)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 4))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("kind", "reading", "where"),
            show="headings",
            selectmode="browse",
            height=12,
        )
        self._tree.heading("kind", text="region")
        self._tree.heading("reading", text="read as")
        self._tree.heading("where", text="saved as")
        self._tree.column("kind", width=120, anchor="w")
        self._tree.column("reading", width=70, anchor="w")
        self._tree.column("where", width=100, anchor="w")
        self._tree.tag_configure("misnumbered", foreground=MISNUMBERED)
        for label, color in COLORS.items():
            self._tree.tag_configure(label, foreground=color)
        self._tree.grid(row=0, column=0, sticky="nsew")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-1>", lambda _e: self._edit_reading())

        tbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        tbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=tbar.set)

        buttons = ttk.Frame(panel)
        buttons.grid(row=2, column=0, sticky="ew")
        ttk.Button(buttons, text="↑", width=3, command=lambda: self._move(-1)).pack(
            side="left"
        )
        ttk.Button(buttons, text="↓", width=3, command=lambda: self._move(1)).pack(
            side="left"
        )
        ttk.Button(buttons, text="Sort by position", command=self._sort).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="Delete", command=self._delete_region).pack(
            side="left"
        )

        self._build_preview(panel)

    def _build_entry_state(self, panel: ttk.Frame) -> None:
        entry = ttk.LabelFrame(panel, text=" Page starts at ", padding=8)
        entry.grid(row=0, column=0, sticky="ew")
        entry.columnconfigure(3, weight=1)

        self._option_var = tk.StringVar(value="0")
        self._part_var = tk.StringVar(value="")
        self._question_var = tk.StringVar(value="0")

        ttk.Label(entry, text="option").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(entry, from_=0, to=99, width=5, textvariable=self._option_var).grid(
            row=0, column=1, sticky="w", padx=(4, 12)
        )
        ttk.Label(entry, text="part").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            entry,
            values=("", "A", "B"),
            width=3,
            state="readonly",
            textvariable=self._part_var,
        ).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(entry, text="questions done").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Spinbox(
            entry, from_=0, to=999, width=5, textvariable=self._question_var
        ).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        for var in (self._option_var, self._part_var, self._question_var):
            var.trace_add("write", lambda *_a: self._on_entry_state_changed())

        self._ends_at = ttk.Label(entry, text="", foreground=MUTED)
        self._ends_at.grid(row=2, column=0, columnspan=4, sticky="w")

        self._continue_button = ttk.Button(
            entry, text="Continue from disk", command=self._continue_from_disk
        )
        self._continue_button.grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0)
        )

    def _build_preview(self, panel: ttk.Frame) -> None:
        frame = ttk.LabelFrame(panel, text=" Crop preview ", padding=4)
        frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, minsize=self._px(PREVIEW_HEIGHT))

        self._preview = tk.Canvas(
            frame,
            height=self._px(PREVIEW_HEIGHT),
            background="#f6f6f6",
            highlightthickness=0,
        )
        self._preview.grid(row=0, column=0, sticky="nsew")
        self._preview_caption = ttk.Label(frame, text="", foreground=MUTED)
        self._preview_caption.grid(row=1, column=0, sticky="w", pady=(2, 0))

    def _build_status(self) -> None:
        bar = ttk.Frame(self.top, padding=(10, 4))
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(1, weight=1)

        self._page_label = ttk.Label(bar, text="", font=("TkDefaultFont", 9, "bold"))
        self._page_label.grid(row=0, column=0, sticky="w")
        self._counts_label = ttk.Label(bar, text="", foreground=MUTED)
        self._counts_label.grid(row=0, column=1, sticky="w", padx=12)
        self._zoom_label = ttk.Label(bar, text="", foreground=MUTED)
        self._zoom_label.grid(row=0, column=2, sticky="e")

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.top, padding=(10, 6))
        footer.grid(row=3, column=0, columnspan=2, sticky="ew")
        footer.columnconfigure(0, weight=1)

        self._problem_label = ttk.Label(
            footer,
            text="",
            foreground=MISNUMBERED,
            wraplength=self._px(PROBLEM_WRAP),
            justify="left",
        )
        self._problem_label.grid(row=0, column=0, sticky="w")

        actions = ttk.Frame(footer)
        actions.grid(row=0, column=1, sticky="e")

        ttk.Button(
            actions, text="Abort run", command=lambda: self._finish("abort")
        ).pack(side="right", padx=(6, 0))
        ttk.Button(
            actions, text="Skip page", command=lambda: self._finish("skip")
        ).pack(side="right")
        # A plain tk.Button, because the primary action wants a colour and no
        # ttk theme on Windows lets a themed one have it.
        self._approve = tk.Button(
            actions,
            text="Approve & save    Ctrl+Enter",
            command=lambda: self._finish("approve"),
            background=ACCENT,
            foreground="white",
            activebackground=ACCENT_ACTIVE,
            activeforeground="white",
            disabledforeground="#eeeeee",
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=5,
            cursor="hand2",
            font=("TkDefaultFont", 9, "bold"),
        )
        self._approve.pack(side="right", padx=8)

    def _bind_keys(self) -> None:
        self.top.bind("<Control-Return>", lambda _e: self._finish("approve"))
        self.top.bind("<Control-z>", lambda _e: self._undo())
        self.top.bind("<Control-y>", lambda _e: self._redo())
        self.top.bind("<Control-Z>", lambda _e: self._redo())  # Ctrl+Shift+Z
        self.top.bind("<Escape>", lambda _e: self._on_escape())
        self.top.bind("<Delete>", lambda _e: self._delete_region())

        # On the canvas, so they cannot fight the tree's own navigation or a
        # spinbox the reviewer is typing into.
        self._canvas.bind("<Key>", self._on_canvas_key)
        for key, delta in (("<Up>", (0, -1)), ("<Down>", (0, 1))):
            self._canvas.bind(key, lambda _e, d=delta: self._on_arrow(d, NUDGE))
            self._canvas.bind(
                f"<Shift-{key[1:-1]}>",
                lambda _e, d=delta: self._on_arrow(d, NUDGE_FAST),
            )
        for key, delta in (("<Left>", (-1, 0)), ("<Right>", (1, 0))):
            self._canvas.bind(key, lambda _e, d=delta: self._on_arrow(d, NUDGE))
            self._canvas.bind(
                f"<Shift-{key[1:-1]}>",
                lambda _e, d=delta: self._on_arrow(d, NUDGE_FAST),
            )

    # --- coordinates and rendering ---

    def _px(self, value: int) -> int:
        """A size written for a 100% display, at this one's scaling."""
        return scaled(value, self._dpi_scale)

    def _to_canvas(self, point: tuple[int, int]) -> tuple[float, float]:
        return (point[0] * self._scale, point[1] * self._scale)

    def _to_image(self, x: float, y: float) -> tuple[int, int]:
        return (round(x / self._scale), round(y / self._scale))

    @property
    def _page_size(self) -> tuple[float, float]:
        return (self._image.width * self._scale, self._image.height * self._scale)

    @property
    def _view_size(self) -> tuple[int, int]:
        return (
            max(self._canvas.winfo_width(), 1),
            max(self._canvas.winfo_height(), 1),
        )

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        self._schedule_page_render()

    def _on_scrolled(self, bar: ttk.Scrollbar, first: str, last: str) -> None:
        bar.set(first, last)
        self._schedule_page_render()

    def _schedule_page_render(self) -> None:
        """Coalesce the redraws a single gesture asks for into one."""
        if self._render_pending:
            return
        self._render_pending = True
        self._canvas.after_idle(self._render_page)

    def _zoom_to_fit(self) -> None:
        self._set_scale(geometry.fit_scale(self._image.size, self._view_size))
        self._canvas.xview_moveto(0)
        self._canvas.yview_moveto(0)

    def _zoom_in(self) -> None:
        self._zoom_about(ZOOM_STEP, None)

    def _zoom_out(self) -> None:
        self._zoom_about(1 / ZOOM_STEP, None)

    def _zoom_about(self, factor: float, pointer: tuple[float, float] | None) -> None:
        """Scale by *factor*, holding the point under *pointer* where it is.

        With no pointer the centre of the view stays put, which is what the
        toolbar's buttons and the keyboard want.
        """
        view_w, view_h = self._view_size
        origin = (self._canvas.canvasx(0), self._canvas.canvasy(0))
        anchor = pointer or (origin[0] + view_w / 2, origin[1] + view_h / 2)

        before = self._scale
        self._set_scale(before * factor)
        ratio = self._scale / before
        if ratio == 1.0:
            return

        self._move_view_to(
            geometry.anchored_origin(origin[0], anchor[0], ratio),
            geometry.anchored_origin(origin[1], anchor[1], ratio),
        )

    def _move_view_to(self, x: float, y: float) -> None:
        """Put canvas point (x, y) at the view's top-left, as far as it goes."""
        left, top, right, bottom = self._region
        self._canvas.xview_moveto(geometry.scroll_fraction(x - left, right - left))
        self._canvas.yview_moveto(geometry.scroll_fraction(y - top, bottom - top))

    def _set_scale(self, scale: float) -> None:
        self._scale = geometry.clamp_scale(scale, MIN_SCALE, MAX_SCALE)
        page_w, page_h = self._page_size
        view_w, view_h = self._view_size
        # Margins on a page smaller than its canvas, so it sits in the middle
        # instead of hugging a corner. They go into the scroll region rather
        # than into an offset on every coordinate, which leaves the mapping
        # between canvas and image pixels exactly as it was.
        margin_x = max(0.0, (view_w - page_w) / 2)
        margin_y = max(0.0, (view_h - page_h) / 2)
        self._region = (-margin_x, -margin_y, page_w + margin_x, page_h + margin_y)

        self._canvas.configure(scrollregion=self._region)
        self._zoom_label.configure(text=f"zoom {self._scale * 100:.0f}%")
        self._schedule_page_render()
        self._render_regions()

    def _render_page(self) -> None:
        """Draw the part of the page that is on screen, and only that.

        Rendering the whole page at every scale is what makes a zoomed-in
        editor crawl: a 3500px scan at 200% is a 50-megapixel bitmap, rebuilt
        on every wheel click. Cropping to the viewport first keeps the cost
        flat however far in the reviewer zooms.
        """
        self._render_pending = False
        box = geometry.visible_box(
            (self._canvas.canvasx(0), self._canvas.canvasy(0)),
            self._view_size,
            self._page_size,
        )
        if box is None:
            return

        scale = self._scale
        left, top = math.floor(box[0] / scale), math.floor(box[1] / scale)
        right = min(self._image.width, math.ceil(box[2] / scale))
        bottom = min(self._image.height, math.ceil(box[3] / scale))
        if right <= left or bottom <= top:
            return

        tile = self._image.crop((left, top, right, bottom))
        size = (
            max(1, round((right - left) * scale)),
            max(1, round((bottom - top) * scale)),
        )
        # Above 1:1 the reviewer is placing a vertex on a pixel, so show the
        # pixels rather than a smoothed guess at them.
        resample = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.BILINEAR
        self._photo = ImageTk.PhotoImage(tile.resize(size, resample))

        self._canvas.delete("page")
        self._canvas.create_image(
            left * scale, top * scale, image=self._photo, anchor="nw", tags="page"
        )
        self._canvas.tag_lower("page")

    def _caption(self, index: int, region: PageRegion) -> str:
        if region.label == "question":
            placement = self._placements.get(index)
            return str(placement) if placement else "?"
        return f"{region.label} {self._reading_text(region)}"

    @staticmethod
    def _reading_text(region: PageRegion) -> str:
        if region.label == "question":
            return ""
        return str(region.reading) if region.reading is not None else "unread"

    def _render_regions(self) -> None:
        self._canvas.delete("region")

        for index, region in enumerate(self.regions):
            color = MISNUMBERED if index in self._misnumbered else COLORS[region.label]
            selected = index == self._selected
            hovered = index == self._hover
            coords = [c for point in region.polygon for c in self._to_canvas(point)]

            self._canvas.create_polygon(
                coords,
                outline=color,
                fill=color,
                stipple="gray12" if selected or hovered else "gray25",
                width=3 if selected else (2 if hovered else 1),
                tags=("region", f"region:{index}"),
            )
            self._draw_caption(index, region, color, selected)

            if selected:
                self._draw_handles(index, region, color)

    def _draw_caption(
        self, index: int, region: PageRegion, color: str, selected: bool
    ) -> None:
        x, y = self._to_canvas(geometry.top_left(list(region.polygon)))
        label = self._canvas.create_text(
            x + 4,
            y - 4,
            text=f"{index + 1}. {self._caption(index, region)}",
            anchor="sw",
            fill=color,
            font=("TkDefaultFont", 10, "bold"),
            tags=("region", f"region:{index}"),
        )
        # A caption on a scan is unreadable without something behind it, and a
        # background can only be sized once the text has been measured.
        bbox = self._canvas.bbox(label)
        if bbox:
            self._canvas.create_rectangle(
                bbox[0] - 3,
                bbox[1] - 1,
                bbox[2] + 3,
                bbox[3] + 1,
                fill="#ffffff",
                outline=color,
                width=2 if selected else 1,
                tags=("region", f"region:{index}"),
            )
            self._canvas.tag_raise(label)

    def _draw_handles(self, index: int, region: PageRegion, color: str) -> None:
        handle = self._px(HANDLE)
        for point_index, point in enumerate(region.polygon):
            px, py = self._to_canvas(point)
            self._canvas.create_rectangle(
                px - handle,
                py - handle,
                px + handle,
                py + handle,
                fill="#ffffff",
                outline=color,
                width=2,
                tags=("region", "handle", f"handle:{index}:{point_index}"),
            )

    # --- the preview, recomputed after every edit ---

    def _refresh(self) -> None:
        """Replay the regions through a copy of the entry state and redraw."""
        preview = replace(self.state)
        self._placements = {}
        self._misnumbered = set()
        self._problem = None

        try:
            placed = place_questions(self.regions, preview)
        except ValueError as exc:
            self._problem = str(exc)
            self._ends_at.configure(text="")
        else:
            questions = [i for i, r in enumerate(self.regions) if r.label == "question"]
            self._placements = {
                index: item.placement
                for index, item in zip(questions, placed, strict=True)
            }

            fault = numbering_fault(placed, self._output_dir)
            if fault is not None:
                self._misnumbered = {questions[fault.position]}
                self._problem = self._fault_message(fault)

            part = preview.part or "?"
            self._ends_at.configure(
                text=f"page ends at {preview.option}/{part}/{preview.question}"
            )

        self._problem_label.configure(text=self._problem or "")
        self._set_approve_enabled(not self._problem)
        self._continue_button.configure(
            state="normal"
            if self._problem and self._entry_state_reaches_first_question()
            else "disabled"
        )
        self._undo_button.configure(
            state="normal" if self._history.can_undo else "disabled"
        )
        self._redo_button.configure(
            state="normal" if self._history.can_redo else "disabled"
        )
        self._render_regions()
        self._fill_tree()
        self._update_status()
        self._schedule_preview()

    def _set_approve_enabled(self, enabled: bool) -> None:
        self._approve.configure(
            state="normal" if enabled else "disabled",
            background=ACCENT if enabled else DISABLED,
        )

    def _update_status(self) -> None:
        where = f"{self._page_name}"
        if self._page_count:
            where += f"    page {self._page_number} of {self._page_count}"
        self._page_label.configure(text=where)

        questions = sum(1 for r in self.regions if r.label == "question")
        markers = len(self.regions) - questions
        span = ""
        if self._placements:
            first = next(iter(self._placements.values()))
            last = list(self._placements.values())[-1]
            span = f"    → {first} … {last}" if first != last else f"    → {first}"
        self._counts_label.configure(
            text=f"{questions} questions, {markers} markers{span}"
        )

    def _fault_message(self, fault: NumberingFault) -> str:
        """Say what is wrong with a number, and which remedy applies."""
        wrong = "already exists" if fault.collides else "would leave a gap"
        remedy = (
            "Use 'Continue from disk'."
            if self._entry_state_reaches_first_question()
            else "A marker starts this group, so the page's own numbering is"
            " right — skip the page if it is already extracted, or correct the"
            " marker above it."
        )
        return (
            f"{fault.placement} {wrong} — the next free number in"
            f" {fault.placement.option}/{fault.placement.part} is {fault.free}."
            f" {remedy}"
        )

    def _entry_state_reaches_first_question(self) -> bool:
        """True when nothing resets the numbering before the first question.

        An option or part marker sets the counter itself, so moving where the
        page starts cannot move a group that begins after one.
        """
        for region in self.regions:
            if region.label == "question":
                return True
            if region.label in ("option", "part"):
                return False
        return False

    def _continue_from_disk(self) -> None:
        """Set the entry counter so the page's first question takes the free slot."""
        first = next(iter(self._placements.values()), None)
        if first is None:
            return
        free = highest_question_number(self._output_dir, first.option, first.part) + 1
        # next_question() hands out question + 1, so the counter sits one below.
        self._question_var.set(str(max(free - 1, 0)))

    def _fill_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for index, region in enumerate(self.regions):
            where = self._caption(index, region) if region.label == "question" else ""
            tag = "misnumbered" if index in self._misnumbered else region.label
            self._tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    f"{index + 1}. {region.label}",
                    self._reading_text(region),
                    where,
                ),
                tags=(tag,),
            )
        if self._selected is not None and self._selected < len(self.regions):
            self._tree.selection_set(str(self._selected))
            self._tree.see(str(self._selected))

    def _select(self, index: int | None) -> None:
        if index == self._selected:
            return
        self._selected = index
        self._render_regions()
        if index is None:
            selection = self._tree.selection()
            if selection:
                self._tree.selection_remove(*selection)
        else:
            self._tree.selection_set(str(index))
            self._tree.see(str(index))
        self._schedule_preview()

    def _set_hover(self, index: int | None) -> None:
        if index == self._hover:
            return
        self._hover = index
        self._render_regions()

    # --- the crop preview ---

    def _schedule_preview(self) -> None:
        self._debounce("preview", PREVIEW_DELAY_MS, self._render_preview)

    def _clear_preview(self) -> None:
        self._preview.delete("all")
        self._preview_caption.configure(text="select a region to preview its crop")

    def _render_preview(self) -> None:
        """Show the selected region as it would be saved."""
        index = self._selected
        if index is None or index >= len(self.regions):
            self._clear_preview()
            return
        self._preview.delete("all")

        region = self.regions[index]
        try:
            image = self._crop_for(region)
        except Exception as exc:  # a polygon can be degenerate mid-edit
            self._preview_caption.configure(text=f"cannot crop this polygon: {exc}")
            return

        width = max(self._preview.winfo_width(), 1)
        height = max(self._preview.winfo_height(), self._px(PREVIEW_HEIGHT))
        scale = min(width / image.width, height / image.height, 1.0)
        shown = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        self._preview_photo = ImageTk.PhotoImage(shown)
        self._preview.create_image(
            width // 2, height // 2, image=self._preview_photo, anchor="center"
        )
        self._preview_caption.configure(text=self._preview_text(index, region, image))

    def _crop_for(self, region: PageRegion) -> Image.Image:
        """The crop for *region* — the extractor's own for a question."""
        if region.label == "question" and self._crop is not None:
            return self._crop(region.polygon)
        left, top, right, bottom = geometry.bounds(list(region.polygon))
        return self._image.crop((left, top, right, bottom))

    def _preview_text(self, index: int, region: PageRegion, image: Image.Image) -> str:
        size = f"{image.width}x{image.height} px"
        if region.label == "question":
            placement = self._placements.get(index)
            saved = f"saved as {placement}" if placement else "not placed"
            return f"{saved} — {size}"
        return f"{region.label} marker — {size}"

    # --- mouse ---

    def _hit(self, x: float, y: float) -> tuple[str, int, int] | None:
        """What sits under the cursor: a vertex handle, a region, or nothing."""
        for prefix, slack in (("handle:", self._px(HANDLE)), ("region:", 1)):
            items = self._canvas.find_overlapping(
                x - slack, y - slack, x + slack, y + slack
            )
            for item in reversed(items):
                for tag in self._canvas.gettags(item):
                    if tag.startswith(prefix):
                        parts = tag.split(":")
                        point = int(parts[2]) if len(parts) > 2 else 0
                        return (prefix[:-1], int(parts[1]), point)
        return None

    def _canvas_point(self, event: tk.Event) -> tuple[float, float]:
        return (self._canvas.canvasx(event.x), self._canvas.canvasy(event.y))

    def _on_hover(self, event: tk.Event) -> None:
        if self._draw_label is not None or self._drag is not None:
            return
        hit = self._hit(*self._canvas_point(event))
        if hit is None:
            self._set_hover(None)
            self._canvas.configure(cursor="")
            return
        kind, index, _ = hit
        self._set_hover(index)
        self._canvas.configure(cursor="tcross" if kind == "handle" else "fleur")

    def _on_press(self, event: tk.Event) -> None:
        self._canvas.focus_set()
        x, y = self._canvas_point(event)

        if self._draw_label is not None:
            self._draw_from = (x, y)
            return

        hit = self._hit(x, y)
        if hit is None:
            self._select(None)
            return

        kind, index, point = hit
        self._select(index)
        self._drag = (kind, index, point)
        self._drag_from = (x, y)

    def _on_motion(self, event: tk.Event) -> None:
        x, y = self._canvas_point(event)

        if self._draw_label is not None and self._draw_from is not None:
            self._canvas.delete("rubber")
            self._canvas.create_rectangle(
                *self._draw_from,
                x,
                y,
                outline=COLORS[self._draw_label],
                width=2,
                dash=(4, 3),
                tags="rubber",
            )
            return

        if self._drag is None or self._drag_from is None:
            return

        kind, index, point = self._drag
        region = self.regions[index]

        if kind == "handle":
            polygon = list(region.polygon)
            polygon[point] = self._to_image(x, y)
            region.polygon = PixelPolygon(polygon)
        else:
            dx, dy = self._to_image(x - self._drag_from[0], y - self._drag_from[1])
            if dx == 0 and dy == 0:
                return
            region.polygon = PixelPolygon(geometry.moved(list(region.polygon), dx, dy))

        self._drag_from = (x, y)
        self._render_regions()

    def _on_release(self, event: tk.Event) -> None:
        if self._draw_label is not None and self._draw_from is not None:
            self._finish_draw(*self._canvas_point(event))
            return

        if self._drag is not None:
            self._drag = None
            self._drag_from = None
            self._commit()

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom_about(
            ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP, self._canvas_point(event)
        )

    def _on_pan_start(self, event: tk.Event) -> None:
        self._canvas.configure(cursor="hand2")
        self._canvas.scan_mark(event.x, event.y)

    def _on_pan(self, event: tk.Event) -> None:
        self._canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_context(self, event: tk.Event) -> None:
        x, y = self._canvas_point(event)
        hit = self._hit(x, y)
        if hit is None:
            return

        kind, index, point = hit
        self._select(index)
        menu = self._context_menu(index, kind, point, (x, y))
        menu.tk_popup(event.x_root, event.y_root)

    def _context_menu(
        self, index: int, kind: str, point: int, at: tuple[float, float]
    ) -> tk.Menu:
        region = self.regions[index]
        menu = tk.Menu(self.top, tearoff=0)

        if kind == "handle":
            menu.add_command(
                label="Delete point", command=lambda: self._delete_point(index, point)
            )
        else:
            menu.add_command(
                label="Insert point here",
                command=lambda: self._insert_point(index, self._to_image(*at)),
            )
        menu.add_command(label="Zoom to region", command=self._zoom_to_selected)
        menu.add_separator()

        label_menu = tk.Menu(menu, tearoff=0)
        for label in LABELS:
            label_menu.add_command(
                label=label, command=lambda lbl=label: self._set_label(index, lbl)
            )
        menu.add_cascade(label="Label", menu=label_menu)

        if region.label == "option":
            menu.add_command(
                label="Set option number…", command=lambda: self._set_option(index)
            )
        elif region.label == "part":
            part_menu = tk.Menu(menu, tearoff=0)
            for value in ("A", "B"):
                part_menu.add_command(
                    label=value, command=lambda v=value: self._set_reading(index, v)
                )
            part_menu.add_command(
                label="unreadable", command=lambda: self._set_reading(index, None)
            )
            menu.add_cascade(label="Set part", menu=part_menu)

        menu.add_separator()
        menu.add_command(label="Delete region", command=self._delete_region)
        return menu

    # --- keyboard ---

    def _on_canvas_key(self, event: tk.Event) -> str | None:
        """Single-key shortcuts, live only while the canvas has the focus."""
        actions: dict[str, Callable[[], object]] = {
            "1": lambda: self._relabel_selected("question"),
            "2": lambda: self._relabel_selected("option"),
            "3": lambda: self._relabel_selected("part"),
            "f": self._zoom_to_fit,
            "z": self._zoom_to_selected,
            "s": self._sort,
            "plus": self._zoom_in,
            "equal": self._zoom_in,
            "minus": self._zoom_out,
            "Tab": lambda: self._cycle_selection(1),
            "ISO_Left_Tab": lambda: self._cycle_selection(-1),
        }
        action = actions.get(event.keysym)
        if action is None:
            return None
        action()
        return "break"

    def _on_arrow(self, delta: tuple[int, int], step: int) -> str:
        """Nudge the selected region, or scroll the page when nothing is selected."""
        dx, dy = delta
        if self._selected is None:
            self._canvas.xview_scroll(dx, "units")
            self._canvas.yview_scroll(dy, "units")
            return "break"

        region = self.regions[self._selected]
        region.polygon = PixelPolygon(
            geometry.moved(list(region.polygon), dx * step, dy * step)
        )
        self._commit()
        return "break"

    def _cycle_selection(self, delta: int) -> None:
        if not self.regions:
            return
        if self._selected is None:
            self._select(0 if delta > 0 else len(self.regions) - 1)
            return
        self._select((self._selected + delta) % len(self.regions))

    def _on_escape(self) -> None:
        if self._draw_label is not None:
            self._cancel_draw()
        else:
            self._select(None)

    # --- editing ---

    def _commit(self) -> None:
        """Record an edit in the undo timeline and redraw everything it touched."""
        self._history.push(self.regions, self.state, self._selected)
        self._refresh()

    def _undo(self) -> None:
        self._restore(self._history.undo())

    def _redo(self) -> None:
        self._restore(self._history.redo())

    def _restore(self, snapshot: EditSnapshot | None) -> None:
        if snapshot is None:
            return
        self.regions = snapshot.regions
        self.state = snapshot.state
        self._selected = snapshot.selected

        self._loading = True
        self._option_var.set(str(self.state.option))
        self._part_var.set(self.state.part)
        self._question_var.set(str(self.state.question))
        self._loading = False
        self._refresh()

    def _start_draw(self, label: PageLabel) -> None:
        self._draw_label = label
        self._canvas.configure(cursor="crosshair")
        self._hint.configure(text=f"drag a box for the new {label} — Esc cancels")

    def _cancel_draw(self) -> None:
        self._draw_label = None
        self._draw_from = None
        self._canvas.delete("rubber")
        self._canvas.configure(cursor="")
        self._hint.configure(text="")

    def _finish_draw(self, x: float, y: float) -> None:
        label, start = self._draw_label, self._draw_from
        if label is None or start is None:
            return

        x0, y0 = self._to_image(*start)
        x1, y1 = self._to_image(x, y)
        self._cancel_draw()

        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < MIN_DRAWN_SIZE or bottom - top < MIN_DRAWN_SIZE:
            return

        # A rectangle is enough: the cropper reduces any polygon to its
        # minimum-area quad before warping, and masks with the polygon itself.
        self.regions.append(
            PageRegion(
                label=label,
                polygon=PixelPolygon(
                    [(left, top), (right, top), (right, bottom), (left, bottom)]
                ),
            )
        )
        self._selected = len(self.regions) - 1
        self._commit()

    def _insert_point(self, index: int, point: tuple[int, int]) -> None:
        """Add a vertex on the polygon edge nearest the click."""
        polygon = list(self.regions[index].polygon)
        polygon.insert(geometry.nearest_edge(polygon, point) + 1, point)
        self.regions[index].polygon = PixelPolygon(polygon)
        self._commit()

    def _delete_point(self, index: int, point: int) -> None:
        polygon = list(self.regions[index].polygon)
        if len(polygon) <= MIN_POINTS:
            self._hint.configure(text=f"a region needs at least {MIN_POINTS} points")
            return
        del polygon[point]
        self.regions[index].polygon = PixelPolygon(polygon)
        self._commit()

    def _relabel_selected(self, label: PageLabel) -> None:
        if self._selected is not None:
            self._set_label(self._selected, label)

    def _set_label(self, index: int, label: PageLabel) -> None:
        region = self.regions[index]
        region.label = label
        # The old reading belongs to the old kind — an option number on a part
        # marker would be ignored anyway, and shown as if it counted.
        region.reading = None
        self._commit()

    def _set_reading(self, index: int, reading: int | str | None) -> None:
        self.regions[index].reading = reading
        self._commit()

    def _set_option(self, index: int) -> None:
        current = self.regions[index].reading
        value = simpledialog.askinteger(
            "Option number",
            "Option this marker starts:",
            parent=self.top,
            initialvalue=current if isinstance(current, int) else None,
            minvalue=1,
            maxvalue=99,
        )
        if value is not None:
            self._set_reading(index, value)

    def _edit_reading(self) -> None:
        """Double-click on a marker row: fix what OCR read off it."""
        if self._selected is None:
            return
        region = self.regions[self._selected]
        if region.label == "option":
            self._set_option(self._selected)
        elif region.label == "part":
            self._set_reading(self._selected, "B" if region.reading == "A" else "A")

    def _delete_region(self) -> None:
        if self._selected is None:
            return
        del self.regions[self._selected]
        self._selected = None
        self._commit()

    def _move(self, delta: int) -> None:
        """Move the selected region in the reading order the numbering follows."""
        if self._selected is None:
            return
        target = self._selected + delta
        if not 0 <= target < len(self.regions):
            return
        regions = self.regions
        regions[self._selected], regions[target] = (
            regions[target],
            regions[self._selected],
        )
        self._selected = target
        self._commit()

    def _sort(self) -> None:
        selected = self.regions[self._selected] if self._selected is not None else None
        self.regions.sort(key=lambda region: reading_order_key(region.polygon))
        # By identity, not equality: two regions can hold equal field values.
        self._selected = next(
            (i for i, r in enumerate(self.regions) if r is selected), None
        )
        self._commit()

    def _zoom_to_selected(self) -> None:
        """Fill the view with the selected region, so its edges can be judged."""
        if self._selected is None:
            return
        left, top, right, bottom = geometry.bounds(
            list(self.regions[self._selected].polygon)
        )
        view_w, view_h = self._view_size
        # Some room around it, so the region is seen in the context it was cut
        # from rather than edge to edge.
        self._set_scale(
            min(view_w / max(right - left, 1), view_h / max(bottom - top, 1)) * 0.8
        )
        self._move_view_to(
            (left + right) / 2 * self._scale - view_w / 2,
            (top + bottom) / 2 * self._scale - view_h / 2,
        )

    def _on_entry_state_changed(self) -> None:
        if self._loading:
            return
        self.state.option = _int_or(self._option_var.get(), self.state.option)
        self.state.part = self._part_var.get()
        self.state.question = _int_or(self._question_var.get(), self.state.question)
        self._refresh()
        # One undo step per settled value, not one per keystroke.
        self._debounce(
            "state",
            STATE_COMMIT_MS,
            lambda: self._history.push(self.regions, self.state, self._selected),
        )

    def _on_tree_select(self, _event: tk.Event) -> None:
        selection = self._tree.selection()
        if selection:
            self._select(int(selection[0]))

    # --- stats tab ---

    def _refresh_stats_if_shown(self) -> None:
        """Recount only when the tab is in front — it walks the whole tree."""
        try:
            current = self._notebook.index(self._notebook.select())
        except tk.TclError:
            return
        if current == 1:
            self._stats.refresh_if_stale()

    # --- verdict ---

    def _debounce(self, name: str, delay_ms: int, action: Callable[[], object]) -> None:
        """Run *action* once *delay_ms* has passed with no further calls."""
        job = self._jobs.pop(name, None)
        if job is not None:
            self.top.after_cancel(job)
        self._jobs[name] = self.top.after(delay_ms, action)

    def _cancel_jobs(self) -> None:
        for job in self._jobs.values():
            self.top.after_cancel(job)
        self._jobs.clear()

    def _finish(self, verdict: Verdict) -> None:
        if verdict == "approve" and self._problem:
            return
        if verdict == "abort" and not messagebox.askokcancel(
            "Abort run",
            "Stop the extraction run? Pages already approved keep their images.",
            parent=self.top,
        ):
            return

        self.verdict = verdict
        self._stats.invalidate()
        self._cancel_jobs()
        # The window stays up while the crops are written — nothing processes
        # its events until the next page arrives, so hiding it would only make
        # the pause between pages look like a crash.
        self._hint.configure(text="saving…" if verdict == "approve" else f"{verdict}…")
        self.top.update_idletasks()
        self._done.set(True)

    def _on_close(self) -> None:
        # Closing the window with the X is an abort, not a skip: a skip writes
        # nothing and says nothing, and losing a page that way is silent.
        self._finish("abort")
