"""Tkinter review window — check a page's regions before they are cropped.

One adapter over the `PageReviewer` seam. The window draws the page, its
polygons and the option/part/number each question would be saved as, and lets
all of it be corrected with the mouse: drag a vertex or a whole polygon, add
and delete points, relabel a region, draw a missing one, reorder them, fix a
misread marker, mark a question as continuing onto the next page, or move where
the page starts numbering. Every edit is undoable, and the pane under the region
list shows the crop that would be written for whichever region is selected.

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
from tkinter import messagebox, simpledialog, ttk
from typing import TYPE_CHECKING, Literal

import structlog
from PIL import Image, ImageTk

from digitex.pipeline.exceptions import ReviewAborted
from digitex.pipeline.review import (
    PageProposal,
    QuestionCrop,
    ReviewedPage,
)
from digitex.ui import geometry
from digitex.ui.display import BASE_DPI, enable_dpi_awareness, scaled
from digitex.ui.edits import MIN_POINTS, Numbering, PageEdits
from digitex.ui.join_editor import JoinEditor, JoinPiece
from digitex.ui.stats_panel import StatsPanel

if TYPE_CHECKING:
    from collections.abc import Callable

    from digitex.pipeline.audit.census import ImageCensus
    from digitex.pipeline.audit.validator import AnswerValidator
    from digitex.pipeline.pieces import HeldPiece
    from digitex.pipeline.placement import PageLabel, PageRegion
    from digitex.pipeline.review import PieceCrop

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

# How far past the page's edge the arrow out of a question continuing onto the
# next page reaches, in screen pixels.
JOIN_TAIL = 40

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


def resolve_verdict(
    verdict: Verdict,
    edits: PageEdits,
    page_name: str,
    discard_carried: bool = False,
) -> ReviewedPage | None:
    """Turn the window's verdict into the reviewer's answer.

    The three ways a review ends, kept out of the widget so the seam's error
    modes run without a display: approve hands back what the reviewer edited,
    skip returns None, abort raises. A skip leaves the pieces carried onto the
    page for the next one — only an approval can throw them away, and only when
    the reviewer said to.

    Raises:
        ReviewAborted: If the reviewer stopped the run.
    """
    if verdict == "abort":
        raise ReviewAborted(page_name)
    if verdict == "skip":
        logger.info("Page skipped in review", page=page_name)
        return None
    logger.info("Page approved in review", page=page_name, regions=len(edits.regions))
    return ReviewedPage(
        regions=edits.regions,
        state=edits.state,
        discard_carried=discard_carried,
    )


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
        try:
            return resolve_verdict(
                verdict,
                window.edits,
                proposal.page_name,
                discard_carried=window.discard_carried,
            )
        except ReviewAborted:
            self.close()
            raise

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

    A view over :class:`~digitex.ui.edits.PageEdits`, which owns the working
    copy of the page and every rule about editing it. The window's own job is
    the widgets: draw what ``edits.numbering()`` reports, turn clicks and keys
    into its operations, and redraw afterwards. Nothing here decides what an
    edit means.
    """

    def __init__(
        self,
        root: tk.Tk,
        subject: str = "",
        census: ImageCensus | None = None,
        validator: AnswerValidator | None = None,
        scale: float = 1.0,
    ) -> None:
        # The page being edited, and every rule about editing it. The window
        # draws what this reports and turns input into its operations.
        self.edits = PageEdits()
        self._numbering = Numbering()

        self._subject = subject
        self._dpi_scale = scale

        self._image = Image.new("RGB", (1, 1), "white")
        self._crop: QuestionCrop | None = None
        self._crop_piece: PieceCrop | None = None
        # The pieces an earlier page left for this page's first question, and
        # whether the reviewer decided to throw them away.
        self._carried: list[HeldPiece] = []
        self.discard_carried = False
        self._page_name = ""
        self._page_number = 0
        self._page_count = 0
        # The year directory the current page's crops land in — what the
        # stats tab is asked to show when it comes to the front.
        self._output_year = ""

        self._hover: int | None = None

        self._scale = 1.0
        self._region = (0.0, 0.0, 1.0, 1.0)
        self._fitted = False
        self._photo: ImageTk.PhotoImage | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._render_pending = False
        self._loading = False
        self._jobs: dict[str, str] = {}

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
        self._crop = proposal.crop
        self._crop_piece = proposal.crop_piece
        self._carried = list(proposal.carried)
        self.discard_carried = False
        self._page_name = proposal.page_name or "page"
        self._page_number = proposal.page_number
        self._page_count = proposal.page_count

        self.edits.load(
            proposal.regions,
            proposal.state,
            proposal.output_dir,
            carried=len(self._carried),
        )
        self._hover = None
        self._cancel_draw()
        # Cleared here rather than left to the debounce: a crop from the page
        # before, sitting under the new page's caption, is a lie for as long as
        # it is on screen.
        self._clear_preview()

        self.top.title(f"Review — {self._page_name} — {self._subject or 'extraction'}")
        self._show_entry_state()
        self._show_carried()
        self._output_year = proposal.output_dir.name

        self._loading = False

        # A book's pages are all the same size, so holding the zoom is what
        # lets a reviewer settle on one and stay there. A page of a different
        # size means a new book, or a fold-out — start it from a fit.
        if resized or not self._fitted:
            self._fit_when_ready()
        else:
            self._set_scale(self._scale)

        self._refresh()
        # A page finishing a question the page before it started opens on that
        # question, so the joined crop is the first thing the reviewer sees.
        if self._carried and self.edits.first_question is not None:
            self._select(self.edits.first_question)
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
        panel.rowconfigure(2, weight=1)
        panel.columnconfigure(0, weight=1)

        self._build_carried(panel)
        self._build_entry_state(panel)

        tree_frame = ttk.Frame(panel)
        tree_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 4))
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
        buttons.grid(row=3, column=0, sticky="ew")
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

    def _build_carried(self, panel: ttk.Frame) -> None:
        """The pieces the page before this one left for this page to finish.

        Gridded away on every page that was handed nothing, which is nearly all
        of them — an empty frame saying so would only take room from the tree.
        """
        frame = self._carried_frame = ttk.LabelFrame(
            panel, text=" Carried over ", padding=8
        )
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        self._carried_label = ttk.Label(
            frame,
            text="",
            foreground=MUTED,
            wraplength=self._px(PANEL_WIDTH - 110),
            justify="left",
        )
        self._carried_label.grid(row=0, column=0, sticky="w")
        ttk.Button(frame, text="Discard", width=9, command=self._discard_carried).grid(
            row=0, column=1, sticky="e", padx=(6, 0)
        )
        frame.grid_remove()

    def _build_entry_state(self, panel: ttk.Frame) -> None:
        entry = ttk.LabelFrame(panel, text=" Page starts at ", padding=8)
        entry.grid(row=1, column=0, sticky="ew")
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
        frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
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

        # The join controls sit with the crop rather than with the region list,
        # because what they change is the image above them.
        joins = ttk.Frame(frame)
        joins.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self._joins_next = tk.BooleanVar(master=self.top, value=False)
        self._joins_button = ttk.Checkbutton(
            joins,
            text="Continues into the next piece  (J)",
            variable=self._joins_next,
            command=self._toggle_join,
        )
        self._joins_button.pack(side="left")
        self._line_up_button = ttk.Button(
            joins, text="Line up…", width=10, command=self._line_up
        )
        self._line_up_button.pack(side="right")

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
        if region.label != "question":
            return f"{region.label} {self._reading_text(region)}"

        piece = self._numbering.pieces.get(index)
        if piece is None:
            return "?"
        if piece.held:
            return f"piece {piece.index} → next page"
        if piece.alone:
            return str(piece.placement)
        return f"{piece.placement}  piece {piece.index} of {piece.count}"

    @staticmethod
    def _reading_text(region: PageRegion) -> str:
        if region.label == "question":
            return ""
        return str(region.reading) if region.reading is not None else "unread"

    def _render_regions(self) -> None:
        self._canvas.delete("region")

        for index, region in enumerate(self.edits.regions):
            color = (
                MISNUMBERED
                if index in self._numbering.misnumbered
                else COLORS[region.label]
            )
            selected = index == self.edits.selected
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

        self._draw_joins()

    def _draw_joins(self) -> None:
        """Link the pieces of a joined question, so the join is visible on the page.

        A question whose next piece is on the next page gets an arrow off its
        bottom edge instead — there is nothing on this page to point at.
        """
        regions = self.edits.regions
        for index, region in enumerate(regions):
            if region.label != "question" or not region.joins_next:
                continue

            left, _, right, bottom = geometry.bounds(list(region.polygon))
            start = self._to_canvas((round((left + right) / 2), bottom))
            following = next(
                (
                    at
                    for at in range(index + 1, len(regions))
                    if regions[at].label == "question"
                ),
                None,
            )
            if following is None:
                end = (start[0], start[1] + self._px(JOIN_TAIL))
            else:
                left, top, right, _ = geometry.bounds(list(regions[following].polygon))
                end = self._to_canvas((round((left + right) / 2), top))

            self._canvas.create_line(
                *start,
                *end,
                fill=COLORS["question"],
                width=3,
                dash=(6, 4),
                arrow="last",
                tags="region",
            )

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
        """Ask the model where everything lands, then redraw what it says."""
        self._numbering = self.edits.numbering()
        numbering = self._numbering

        self._ends_at.configure(text=numbering.ends_at)
        self._problem_label.configure(text=numbering.problem or "")
        self._set_approve_enabled(numbering.ok)
        self._continue_button.configure(
            state="normal" if numbering.continue_helps else "disabled"
        )
        self._undo_button.configure(
            state="normal" if self.edits.history.can_undo else "disabled"
        )
        self._redo_button.configure(
            state="normal" if self.edits.history.can_redo else "disabled"
        )
        self._render_regions()
        self._fill_tree()
        self._update_status()
        self._show_join_controls()
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

        placements = self._numbering.placements
        span = ""
        if placements:
            first, last = placements[0], placements[-1]
            span = f"    → {first} … {last}" if first != last else f"    → {first}"

        counts = f"{len(placements)} questions, {self.edits.marker_count} markers"
        held = self._numbering.held
        if held:
            counts += f", {held} piece{'s' if held > 1 else ''} held"
        self._counts_label.configure(text=f"{counts}{span}")

    def _show_entry_state(self) -> None:
        """Put the model's entry state into the spinboxes without echoing back."""
        self._loading = True
        state = self.edits.state
        self._option_var.set(str(state.option))
        self._part_var.set(state.part)
        self._question_var.set(str(state.question))
        self._loading = False

    def _show_carried(self) -> None:
        """Say what was carried onto this page, or hide the row saying it."""
        if not self._carried:
            self._carried_frame.grid_remove()
            return

        pages = ", ".join(dict.fromkeys(piece.page_name for piece in self._carried))
        count = len(self._carried)
        self._carried_label.configure(
            text=f"{count} piece{'s' if count > 1 else ''} from {pages}, saved as the"
            " top of this page's first question."
        )
        self._carried_frame.grid()

    def _discard_carried(self) -> None:
        """Throw away what was carried here — this page does not continue it."""
        if not self._carried:
            return
        if not messagebox.askokcancel(
            "Discard carried pieces",
            "Throw away the pieces carried onto this page? Nothing will be"
            " written for them, and the page they came from is behind us.",
            parent=self.top,
        ):
            return

        self._carried = []
        self.discard_carried = True
        self.edits.carried = 0
        self._show_carried()
        self._refresh()

    def _continue_from_disk(self) -> None:
        """Set the entry counter so the page's first question takes the free slot."""
        counter = self.edits.continue_from_disk()
        if counter is not None:
            self._question_var.set(str(counter))

    def _fill_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for index, region in enumerate(self.edits.regions):
            where = self._caption(index, region) if region.label == "question" else ""
            tag = (
                "misnumbered" if index in self._numbering.misnumbered else region.label
            )
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
        selected = self.edits.selected
        if selected is not None and selected < len(self.edits.regions):
            self._tree.selection_set(str(selected))
            self._tree.see(str(selected))

    def _select(self, index: int | None) -> None:
        if index == self.edits.selected:
            return
        self.edits.selected = index
        self._render_regions()
        if index is None:
            selection = self._tree.selection()
            if selection:
                self._tree.selection_remove(*selection)
        else:
            self._tree.selection_set(str(index))
            self._tree.see(str(index))
        self._show_join_controls()
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
        index = self.edits.selected
        if index is None or index >= len(self.edits.regions):
            self._clear_preview()
            return
        self._preview.delete("all")

        region = self.edits.regions[index]
        try:
            image = self._crop_for(index, region)
        except Exception as exc:  # a polygon can be degenerate mid-edit
            # A real cropping bug lands here too, so keep its traceback.
            logger.debug("Preview crop failed", exc_info=True)
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

    def _crop_for(self, index: int, region: PageRegion) -> Image.Image:
        """The crop for *region* — the extractor's own for a question.

        A question in pieces is shown whole: every piece of it on this page,
        under whatever was carried onto the page, exactly as the file is built.
        """
        if region.label == "question" and self._crop is not None:
            pieces = [
                self.edits.regions[at] for at in self.edits.question_pieces(index)
            ]
            carried = self._carried if self.edits.takes_carried(index) else []
            return self._crop(pieces, carried)
        left, top, right, bottom = geometry.bounds(list(region.polygon))
        return self._image.crop((left, top, right, bottom))

    def _preview_text(self, index: int, region: PageRegion, image: Image.Image) -> str:
        size = f"{image.width}x{image.height} px"
        if region.label != "question":
            return f"{region.label} marker — {size}"

        piece = self._numbering.pieces.get(index)
        if piece is None:
            return f"not placed — {size}"
        if piece.held:
            return f"piece {piece.index}, held for the next page — {size}"
        if piece.alone:
            return f"saved as {piece.placement} — {size}"
        return f"saved as {piece.placement}, {piece.count} pieces joined — {size}"

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

        if kind == "handle":
            self.edits.drag_vertex(index, point, self._to_image(x, y))
        else:
            dx, dy = self._to_image(x - self._drag_from[0], y - self._drag_from[1])
            if dx == 0 and dy == 0:
                return
            self.edits.drag_polygon(index, dx, dy)

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
        region = self.edits.regions[index]
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

        if region.label == "question":
            menu.add_command(
                label="Stop continuing into the next piece"
                if region.joins_next
                else "Continues into the next piece",
                command=self._toggle_join,
            )
            if self._piece_count(index) > 1:
                menu.add_command(label="Line up the pieces…", command=self._line_up)
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
            "j": self._toggle_join,
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
        selected = self.edits.selected
        if selected is None:
            self._canvas.xview_scroll(dx, "units")
            self._canvas.yview_scroll(dy, "units")
            return "break"

        self.edits.nudge(selected, dx * step, dy * step)
        self._refresh()
        return "break"

    def _cycle_selection(self, delta: int) -> None:
        regions = self.edits.regions
        if not regions:
            return
        if self.edits.selected is None:
            self._select(0 if delta > 0 else len(regions) - 1)
            return
        self._select((self.edits.selected + delta) % len(regions))

    def _on_escape(self) -> None:
        if self._draw_label is not None:
            self._cancel_draw()
        else:
            self._select(None)

    # --- editing ---

    def _commit(self) -> None:
        """Record an edit in the undo timeline and redraw everything it touched."""
        self.edits.commit()
        self._refresh()

    def _undo(self) -> None:
        if self.edits.undo():
            self._show_entry_state()
            self._refresh()

    def _redo(self) -> None:
        if self.edits.redo():
            self._show_entry_state()
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

        corner = self._to_image(*start)
        opposite = self._to_image(x, y)
        self._cancel_draw()

        if self.edits.add_box(label, corner, opposite):
            self._refresh()

    def _insert_point(self, index: int, point: tuple[int, int]) -> None:
        """Add a vertex on the polygon edge nearest the click."""
        self.edits.insert_point(index, point)
        self._refresh()

    def _delete_point(self, index: int, point: int) -> None:
        if self.edits.delete_point(index, point):
            self._refresh()
        else:
            self._hint.configure(text=f"a region needs at least {MIN_POINTS} points")

    def _relabel_selected(self, label: PageLabel) -> None:
        if self.edits.selected is not None:
            self._set_label(self.edits.selected, label)

    def _set_label(self, index: int, label: PageLabel) -> None:
        self.edits.set_label(index, label)
        self._refresh()

    def _set_reading(self, index: int, reading: int | str | None) -> None:
        self.edits.set_reading(index, reading)
        self._refresh()

    def _set_option(self, index: int) -> None:
        current = self.edits.regions[index].reading
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
        selected = self.edits.selected
        if selected is None:
            return
        region = self.edits.regions[selected]
        if region.label == "option":
            self._set_option(selected)
        elif region.label == "part":
            self._set_reading(selected, "B" if region.reading == "A" else "A")

    def _piece_count(self, index: int) -> int:
        """How many pieces the question at *index* is made of on this page.

        Counts what was carried onto the page when the question is the one it
        was carried for. Cheap on purpose — it decides whether a button is
        enabled, and cutting the pieces out to count them would deskew each of
        them first.
        """
        if self.edits.regions[index].label != "question":
            return 0
        carried = self.edits.carried if self.edits.takes_carried(index) else 0
        return len(self.edits.question_pieces(index)) + carried

    def _show_join_controls(self) -> None:
        """Put the join controls where the selected region actually stands."""
        index = self.edits.selected
        region = (
            self.edits.regions[index]
            if index is not None and index < len(self.edits.regions)
            else None
        )
        question = region is not None and region.label == "question"

        self._joins_next.set(question and bool(region and region.joins_next))
        self._joins_button.configure(state="normal" if question else "disabled")
        joinable = (
            question
            and index is not None
            # Without the extractor's own piece crop there is nothing honest to
            # line up against.
            and self._crop_piece is not None
            and self._piece_count(index) > 1
        )
        self._line_up_button.configure(state="normal" if joinable else "disabled")

    def _toggle_join(self) -> None:
        """Mark the selected question as continuing into the next piece, or stop.

        The checkbox has already flipped itself by the time this runs, so a
        refused toggle is put back from the model rather than assumed.
        """
        selected = self.edits.selected
        if selected is None:
            self._show_join_controls()
            return
        if not self.edits.toggle_join_next(selected):
            self._show_join_controls()
            self._hint.configure(text="only a question can be half of one")
            return
        self._refresh()

    def _line_up(self) -> None:
        """Line up the pieces of the selected question, by hand, in their own window."""
        index = self.edits.selected
        if index is None:
            return

        pieces, origins = self._join_pieces(index)
        if len(pieces) < 2:
            return

        offsets = JoinEditor(self.top, pieces, scale=self._dpi_scale).run()
        # The editor took the grab for itself; this window needs it back.
        self.top.grab_set()
        self._canvas.focus_set()
        if offsets is None:
            return

        self.edits.set_join_offsets(
            {
                at: offset
                for at, offset in zip(origins, offsets, strict=True)
                if at is not None
            }
        )
        self._refresh()

    def _join_pieces(self, index: int) -> tuple[list[JoinPiece], list[int | None]]:
        """The pieces of the question at *index*, and which region each came from.

        None where a piece was carried onto this page: its offset was settled
        while the page it was cut from was being reviewed, and this page cannot
        move it.
        """
        if self.edits.regions[index].label != "question" or self._crop_piece is None:
            return [], []

        pieces: list[JoinPiece] = []
        origins: list[int | None] = []
        if self.edits.takes_carried(index):
            for carried in self._carried:
                pieces.append(
                    JoinPiece(
                        image=carried.image,
                        offset=carried.offset,
                        caption=f"from {carried.page_name}",
                        movable=False,
                    )
                )
                origins.append(None)

        for at in self.edits.question_pieces(index):
            region = self.edits.regions[at]
            pieces.append(
                JoinPiece(
                    image=self._crop_piece(region.polygon),
                    offset=region.join_offset,
                    caption=f"region {at + 1}, this page",
                )
            )
            origins.append(at)
        return pieces, origins

    def _delete_region(self) -> None:
        if self.edits.selected is None:
            return
        self.edits.delete(self.edits.selected)
        self._refresh()

    def _move(self, delta: int) -> None:
        """Move the selected region in the reading order the numbering follows."""
        if self.edits.selected is None:
            return
        if self.edits.reorder(self.edits.selected, delta) is not None:
            self._refresh()

    def _sort(self) -> None:
        self.edits.sort_by_reading_order()
        self._refresh()

    def _zoom_to_selected(self) -> None:
        """Fill the view with the selected region, so its edges can be judged."""
        selected = self.edits.selected
        if selected is None:
            return
        left, top, right, bottom = geometry.bounds(
            list(self.edits.regions[selected].polygon)
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
        state = self.edits.state
        self.edits.set_entry_state(
            option=_int_or(self._option_var.get(), state.option),
            part=self._part_var.get(),
            question=_int_or(self._question_var.get(), state.question),
        )
        self._refresh()
        # One undo step per settled value, not one per keystroke.
        self._debounce("state", STATE_COMMIT_MS, self.edits.commit)

    def _on_tree_select(self, _event: tk.Event) -> None:
        selection = self._tree.selection()
        if selection:
            self._select(int(selection[0]))

    # --- stats tab ---

    def _refresh_stats_if_shown(self) -> None:
        """The stats tab is brought up to date only when it is in front."""
        try:
            current = self._notebook.select()
        except tk.TclError:
            return
        if current == str(self._stats):
            self._stats.show(self._subject, self._output_year)

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
        if verdict == "approve" and self._numbering.problem:
            return
        if verdict == "abort" and not messagebox.askokcancel(
            "Abort run",
            "Stop the extraction run? Pages already approved keep their images.",
            parent=self.top,
        ):
            return

        self.verdict = verdict
        if verdict == "approve":
            # The extractor writes this page's crops next; the tally moved.
            self._stats.page_written()
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
