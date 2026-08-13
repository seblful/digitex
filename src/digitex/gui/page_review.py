"""Tkinter review window — check a page's regions before they are cropped.

One adapter over the `PageReviewer` seam. The window draws the page, its
polygons and the option/part/number each question would be saved as, and lets
all of it be corrected with the mouse: drag a vertex or a whole polygon, add
and delete points, relabel a region, draw a missing one, reorder them, fix a
misread marker, or move where the page starts numbering.

Every placement it shows comes from :func:`place_questions`, the same walk the
extractor writes with, so the preview cannot drift from what lands on disk.

Its second tab is the running tally of what the subject has produced so far —
the same `ImageCensus` and `AnswerValidator` the ``count-questions`` and
``check-answers`` commands render, so the check that used to be a separate
command after the run happens beside the page that is causing it.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, simpledialog, ttk
from typing import TYPE_CHECKING, Literal

import structlog
from PIL import Image, ImageTk

from digitex.core.domain import PixelPolygon
from digitex.extractors.exceptions import ReviewAborted
from digitex.extractors.placement import (
    PageLabel,
    PageRegion,
    QuestionPlacement,
    place_questions,
    reading_order_key,
)
from digitex.extractors.review import PageProposal, ReviewedPage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from digitex.services.answer_validator import AnswerValidator, ValidationReport
    from digitex.services.image_census import ImageCensus

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

# Half-width of a vertex handle, in screen pixels.
HANDLE = 4

# A crop needs four points (ImageCropper refuses fewer), and the rendered page
# is capped so a deep zoom on a 3500px scan cannot allocate a 150MB bitmap.
MIN_POINTS = 4
MAX_RENDERED_PIXELS = 6000

PANEL_WIDTH = 380
FOOTER_HEIGHT = 44


def _fit_scale(image: Image.Image, width: int, height: int) -> float:
    return min(width / image.width, height / image.height)


def _int_or(raw: str, fallback: int) -> int:
    """Read a spinbox that the user may have emptied or typed junk into."""
    try:
        return int(raw)
    except ValueError:
        return fallback


def _copy_regions(regions: Sequence[PageRegion]) -> list[PageRegion]:
    """Deep-copy the proposal's regions, so skipping leaves the originals alone."""
    return [
        PageRegion(
            label=r.label, polygon=PixelPolygon(list(r.polygon)), reading=r.reading
        )
        for r in regions
    ]


class TkPageReviewer:
    """Interactive `PageReviewer`: opens a window per page and waits for a verdict.

    One hidden root is created on first use and reused, and the window geometry
    carries from page to page so the window stays where it was put.

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
        self._geometry: str | None = None

    def _ensure_root(self) -> tk.Tk:
        if self._root is None:
            root = tk.Tk()
            root.withdraw()
            self._root = root
        return self._root

    def __call__(self, proposal: PageProposal) -> ReviewedPage | None:
        """Show *proposal* and block until the page is approved, skipped or aborted.

        Raises:
            ReviewAborted: If the reviewer stopped the run.
        """
        root = self._ensure_root()
        window = _ReviewWindow(
            root,
            proposal,
            geometry=self._geometry,
            subject=self._subject,
            census=self._census,
            validator=self._validator,
        )
        root.wait_window(window.top)
        self._geometry = window.geometry

        if window.verdict == "abort":
            raise ReviewAborted(proposal.page_name)
        if window.verdict == "skip":
            logger.info("Page skipped in review", page=proposal.page_name)
            return None

        logger.info(
            "Page approved in review",
            page=proposal.page_name,
            regions=len(window.regions),
        )
        return ReviewedPage(regions=window.regions, state=window.state)


class _ReviewWindow:
    """The window itself. Owns a working copy of the page's regions and state."""

    def __init__(
        self,
        root: tk.Tk,
        proposal: PageProposal,
        geometry: str | None = None,
        subject: str = "",
        census: ImageCensus | None = None,
        validator: AnswerValidator | None = None,
    ) -> None:
        self.regions = _copy_regions(proposal.regions)
        self.state = replace(proposal.state)
        self._subject = subject
        self._census = census
        self._validator = validator
        self._year = proposal.output_dir.name
        # Closing the window with the X is an abort, not a skip: a skip writes
        # nothing and says nothing, and losing a page that way is silent.
        self.verdict: Verdict = "abort"
        self.geometry = geometry

        self._image = proposal.image.convert("RGB")
        self._selected: int | None = None
        self._placements: dict[int, QuestionPlacement] = {}
        self._problem: str | None = None
        self._fitted = False
        self._scale = 1.0
        self._photo: ImageTk.PhotoImage | None = None
        self._drag: tuple[str, int, int] | None = None
        self._drag_from: tuple[float, float] | None = None
        self._draw_label: PageLabel | None = None
        self._draw_from: tuple[float, float] | None = None

        self.top = tk.Toplevel(root)
        self.top.title(f"Review — {proposal.page_name or 'page'}")
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)
        if geometry:
            self.top.geometry(geometry)
        else:
            # Sized off the screen rather than fixed: a window taller than the
            # display puts the footer's buttons out of reach.
            width = int(self.top.winfo_screenwidth() * 0.92)
            height = int(self.top.winfo_screenheight() * 0.88)
            self.top.geometry(f"{width}x{height}+20+20")

        self._build()
        self._refresh()
        self._refresh_stats()

        # No transient(): the reviewer's root is withdrawn, and a toplevel made
        # transient for an unmapped master never maps itself on Windows.
        self.top.deiconify()
        self.top.lift()
        self.top.grab_set()
        self.top.focus_force()

    # --- construction ---

    def _build(self) -> None:
        self.top.rowconfigure(1, weight=1)
        self.top.columnconfigure(0, weight=1)
        # The canvas is the only cell that grows; without floors of their own
        # the side panel and the footer buttons are squeezed out of the window.
        self.top.columnconfigure(1, minsize=PANEL_WIDTH)
        self.top.rowconfigure(2, minsize=FOOTER_HEIGHT)

        self._build_toolbar()
        self._build_canvas()
        self._build_panel()
        self._build_footer()

        self.top.bind("<Delete>", lambda _e: self._delete_region())
        self.top.bind("<Control-Return>", lambda _e: self._finish("approve"))
        self.top.bind("<Escape>", lambda _e: self._cancel_draw())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.top, padding=(6, 4))
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Button(bar, text="-", width=3, command=lambda: self._zoom(1 / 1.25)).pack(
            side="left"
        )
        ttk.Button(bar, text="+", width=3, command=lambda: self._zoom(1.25)).pack(
            side="left"
        )
        ttk.Button(bar, text="Fit", width=5, command=self._zoom_to_fit).pack(
            side="left"
        )

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(bar, text="Draw:").pack(side="left")
        for label in LABELS:
            ttk.Button(
                bar,
                text=label,
                width=9,
                command=lambda lbl=label: self._start_draw(lbl),
            ).pack(side="left", padx=1)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)

        self._hint = ttk.Label(bar, text="", foreground="#666")
        self._hint.pack(side="left")

    def _build_canvas(self) -> None:
        frame = ttk.Frame(self.top)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(frame, background="#3a3a3a", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        vbar = ttk.Scrollbar(frame, orient="vertical", command=self._canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = ttk.Scrollbar(frame, orient="horizontal", command=self._canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        # The first fit has to wait for the canvas to have a size: at build
        # time it is 1x1, and fitting to that pins the page at minimum zoom.
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Button-3>", self._on_context)
        self._canvas.bind("<MouseWheel>", self._on_wheel)

    def _build_panel(self) -> None:
        notebook = self._notebook = ttk.Notebook(self.top)
        notebook.grid(row=1, column=1, sticky="ns")

        page_tab = ttk.Frame(notebook, padding=6)
        stats_tab = ttk.Frame(notebook, padding=6)
        notebook.add(page_tab, text="This page")
        notebook.add(stats_tab, text="Extracted so far")

        self._build_page_tab(page_tab)
        self._build_stats_tab(stats_tab)

    def _build_page_tab(self, panel: ttk.Frame) -> None:
        panel.rowconfigure(1, weight=1)

        entry = ttk.LabelFrame(panel, text="Page starts at", padding=6)
        entry.grid(row=0, column=0, sticky="ew")

        self._option_var = tk.StringVar(value=str(self.state.option))
        self._part_var = tk.StringVar(value=self.state.part)
        self._question_var = tk.StringVar(value=str(self.state.question))

        ttk.Label(entry, text="option").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(entry, from_=0, to=99, width=5, textvariable=self._option_var).grid(
            row=0, column=1, sticky="w", padx=4
        )

        ttk.Label(entry, text="part").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            entry,
            values=("", "A", "B"),
            width=3,
            state="readonly",
            textvariable=self._part_var,
        ).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(entry, text="questions done").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            entry, from_=0, to=999, width=5, textvariable=self._question_var
        ).grid(row=2, column=1, sticky="w", padx=4)

        for var in (self._option_var, self._part_var, self._question_var):
            var.trace_add("write", lambda *_a: self._on_entry_state_changed())

        self._ends_at = ttk.Label(entry, text="", foreground="#555")
        self._ends_at.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        tree_frame = ttk.Frame(panel)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 4))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("kind", "where"),
            show="headings",
            selectmode="browse",
            height=22,
        )
        self._tree.heading("kind", text="region")
        self._tree.heading("where", text="saved as")
        self._tree.column("kind", width=110, anchor="w")
        self._tree.column("where", width=110, anchor="w")
        self._tree.grid(row=0, column=0, sticky="nsew")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

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

        self._problem_label = ttk.Label(
            panel, text="", foreground="#c00", wraplength=300, justify="left"
        )
        self._problem_label.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def _build_stats_tab(self, panel: ttk.Frame) -> None:
        panel.rowconfigure(1, weight=1)
        panel.rowconfigure(3, weight=1)
        panel.columnconfigure(0, weight=1)

        head = ttk.Frame(panel)
        head.grid(row=0, column=0, sticky="ew")
        self._stats_title = ttk.Label(head, text="", font=("TkDefaultFont", 10, "bold"))
        self._stats_title.pack(side="left")
        ttk.Button(head, text="Recount", width=8, command=self._refresh_stats).pack(
            side="right"
        )

        self._stats_tree = ttk.Treeview(
            panel, columns=("images",), show="tree headings", height=14
        )
        self._stats_tree.heading("#0", text="year / option / part")
        self._stats_tree.heading("images", text="images")
        self._stats_tree.column("#0", width=210, anchor="w")
        self._stats_tree.column("images", width=70, anchor="e")
        self._stats_tree.tag_configure("complete", foreground="#1f9d55")
        self._stats_tree.tag_configure("off", foreground="#c00")
        self._stats_tree.grid(row=1, column=0, sticky="nsew", pady=(4, 6))

        ttk.Button(panel, text="Check answers", command=self._check_answers).grid(
            row=2, column=0, sticky="ew"
        )

        self._answers_text = tk.Text(
            panel, width=40, height=12, wrap="word", font=("TkFixedFont", 9)
        )
        self._answers_text.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        self._answers_text.configure(state="disabled")

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.top, padding=(6, 6))
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Label(
            footer,
            text=(
                "drag polygon or handle to move · right-click for points and labels"
                " · Del removes · Ctrl+Enter approves"
            ),
            foreground="#666",
        ).pack(side="left")

        ttk.Button(
            footer, text="Abort run", command=lambda: self._finish("abort")
        ).pack(side="right", padx=(6, 0))
        ttk.Button(footer, text="Skip page", command=lambda: self._finish("skip")).pack(
            side="right"
        )
        self._approve = ttk.Button(
            footer, text="Approve & save", command=lambda: self._finish("approve")
        )
        self._approve.pack(side="right", padx=6)

    # --- coordinates and rendering ---

    def _to_canvas(self, point: tuple[int, int]) -> tuple[float, float]:
        return (point[0] * self._scale, point[1] * self._scale)

    def _to_image(self, x: float, y: float) -> tuple[int, int]:
        return (round(x / self._scale), round(y / self._scale))

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if not self._fitted and self._canvas.winfo_width() > 1:
            self._fitted = True
            self._zoom_to_fit()

    def _zoom_to_fit(self) -> None:
        width = max(self._canvas.winfo_width(), 1)
        height = max(self._canvas.winfo_height(), 1)
        self._set_scale(_fit_scale(self._image, width, height))

    def _zoom(self, factor: float) -> None:
        self._set_scale(self._scale * factor)

    def _set_scale(self, scale: float) -> None:
        largest = max(self._image.width, self._image.height)
        self._scale = max(0.05, min(scale, MAX_RENDERED_PIXELS / largest))
        self._render_page()
        self._render_regions()

    def _render_page(self) -> None:
        width = max(1, round(self._image.width * self._scale))
        height = max(1, round(self._image.height * self._scale))
        self._photo = ImageTk.PhotoImage(
            self._image.resize((width, height), Image.Resampling.BILINEAR)
        )
        self._canvas.delete("page")
        self._canvas.create_image(0, 0, image=self._photo, anchor="nw", tags="page")
        self._canvas.tag_lower("page")
        self._canvas.configure(scrollregion=(0, 0, width, height))

    def _caption(self, index: int, region: PageRegion) -> str:
        if region.label == "question":
            placement = self._placements.get(index)
            return str(placement) if placement else "?"
        return f"{region.label} {region.reading if region.reading is not None else '?'}"

    def _render_regions(self) -> None:
        self._canvas.delete("region")

        for index, region in enumerate(self.regions):
            color = COLORS[region.label]
            selected = index == self._selected
            coords = [c for point in region.polygon for c in self._to_canvas(point)]

            self._canvas.create_polygon(
                coords,
                outline=color,
                fill=color,
                stipple="gray12",
                width=3 if selected else 1,
                tags=("region", f"region:{index}"),
            )

            anchor = min(region.polygon, key=_caption_anchor)
            x, y = self._to_canvas(anchor)
            self._canvas.create_text(
                x + 3,
                y - 3,
                text=f"{index + 1}. {self._caption(index, region)}",
                anchor="sw",
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
                tags=("region", f"region:{index}"),
            )

            if selected:
                for point_index, point in enumerate(region.polygon):
                    px, py = self._to_canvas(point)
                    self._canvas.create_rectangle(
                        px - HANDLE,
                        py - HANDLE,
                        px + HANDLE,
                        py + HANDLE,
                        fill="#ffffff",
                        outline=color,
                        tags=("region", "handle", f"handle:{index}:{point_index}"),
                    )

    # --- the preview, recomputed after every edit ---

    def _refresh(self) -> None:
        """Replay the regions through a copy of the entry state and redraw."""
        preview = replace(self.state)
        self._placements = {}
        self._problem = None

        try:
            placed = place_questions(self.regions, preview)
        except ValueError as exc:
            self._problem = str(exc)
        else:
            questions = [i for i, r in enumerate(self.regions) if r.label == "question"]
            self._placements = {
                index: item.placement
                for index, item in zip(questions, placed, strict=True)
            }
            part = preview.part or "?"
            self._ends_at.configure(
                text=f"ends at {preview.option}/{part}/{preview.question}"
            )

        self._problem_label.configure(text=self._problem or "")
        self._approve.configure(state="disabled" if self._problem else "normal")
        self._render_regions()
        self._fill_tree()

    def _fill_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for index, region in enumerate(self.regions):
            where = self._caption(index, region) if region.label == "question" else ""
            self._tree.insert(
                "",
                "end",
                iid=str(index),
                values=(f"{index + 1}. {region.label}", where),
            )
        if self._selected is not None and self._selected < len(self.regions):
            self._tree.selection_set(str(self._selected))
            self._tree.see(str(self._selected))

    def _select(self, index: int | None) -> None:
        self._selected = index
        self._render_regions()
        if index is None:
            self._tree.selection_remove(*self._tree.selection())
        else:
            self._tree.selection_set(str(index))
            self._tree.see(str(index))

    # --- mouse ---

    def _hit(self, x: float, y: float) -> tuple[str, int, int] | None:
        """What sits under the cursor: a vertex handle, a region, or nothing."""
        for prefix, slack in (("handle:", HANDLE), ("region:", 1)):
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

    def _on_press(self, event: tk.Event) -> None:
        x, y = self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)

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
        x, y = self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)

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
            region.polygon = PixelPolygon(
                [(px + dx, py + dy) for px, py in region.polygon]
            )

        self._drag_from = (x, y)
        self._render_regions()

    def _on_release(self, event: tk.Event) -> None:
        if self._draw_label is not None and self._draw_from is not None:
            self._finish_draw(
                self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)
            )
            return

        if self._drag is not None:
            self._drag = None
            self._drag_from = None
            self._refresh()

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom(1.25 if event.delta > 0 else 1 / 1.25)

    def _on_context(self, event: tk.Event) -> None:
        x, y = self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)
        hit = self._hit(x, y)
        if hit is None:
            return

        kind, index, point = hit
        self._select(index)
        region = self.regions[index]

        menu = tk.Menu(self.top, tearoff=0)
        if kind == "handle":
            menu.add_command(
                label="Delete point", command=lambda: self._delete_point(index, point)
            )
        else:
            menu.add_command(
                label="Insert point here",
                command=lambda: self._insert_point(index, self._to_image(x, y)),
            )
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
        menu.tk_popup(event.x_root, event.y_root)

    # --- editing ---

    def _start_draw(self, label: PageLabel) -> None:
        self._draw_label = label
        self._hint.configure(text=f"drag a box for the new {label} — Esc cancels")

    def _cancel_draw(self) -> None:
        self._draw_label = None
        self._draw_from = None
        self._canvas.delete("rubber")
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
        if right - left < 5 or bottom - top < 5:
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
        self._refresh()

    def _insert_point(self, index: int, point: tuple[int, int]) -> None:
        """Add a vertex on the polygon edge nearest the click."""
        polygon = list(self.regions[index].polygon)
        best = min(
            range(len(polygon)),
            key=lambda i: _distance_to_segment(
                point, polygon[i], polygon[(i + 1) % len(polygon)]
            ),
        )
        polygon.insert(best + 1, point)
        self.regions[index].polygon = PixelPolygon(polygon)
        self._refresh()

    def _delete_point(self, index: int, point: int) -> None:
        polygon = list(self.regions[index].polygon)
        if len(polygon) <= MIN_POINTS:
            self._hint.configure(text=f"a region needs at least {MIN_POINTS} points")
            return
        del polygon[point]
        self.regions[index].polygon = PixelPolygon(polygon)
        self._refresh()

    def _set_label(self, index: int, label: PageLabel) -> None:
        region = self.regions[index]
        region.label = label
        # The old reading belongs to the old kind — an option number on a part
        # marker would be ignored anyway, and shown as if it counted.
        region.reading = None
        self._refresh()

    def _set_reading(self, index: int, reading: int | str | None) -> None:
        self.regions[index].reading = reading
        self._refresh()

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

    def _delete_region(self) -> None:
        if self._selected is None:
            return
        del self.regions[self._selected]
        self._selected = None
        self._refresh()

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
        self._refresh()

    def _sort(self) -> None:
        selected = self.regions[self._selected] if self._selected is not None else None
        self.regions.sort(key=lambda region: reading_order_key(region.polygon))
        # By identity, not equality: two regions can hold equal field values.
        self._selected = next(
            (i for i, r in enumerate(self.regions) if r is selected), None
        )
        self._refresh()

    def _on_entry_state_changed(self) -> None:
        self.state.option = _int_or(self._option_var.get(), self.state.option)
        self.state.part = self._part_var.get()
        self.state.question = _int_or(self._question_var.get(), self.state.question)
        self._refresh()

    def _on_tree_select(self, _event: tk.Event) -> None:
        selection = self._tree.selection()
        if selection:
            index = int(selection[0])
            if index != self._selected:
                self._selected = index
                self._render_regions()

    # --- what the subject has produced so far ---

    def _refresh_stats(self) -> None:
        """Recount the subject's output tree — what is on disk, this page aside."""
        self._stats_tree.delete(*self._stats_tree.get_children())

        if self._census is None:
            self._stats_title.configure(text="no census available")
            return

        try:
            census = self._census.take(self._subject)
        except FileNotFoundError:
            self._stats_title.configure(text=f"{self._subject}: nothing extracted yet")
            return

        self._stats_title.configure(
            text=f"{census.subject}: {census.images} images, {census.folders} folders"
        )

        for year in census.years:
            node = self._stats_tree.insert(
                "",
                "end",
                text=f"{year.year} — {year.options} options",
                values=(year.images,),
                tags=("complete",) if year.is_complete else ("off",),
                open=year.year == self._year,
            )
            for part in year.parts:
                self._stats_tree.insert(
                    node,
                    "end",
                    text=f"{part.option}/{part.part}",
                    values=(part.images,),
                    tags=("off",) if part.off_mode else (),
                )

    def _check_answers(self) -> None:
        """Validate answers.json against the images, and show the report."""
        if self._validator is None:
            self._show_answers("no validator available")
            return

        try:
            report = self._validator.validate(self._subject)
        except FileNotFoundError:
            self._show_answers(f"{self._subject} has no extraction output yet")
            return

        self._show_answers(_format_validation(report))

    def _show_answers(self, text: str) -> None:
        self._answers_text.configure(state="normal")
        self._answers_text.delete("1.0", "end")
        self._answers_text.insert("1.0", text)
        self._answers_text.configure(state="disabled")

    # --- verdict ---

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
        self.geometry = self.top.geometry()
        self.top.destroy()

    def _on_close(self) -> None:
        self._finish("abort")


def _caption_anchor(point: tuple[int, int]) -> tuple[int, int]:
    """Sort key picking a polygon's topmost point, to hang its caption off."""
    return (point[1], point[0])


# How many missing question keys to name before saying "and N more" — a year
# whose answers.json never arrived would otherwise print hundreds.
_MAX_LISTED = 12


def _listed(keys: list[str]) -> str:
    shown = ", ".join(keys[:_MAX_LISTED])
    rest = len(keys) - _MAX_LISTED
    return f"{shown}, +{rest} more" if rest > 0 else shown


def _format_validation(report: ValidationReport) -> str:
    """Render an answers report as plain text, the way check-answers renders it."""
    lines = [f"Answers for {report.subject}"]

    for year in report.years:
        if not year.answers_file_present:
            lines.append(f"\n{year.year}: answers.json NOT FOUND")
            continue
        if not year.answers_file_valid:
            lines.append(f"\n{year.year}: answers.json IS UNREADABLE")
            continue

        if year.has_mismatch:
            status = "MISMATCH"
        elif year.options_differ:
            status = "OPTIONS DIFFER"
        else:
            status = "OK"

        lines.append(f"\n{year.year}: {status}")
        lines.append(f"  A-part {year.a_count}, B-part {year.b_count}")
        lines.append(
            f"  in images {year.image_question_count},"
            f" in answers.json {year.answer_question_count}"
        )
        if year.options_with_differing_questions:
            lines.append(
                f"  options differing: {_listed(year.options_with_differing_questions)}"
            )
        if year.missing_in_answers:
            lines.append(f"  missing in answers: {_listed(year.missing_in_answers)}")
        if year.missing_in_images:
            lines.append(f"  missing in images: {_listed(year.missing_in_images)}")
        lines.append(
            f"  Part B 'Б': {year.part_b_coverage}"
            f" ({year.options_with_b}/{year.total_options} options)"
        )

    issues = report.total_issues
    lines.append(f"\n{issues} issue(s) found" if issues else "\nAll years match")
    return "\n".join(lines)


def _distance_to_segment(
    point: tuple[int, int], start: tuple[int, int], end: tuple[int, int]
) -> float:
    """Squared distance from *point* to the segment *start*-*end*."""
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    if length == 0:
        return (px - x0) ** 2 + (py - y0) ** 2
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length))
    return (px - x0 - t * dx) ** 2 + (py - y0 - t * dy) ** 2
