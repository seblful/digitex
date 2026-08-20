"""A page review, minus the pixels.

The review window was 95 methods over 1,499 lines, and everything it knew — the
page being reviewed, which region is selected, what each row says, whether the
page may be approved, what verdict was reached — it knew as widget state. So
none of it could be asserted without a display, and the 43 tests written for it
skipped on every headless machine, including CI. The largest class in the
project was the least covered.

This holds that state instead. It builds no widget and imports no tkinter, so
what a review *is* can be exercised directly:

    controller = ReviewController()
    controller.load(proposal)
    controller.select(0)
    assert controller.approve_enabled

The window keeps what is genuinely a window's: pixels on a canvas, coordinate
spaces, event translation, and the modal loop. It asks this object what to draw
and tells it what the reviewer did.

The editing rules are one layer further down still, in :class:`PageEdits`, which
this delegates to and does not second-guess. Three layers, each testable without
the one above it: what an edit means, what a review is, what it looks like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog
from PIL import Image

from digitex.pipeline.exceptions import ReviewAborted
from digitex.pipeline.review import ReviewedPage
from digitex.ui import geometry
from digitex.ui.edits import Numbering, PageEdits
from digitex.ui.join_editor import JoinPiece

if TYPE_CHECKING:
    from digitex.domain.placement import PageLabel, PageRegion
    from digitex.pipeline.pieces import HeldPiece
    from digitex.pipeline.review import PageProposal, PieceCrop, QuestionCrop

logger = structlog.get_logger()

Verdict = Literal["approve", "skip", "abort"]


@dataclass(frozen=True)
class RegionRow:
    """One line of the region list, already worded.

    The window puts these in a treeview; a test reads them as text. Neither
    needs to know how a piece caption is phrased.
    """

    index: int
    label: str
    reading: str
    where: str
    misnumbered: bool


@dataclass(frozen=True)
class JoinControls:
    """What the join controls should show for the current selection."""

    joins_next: bool
    can_toggle: bool
    can_line_up: bool


@dataclass(frozen=True)
class Status:
    """The two lines above the canvas."""

    where: str
    counts: str


NO_SELECTION_CAPTION = "select a region to preview its crop"


@dataclass(frozen=True)
class Preview:
    """The crop the selection would be saved as, and its caption.

    ``image`` is None when there is nothing to show — no selection, or a polygon
    too degenerate to cut — and the caption says which.
    """

    image: Image.Image | None
    caption: str


def resolve_verdict(
    verdict: Verdict,
    edits: PageEdits,
    page_name: str,
    discard_carried: bool = False,
) -> ReviewedPage | None:
    """Turn a verdict into the reviewer's answer.

    The three ways a review ends: approve hands back what the reviewer edited,
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


class ReviewController:
    """The page under review, and every question that can be asked about it.

    Mutable and reused: one controller serves a whole run, taking page after page
    through :meth:`load`, which is what lets the window carry zoom and selection
    habits across pages without this object knowing about either.

    Every operation that changes the page re-derives the numbering before it
    returns, so what this reports is never one edit behind what it holds. The
    caller redraws; it does not have to remember to recompute first.
    """

    def __init__(self) -> None:
        self.edits = PageEdits()
        self.numbering = Numbering()

        # A 1x1 white page until the first proposal arrives, so the window can
        # measure and draw something before it has anything to review.
        self.image = Image.new("RGB", (1, 1), "white")
        self.crop: QuestionCrop | None = None
        self.crop_piece: PieceCrop | None = None
        self.carried: list[HeldPiece] = []
        self.discard_carried = False

        self.page_name = ""
        self.page_number = 0
        self.page_count = 0
        self.output_year = ""

        self.hover: int | None = None
        self.draw_label: PageLabel | None = None
        # Abort until the reviewer says otherwise: a window closed or a run
        # interrupted must never read as an approval.
        self.verdict: Verdict = "abort"

    # --- taking on a page ---

    def load(self, proposal: PageProposal) -> None:
        """Take on a new page. Returns nothing; ask for what you need after."""
        self.image = proposal.image.convert("RGB")
        self.crop = proposal.crop
        self.crop_piece = proposal.crop_piece
        self.carried = list(proposal.carried)
        self.discard_carried = False
        self.page_name = proposal.page_name or "page"
        self.page_number = proposal.page_number
        self.page_count = proposal.page_count
        self.output_year = proposal.output_dir.name

        self.edits.load(
            proposal.regions,
            proposal.state,
            proposal.output_dir,
            carried=len(self.carried),
        )
        self.hover = None
        self.draw_label = None
        self.refresh()
        # A page finishing a question the page before it started opens on that
        # question, so the joined crop is the first thing the reviewer sees.
        if self.carried and self.edits.first_question is not None:
            self.select(self.edits.first_question)

    def refresh(self) -> Numbering:
        """Ask the model where everything lands. Every edit ends here."""
        self.numbering = self.edits.numbering()
        return self.numbering

    def title(self, subject: str) -> str:
        return f"Review — {self.page_name} — {subject or 'extraction'}"

    # --- what the window draws ---

    @property
    def approve_enabled(self) -> bool:
        return self.numbering.ok

    @property
    def continue_helps(self) -> bool:
        return self.numbering.continue_helps

    @property
    def problem(self) -> str:
        return self.numbering.problem or ""

    @property
    def ends_at(self) -> str:
        return self.numbering.ends_at

    @property
    def can_undo(self) -> bool:
        return self.edits.history.can_undo

    @property
    def can_redo(self) -> bool:
        return self.edits.history.can_redo

    @staticmethod
    def reading_text(region: PageRegion) -> str:
        """What OCR made of a marker. Empty for a question, which carries none."""
        if region.label == "question":
            return ""
        return str(region.reading) if region.reading is not None else "unread"

    def caption(self, index: int, region: PageRegion) -> str:
        """What the region at *index* is labelled on the page and in the list."""
        if region.label != "question":
            return f"{region.label} {self.reading_text(region)}"

        piece = self.numbering.pieces.get(index)
        if piece is None:
            return "?"
        if piece.held:
            return f"piece {piece.index} → next page"
        if piece.alone:
            return str(piece.placement)
        return f"{piece.placement}  piece {piece.index} of {piece.count}"

    def rows(self) -> list[RegionRow]:
        """Every region as a list row, in reading order."""
        return [
            RegionRow(
                index=index,
                label=f"{index + 1}. {region.label}",
                reading=self.reading_text(region),
                # A marker is saved nowhere, so the column stays empty for one.
                where=self.caption(index, region) if region.label == "question" else "",
                misnumbered=index in self.numbering.misnumbered,
            )
            for index, region in enumerate(self.edits.regions)
        ]

    def status(self) -> Status:
        """The page's place in the run, and what it is about to write."""
        where = self.page_name
        if self.page_count:
            where += f"    page {self.page_number} of {self.page_count}"

        placements = self.numbering.placements
        counts = f"{len(placements)} questions, {self.edits.marker_count} markers"
        held = self.numbering.held
        if held:
            counts += f", {held} piece{'s' if held > 1 else ''} held"

        span = ""
        if placements:
            first, last = placements[0], placements[-1]
            span = f"    → {first} … {last}" if first != last else f"    → {first}"
        return Status(where=where, counts=f"{counts}{span}")

    def carried_summary(self) -> str | None:
        """What was carried onto this page, or None when nothing was."""
        if not self.carried:
            return None
        # dict.fromkeys rather than a set: the pages keep the order the pieces
        # came in, and a question spanning three pages names two of them.
        pages = ", ".join(dict.fromkeys(piece.page_name for piece in self.carried))
        count = len(self.carried)
        return (
            f"{count} piece{'s' if count > 1 else ''} from {pages}, saved as the"
            " top of this page's first question."
        )

    # --- selection ---

    def select(self, index: int | None) -> bool:
        """Select a region. False when it was already selected."""
        if index == self.edits.selected:
            return False
        self.edits.selected = index
        return True

    def set_hover(self, index: int | None) -> bool:
        """Point at a region. False when it was already hovered."""
        if index == self.hover:
            return False
        self.hover = index
        return True

    def cycle_selection(self, delta: int) -> bool:
        """Step through the regions, wrapping. False on a page with none."""
        regions = self.edits.regions
        if not regions:
            return False
        if self.edits.selected is None:
            return self.select(0 if delta > 0 else len(regions) - 1)
        return self.select((self.edits.selected + delta) % len(regions))

    def selected_region(self) -> PageRegion | None:
        index = self.edits.selected
        if index is None or index >= len(self.edits.regions):
            return None
        return self.edits.regions[index]

    # --- the crop preview ---

    def crop_for(self, index: int, region: PageRegion) -> Image.Image:
        """The selected region as it would be saved.

        A question goes through the extractor's own cropping pipeline, so the
        preview is the file that would be written rather than a lookalike of it.
        A marker is only ever read, so it is just cut out of the page.
        """
        if region.label == "question" and self.crop is not None:
            pieces = [
                self.edits.regions[at] for at in self.edits.question_pieces(index)
            ]
            carried = self.carried if self.edits.takes_carried(index) else []
            return self.crop(pieces, carried)
        left, top, right, bottom = geometry.bounds(list(region.polygon))
        return self.image.crop((left, top, right, bottom))

    def preview(self) -> Preview:
        """The selected region as it would be saved, and what to say about it.

        A degenerate polygon mid-edit is a caption, not an exception: the
        reviewer is dragging a vertex and the crop is momentarily impossible. A
        real cropping bug lands here too, so the caller keeps the traceback.
        """
        index = self.edits.selected
        if index is None or index >= len(self.edits.regions):
            return Preview(image=None, caption=NO_SELECTION_CAPTION)

        region = self.edits.regions[index]
        try:
            image = self.crop_for(index, region)
        except Exception as exc:
            return Preview(image=None, caption=f"cannot crop this polygon: {exc}")
        return Preview(image=image, caption=self.preview_text(index, region, image))

    def preview_text(self, index: int, region: PageRegion, image: Image.Image) -> str:
        """The caption under the crop: what it would be written as, and how big."""
        size = f"{image.width}x{image.height} px"
        if region.label != "question":
            return f"{region.label} marker — {size}"

        piece = self.numbering.pieces.get(index)
        if piece is None:
            return f"not placed — {size}"
        if piece.held:
            return f"piece {piece.index}, held for the next page — {size}"
        if piece.alone:
            return f"saved as {piece.placement} — {size}"
        return f"saved as {piece.placement}, {piece.count} pieces joined — {size}"

    # --- pieces and joins ---

    def piece_count(self, index: int) -> int:
        """How many pieces the question at *index* is made of on this page.

        Cheap on purpose — it decides whether a button is enabled, and cutting
        the pieces out to count them would deskew each of them first.
        """
        if self.edits.regions[index].label != "question":
            return 0
        carried = self.edits.carried if self.edits.takes_carried(index) else 0
        return len(self.edits.question_pieces(index)) + carried

    def join_controls(self) -> JoinControls:
        """Where the join controls stand for the current selection."""
        index = self.edits.selected
        region = self.selected_region()
        question = region is not None and region.label == "question"
        return JoinControls(
            joins_next=question and bool(region and region.joins_next),
            can_toggle=question,
            # Without the extractor's own piece crop there is nothing honest to
            # line the pieces up against.
            can_line_up=(
                question
                and index is not None
                and self.crop_piece is not None
                and self.piece_count(index) > 1
            ),
        )

    def toggle_join(self) -> bool:
        """Mark the selection as continuing into the next piece. False if refused."""
        selected = self.edits.selected
        if selected is None:
            return False
        if not self.edits.toggle_join_next(selected):
            return False
        self.refresh()
        return True

    def join_pieces(self, index: int) -> tuple[list[JoinPiece], list[int | None]]:
        """The pieces of the question at *index*, and which region each came from.

        None where a piece was carried onto this page: its offset was settled
        while the page it was cut from was being reviewed, and this page cannot
        move it.
        """
        if self.edits.regions[index].label != "question" or self.crop_piece is None:
            return [], []

        pieces: list[JoinPiece] = []
        origins: list[int | None] = []

        if self.edits.takes_carried(index):
            for carried in self.carried:
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
                    image=self.crop_piece(region.polygon),
                    offset=region.join_offset,
                    caption=f"region {at + 1}, this page",
                )
            )
            origins.append(at)
        return pieces, origins

    def apply_join_offsets(
        self, offsets: list[tuple[int, int]], origins: list[int | None]
    ) -> None:
        """Take the offsets the join editor handed back, dropping carried pieces."""
        self.edits.set_join_offsets(
            {
                at: offset
                for at, offset in zip(origins, offsets, strict=True)
                if at is not None
            }
        )
        self.refresh()

    # --- editing ---

    def delete_selected(self) -> bool:
        if self.edits.selected is None:
            return False
        self.edits.delete(self.edits.selected)
        self.refresh()
        return True

    def move_selected(self, delta: int) -> bool:
        """Move the selection in the reading order the numbering follows."""
        if self.edits.selected is None:
            return False
        moved = self.edits.reorder(self.edits.selected, delta) is not None
        if moved:
            self.refresh()
        return moved

    def sort_by_reading_order(self) -> None:
        self.edits.sort_by_reading_order()
        self.refresh()

    def relabel_selected(self, label: PageLabel) -> bool:
        if self.edits.selected is None:
            return False
        self.edits.set_label(self.edits.selected, label)
        self.refresh()
        return True

    def nudge_selected(self, dx: int, dy: int) -> bool:
        if self.edits.selected is None:
            return False
        self.edits.nudge(self.edits.selected, dx, dy)
        self.refresh()
        return True

    def undo(self) -> bool:
        if not self.edits.undo():
            return False
        self.refresh()
        return True

    def redo(self) -> bool:
        if not self.edits.redo():
            return False
        self.refresh()
        return True

    def commit(self) -> None:
        self.edits.commit()
        self.refresh()

    def continue_from_disk(self) -> int | None:
        """Move the entry counter so the first question takes the free slot.

        Applied here rather than returned for the caller to apply. It used to
        reach the model only by being written into a spinbox, which meant the
        rule ran through a widget's change event — and could not be exercised
        without one.
        """
        counter = self.edits.continue_from_disk()
        if counter is None:
            return None
        state = self.edits.state
        self.edits.set_entry_state(state.option, state.part, counter)
        self.refresh()
        return counter

    def discard_carried_pieces(self) -> None:
        """Throw away what was carried here — this page does not continue it."""
        self.carried = []
        self.discard_carried = True
        self.edits.carried = 0
        self.refresh()

    # --- ending ---

    def finish(self, verdict: Verdict) -> bool:
        """Settle the page. False when the verdict is refused.

        Only approval can be refused, and only for a page whose numbering would
        not continue the output tree — the same rule the extractor applies before
        writing, which is why it is here and not in the widget that happens to
        own the button.
        """
        if verdict == "approve" and not self.approve_enabled:
            return False
        self.verdict = verdict
        return True

    def answer(self) -> ReviewedPage | None:
        """The reviewer's answer for the page just reviewed.

        Raises:
            ReviewAborted: If the reviewer stopped the run.
        """
        return resolve_verdict(
            self.verdict,
            self.edits,
            self.page_name,
            discard_carried=self.discard_carried,
        )
