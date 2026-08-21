"""One page in, question images out.

Reading a page is split from writing it. :meth:`PageExtractor.read_page` turns a
page image into labelled regions with their markers read; :meth:`~PageExtractor.
extract` walks those regions through the numbering in
:mod:`digitex.domain.placement` and saves a crop for each question. A
``PageReviewer`` sits between the two, which is where a human gets to correct a
polygon or a misread marker before anything lands on disk.

Both collaborators arrive as :mod:`digitex.pipeline.ports` protocols, never as
classes: importing this module loads neither torch nor ultralytics, because
everything here after the detections is arithmetic on pixels.

A question printed across a page break is saved by the page that finishes it —
the page before hands its piece over in a :class:`PageCarry`, and the crop
written is the pieces stacked. See :mod:`digitex.pipeline.pieces`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from digitex.domain.corpus import question_image_path, question_slot_taken
from digitex.domain.entities import normalize_option_number
from digitex.domain.numbering import preview
from digitex.domain.placement import (
    PageRegion,
    copy_regions,
    place_questions,
    reading_order_key,
)
from digitex.imaging import (
    add_white_background,
    cut_out_image_by_polygon,
    resize_image,
    rotate_image,
    stack_vertically,
)
from digitex.pipeline.pieces import PIECE_GAP, HeldPiece, PageCarry
from digitex.pipeline.review import PageProposal, accept_page

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

    from digitex.domain.entities import Detection, PixelPolygon
    from digitex.domain.placement import PageExtractionState, QuestionPlacement
    from digitex.pipeline.base import ExtractionConfig
    from digitex.pipeline.ports import RegionDetector, TextReader
    from digitex.pipeline.review import PageReviewer

logger = structlog.get_logger()


class PageExtractor:
    """Extract question images from a single page using YOLO segmentation."""

    def __init__(
        self,
        config: ExtractionConfig,
        detector: RegionDetector,
        text_reader: TextReader,
        on_review: PageReviewer | None = None,
    ) -> None:
        self.config = config

        self._detector = detector
        self._text_reader = text_reader
        self._on_review = on_review or accept_page

    # -- reading a page ----------------------------------------------------

    def _detect(self, image: Image.Image) -> list[Detection]:
        """Every region the model found, sorted into reading order.

        Reading order is the numbering's business rather than the model's, so it
        is imposed here and not asked of the detector.

        Raises:
            ValueError: If no detections are found on the page.
        """
        detections = self._detector.predict(image)

        if not detections:
            raise ValueError("No detections found on page")

        logger.debug(
            "Predictions",
            class_counts=dict(Counter(det.label for det in detections)),
        )

        return sorted(detections, key=lambda det: reading_order_key(det.polygon))

    def _read_option_number(
        self, image: Image.Image, polygon: PixelPolygon
    ) -> int | None:
        """The option a marker names, folded onto 1..10. None when unreadable."""
        cropped = cut_out_image_by_polygon(image, polygon)
        digits = self._text_reader.extract_digits(cropped)
        if digits:
            return normalize_option_number(digits[0])
        return None

    def _read_part_letter(
        self, image: Image.Image, polygon: PixelPolygon
    ) -> str | None:
        """The part a marker names as ``"A"`` or ``"B"``. None when unreadable."""
        cropped = cut_out_image_by_polygon(image, polygon)
        text = self._text_reader.extract_text(cropped).upper()
        # Uppercase and drop the part word before transliterating. Its second
        # letter is a Cyrillic A, which maps to a Latin "A" and would win the
        # Part A test below for every marker, Part B included.
        text = text.replace("ЧАСТЬ", "").strip()
        text_normalized = text.translate(str.maketrans("АБВ", "ABB"))

        if "A" in text_normalized:
            return "A"
        if "B" in text_normalized:
            return "B"
        return None

    def _as_region(self, image: Image.Image, detection: Detection) -> PageRegion | None:
        """*detection* as a region with its marker read, or None for a stray label.

        A label outside the model's class map has no place in the numbering
        either way, so there is nothing to build from it.
        """
        polygon = detection.polygon
        if detection.label == "option":
            return PageRegion(
                label="option",
                polygon=polygon,
                reading=self._read_option_number(image, polygon),
            )
        if detection.label == "part":
            return PageRegion(
                label="part",
                polygon=polygon,
                reading=self._read_part_letter(image, polygon),
            )
        if detection.label == "question":
            return PageRegion(label="question", polygon=polygon)
        return None

    def read_page(self, image: Image.Image) -> list[PageRegion]:
        """Detect the page's regions and read what its markers say.

        Returned in reading order, which is the order the numbering consumes
        them in.

        Raises:
            ValueError: If the page has no detections.
        """
        regions: list[PageRegion] = []
        for detection in self._detect(image):
            region = self._as_region(image, detection)
            if region is None:
                logger.warning(
                    "Ignoring region with unknown label", label=detection.label
                )
                continue
            regions.append(region)
        return regions

    # -- cutting the pixels ------------------------------------------------

    def _crop_piece(self, image: Image.Image, polygon: PixelPolygon) -> Image.Image:
        """Cut *polygon* out of the page and deskew it, at the page's own scale.

        Deskew comes from tesseract: the crop is flattened first so the baseline
        read sees white behind the polygon mask, and rotated before anything
        stacks or resizes it, so the grown canvas is measured once.
        """
        cropped = cut_out_image_by_polygon(image, polygon)
        piece = add_white_background(cropped)

        angle = self._text_reader.detect_skew(piece)
        if angle:
            logger.debug("Correcting skew", angle=angle)
            piece = rotate_image(piece, angle)

        return piece

    def _crop_question(
        self,
        image: Image.Image,
        regions: Sequence[PageRegion],
        carried: Sequence[HeldPiece] = (),
    ) -> Image.Image:
        """The image a question is saved as: its pieces stacked, then capped.

        The size cap belongs to the whole question, not to each of its pieces —
        two pieces capped separately and then stacked would meet at a seam where
        the text changes size. *carried* is the pieces cut from an earlier page,
        which go on top. Each piece is laid against the one above it by the
        offset the reviewer lined them up with.
        """
        pieces: list[tuple[Image.Image, tuple[int, int]]] = [
            *((piece.image, piece.offset) for piece in carried),
            *(
                (self._crop_piece(image, region.polygon), region.join_offset)
                for region in regions
            ),
        ]
        return resize_image(
            stack_vertically(
                [piece for piece, _ in pieces],
                PIECE_GAP,
                [offset for _, offset in pieces],
            ),
            self.config.question_max_width,
            self.config.question_max_height,
        )

    def _write_question(
        self,
        image: Image.Image,
        regions: list[PageRegion],
        placement: QuestionPlacement,
        output_dir: Path,
        carried: Sequence[HeldPiece] = (),
    ) -> bool:
        """Save one placed question's crop, unless its slot is already taken.

        A taken slot means this page's numbering has run into output an earlier
        page already wrote. Overwriting would destroy an extracted question, so
        the existing file wins and False comes back for the caller to report —
        and ``--review`` marks it on the page before anything is written.

        *regions* is the question's pieces on this page and *carried* the pieces
        cut from the page before, which is what a question printed across a page
        break amounts to: one file, stacked from both.
        """
        logger.debug(
            "Extracting question",
            option=placement.option,
            part=placement.part,
            question=placement.number,
            pieces=len(carried) + len(regions),
        )

        if question_slot_taken(
            output_dir, placement.option, placement.part, placement.number
        ):
            logger.error(
                "Question slot already taken, keeping the existing image",
                option=placement.option,
                part=placement.part,
                question=placement.number,
            )
            return False

        output_path = question_image_path(
            output_dir,
            placement.option,
            placement.part,
            placement.number,
            self.config.image_format,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._crop_question(image, regions, carried).save(output_path)
        return True

    # -- the run ------------------------------------------------------------

    def _refuse_a_gap(
        self,
        regions: list[PageRegion],
        state: PageExtractionState,
        output_dir: Path,
    ) -> None:
        """Refuse the page if its numbering would leave a hole in the tree.

        The legality rule itself lives in :func:`digitex.domain.numbering.
        preview`, which the review window consults too — only the policy is
        this method's. A collision is survivable — the write walk keeps the
        existing file and reports it, which is what lets a resumed book replay
        pages over their own output — but a gap would put a hole in the output
        tree that no renumbering pass exists to close.

        Raises:
            ValueError: If any question would land past its folder's end.
        """
        fault = preview(regions, state, output_dir).fault
        if fault is None or fault.collides:
            return
        folder = f"{fault.placement.option}/{fault.placement.part}"
        raise ValueError(
            f"Question numbering leaves a gap: {fault.placement} — the next"
            f" free number in {folder} is {fault.free}"
        )

    def extract(
        self,
        image: Image.Image,
        output_dir: Path,
        state: PageExtractionState,
        page_number: int = 0,
        page_count: int = 0,
        carry: PageCarry | None = None,
    ) -> list[QuestionPlacement]:
        """Extract questions from a single page image, advancing *state*.

        *state* is mutated in place — it belongs to the caller, which threads one
        state across a whole book so question numbering continues across page
        boundaries. If a page raises partway through, the state reflects the
        detections handled up to that point; the caller decides whether that is
        recoverable.

        The reviewer sees the page's regions before any of them is cropped, and
        may correct them, move where the page starts numbering, or skip the page
        entirely — in which case nothing is written and *state* is left exactly
        where it was.

        A question the reviewer marked as continuing into the next piece is not
        written here and takes no number: its crop goes into *carry* for the page
        that finishes it, which saves the pieces as one image.

        Args:
            image: PIL Image of the page.
            output_dir: Base output directory.
            state: Question-numbering state, advanced by this call.
            page_number: This page's 1-based place in its book, for the reviewer
                to report progress with. 0 outside a book.
            page_count: How many pages the book holds. 0 outside a book.
            carry: The pieces the page before this one could not finish, and
                where this page leaves its own. Mutated in place, like *state* —
                the caller threads one carry across a book. A page extracted on
                its own gets a carry of its own, so a piece it holds goes
                nowhere.

        Returns:
            The placements whose slot was already taken. Their crops were not
            written — the existing files were kept — and a caller reporting an
            honest result must say so.

        Raises:
            ValueError: If the page has no detections, a question comes before
                any option/part marker, or the page's numbering would leave a
                gap in its option/part folder.
            ReviewAborted: If the reviewer stopped the run.
        """
        regions = self.read_page(image)
        carry = PageCarry() if carry is None else carry
        # BookExtractor opens pages from disk, so PIL knows the filename —
        # though only ImageFile declares it.
        page_name = Path(str(getattr(image, "filename", ""))).name

        reviewed = self._on_review(
            PageProposal(
                image=image,
                # Copies: whatever the reviewer does to them — edit, drop, keep
                # — the extractor's own regions and state move only when it
                # adopts the verdict below.
                regions=copy_regions(regions),
                state=replace(state),
                output_dir=output_dir,
                page_name=page_name,
                # Bound to this page, so a reviewer previewing a question sees
                # the file that would be written rather than a lookalike.
                crop=partial(self._crop_question, image),
                crop_piece=partial(self._crop_piece, image),
                page_number=page_number,
                page_count=page_count,
                # Copies of the pieces, not the carry itself: a skipped page
                # leaves them for the next one to finish.
                carried=list(carry.pieces),
            )
        )
        if reviewed is None:
            logger.info("Page skipped by reviewer")
            return []

        state.adopt(reviewed.state)

        # Taken now: whatever happens below, these pieces are this page's
        # business and not the next page's.
        pending = carry.take()
        if reviewed.discard_carried and pending:
            logger.warning(
                "Carried question pieces discarded by reviewer",
                pieces=len(pending),
                page=page_name,
            )
            pending = []

        self._refuse_a_gap(reviewed.regions, state, output_dir)

        collisions: list[QuestionPlacement] = []

        def write(regions: list[PageRegion], placement: QuestionPlacement) -> None:
            # The carried pieces belong to the first question written on the
            # page, and to that one only.
            carried = list(pending)
            pending.clear()
            if not self._write_question(image, regions, placement, output_dir, carried):
                collisions.append(placement)

        written = place_questions(reviewed.regions, state, write=write)

        # Anything still pending was carried onto a page with no question to
        # finish it — a question spanning three pages — and travels on ahead of
        # whatever this page leaves behind.
        pending.extend(
            HeldPiece(
                image=self._crop_piece(image, region.polygon),
                page_name=page_name,
                offset=region.join_offset,
            )
            for region in written.held
        )
        if pending:
            logger.info(
                "Question continues on the next page",
                pieces=len(pending),
                page=page_name,
            )
        carry.hold(pending)
        return collisions
