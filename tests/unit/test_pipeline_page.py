"""Tests for the page extractor: reading a page, reviewing it, writing it.

PageExtractor takes every collaborator via its constructor, so these tests
inject fakes for the YOLO predictor and the OCR text extractor and observe
only the interface: which files land where on disk, and what state comes back.
The numbering itself is exercised in ``test_placement``.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from digitex.domain.entities import Detection, PixelPolygon
from digitex.pipeline.base import ExtractionConfig
from digitex.pipeline.page import PageExtractor
from digitex.pipeline.pieces import HeldPiece, PageCarry
from digitex.pipeline.placement import PageExtractionState, PageRegion
from digitex.pipeline.review import PageProposal, ReviewedPage

OPTION_REGION = PixelPolygon([(10, 0), (40, 0), (40, 10), (10, 10)])
PART_REGION = PixelPolygon([(10, 20), (40, 20), (40, 30), (10, 30)])
QUESTION_REGION = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])
SECOND_QUESTION_REGION = PixelPolygon([(10, 90), (200, 90), (200, 130), (10, 130)])


def _dets(*pairs: tuple[str, PixelPolygon]) -> list[Detection]:
    """Detections in the order the predictor would report them."""
    return [
        Detection(label=label, polygon=polygon, score=0.9) for label, polygon in pairs
    ]


class _FakePredictor:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def predict(self, image: Image.Image) -> list[Detection]:
        return self._detections


class _FakeTextExtractor:
    def __init__(self, digits: list[int] | None = None, text: str = "") -> None:
        self._digits = digits or []
        self._text = text

    def extract_digits(self, image: Image.Image) -> list[int]:
        return self._digits

    def extract_text(self, image: Image.Image) -> str:
        return self._text

    def detect_skew(self, image: Image.Image) -> float:
        return 0.0


def _extractor(
    detections: list[Detection],
    *,
    digits: list[int] | None = None,
    text: str = "",
    on_review=None,
    max_size: int = 50,
) -> PageExtractor:
    # The fakes satisfy the ports structurally — no cast, no registration.
    return PageExtractor(
        ExtractionConfig(
            image_format="jpg",
            question_max_width=max_size,
            question_max_height=max_size,
        ),
        detector=_FakePredictor(detections),
        text_reader=_FakeTextExtractor(digits=digits, text=text),
        on_review=on_review,
    )


class TestPageExtractorReadPage:
    """What the reviewer is shown: the page's regions, with the markers read."""

    def test_markers_carry_what_ocr_read_off_them(self) -> None:
        detections = _dets(
            ("option", OPTION_REGION),
            ("part", PART_REGION),
            ("question", QUESTION_REGION),
        )
        image = Image.new("RGB", (300, 300), color="white")

        regions = _extractor(detections, digits=[11], text="Часть Б").read_page(image)

        assert [(r.label, r.reading) for r in regions] == [
            ("option", 1),  # 11 folds onto the 1..10 range
            ("part", "B"),
            ("question", None),
        ]

    def test_regions_come_back_in_reading_order(self) -> None:
        detections = _dets(("question", QUESTION_REGION), ("part", PART_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        regions = _extractor(detections, text="Часть B").read_page(image)

        assert [r.label for r in regions] == ["part", "question"]

    def test_regions_the_class_map_does_not_cover_are_dropped(self) -> None:
        detections = _dets(("unknown", OPTION_REGION), ("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        regions = _extractor(detections).read_page(image)

        assert [r.label for r in regions] == ["question"]


class TestPageExtractorExtract:
    """Behavior tests of extract() through its interface — no YOLO, no OCR.

    ``extract`` advances the state it is handed, so each test builds the state
    it wants and reads it back afterwards.
    """

    def test_questions_saved_under_detected_option_and_part(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(
            ("option", OPTION_REGION),
            ("part", PART_REGION),
            ("question", QUESTION_REGION),
            ("question", SECOND_QUESTION_REGION),
        )
        image = Image.new("RGB", (300, 300), color="white")
        state = PageExtractionState()

        _extractor(detections, digits=[1], text="Часть A").extract(
            image, tmp_path, state
        )

        assert (tmp_path / "1" / "A" / "1.jpg").exists()
        assert (tmp_path / "1" / "A" / "2.jpg").exists()
        assert (state.option, state.part, state.question) == (1, "A", 2)

    def test_option_digits_normalized_to_one_to_ten_range(self, tmp_path: Path) -> None:
        """Book pages number options 11-20 / 31-40; OCR reads map back to 1-10."""
        detections = _dets(
            ("option", OPTION_REGION),
            ("part", PART_REGION),
            ("question", QUESTION_REGION),
        )
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(detections, digits=[11], text="Часть A").extract(
            image, tmp_path, PageExtractionState()
        )

        assert (tmp_path / "1" / "A" / "1.jpg").exists()

    def test_cyrillic_part_marker_maps_to_latin_b(self, tmp_path: Path) -> None:
        detections = _dets(("part", PART_REGION), ("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(detections, text="Часть Б").extract(
            image, tmp_path, PageExtractionState(option=1, part="A", question=5)
        )

        assert (tmp_path / "1" / "B" / "1.jpg").exists()

    @pytest.mark.parametrize(
        "text",
        ["Часть Б", "ЧАСТЬ Б", "часть б", "ЧАСТЬ B"],
        ids=["title-case", "upper-case", "lower-case", "latin-b"],
    )
    def test_part_b_marker_is_read_whatever_its_casing(
        self, tmp_path: Path, text: str
    ) -> None:
        """The part word's second letter is a Cyrillic A.

        It transliterates to a Latin "A", so stripping the word has to happen
        after the uppercase — otherwise every Part B marker reads as Part A.
        """
        detections = _dets(("part", PART_REGION), ("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(detections, text=text).extract(
            image, tmp_path, PageExtractionState(option=1, part="A", question=5)
        )

        assert (tmp_path / "1" / "B" / "1.jpg").exists()

    def test_unreadable_markers_leave_state_untouched(self, tmp_path: Path) -> None:
        detections = _dets(
            ("option", OPTION_REGION),
            ("part", PART_REGION),
            ("question", QUESTION_REGION),
        )
        image = Image.new("RGB", (300, 300), color="white")

        # The state says 2/B/1 was already written by an earlier page.
        earlier = tmp_path / "2" / "B" / "1.jpg"
        earlier.parent.mkdir(parents=True)
        earlier.write_bytes(b"earlier page")
        state = PageExtractionState(option=2, part="B", question=1)

        _extractor(detections, digits=[], text="smudge").extract(image, tmp_path, state)

        assert (tmp_path / "2" / "B" / "2.jpg").exists()
        assert (state.option, state.part) == (2, "B")

    def test_no_detections_raises(self, tmp_path: Path) -> None:
        image = Image.new("RGB", (300, 300), color="white")

        with pytest.raises(ValueError, match="No detections found on page"):
            _extractor([]).extract(image, tmp_path, PageExtractionState())

    def test_question_before_any_marker_raises(self, tmp_path: Path) -> None:
        """A crop placed from the pristine state would lose its Part directory.

        ``Path(out) / "0" / "" / "1.jpg"`` collapses to ``out/0/1.jpg``, which
        every reader of the output tree skips — so the page must fail loudly.
        """
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        with pytest.raises(ValueError, match="before any option/part marker"):
            _extractor(detections).extract(image, tmp_path, PageExtractionState())

        assert list(tmp_path.rglob("*.jpg")) == []

    def test_detections_processed_in_reading_order(self, tmp_path: Path) -> None:
        """A part marker above a question applies to it, whatever the predict order."""
        # Question reported first, but it sits BELOW the marker on the page.
        detections = _dets(("question", QUESTION_REGION), ("part", PART_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(detections, text="Часть B").extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert (tmp_path / "1" / "B" / "1.jpg").exists()

    def test_a_taken_slot_keeps_the_image_that_is_already_there(
        self, tmp_path: Path
    ) -> None:
        """Overwriting would destroy an extracted question, so the page yields."""
        detections = _dets(("question", QUESTION_REGION))
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        state = PageExtractionState(option=1, part="A")

        _extractor(detections).extract(image, tmp_path, state)

        assert existing.read_bytes() == b"original"
        # The number is still consumed: the question exists, it just was not
        # this run that wrote it.
        assert (state.option, state.question) == (1, 1)

    def test_a_taken_slot_is_reported_to_the_caller(self, tmp_path: Path) -> None:
        """The crop that was not written must not disappear from the result."""
        detections = _dets(("question", QUESTION_REGION))
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        collisions = _extractor(detections).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert [str(placement) for placement in collisions] == ["1/A/1"]

    def test_a_clean_page_reports_no_collisions(self, tmp_path: Path) -> None:
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        collisions = _extractor(detections).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert collisions == []

    def test_a_page_that_would_leave_a_gap_is_refused(self, tmp_path: Path) -> None:
        """The rule the review window applies, applied without a reviewer too.

        A crop landing past its folder's next free number would leave a hole
        nothing can renumber away — the reviewed path refuses to approve it,
        so the unreviewed path must refuse to write it.
        """
        detections = _dets(("question", QUESTION_REGION))
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        # The folder's free number is 2; this state would write 1/A/5.
        state = PageExtractionState(option=1, part="A", question=4)

        with pytest.raises(ValueError, match="leaves a gap"):
            _extractor(detections).extract(image, tmp_path, state)

        assert list(tmp_path.rglob("*.jpg")) == [existing]

    def test_a_gap_is_refused_before_anything_is_written(self, tmp_path: Path) -> None:
        """The check replays the whole page first — no partial page on disk."""
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        # An empty folder's free number is 1; this state would write 2/B/4.
        with pytest.raises(ValueError, match="leaves a gap"):
            _extractor(detections).extract(
                image, tmp_path, PageExtractionState(option=2, part="B", question=3)
            )

        assert list(tmp_path.rglob("*.jpg")) == []

    def test_a_slot_taken_in_another_format_is_still_taken(
        self, tmp_path: Path
    ) -> None:
        """An earlier run's png must not be shadowed by a second jpg copy."""
        detections = _dets(("question", QUESTION_REGION))
        existing = tmp_path / "1" / "A" / "1.png"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(detections).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert not (tmp_path / "1" / "A" / "1.jpg").exists()
        assert existing.read_bytes() == b"original"


class TestQuestionsInPieces:
    """A question printed across a page break: held by one page, saved by the next.

    The reviewer is the only one who can say a question is in pieces, so these
    stand a fake one in that marks the flag the review window would.
    """

    @staticmethod
    def _joins_next(proposal: PageProposal) -> ReviewedPage:
        """A reviewer marking the page's last question as continuing."""
        for region in reversed(proposal.regions):
            if region.label == "question":
                region.joins_next = True
                break
        return ReviewedPage(regions=proposal.regions, state=proposal.state)

    @staticmethod
    def _carried(page_name: str = "001.jpg") -> PageCarry:
        """A carry holding one black piece, so the join is visible in the file."""
        return PageCarry(
            pieces=[
                HeldPiece(
                    image=Image.new("RGB", (60, 30), color="black"),
                    page_name=page_name,
                )
            ]
        )

    def test_a_held_piece_is_not_written_and_takes_no_number(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        state = PageExtractionState(option=1, part="A")
        carry = PageCarry()

        _extractor(detections, on_review=self._joins_next).extract(
            image, tmp_path, state, carry=carry
        )

        assert list(tmp_path.rglob("*.jpg")) == []
        # The page that finishes the question is the page that numbers it.
        assert state.question == 0
        assert len(carry.pieces) == 1

    def test_the_page_that_finishes_a_question_saves_both_pieces_as_one(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        carry = self._carried()

        _extractor(detections, max_size=400).extract(
            image, tmp_path, PageExtractionState(option=1, part="A"), carry=carry
        )

        with Image.open(tmp_path / "1" / "A" / "1.jpg") as saved:
            # The carried piece sits on top of this page's crop: the file opens
            # in the carried piece's black and ends in the page's white.
            pixels = np.array(saved)
        assert pixels[:5, :5].max() < 60
        assert pixels[-5:, :5].min() > 200
        assert carry.pieces == []

    def test_a_skipped_page_leaves_the_carried_pieces_for_the_next_one(
        self, tmp_path: Path
    ) -> None:
        """Skipping writes nothing, so it must not consume anything either."""
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        carry = self._carried()

        _extractor(detections, on_review=lambda proposal: None).extract(
            image, tmp_path, PageExtractionState(option=1, part="A"), carry=carry
        )

        assert len(carry.pieces) == 1

    def test_a_page_with_no_question_hands_the_pieces_on(self, tmp_path: Path) -> None:
        """A question can span three pages; the middle one has nothing to place."""
        detections = _dets(("option", OPTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        carry = self._carried()

        _extractor(detections, digits=[1]).extract(
            image, tmp_path, PageExtractionState(), carry=carry
        )

        assert [piece.page_name for piece in carry.pieces] == ["001.jpg"]

    def test_the_reviewer_can_discard_what_was_carried(self, tmp_path: Path) -> None:
        def discard(proposal: PageProposal) -> ReviewedPage:
            assert [piece.page_name for piece in proposal.carried] == ["001.jpg"]
            return ReviewedPage(
                regions=proposal.regions,
                state=proposal.state,
                discard_carried=True,
            )

        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        carry = self._carried()

        _extractor(detections, on_review=discard, max_size=400).extract(
            image, tmp_path, PageExtractionState(option=1, part="A"), carry=carry
        )

        with Image.open(tmp_path / "1" / "A" / "1.jpg") as saved:
            pixels = np.array(saved)
        # No black at the top: the carried piece is not in the file.
        assert pixels[:5, :5].min() > 200
        assert carry.pieces == []

    def _joined_size(
        self, output_dir: Path, offset: tuple[int, int]
    ) -> tuple[int, int]:
        """The size of the file a page joining its piece to a carried one writes."""

        def review(proposal: PageProposal) -> ReviewedPage:
            for region in proposal.regions:
                if region.label == "question":
                    region.join_offset = offset
            return ReviewedPage(regions=proposal.regions, state=proposal.state)

        _extractor(
            _dets(("question", QUESTION_REGION)), on_review=review, max_size=400
        ).extract(
            Image.new("RGB", (300, 300), color="white"),
            output_dir,
            PageExtractionState(option=1, part="A"),
            carry=self._carried(),
        )
        with Image.open(output_dir / "1" / "A" / "1.jpg") as saved:
            return saved.size

    def test_lining_a_piece_up_moves_it_in_the_saved_file(self, tmp_path: Path) -> None:
        """What the join editor settles is what the file is built with."""
        plain = self._joined_size(tmp_path / "plain", offset=(0, 0))
        nudged = self._joined_size(tmp_path / "nudged", offset=(10, -8))

        # Nudging the lower piece right widens the stack and closing the seam
        # shortens it, so the file comes out a wider shape than the plain join.
        assert nudged[0] / nudged[1] > plain[0] / plain[1]


class TestPageExtractorReview:
    """The reviewer's verdict is what gets written — not what was detected."""

    def test_regions_the_reviewer_edited_are_the_ones_written(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        def relabel(proposal: PageProposal) -> ReviewedPage:
            """The crop was really a Part B marker, plus a question drawn by hand."""
            return ReviewedPage(
                regions=[
                    PageRegion(label="part", polygon=PART_REGION, reading="B"),
                    PageRegion(label="question", polygon=SECOND_QUESTION_REGION),
                ],
                state=proposal.state,
            )

        state = PageExtractionState(option=1, part="A", question=9)
        _extractor(detections, on_review=relabel).extract(image, tmp_path, state)

        assert (tmp_path / "1" / "B" / "1.jpg").exists()
        assert not (tmp_path / "1" / "A" / "10.jpg").exists()

    def test_a_corrected_entry_state_moves_where_the_page_starts(
        self, tmp_path: Path
    ) -> None:
        """OCR read no option marker at all; the reviewer says where the page is."""
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")

        # The reviewer's entry point continues what is already on disk.
        earlier = tmp_path / "4" / "B" / "6.jpg"
        earlier.parent.mkdir(parents=True)
        earlier.write_bytes(b"earlier page")

        def restart(proposal: PageProposal) -> ReviewedPage:
            return ReviewedPage(
                regions=proposal.regions,
                state=PageExtractionState(option=4, part="B", question=6),
            )

        state = PageExtractionState()
        _extractor(detections, on_review=restart).extract(image, tmp_path, state)

        assert (tmp_path / "4" / "B" / "7.jpg").exists()
        # The book's own state moved with it, so the next page carries on here.
        assert (state.option, state.part, state.question) == (4, "B", 7)

    def test_a_skipped_page_writes_nothing_and_leaves_the_state_alone(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        state = PageExtractionState(option=1, part="A", question=3)

        _extractor(detections, on_review=lambda proposal: None).extract(
            image, tmp_path, state
        )

        assert list(tmp_path.rglob("*.jpg")) == []
        assert (state.option, state.part, state.question) == (1, "A", 3)

    def test_a_reviewer_may_maul_the_proposal_and_still_skip_cleanly(
        self, tmp_path: Path
    ) -> None:
        """The proposal carries copies, so no discipline is asked of a reviewer."""
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        state = PageExtractionState(option=1, part="A", question=3)

        def maul(proposal: PageProposal) -> None:
            proposal.regions[0].label = "part"
            proposal.regions.clear()
            proposal.state.adopt(PageExtractionState(option=9, part="B", question=99))

        _extractor(detections, on_review=maul).extract(image, tmp_path, state)

        assert list(tmp_path.rglob("*.jpg")) == []
        assert (state.option, state.part, state.question) == (1, "A", 3)

    def test_the_reviewer_sees_the_page_it_is_reviewing(self, tmp_path: Path) -> None:
        detections = _dets(("part", PART_REGION), ("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        seen: list[PageProposal] = []

        def record(proposal: PageProposal) -> None:
            seen.append(proposal)

        _extractor(detections, text="Часть Б", on_review=record).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert [(r.label, r.reading) for r in seen[0].regions] == [
            ("part", "B"),
            ("question", None),
        ]
        assert seen[0].image is image

    def test_the_reviewer_is_told_where_the_page_sits_in_its_book(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        seen: list[PageProposal] = []

        _extractor(detections, on_review=seen.append).extract(
            image,
            tmp_path,
            PageExtractionState(option=1, part="A"),
            page_number=7,
            page_count=42,
        )

        assert (seen[0].page_number, seen[0].page_count) == (7, 42)

    def test_a_page_extracted_outside_a_book_has_no_position(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        seen: list[PageProposal] = []

        _extractor(detections, on_review=seen.append).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert (seen[0].page_number, seen[0].page_count) == (0, 0)

    def test_the_reviewer_can_crop_exactly_what_would_be_saved(
        self, tmp_path: Path
    ) -> None:
        """A preview built any other way could disagree with the file written."""
        detections = _dets(("question", QUESTION_REGION))
        image = Image.new("RGB", (300, 300), color="white")
        seen: list[PageProposal] = []

        extractor = _extractor(detections, on_review=seen.append)
        extractor.extract(image, tmp_path, PageExtractionState(option=1, part="A"))

        crop = seen[0].crop
        assert crop is not None
        # The same pipeline the write goes through, on the same page.
        region = PageRegion(label="question", polygon=QUESTION_REGION)
        assert crop([region], []).size == extractor._crop_question(image, [region]).size
