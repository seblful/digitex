"""Tests for the page extractor: reading a page, reviewing it, writing it.

PageExtractor takes every collaborator via its constructor, so these tests
inject fakes for the YOLO predictor and the OCR text extractor and observe
only the interface: which files land where on disk, and what state comes back.
The numbering itself is exercised in ``test_placement``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from PIL import Image

from digitex.domain.entities import Detection, PixelPolygon
from digitex.pipeline.base import ExtractionConfig
from digitex.pipeline.page import PageExtractor
from digitex.pipeline.placement import PageExtractionState, PageRegion
from digitex.pipeline.review import PageProposal, ReviewedPage

if TYPE_CHECKING:
    from digitex.imaging.ocr import TextExtractor
    from digitex.ml.predictors import YOLO_SegmentationPredictor

OPTION_REGION = PixelPolygon([(10, 0), (40, 0), (40, 10), (10, 10)])
PART_REGION = PixelPolygon([(10, 20), (40, 20), (40, 30), (10, 30)])
QUESTION_REGION = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])
SECOND_QUESTION_REGION = PixelPolygon([(10, 90), (200, 90), (200, 130), (10, 130)])


def _dets(*pairs: tuple[str, PixelPolygon]) -> list[Detection]:
    """Detections in the order the predictor would report them."""
    return [Detection(label=label, polygon=polygon) for label, polygon in pairs]


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


def _extractor(
    detections: list[Detection],
    *,
    digits: list[int] | None = None,
    text: str = "",
    on_review=None,
) -> PageExtractor:
    # The fakes satisfy the collaborators' contracts structurally.
    return PageExtractor(
        ExtractionConfig(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=50,
            question_max_height=50,
        ),
        predictor=cast("YOLO_SegmentationPredictor", _FakePredictor(detections)),
        text_extractor=cast(
            "TextExtractor", _FakeTextExtractor(digits=digits, text=text)
        ),
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
        assert (
            crop(QUESTION_REGION).size == extractor._crop(image, QUESTION_REGION).size
        )
