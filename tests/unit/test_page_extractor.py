"""Tests for the page extractor and its question-numbering state machine.

PageExtractor takes every collaborator via its constructor, so these tests
inject fakes for the YOLO predictor and the OCR text extractor and observe
only the interface: which files land where on disk, and what state comes back.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from PIL import Image

from digitex.core.domain import Detection, PixelPolygon
from digitex.extractors.base import ExtractionConfig
from digitex.extractors.page_extractor import (
    PageExtractionState,
    PageExtractor,
    QuestionPlacement,
)

if TYPE_CHECKING:
    from digitex.core import TextExtractor
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
    on_conflict=None,
) -> PageExtractor:
    # The fakes satisfy the collaborators' contracts structurally.
    return PageExtractor(
        ExtractionConfig(
            model_path=Path("model.pt"),
            question_max_width=50,
            question_max_height=50,
        ),
        predictor=cast("YOLO_SegmentationPredictor", _FakePredictor(detections)),
        text_extractor=cast(
            "TextExtractor", _FakeTextExtractor(digits=digits, text=text)
        ),
        on_conflict=on_conflict,
    )


class TestPageExtractionState:
    """The question-numbering state machine through its interface."""

    def test_option_marker_advances_sequentially(self) -> None:
        state = PageExtractionState()
        assert state.on_option(1) is True
        assert (state.option, state.part, state.question) == (1, "A", 0)

    def test_non_sequential_option_marker_ignored(self) -> None:
        state = PageExtractionState(option=1, part="B", question=3)
        assert state.on_option(5) is False
        assert state.on_option(None) is False
        assert (state.option, state.part, state.question) == (1, "B", 3)

    def test_part_marker_switches_and_resets_numbering(self) -> None:
        state = PageExtractionState(option=1, part="A", question=7)
        assert state.on_part("B") is True
        assert (state.part, state.question) == ("B", 0)

    def test_same_or_missing_part_marker_ignored(self) -> None:
        state = PageExtractionState(option=1, part="A", question=7)
        assert state.on_part("A") is False
        assert state.on_part(None) is False
        assert state.question == 7

    def test_placements_number_sequentially_after_commit(self) -> None:
        state = PageExtractionState(option=1, part="A")
        assert state.next_question() == QuestionPlacement(option=1, part="A", number=1)
        state.commit_question()
        assert state.next_question() == QuestionPlacement(option=1, part="A", number=2)

    def test_next_question_without_commit_does_not_consume(self) -> None:
        state = PageExtractionState(option=1, part="A")
        assert state.next_question().number == 1
        assert state.next_question().number == 1

    def test_correct_option_moves_and_keeps_numbering(self) -> None:
        state = PageExtractionState(option=1, part="B", question=3)
        assert state.correct_option(2) is True
        assert (state.option, state.part, state.question) == (2, "A", 3)
        assert state.next_question() == QuestionPlacement(option=2, part="A", number=4)

    def test_correct_option_same_option_is_noop(self) -> None:
        state = PageExtractionState(option=1, part="A", question=3)
        assert state.correct_option(1) is False
        assert (state.option, state.part, state.question) == (1, "A", 3)

    def test_full_page_event_sequence(self) -> None:
        state = PageExtractionState()
        state.on_option(1)
        state.on_part("A")
        placements = [state.next_question()]
        state.commit_question()
        placements.append(state.next_question())
        state.commit_question()
        state.on_part("B")
        placements.append(state.next_question())
        state.commit_question()
        assert placements == [
            QuestionPlacement(option=1, part="A", number=1),
            QuestionPlacement(option=1, part="A", number=2),
            QuestionPlacement(option=1, part="B", number=1),
        ]


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

    def test_conflict_with_default_resolver_keeps_existing_file(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        state = PageExtractionState(option=1, part="A")

        _extractor(detections).extract(image, tmp_path, state)

        assert existing.read_bytes() == b"original"
        assert (state.option, state.question) == (1, 1)

    def test_conflict_resolver_correction_moves_question_and_state(
        self, tmp_path: Path
    ) -> None:
        detections = _dets(("question", QUESTION_REGION))
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        state = PageExtractionState(option=1, part="A")

        _extractor(detections, on_conflict=lambda conflict: 2).extract(
            image, tmp_path, state
        )

        assert not existing.exists()
        assert (tmp_path / "2" / "A" / "1.jpg").exists()
        assert (state.option, state.part, state.question) == (2, "A", 1)

    def test_correction_into_an_occupied_slot_keeps_both_files(
        self, tmp_path: Path
    ) -> None:
        """The resolver's option is already taken, so nothing is overwritten."""
        detections = _dets(("question", QUESTION_REGION))
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        occupied = tmp_path / "2" / "A" / "1.jpg"
        occupied.parent.mkdir(parents=True)
        occupied.write_bytes(b"someone else")
        image = Image.new("RGB", (300, 300), color="white")
        state = PageExtractionState(option=1, part="A")

        _extractor(detections, on_conflict=lambda conflict: 2).extract(
            image, tmp_path, state
        )

        assert existing.read_bytes() == b"original"
        assert occupied.read_bytes() == b"someone else"
        assert (state.option, state.part) == (1, "A")
