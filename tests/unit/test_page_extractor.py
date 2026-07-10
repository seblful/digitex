"""Tests for the page extractor and its question-numbering state machine.

PageExtractor takes every collaborator via its constructor, so these tests
inject fakes for the YOLO predictor and the OCR text extractor and observe
only the interface: which files land where on disk, and what state comes back.
"""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import pytest
from PIL import Image

from digitex.extractors.page_extractor import (
    PageExtractionState,
    PageExtractor,
    QuestionPlacement,
)
from digitex.ml.predictors import SegmentationPredictionResult

if TYPE_CHECKING:
    from digitex.core import TextExtractor
    from digitex.ml.predictors import YOLO_SegmentationPredictor

ID2LABEL = {0: "question", 1: "option", 2: "part"}

OPTION_REGION = [(10, 0), (40, 0), (40, 10), (10, 10)]
PART_REGION = [(10, 20), (40, 20), (40, 30), (10, 30)]
QUESTION_REGION = [(10, 40), (200, 40), (200, 80), (10, 80)]
SECOND_QUESTION_REGION = [(10, 90), (200, 90), (200, 130), (10, 130)]


class _FakePredictor:
    def __init__(self, result: SegmentationPredictionResult) -> None:
        self._result = result

    def predict(self, image: Image.Image) -> SegmentationPredictionResult:
        return self._result


class _FakeTextExtractor:
    def __init__(self, digits: list[int] | None = None, text: str = "") -> None:
        self._digits = digits or []
        self._text = text

    def extract_digits(self, image: Image.Image) -> list[int]:
        return self._digits

    def extract_text(self, image: Image.Image) -> str:
        return self._text


def _extractor(
    result: SegmentationPredictionResult,
    *,
    digits: list[int] | None = None,
    text: str = "",
    on_conflict=None,
) -> PageExtractor:
    # The fakes satisfy the collaborators' contracts structurally.
    return PageExtractor(
        model_path=Path("model.pt"),
        image_format="jpg",
        question_max_width=50,
        question_max_height=50,
        predictor=cast("YOLO_SegmentationPredictor", _FakePredictor(result)),
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
    """Behavior tests of extract() through its interface — no YOLO, no OCR."""

    ID2LABEL: ClassVar[dict[int, str]] = ID2LABEL

    def test_questions_saved_under_detected_option_and_part(
        self, tmp_path: Path
    ) -> None:
        result = SegmentationPredictionResult(
            ids=[1, 2, 0, 0],
            polygons=[
                OPTION_REGION,
                PART_REGION,
                QUESTION_REGION,
                SECOND_QUESTION_REGION,
            ],
            id2label=self.ID2LABEL,
        )
        image = Image.new("RGB", (300, 300), color="white")

        state = _extractor(result, digits=[1], text="Часть A").extract(image, tmp_path)

        assert (tmp_path / "1" / "A" / "1.jpg").exists()
        assert (tmp_path / "1" / "A" / "2.jpg").exists()
        assert (state.option, state.part, state.question) == (1, "A", 2)

    def test_option_digits_normalized_to_one_to_ten_range(self, tmp_path: Path) -> None:
        """Book pages number options 11-20 / 31-40; OCR reads map back to 1-10."""
        result = SegmentationPredictionResult(
            ids=[1, 2, 0],
            polygons=[OPTION_REGION, PART_REGION, QUESTION_REGION],
            id2label=self.ID2LABEL,
        )
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(result, digits=[11], text="Часть A").extract(image, tmp_path)

        assert (tmp_path / "1" / "A" / "1.jpg").exists()

    def test_cyrillic_part_marker_maps_to_latin_b(self, tmp_path: Path) -> None:
        result = SegmentationPredictionResult(
            ids=[2, 0],
            polygons=[PART_REGION, QUESTION_REGION],
            id2label=self.ID2LABEL,
        )
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(result, text="Часть Б").extract(
            image, tmp_path, PageExtractionState(option=1, part="A", question=5)
        )

        assert (tmp_path / "1" / "B" / "1.jpg").exists()

    def test_unreadable_markers_leave_state_untouched(self, tmp_path: Path) -> None:
        result = SegmentationPredictionResult(
            ids=[1, 2, 0],
            polygons=[OPTION_REGION, PART_REGION, QUESTION_REGION],
            id2label=self.ID2LABEL,
        )
        image = Image.new("RGB", (300, 300), color="white")

        state = _extractor(result, digits=[], text="smudge").extract(
            image, tmp_path, PageExtractionState(option=2, part="B", question=1)
        )

        assert (tmp_path / "2" / "B" / "2.jpg").exists()
        assert (state.option, state.part) == (2, "B")

    def test_no_detections_raises(self, tmp_path: Path) -> None:
        result = SegmentationPredictionResult(
            ids=[], polygons=[], id2label=self.ID2LABEL
        )
        image = Image.new("RGB", (300, 300), color="white")

        with pytest.raises(ValueError, match="No detections found on page"):
            _extractor(result).extract(image, tmp_path)

    def test_detections_processed_in_reading_order(self, tmp_path: Path) -> None:
        """A part marker above a question applies to it, whatever the predict order."""
        result = SegmentationPredictionResult(
            ids=[0, 2],  # question predicted first, but it sits BELOW the marker
            polygons=[QUESTION_REGION, PART_REGION],
            id2label=self.ID2LABEL,
        )
        image = Image.new("RGB", (300, 300), color="white")

        _extractor(result, text="Часть B").extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert (tmp_path / "1" / "B" / "1.jpg").exists()

    def test_conflict_with_default_resolver_keeps_existing_file(
        self, tmp_path: Path
    ) -> None:
        result = SegmentationPredictionResult(
            ids=[0],
            polygons=[QUESTION_REGION],
            id2label=self.ID2LABEL,
        )
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        state = _extractor(result).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert existing.read_bytes() == b"original"
        assert (state.option, state.question) == (1, 1)

    def test_conflict_resolver_correction_moves_question_and_state(
        self, tmp_path: Path
    ) -> None:
        result = SegmentationPredictionResult(
            ids=[0],
            polygons=[QUESTION_REGION],
            id2label=self.ID2LABEL,
        )
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        state = _extractor(result, on_conflict=lambda conflict: 2).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert not existing.exists()
        assert (tmp_path / "2" / "A" / "1.jpg").exists()
        assert (state.option, state.part, state.question) == (2, "A", 1)
