"""Tests for the replay fixture format the differential suite runs on.

The differential suite itself needs a recorded book and skips without one, so
the harness would otherwise be the one untested thing the whole restructuring
depends on. Here it is exercised end to end over a synthetic book instead: a
page of drawn regions, a fake predictor and a fake OCR, recorded and then
replayed, with the written files compared byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from digitex.domain.entities import Detection, PixelPolygon
from digitex.pipeline.base import ExtractionConfig
from digitex.pipeline.page import PageExtractor
from digitex.pipeline.placement import PageExtractionState
from digitex.pipeline.recording import (
    MissingAnswer,
    Recording,
    RecordingPredictor,
    RecordingTextExtractor,
    ReplayPredictor,
    ReplayTextExtractor,
    directory_digests,
    image_digest,
    recorded_output_dir,
    recording_path,
    replay_config,
)

OPTION_REGION = PixelPolygon([(10, 0), (40, 0), (40, 10), (10, 10)])
PART_REGION = PixelPolygon([(10, 20), (40, 20), (40, 30), (10, 30)])
QUESTION_REGION = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])
SECOND_QUESTION_REGION = PixelPolygon([(10, 90), (200, 90), (200, 130), (10, 130)])

DETECTIONS = [
    Detection(label="option", polygon=OPTION_REGION, score=0.91),
    Detection(label="part", polygon=PART_REGION, score=0.88),
    Detection(label="question", polygon=QUESTION_REGION, score=0.95),
    Detection(label="question", polygon=SECOND_QUESTION_REGION, score=0.93),
]


def _page(seed: int = 0) -> Image.Image:
    """A page with something on it, so crops are not uniform white.

    Uniform pages would digest identically whatever was cut out of them, and
    every assertion below would hold for the wrong reason.
    """
    image = Image.new("RGB", (300, 300), color="white")
    for x in range(0, 300, 7):
        for y in range((x + seed) % 5, 300, 11):
            image.putpixel((x, y), (seed * 37 % 256, x % 256, y % 256))
    return image


class _FakePredictor:
    def predict(self, image: Image.Image) -> list[Detection]:
        return list(DETECTIONS)


class _FakeTextExtractor:
    """Answers that vary with the crop, the way real OCR does."""

    language = "rus"

    def extract_digits(self, image: Image.Image) -> list[int]:
        return [1]

    def extract_text(self, image: Image.Image) -> str:
        return "Часть A"

    def detect_skew(self, image: Image.Image) -> float:
        # Varies per crop, so a replay that mixed two crops up would show it.
        return round(image.size[0] % 3 * 0.5, 3)


def _config() -> ExtractionConfig:
    return ExtractionConfig(
        image_format="jpg",
        question_max_width=120,
        question_max_height=160,
    )


class TestImageDigest:
    def test_the_same_pixels_digest_the_same(self) -> None:
        assert image_digest(_page(3)) == image_digest(_page(3))

    def test_different_pixels_digest_differently(self) -> None:
        assert image_digest(_page(1)) != image_digest(_page(2))

    def test_size_is_part_of_the_digest(self) -> None:
        wide = Image.new("RGB", (4, 1), color="white")
        tall = Image.new("RGB", (1, 4), color="white")

        assert image_digest(wide) != image_digest(tall)


class TestRecordingRoundTrip:
    def test_a_written_recording_reads_back_equal(self, tmp_path: Path) -> None:
        recording = Recording(
            source="biology/2024",
            image_format="png",
            question_max_width=11,
            question_max_height=22,
            pages=["001.jpg"],
            detections={"abc": list(DETECTIONS)},
            text={"def": "Часть Б"},
            digits={"ghi": [4, 5]},
            skew={"jkl": -0.25},
            outputs={"1/A/1.jpg": "0" * 64},
        )
        path = tmp_path / "book.json"

        recording.write(path)

        assert Recording.load(path) == recording

    def test_a_recording_from_another_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "book.json"
        path.write_text(json.dumps({"version": 999}), encoding="utf-8")

        with pytest.raises(ValueError, match="version 999"):
            Recording.load(path)

    def test_polygons_survive_as_integer_points(self, tmp_path: Path) -> None:
        """JSON has no tuples, and a float point would shift every crop."""
        path = tmp_path / "book.json"
        Recording(detections={"abc": list(DETECTIONS)}).write(path)

        loaded = Recording.load(path)

        assert loaded.detections["abc"][0].polygon == OPTION_REGION
        assert all(
            isinstance(x, int) and isinstance(y, int)
            for x, y in loaded.detections["abc"][0].polygon
        )


class TestReplayRefusesWhatItDoesNotHold:
    def test_an_unrecorded_page_is_not_silently_empty(self) -> None:
        """Empty detections are a legal answer, so guessing one would mislead."""
        with pytest.raises(MissingAnswer, match="detections"):
            ReplayPredictor(Recording()).predict(_page())

    @pytest.mark.parametrize(
        ("call", "kind"),
        [
            (lambda ocr: ocr.extract_text(_page()), "text"),
            (lambda ocr: ocr.extract_digits(_page()), "digits"),
            (lambda ocr: ocr.detect_skew(_page()), "skew"),
        ],
    )
    def test_an_unrecorded_crop_is_refused(self, call, kind: str) -> None:
        with pytest.raises(MissingAnswer, match=kind):
            call(ReplayTextExtractor(Recording()))


class TestRecordThenReplay:
    """The property the differential suite rests on, on a synthetic book."""

    def _record(self, tmp_path: Path) -> tuple[Recording, Path]:
        recording = Recording(
            source="synthetic/2024",
            image_format="jpg",
            question_max_width=120,
            question_max_height=160,
        )
        output_dir = tmp_path / "recorded"
        output_dir.mkdir()
        extractor = PageExtractor(
            _config(),
            detector=RecordingPredictor(_FakePredictor(), recording),
            text_reader=RecordingTextExtractor(_FakeTextExtractor(), recording),
        )
        state = PageExtractionState()
        for page_number in range(1, 3):
            extractor.extract(_page(page_number), output_dir, state)
        recording.outputs = directory_digests(output_dir)
        return recording, output_dir

    def _replay(self, recording: Recording, tmp_path: Path) -> Path:
        output_dir = tmp_path / "replayed"
        output_dir.mkdir()
        extractor = PageExtractor(
            replay_config(recording),
            detector=ReplayPredictor(recording),
            text_reader=ReplayTextExtractor(recording),
        )
        state = PageExtractionState()
        for page_number in range(1, 3):
            extractor.extract(_page(page_number), output_dir, state)
        return output_dir

    def test_a_replay_writes_the_same_files(self, tmp_path: Path) -> None:
        recording, _ = self._record(tmp_path)

        replayed = self._replay(recording, tmp_path)

        assert directory_digests(replayed) == recording.outputs

    def test_the_recording_actually_captured_something(self, tmp_path: Path) -> None:
        recording, _ = self._record(tmp_path)

        assert len(recording.outputs) == 4  # two questions on each of two pages
        assert len(recording.detections) == 2  # one answer per page
        assert recording.skew, "piece deskew was never recorded"

    def test_a_replay_survives_a_write_and_reload(self, tmp_path: Path) -> None:
        """The fixture is used from disk, so serialization is on the path."""
        recording, _ = self._record(tmp_path)
        path = tmp_path / "book.json"
        recording.write(path)

        replayed = self._replay(Recording.load(path), tmp_path)

        assert directory_digests(replayed) == recording.outputs

    def test_a_changed_size_cap_changes_the_files(self, tmp_path: Path) -> None:
        """The comparison can fail — otherwise it proves nothing."""
        recording, _ = self._record(tmp_path)
        recording.question_max_width = 40
        recording.question_max_height = 60

        replayed = self._replay(recording, tmp_path)

        assert directory_digests(replayed) != recording.outputs


class TestGoldenPaths:
    def test_a_recording_never_lands_in_the_extraction_tree(self) -> None:
        """A recording run starts by emptying its output folder."""
        data_root = Path("/data")

        output = recorded_output_dir(data_root, "biology", "2024")

        assert data_root / "extraction" not in output.parents

    def test_each_book_gets_its_own_recording(self) -> None:
        data_root = Path("/data")

        assert recording_path(data_root, "biology", "2024") != recording_path(
            data_root, "biology", "2025"
        )
