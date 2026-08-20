"""The two ports page extraction names its collaborators by.

Two things are worth asserting about an interface introduced to break a
dependency: that the classes on both sides still answer to it, and that the
dependency actually broke. The second is the one that silently comes back — a
convenience import at the top of a module is all it takes.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from digitex.pipeline.ports import RegionDetector, TextReader
from digitex.pipeline.recording import (
    Recording,
    RecordingPredictor,
    RecordingTextExtractor,
    ReplayPredictor,
    ReplayTextExtractor,
)

# Importing page extraction must not drag in the ML stack. Anything here that
# ends up in `sys.modules` is several hundred megabytes of wheel loaded to do
# arithmetic on pixels — and, on a machine that installed only the pipeline
# extra, an ImportError.
FORBIDDEN_BY_PAGE_EXTRACTION = ("torch", "ultralytics")


def _modules_after_importing(module: str) -> set[str]:
    """Top-level modules present after importing *module* in a fresh process.

    A subprocess because the test session has already imported most of the
    project — asking this question in-process would answer about the session.
    """
    code = (
        f"import importlib, sys; importlib.import_module({module!r});"
        " print('\\n'.join(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


class TestTheConcreteClassesAnswerToThePorts:
    """Whichever way a rename goes, one side of it fails here."""

    def test_the_yolo_predictor_is_a_region_detector(self) -> None:
        predictors = pytest.importorskip("digitex.ml.predictors")

        assert issubclass(predictors.YOLO_SegmentationPredictor, RegionDetector)

    def test_the_tesseract_extractor_is_a_text_reader(self) -> None:
        from digitex.imaging.ocr import TextExtractor

        assert issubclass(TextExtractor, TextReader)

    @pytest.mark.parametrize("stand_in", [RecordingPredictor, ReplayPredictor])
    def test_the_recording_detectors_are_region_detectors(self, stand_in: type) -> None:
        assert issubclass(stand_in, RegionDetector)

    @pytest.mark.parametrize("stand_in", [RecordingTextExtractor, ReplayTextExtractor])
    def test_the_recording_readers_are_text_readers(self, stand_in: type) -> None:
        assert issubclass(stand_in, TextReader)

    def test_a_replay_detector_is_accepted_where_a_detector_is_asked_for(
        self,
    ) -> None:
        """The substitution the differential suite depends on, at runtime."""
        detector: RegionDetector = ReplayPredictor(Recording())
        reader: TextReader = ReplayTextExtractor(Recording())

        assert isinstance(detector, RegionDetector)
        assert isinstance(reader, TextReader)


class TestPageExtractionDoesNotReachForTheModel:
    def test_importing_it_loads_neither_torch_nor_ultralytics(self) -> None:
        """The point of the ports: extraction is arithmetic, not inference.

        Before they existed, `pipeline.page` named `YOLO_SegmentationPredictor`
        at module scope to build one lazily, so replaying a recorded run — which
        loads no model at all — still imported the whole CUDA stack.
        """
        loaded = _modules_after_importing("digitex.pipeline.page")

        assert loaded.isdisjoint(FORBIDDEN_BY_PAGE_EXTRACTION)

    def test_the_check_would_notice_the_ml_package(self) -> None:
        """A guard that cannot fire is not a guard.

        `digitex.ml.predictors` does import torch, so the same question asked
        about it has to come back the other way.
        """
        loaded = _modules_after_importing("digitex.ml.predictors")

        assert not loaded.isdisjoint(FORBIDDEN_BY_PAGE_EXTRACTION)
