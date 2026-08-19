"""Tests for the ML predictors.

YOLO's ``Results`` objects are stood in by small fakes with the same shape
(``pred.boxes[i].cls.item()`` / ``pred.masks.xyn``); the ``YOLO`` constructor
itself is patched only in the lazy-loading tests. ``detections_from`` takes the
class map as an argument, so exercising it needs no model at all.
"""

from __future__ import annotations

import os
import pathlib
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from structlog.testing import capture_logs

from digitex.ml.predictors import (
    YOLO_SegmentationPredictor,
    detections_from,
    foreign_paths_readable,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from ultralytics import YOLO
    from ultralytics.engine.results import Results


_FOREIGN = "PosixPath" if os.name == "nt" else "WindowsPath"

# Where a checkpoint can name the class, depending on the Python that wrote it.
_SPELLINGS = ("pathlib", "pathlib._local")


def _foreign_path_payload(module: str) -> bytes:
    """A pickle that rebuilds a path by calling ``module.PosixPath("runs")``.

    Written out as protocol-0 opcodes rather than pickled from a real object,
    because a real one cannot be built on this platform — which is the very
    problem under test. Nothing here is read from disk or from a caller.
    """
    return f"c{module}\n{_FOREIGN}\n(S'runs'\ntR.".encode()


class _Scalar:
    """Mimics a 0-d tensor: exposes ``.item()``."""

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeBox:
    def __init__(self, class_id: int, score: float = 0.9) -> None:
        self.cls = _Scalar(class_id)
        self.conf = _Scalar(score)


@dataclass
class _FakeMasks:
    xyn: list[np.ndarray]


@dataclass
class _FakePrediction:
    boxes: list[_FakeBox] | None
    masks: _FakeMasks | None


class _FakeModel:
    def __init__(self, names: dict[int, str], preds: list[Any]) -> None:
        self.names = names
        self._preds = preds
        self.predict_kwargs: dict[str, Any] = {}

    def predict(self, image: Image.Image, **kwargs: Any) -> list[Any]:
        self.predict_kwargs = kwargs
        return self._preds


def _prediction(*detections: tuple[int, np.ndarray]) -> _FakePrediction:
    """Build a fake YOLO prediction from (class_id, normalized-polygon) pairs."""
    return _FakePrediction(
        boxes=[_FakeBox(class_id) for class_id, _ in detections],
        masks=_FakeMasks(xyn=[xyn for _, xyn in detections]),
    )


def _as_results(preds: list[Any]) -> list[Results]:
    """The fakes match the ``Results`` shape the predictor reads; cast for ty."""
    return cast("list[Results]", preds)


def _as_model(fake: _FakeModel) -> YOLO:
    return cast("YOLO", fake)


def _event(
    logs: list[MutableMapping[str, Any]], event: str
) -> MutableMapping[str, Any]:
    """The one captured log entry with this event, or fail saying what was there."""
    matches = [entry for entry in logs if entry["event"] == event]
    assert matches, f"{event!r} not logged; got {[e['event'] for e in logs]}"
    return matches[0]


class TestYOLOSegmentationPredictorModelLoading:
    def test_model_loads_lazily_on_access(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.pt"
        model_path.touch()
        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))
        assert predictor._model is None

        with patch("digitex.ml.predictors.YOLO") as mock_yolo:
            _ = predictor.model

        mock_yolo.assert_called_once_with(str(model_path), verbose=False)

    def test_model_cached_after_first_access(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.pt"
        model_path.touch()
        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        with patch("digitex.ml.predictors.YOLO") as mock_yolo:
            model1 = predictor.model
            model2 = predictor.model

        assert model1 is model2
        mock_yolo.assert_called_once()

    def test_load_failure_raises_runtime_error(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.pt"
        model_path.touch()
        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        with (
            patch(
                "digitex.ml.predictors.YOLO",
                side_effect=FileNotFoundError("Model not found"),
            ),
            pytest.raises(RuntimeError, match="Failed to load model"),
        ):
            _ = predictor.model


class TestForeignPathsReadable:
    """A checkpoint pickled on the other kind of platform still has to load."""

    @pytest.mark.parametrize("module", _SPELLINGS)
    def test_a_foreign_path_can_be_unpickled_inside(self, module: str) -> None:
        """The failure this exists for: a Linux-trained model, loaded here.

        The checkpoint holds the training run's directory, which pickle
        rebuilds by calling the class it was saved as — and that class refuses
        to instantiate on the other platform. Which module it names depends on
        the Python that trained the model, so both are covered.
        """
        payload = _foreign_path_payload(module)

        with pytest.raises(NotImplementedError):
            pickle.loads(payload)

        with foreign_paths_readable():
            restored = pickle.loads(payload)

        assert restored.name == "runs"

    def test_the_name_is_put_back_afterwards(self) -> None:
        original = getattr(pathlib, _FOREIGN)

        with foreign_paths_readable():
            assert getattr(pathlib, _FOREIGN) is not original

        assert getattr(pathlib, _FOREIGN) is original

    def test_the_name_is_put_back_after_a_failed_load(self) -> None:
        original = getattr(pathlib, _FOREIGN)

        with pytest.raises(RuntimeError), foreign_paths_readable():
            raise RuntimeError("corrupt checkpoint")

        assert getattr(pathlib, _FOREIGN) is original

    def test_a_model_is_loaded_with_the_names_patched(self, tmp_path: Path) -> None:
        """The guard is worth nothing if the load happens outside it."""
        model_path = tmp_path / "model.pt"
        model_path.touch()
        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))
        seen: list[object] = []

        with patch(
            "digitex.ml.predictors.YOLO",
            side_effect=lambda *_a, **_k: seen.append(getattr(pathlib, _FOREIGN)),
        ):
            _ = predictor.model

        assert seen == [pathlib.Path]


class TestDetectionsFrom:
    """``detections_from`` is pure — no predictor, no model."""

    def test_empty_predictions_raise(self) -> None:
        with pytest.raises(ValueError, match="Empty predictions received"):
            detections_from(_as_results([]), 100, 100, {0: "question"})

    def test_prediction_without_boxes_attr_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid prediction format"):
            detections_from(_as_results([object()]), 100, 100, {0: "question"})

    def test_a_dropped_detection_is_counted_in_the_log(self) -> None:
        """A silently dropped marker re-files the rest of a book, so say so."""
        good = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])
        pred = _prediction((0, good), (0, good))
        # A mask that cannot be scaled fails inside the loop, not before it.
        pred.masks = _FakeMasks(xyn=[good, cast("Any", "not an array")])

        with capture_logs() as logs:
            detections = detections_from(_as_results([pred]), 100, 100, {0: "question"})

        assert len(detections) == 1
        summary = _event(logs, "Dropped detections on this page")
        assert (summary["dropped"], summary["kept"]) == (1, 1)

    def test_a_box_mask_count_mismatch_is_reported(self) -> None:
        """Truncating to the shorter of the two used to happen in silence."""
        good = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])
        pred = _prediction((0, good), (0, good))
        pred.masks = _FakeMasks(xyn=[good])

        with capture_logs() as logs:
            detections = detections_from(_as_results([pred]), 100, 100, {0: "question"})

        assert len(detections) == 1
        mismatch = _event(
            logs, "Box and mask counts differ, pairing only what lines up"
        )
        assert (mismatch["boxes"], mismatch["masks"]) == (2, 1)

    def test_scales_normalized_polygons_to_pixels(self) -> None:
        pred = _prediction((0, np.array([[0.1, 0.1], [0.5, 0.5], [0.5, 0.1]])))

        detections = detections_from(_as_results([pred]), 100, 200, {0: "question"})

        assert len(detections) == 1
        assert detections[0].label == "question"
        assert detections[0].polygon == [(10, 20), (50, 100), (50, 20)]

    def test_unknown_class_id_falls_back_to_unknown(self) -> None:
        pred = _prediction((7, np.array([[0.1, 0.1], [0.5, 0.5], [0.5, 0.1]])))

        detections = detections_from(_as_results([pred]), 100, 100, {0: "question"})

        assert detections[0].label == "unknown"

    def test_none_boxes_or_masks_yield_no_detections(self) -> None:
        pred = _FakePrediction(boxes=None, masks=None)

        assert detections_from(_as_results([pred]), 100, 100, {0: "question"}) == []

    def test_the_confidence_rides_along_with_the_detection(self) -> None:
        """Label Studio sorts a review queue by it."""
        pred = _FakePrediction(
            boxes=[_FakeBox(0, score=0.42)],
            masks=_FakeMasks(xyn=[np.array([[0.1, 0.1], [0.5, 0.5], [0.5, 0.1]])]),
        )

        detections = detections_from(_as_results([pred]), 100, 100, {0: "question"})

        assert detections[0].score == pytest.approx(0.42)

    @pytest.mark.parametrize(
        ("xyn", "reason"),
        [
            (np.zeros((0, 2)), "no-contour"),
            (np.array([[0.1, 0.1]]), "one-point"),
            (np.array([[0.1, 0.1], [0.4, 0.4]]), "two-points"),
        ],
    )
    @pytest.mark.parametrize("simplify", [False, True])
    def test_a_mask_too_thin_to_be_a_ring_is_dropped(
        self, xyn: np.ndarray, reason: str, simplify: bool
    ) -> None:
        """Ultralytics returns an empty (0, 2) array for a mask with no contour.

        Nothing raises on it, so it used to reach Label Studio as a polygon
        with no points, and page extraction as a reading order over min() of
        an empty sequence.
        """
        pred = _prediction((0, xyn))

        with capture_logs() as logs:
            detections = detections_from(
                _as_results([pred]), 100, 100, {0: "question"}, simplify=simplify
            )

        assert detections == []
        assert _event(logs, "Dropped detections on this page")["dropped"] == 1

    def test_simplify_drops_collinear_points(self) -> None:
        xyn = np.array(
            [
                [0.0, 0.0],
                [0.25, 0.0],
                [0.5, 0.0],
                [0.75, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        pred = _prediction((0, xyn))

        detections = detections_from(
            _as_results([pred]), 100, 100, {0: "question"}, simplify=True
        )

        assert len(detections) == 1
        assert len(detections[0].polygon) < len(xyn)


class TestYOLOSegmentationPredictorPredict:
    def test_full_prediction_pipeline(self) -> None:
        pred = _prediction(
            (0, np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]])),
            (1, np.array([[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]])),
        )
        predictor = YOLO_SegmentationPredictor(model_path="model.pt")
        predictor._model = _as_model(
            _FakeModel({0: "question", 1: "option"}, preds=[pred])
        )

        detections = predictor.predict(Image.new("RGB", (640, 480), color="white"))

        assert [det.label for det in detections] == ["question", "option"]
        assert all(det.polygon for det in detections)

    def test_the_nms_free_head_stays_switched_off(self) -> None:
        """YOLO26 is end2end by default, and this model duplicates regions that way.

        Measured over the pooled pages, the one2one head takes overlapping
        region pairs from 28 to 283 — so inference stays on the one2many branch
        behind ordinary NMS, class-agnostic so one anchor cannot return twice
        under two labels. YOLO reads ``end2end`` once per model instance, off
        the first predict() call, which is why it is pinned rather than passed.
        """
        pred = _prediction((0, np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]])))
        model = _FakeModel({0: "question"}, preds=[pred])
        predictor = YOLO_SegmentationPredictor(model_path="model.pt")
        predictor._model = _as_model(model)

        predictor.predict(Image.new("RGB", (640, 480), color="white"))

        assert model.predict_kwargs["end2end"] is False
        assert model.predict_kwargs["agnostic_nms"] is True
        assert model.predict_kwargs["imgsz"] == 640
        assert model.predict_kwargs["max_det"] == 50
