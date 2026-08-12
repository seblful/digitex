"""Tests for the ML predictors.

YOLO's ``Results`` objects are stood in by small fakes with the same shape
(``pred.boxes[i].cls.item()`` / ``pred.masks.xyn``); the ``YOLO`` constructor
itself is patched only in the lazy-loading tests. ``detections_from`` takes the
class map as an argument, so exercising it needs no model at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from digitex.ml.predictors import YOLO_SegmentationPredictor, detections_from

if TYPE_CHECKING:
    from pathlib import Path

    from ultralytics import YOLO
    from ultralytics.engine.results import Results


class _Scalar:
    """Mimics a 0-d tensor: exposes ``.item()``."""

    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


class _FakeBox:
    def __init__(self, class_id: int) -> None:
        self.cls = _Scalar(class_id)


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

    def predict(self, image: Image.Image, **kwargs: Any) -> list[Any]:
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


class TestDetectionsFrom:
    """``detections_from`` is pure — no predictor, no model."""

    def test_empty_predictions_raise(self) -> None:
        with pytest.raises(ValueError, match="Empty predictions received"):
            detections_from(_as_results([]), 100, 100, {0: "question"})

    def test_prediction_without_boxes_attr_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid prediction format"):
            detections_from(_as_results([object()]), 100, 100, {0: "question"})

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
