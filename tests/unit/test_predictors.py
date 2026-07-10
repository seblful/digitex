"""Tests for the ML predictors.

YOLO's ``Results`` objects are stood in by small fakes with the same shape
(``pred.boxes[i].cls.item()`` / ``pred.masks.xyn``); the ``YOLO`` constructor
itself is patched only in the lazy-loading tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from digitex.ml.predictors import (
    SegmentationPredictionResult,
    YOLO_SegmentationPredictor,
)

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


class TestSegmentationPredictionResult:
    def test_init_valid(self) -> None:
        ids = [0, 1, 2]
        polygons = [
            [(10, 10), (50, 10), (50, 50), (10, 50)],
            [(20, 20), (60, 20), (60, 60), (20, 60)],
            [(30, 30), (70, 30), (70, 70), (30, 70)],
        ]
        id2label = {0: "question", 1: "option", 2: "part"}

        result = SegmentationPredictionResult(
            ids=ids, polygons=polygons, id2label=id2label
        )

        assert result.ids == ids
        assert result.polygons == polygons
        assert result.id2label == id2label

    def test_init_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="Number of polygons must be equal"):
            SegmentationPredictionResult(
                ids=[0, 1],
                polygons=[
                    [(10, 10), (50, 10), (50, 50), (10, 50)],
                    [(20, 20), (60, 20), (60, 60), (20, 60)],
                    [(30, 30), (70, 30), (70, 70), (30, 70)],
                ],
                id2label={0: "question", 1: "option"},
            )

    def test_id2polygons_groups_by_class(self) -> None:
        polygons = [
            [(10, 10), (50, 10), (50, 50), (10, 50)],
            [(20, 20), (60, 20), (60, 60), (20, 60)],
            [(30, 30), (70, 30), (70, 70), (30, 70)],
            [(40, 40), (80, 40), (80, 80), (40, 80)],
        ]

        result = SegmentationPredictionResult(
            ids=[0, 1, 0, 2],
            polygons=polygons,
            id2label={0: "question", 1: "option", 2: "part"},
        )
        grouped = result.id2polygons

        assert len(grouped[0]) == 2
        assert len(grouped[1]) == 1
        assert len(grouped[2]) == 1
        assert grouped[0][0] == polygons[0]
        assert grouped[0][1] == polygons[2]


class TestYOLOSegmentationPredictorModelLoading:
    def test_model_loads_lazily_on_access(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.pt"
        model_path.touch()
        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))
        assert predictor._model is None

        with patch("digitex.ml.predictors.segmentation.YOLO") as mock_yolo:
            _ = predictor.model

        mock_yolo.assert_called_once_with(str(model_path), verbose=False)

    def test_model_cached_after_first_access(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.pt"
        model_path.touch()
        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        with patch("digitex.ml.predictors.segmentation.YOLO") as mock_yolo:
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
                "digitex.ml.predictors.segmentation.YOLO",
                side_effect=FileNotFoundError("Model not found"),
            ),
            pytest.raises(RuntimeError, match="Failed to load model"),
        ):
            _ = predictor.model


class TestYOLOSegmentationPredictorCreateResult:
    def _predictor(
        self, names: dict[int, str] | None = None, *, simplify: bool = False
    ) -> YOLO_SegmentationPredictor:
        predictor = YOLO_SegmentationPredictor(model_path="model.pt", simplify=simplify)
        predictor._model = _as_model(_FakeModel(names or {0: "question"}, preds=[]))
        return predictor

    def test_empty_predictions_raise(self) -> None:
        with pytest.raises(ValueError, match="Empty predictions received"):
            self._predictor().create_result(_as_results([]), 100, 100)

    def test_prediction_without_boxes_attr_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid prediction format"):
            self._predictor().create_result(_as_results([object()]), 100, 100)

    def test_scales_normalized_polygons_to_pixels(self) -> None:
        pred = _prediction((0, np.array([[0.1, 0.1], [0.5, 0.5], [0.5, 0.1]])))

        result = self._predictor().create_result(_as_results([pred]), 100, 200)

        assert result.ids == [0]
        assert result.polygons == [[(10, 20), (50, 100), (50, 20)]]

    def test_none_boxes_or_masks_yield_empty_result(self) -> None:
        pred = _FakePrediction(boxes=None, masks=None)

        result = self._predictor().create_result(_as_results([pred]), 100, 100)

        assert result.ids == []
        assert result.polygons == []

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

        result = self._predictor(simplify=True).create_result(
            _as_results([pred]), 100, 100
        )

        assert result.ids == [0]
        assert len(result.polygons[0]) < len(xyn)


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

        result = predictor.predict(Image.new("RGB", (640, 480), color="white"))

        assert result.ids == [0, 1]
        assert len(result.polygons) == 2
        assert result.id2label == {0: "question", 1: "option"}
