"""Tests for the ML Predictors module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from digitex.ml.predictors import (
    SegmentationPredictionResult,
    YOLO_SegmentationPredictor,
)


class TestSegmentationPredictionResult:
    """Test suite for SegmentationPredictionResult class."""

    def test_init_valid(self) -> None:
        """Test initialization with valid data."""
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

    def test_init_mismatched_lengths(self) -> None:
        """Test initialization raises ValueError when lengths don't match."""
        ids = [0, 1]
        polygons = [
            [(10, 10), (50, 10), (50, 50), (10, 50)],
            [(20, 20), (60, 20), (60, 60), (20, 60)],
            [(30, 30), (70, 30), (70, 70), (30, 70)],
        ]
        id2label = {0: "question", 1: "option"}

        with pytest.raises(ValueError, match="Number of polygons must be equal"):
            SegmentationPredictionResult(ids=ids, polygons=polygons, id2label=id2label)

    def test_id2polygons(self) -> None:
        """Test id2polygons groups polygons by class ID."""
        ids = [0, 1, 0, 2]
        polygons = [
            [(10, 10), (50, 10), (50, 50), (10, 50)],
            [(20, 20), (60, 20), (60, 60), (20, 60)],
            [(30, 30), (70, 30), (70, 70), (30, 70)],
            [(40, 40), (80, 40), (80, 80), (40, 80)],
        ]
        id2label = {0: "question", 1: "option", 2: "part"}

        result = SegmentationPredictionResult(
            ids=ids, polygons=polygons, id2label=id2label
        )
        grouped = result.id2polygons

        assert len(grouped[0]) == 2
        assert len(grouped[1]) == 1
        assert len(grouped[2]) == 1
        assert grouped[0][0] == polygons[0]
        assert grouped[0][1] == polygons[2]


class TestYOLOSegmentationPredictor:
    """Test suite for YOLO_SegmentationPredictor class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test YOLO_SegmentationPredictor initialization."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(
            model_path=str(model_path),
            simplify=False,
        )

        assert predictor.model_path == str(model_path)
        assert predictor.simplify is False
        assert predictor._model is None

    def test_init_with_simplify(self, tmp_path: Path) -> None:
        """Test initialization with simplify enabled."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(
            model_path=str(model_path),
            simplify=True,
        )

        assert predictor.simplify is True

    def test_model_loads_on_access(self, tmp_path: Path) -> None:
        """Test that model is loaded when accessed."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        with patch("digitex.ml.predictors.segmentation.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            _ = predictor.model

            mock_yolo.assert_called_once_with(str(model_path), verbose=False)

    def test_model_caches_after_first_access(self, tmp_path: Path) -> None:
        """Test that model is cached after first access."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        with patch("digitex.ml.predictors.segmentation.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            model1 = predictor.model
            model2 = predictor.model

            assert model1 is model2
            mock_yolo.assert_called_once()

    def test_model_raises_on_load_failure(self, tmp_path: Path) -> None:
        """Test that RuntimeError is raised when model fails to load."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        with patch("digitex.ml.predictors.segmentation.YOLO") as mock_yolo:
            mock_yolo.side_effect = FileNotFoundError("Model not found")

            with pytest.raises(RuntimeError, match="Failed to load model"):
                _ = predictor.model

    def test_preprocess_image(self, tmp_path: Path) -> None:
        """Test preprocess_image converts PIL Image to numpy array."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        img = Image.new("RGB", (100, 100), color="red")
        result = predictor.preprocess_image(img)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_create_result_empty_predictions(self, tmp_path: Path) -> None:
        """Test create_result raises ValueError on empty predictions."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        with pytest.raises(ValueError, match="Empty predictions received"):
            predictor.create_result([], 100, 100)

    def test_create_result_invalid_prediction_format(self, tmp_path: Path) -> None:
        """Test create_result raises ValueError on invalid prediction format."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        mock_pred = MagicMock(spec=[])

        with pytest.raises(ValueError, match="Invalid prediction format"):
            predictor.create_result([mock_pred], 100, 100)

    def test_create_result_with_boxes_and_masks(self, tmp_path: Path) -> None:
        """Test create_result correctly parses boxes and masks."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0

        mock_mask = MagicMock()
        mock_mask.xyn = np.array([[0.1, 0.1], [0.5, 0.5], [0.5, 0.1]])

        mock_pred = MagicMock()
        mock_pred.boxes = MagicMock()
        mock_pred.boxes.__iter__ = MagicMock(return_value=iter([mock_box]))
        mock_pred.masks = MagicMock()
        mock_pred.masks.xyn = [mock_mask.xyn]

        mock_model = MagicMock()
        mock_model.names = {0: "question"}
        predictor._model = mock_model

        result = predictor.create_result([mock_pred], 100, 100)

        assert len(result.ids) == 1
        assert len(result.polygons) == 1
        assert result.ids[0] == 0

    def test_create_result_with_none_boxes_or_masks(self, tmp_path: Path) -> None:
        """Test create_result handles None boxes or masks gracefully."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        mock_pred = MagicMock()
        mock_pred.boxes = None
        mock_pred.masks = None

        mock_model = MagicMock()
        mock_model.names = {0: "question"}
        predictor._model = mock_model

        result = predictor.create_result([mock_pred], 100, 100)

        assert result.ids == []
        assert result.polygons == []

    def test_create_result_simplifies_polygon(self, tmp_path: Path) -> None:
        """Test create_result simplifies polygon when simplify=True."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(
            model_path=str(model_path),
            simplify=True,
        )

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0

        mock_mask = MagicMock()
        mock_mask.xyn = np.array(
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

        mock_pred = MagicMock()
        mock_pred.boxes = MagicMock()
        mock_pred.boxes.__iter__ = MagicMock(return_value=iter([mock_box]))
        mock_pred.masks = MagicMock()
        mock_pred.masks.xyn = [mock_mask.xyn]

        mock_model = MagicMock()
        mock_model.names = {0: "question"}
        predictor._model = mock_model

        result = predictor.create_result([mock_pred], 100, 100)

        assert len(result.ids) == 1
        assert len(result.polygons) == 1

    def test_predict(self, tmp_path: Path) -> None:
        """Test predict runs full prediction pipeline."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        img = Image.new("RGB", (100, 100), color="white")

        mock_model = MagicMock()
        mock_pred = MagicMock()
        mock_pred.boxes = None
        mock_pred.masks = None

        mock_model.names = {}
        mock_model.predict.return_value = [mock_pred]
        predictor._model = mock_model

        with patch.object(predictor, "preprocess_image") as mock_preprocess:
            mock_preprocess.return_value = np.array(img)
            with patch.object(predictor, "create_result") as mock_create:
                mock_result = MagicMock()
                mock_create.return_value = mock_result

                predictor.predict(img)

                mock_preprocess.assert_called_once_with(img)
                mock_create.assert_called_once()


class TestYOLOSegmentationPredictorIntegration:
    """Integration tests for YOLO_SegmentationPredictor with mocked YOLO."""

    def test_full_prediction_pipeline(self, tmp_path: Path) -> None:
        """Test full prediction pipeline with mocked YOLO model."""
        model_path = tmp_path / "model.pt"
        model_path.touch()

        predictor = YOLO_SegmentationPredictor(model_path=str(model_path))

        img = Image.new("RGB", (640, 480), color="white")

        mock_box1 = MagicMock()
        mock_box1.cls.item.return_value = 0
        mock_box2 = MagicMock()
        mock_box2.cls.item.return_value = 1

        mock_mask1 = MagicMock()
        mock_mask1.xyn = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]])
        mock_mask2 = MagicMock()
        mock_mask2.xyn = np.array([[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]])

        mock_pred = MagicMock()
        mock_pred.boxes = MagicMock()
        mock_pred.boxes.__iter__ = MagicMock(return_value=iter([mock_box1, mock_box2]))
        mock_pred.masks = MagicMock()
        mock_pred.masks.xyn = [mock_mask1.xyn, mock_mask2.xyn]

        mock_model = MagicMock()
        mock_model.names = {0: "question", 1: "option"}
        mock_model.predict.return_value = [mock_pred]
        predictor._model = mock_model

        result = predictor.predict(img)

        assert len(result.ids) == 2
        assert 0 in result.ids
        assert 1 in result.ids
        assert len(result.polygons) == 2

        assert result.id2label == {0: "question", 1: "option"}
        assert result.label2id == {"question": 0, "option": 1}
