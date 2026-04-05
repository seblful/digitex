"""Tests for Label Studio module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from digitex.label_studio.client import LabelStudioClient
from digitex.label_studio.predictor import TaskPredictor


class TestLabelStudioClient:
    """Test LabelStudioClient class."""

    def test_init(self) -> None:
        """Test client initialization."""
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            client = LabelStudioClient("http://localhost:8080", "api-key")
            mock_ls.assert_called_once_with(
                base_url="http://localhost:8080", api_key="api-key"
            )

    def test_get_tasks(self) -> None:
        """Test getting tasks for a project."""
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            mock_instance = mock_ls.return_value
            mock_instance.tasks.list.return_value = [MagicMock(), MagicMock()]
            client = LabelStudioClient("http://localhost:8080", "api-key")
            tasks = client.get_tasks(project_id=1)
            assert len(tasks) == 2

    def test_get_unlabeled_tasks(self) -> None:
        """Test getting unlabeled tasks."""
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            mock_instance = mock_ls.return_value
            task1 = MagicMock()
            task1.is_labeled = False
            task2 = MagicMock()
            task2.is_labeled = True
            task3 = MagicMock()
            task3.is_labeled = False
            mock_instance.tasks.list.return_value = [task1, task2, task3]
            client = LabelStudioClient("http://localhost:8080", "api-key")
            unlabeled = client.get_unlabeled_tasks(project_id=1)
            assert len(unlabeled) == 2

    def test_get_local_path_with_file_param(self) -> None:
        """Test extracting local path from file parameter."""
        task = MagicMock()
        task.data = {"image": "/data/local-files/?d=path/to/image.png"}
        path = LabelStudioClient.get_local_path(task)
        assert path == Path("path/to/image.png")

    def test_get_local_path_with_d_param(self) -> None:
        """Test extracting local path from d parameter."""
        task = MagicMock()
        task.data = {"image": "/data/local-files/?d=path/to/image.png"}
        path = LabelStudioClient.get_local_path(task)
        assert path is not None

    def test_get_local_path_no_image(self) -> None:
        """Test get_local_path returns None when no image in task data."""
        task = MagicMock()
        task.data = {}
        path = LabelStudioClient.get_local_path(task)
        assert path is None

    def test_get_local_path_invalid_uri(self) -> None:
        """Test get_local_path returns None for invalid URI."""
        task = MagicMock()
        task.data = {"image": "http://example.com/image.png"}
        path = LabelStudioClient.get_local_path(task)
        assert path is None

    def test_upload_predictions_empty(self) -> None:
        """Test uploading empty predictions."""
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            mock_instance = mock_ls.return_value
            client = LabelStudioClient("http://localhost:8080", "api-key")
            client.upload_predictions(project_id=1, predictions=[])
            mock_instance.projects.import_predictions.assert_not_called()

    def test_upload_predictions(self) -> None:
        """Test uploading predictions."""
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            mock_instance = mock_ls.return_value
            client = LabelStudioClient("http://localhost:8080", "api-key")
            predictions = [{"task": 1, "result": [], "model_version": "v1"}]
            client.upload_predictions(project_id=1, predictions=predictions)
            mock_instance.projects.import_predictions.assert_called_once()

    def test_get_label_config(self) -> None:
        """Test getting label config."""
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            mock_instance = mock_ls.return_value
            mock_project = MagicMock()
            mock_project.label_config = "<View></View>"
            mock_instance.projects.get.return_value = mock_project
            client = LabelStudioClient("http://localhost:8080", "api-key")
            config = client.get_label_config(project_id=1)
            assert config == "<View></View>"

    def test_cancel_task(self) -> None:
        """Test cancelling a task."""
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            mock_instance = mock_ls.return_value
            client = LabelStudioClient("http://localhost:8080", "api-key")
            client.cancel_task(task_id=42)
            mock_instance.tasks.update.assert_called_once_with(
                id="42", meta={"is_cancelled": True}
            )


class TestTaskPredictor:
    """Test TaskPredictor class."""

    def test_init(self) -> None:
        """Test predictor initialization."""
        with (
            patch("digitex.label_studio.predictor.YOLO_SegmentationPredictor"),
            patch("digitex.label_studio.predictor.LabelStudioClient"),
        ):
            predictor = TaskPredictor(
                model_path="model.pt",
                url="http://localhost:8080",
                api_key="api-key",
                epsilon=5.0,
            )
            assert predictor._model_version == "model"

    def test_init_with_model_version(self) -> None:
        """Test predictor initialization with custom model version."""
        with (
            patch("digitex.label_studio.predictor.YOLO_SegmentationPredictor"),
            patch("digitex.label_studio.predictor.LabelStudioClient"),
        ):
            predictor = TaskPredictor(
                model_path="model.pt",
                url="http://localhost:8080",
                api_key="api-key",
                model_version="custom-v1",
            )
            assert predictor._model_version == "custom-v1"

    def test_classes_lazy_load(self) -> None:
        """Test that classes are loaded lazily."""
        with (
            patch(
                "digitex.label_studio.predictor.YOLO_SegmentationPredictor"
            ) as mock_pred_class,
            patch("digitex.label_studio.predictor.LabelStudioClient"),
        ):
            mock_model = MagicMock()
            mock_model.names = {0: "question", 1: "answer"}
            mock_pred_instance = mock_pred_class.return_value
            mock_pred_instance.model = mock_model
            predictor = TaskPredictor(
                model_path="model.pt",
                url="http://localhost:8080",
                api_key="api-key",
            )
            classes = predictor.classes
            assert classes == {0: "question", 1: "answer"}

    def test_to_ls_results(self) -> None:
        """Test converting prediction to Label Studio results."""
        with (
            patch("digitex.label_studio.predictor.YOLO_SegmentationPredictor"),
            patch("digitex.label_studio.predictor.LabelStudioClient"),
        ):
            predictor = TaskPredictor(
                model_path="model.pt",
                url="http://localhost:8080",
                api_key="api-key",
            )
            predictor._classes = {0: "question"}

            from digitex.ml.predictors.prediction_result import (
                SegmentationPredictionResult,
            )

            result = SegmentationPredictionResult(
                ids=[0],
                polygons=[[(10, 10), (50, 10), (50, 50), (10, 50)]],
                id2label={0: "question"},
            )
            ls_results = predictor._to_ls_results(result, img_width=100, img_height=100)
            assert len(ls_results) == 1
            assert ls_results[0]["value"]["polygonlabels"] == ["question"]
            assert ls_results[0]["value"]["points"] == [
                [10.0, 10.0],
                [50.0, 10.0],
                [50.0, 50.0],
                [10.0, 50.0],
            ]

    def test_predict_task_no_path(self) -> None:
        """Test predicting task with no image path."""
        with (
            patch("digitex.label_studio.predictor.YOLO_SegmentationPredictor"),
            patch("digitex.label_studio.predictor.LabelStudioClient") as mock_client,
        ):
            mock_client.return_value.get_local_path.return_value = None
            predictor = TaskPredictor(
                model_path="model.pt",
                url="http://localhost:8080",
                api_key="api-key",
            )
            task = MagicMock()
            task.id = 1
            result = predictor._predict_task(task)
            assert result is None

    def test_predict_task_file_missing(self) -> None:
        """Test predicting task with missing file."""
        with (
            patch("digitex.label_studio.predictor.YOLO_SegmentationPredictor"),
            patch("digitex.label_studio.predictor.LabelStudioClient") as mock_client,
        ):
            mock_client.return_value.get_local_path.return_value = Path(
                "/nonexistent/image.png"
            )
            predictor = TaskPredictor(
                model_path="model.pt",
                url="http://localhost:8080",
                api_key="api-key",
            )
            task = MagicMock()
            task.id = 1
            result = predictor._predict_task(task)
            assert result is None
