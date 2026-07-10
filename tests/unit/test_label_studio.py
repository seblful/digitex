"""Tests for the Label Studio client adapter and task predictor.

``LabelStudioClient`` is the project's thin adapter around the Label Studio
SDK, so the SDK itself is patched here; everything else is exercised through
the adapter's interface.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from digitex.label_studio.client import LabelStudioClient
from digitex.label_studio.predictor import TaskPredictor
from digitex.ml.predictors import SegmentationPredictionResult


@pytest.fixture
def ls_sdk() -> Iterator[MagicMock]:
    """Patch the Label Studio SDK class and yield its mocked instance."""
    with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
        yield mock_ls.return_value


class TestLabelStudioClient:
    def test_init_passes_credentials_to_sdk(self) -> None:
        with patch("digitex.label_studio.client.LabelStudio") as mock_ls:
            LabelStudioClient("http://localhost:8080", "api-key")
        mock_ls.assert_called_once_with(
            base_url="http://localhost:8080", api_key="api-key"
        )

    def test_get_tasks(self, ls_sdk: MagicMock) -> None:
        ls_sdk.tasks.list.return_value = [MagicMock(), MagicMock()]
        client = LabelStudioClient("http://localhost:8080", "api-key")
        tasks = client.get_tasks(project_id=1)
        assert len(tasks) == 2

    def test_get_unlabeled_tasks_filters_labeled(self, ls_sdk: MagicMock) -> None:
        task1 = MagicMock(is_labeled=False)
        task2 = MagicMock(is_labeled=True)
        task3 = MagicMock(is_labeled=False)
        ls_sdk.tasks.list.return_value = [task1, task2, task3]
        client = LabelStudioClient("http://localhost:8080", "api-key")
        unlabeled = client.get_unlabeled_tasks(project_id=1)
        assert unlabeled == [task1, task3]

    def test_get_local_path_from_local_files_uri(self) -> None:
        task = MagicMock()
        task.data = {"image": "/data/local-files/?d=path/to/image.png"}
        assert LabelStudioClient.get_local_path(task) == Path("path/to/image.png")

    def test_get_local_path_none_without_image(self) -> None:
        task = MagicMock()
        task.data = {}
        assert LabelStudioClient.get_local_path(task) is None

    def test_get_local_path_none_for_remote_uri(self) -> None:
        task = MagicMock()
        task.data = {"image": "http://example.com/image.png"}
        assert LabelStudioClient.get_local_path(task) is None

    def test_upload_predictions_skips_empty(self, ls_sdk: MagicMock) -> None:
        client = LabelStudioClient("http://localhost:8080", "api-key")
        client.upload_predictions(project_id=1, predictions=[])
        ls_sdk.projects.import_predictions.assert_not_called()

    def test_upload_predictions(self, ls_sdk: MagicMock) -> None:
        client = LabelStudioClient("http://localhost:8080", "api-key")
        predictions = [{"task": 1, "result": [], "model_version": "v1"}]
        client.upload_predictions(project_id=1, predictions=predictions)
        ls_sdk.projects.import_predictions.assert_called_once()

    def test_get_label_config(self, ls_sdk: MagicMock) -> None:
        ls_sdk.projects.get.return_value = MagicMock(label_config="<View></View>")
        client = LabelStudioClient("http://localhost:8080", "api-key")
        assert client.get_label_config(project_id=1) == "<View></View>"

    def test_cancel_task(self, ls_sdk: MagicMock) -> None:
        client = LabelStudioClient("http://localhost:8080", "api-key")
        client.cancel_task(task_id=42)
        ls_sdk.tasks.update.assert_called_once_with(
            id="42", meta={"is_cancelled": True}
        )


@pytest.fixture
def predictor_deps() -> Iterator[tuple[MagicMock, MagicMock]]:
    """Patch TaskPredictor's collaborators; yield (predictor_cls, client_cls)."""
    with (
        patch(
            "digitex.label_studio.predictor.YOLO_SegmentationPredictor"
        ) as mock_pred_cls,
        patch("digitex.label_studio.predictor.LabelStudioClient") as mock_client_cls,
    ):
        yield mock_pred_cls, mock_client_cls


def _task_predictor(model_version: str | None = None) -> TaskPredictor:
    kwargs = {"model_version": model_version} if model_version else {}
    return TaskPredictor(
        model_path="model.pt",
        url="http://localhost:8080",
        api_key="api-key",
        **kwargs,
    )


class TestTaskPredictor:
    def test_model_version_defaults_to_model_stem(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        assert _task_predictor()._model_version == "model"

    def test_custom_model_version(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        assert _task_predictor("custom-v1")._model_version == "custom-v1"

    def test_classes_lazy_load_from_model(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        mock_pred_cls, _ = predictor_deps
        mock_pred_cls.return_value.model.names = {0: "question", 1: "answer"}
        assert _task_predictor().classes == {0: "question", 1: "answer"}

    def test_to_ls_results_converts_pixels_to_percent(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        predictor = _task_predictor()
        predictor._classes = {0: "question"}
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

    def test_predict_task_none_without_local_path(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        _, mock_client_cls = predictor_deps
        mock_client_cls.return_value.get_local_path.return_value = None
        task = MagicMock(id=1)
        assert _task_predictor()._predict_task(task) is None

    def test_predict_task_none_when_file_missing(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        _, mock_client_cls = predictor_deps
        mock_client_cls.return_value.get_local_path.return_value = Path(
            "/nonexistent/image.png"
        )
        task = MagicMock(id=1)
        assert _task_predictor()._predict_task(task) is None
