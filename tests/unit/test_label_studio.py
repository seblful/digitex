"""Tests for the Label Studio client adapter and task predictor.

``LabelStudioClient`` is the project's thin adapter around the Label Studio
SDK, so the SDK itself is patched here; everything else is exercised through
the adapter's interface.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import pytest
from PIL import Image

from digitex.domain.entities import Detection, PixelPolygon
from digitex.label_studio.client import LabelStudioClient
from digitex.label_studio.predictor import TaskPredictor


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

    def test_get_unlabeled_tasks_filters_labeled(self, ls_sdk: MagicMock) -> None:
        task1 = MagicMock(is_labeled=False)
        task2 = MagicMock(is_labeled=True)
        task3 = MagicMock(is_labeled=False)
        ls_sdk.tasks.list.return_value = [task1, task2, task3]
        client = LabelStudioClient("http://localhost:8080", "api-key")
        unlabeled = client.get_unlabeled_tasks(project_id=1)
        assert unlabeled == [task1, task3]

    def test_upload_predictions_skips_empty(self, ls_sdk: MagicMock) -> None:
        client = LabelStudioClient("http://localhost:8080", "api-key")
        client.upload_predictions(project_id=1, predictions=[])
        ls_sdk.projects.import_predictions.assert_not_called()

    def test_upload_predictions(self, ls_sdk: MagicMock) -> None:
        client = LabelStudioClient("http://localhost:8080", "api-key")
        predictions = [{"task": 1, "result": [], "model_version": "v1"}]
        client.upload_predictions(project_id=1, predictions=predictions)
        ls_sdk.projects.import_predictions.assert_called_once()


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

    def test_to_ls_results_converts_pixels_to_percent(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        detections = [
            Detection(
                label="question",
                polygon=PixelPolygon([(10, 10), (50, 10), (50, 50), (10, 50)]),
            )
        ]

        ls_results = _task_predictor()._to_ls_results(
            detections, img_width=100, img_height=100
        )

        assert len(ls_results) == 1
        assert ls_results[0]["value"]["polygonlabels"] == ["question"]
        assert ls_results[0]["value"]["points"] == [
            [10.0, 10.0],
            [50.0, 10.0],
            [50.0, 50.0],
            [10.0, 50.0],
        ]

    def test_predict_tasks_uploads_one_prediction_per_task(
        self, predictor_deps: tuple[MagicMock, MagicMock], tmp_path: Path
    ) -> None:
        """The whole run: unlabeled tasks in, uploaded predictions out."""
        mock_pred_cls, mock_client_cls = predictor_deps
        image_path = tmp_path / "page.png"
        Image.new("RGB", (100, 100), color="white").save(image_path)
        task = MagicMock(id=7)
        task.data = {"image": f"/data/local-files/?d={quote(str(image_path))}"}
        mock_client_cls.return_value.get_unlabeled_tasks.return_value = [task]
        mock_pred_cls.return_value.predict.return_value = [
            Detection(
                label="question",
                polygon=PixelPolygon([(10, 10), (50, 10), (50, 50), (10, 50)]),
            )
        ]

        predicted = _task_predictor("v3").predict_tasks(project_id=1)

        assert predicted == 1
        mock_client_cls.return_value.upload_predictions.assert_called_once()
        project_id, predictions = (
            mock_client_cls.return_value.upload_predictions.call_args.args
        )
        assert project_id == 1
        assert predictions[0]["task"] == 7
        assert predictions[0]["model_version"] == "v3"
        assert predictions[0]["result"][0]["value"]["polygonlabels"] == ["question"]

    def test_predict_tasks_skips_tasks_it_cannot_read(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        _, mock_client_cls = predictor_deps
        task = MagicMock(id=1)
        task.data = {"image": "http://example.com/remote.png"}
        mock_client_cls.return_value.get_unlabeled_tasks.return_value = [task]

        assert _task_predictor().predict_tasks(project_id=1) == 0
        mock_client_cls.return_value.upload_predictions.assert_not_called()

    def test_predict_task_none_without_local_path(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        task = MagicMock(id=1)
        task.data = {"image": "http://example.com/remote.png"}
        assert _task_predictor()._predict_task(task) is None

    def test_predict_task_none_when_file_missing(
        self, predictor_deps: tuple[MagicMock, MagicMock]
    ) -> None:
        task = MagicMock(id=1)
        task.data = {"image": "/data/local-files/?d=nonexistent%5Cimage.png"}
        assert _task_predictor()._predict_task(task) is None
