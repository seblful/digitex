"""Tests for the Label Studio adapter and the prediction run over its tasks.

``LabelStudioClient`` is the project's thin adapter around the Label Studio
SDK — the SDK is patched and the adapter's own narrow contract is asserted.
``TaskPredictor`` takes both of its collaborators through its constructor, so
these tests hand in stubs at that seam and check the run: which tasks are
read, what is uploaded for them, and which ones are passed over.

A task whose image the predictor cannot reach is skipped rather than fatal —
the annotation server holds tasks for images this machine may not have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, cast
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import pytest
from PIL import Image

from digitex.domain.entities import Detection, PixelPolygon
from digitex.labeling.client import LabelStudioClient
from digitex.labeling.predictor import TaskPredictor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

QUAD = PixelPolygon([(10, 10), (50, 10), (50, 50), (10, 50)])


class TestLabelStudioClient:
    @pytest.fixture
    def sdk_class(self) -> Iterator[MagicMock]:
        with patch("digitex.labeling.client.LabelStudio") as mock_ls:
            yield mock_ls

    def test_the_credentials_reach_the_sdk(self, sdk_class: MagicMock) -> None:
        LabelStudioClient("http://localhost:8080", "api-key")

        sdk_class.assert_called_once_with(
            base_url="http://localhost:8080", api_key="api-key"
        )

    def test_only_the_unlabeled_tasks_come_back(self, sdk_class: MagicMock) -> None:
        """Predicting over an annotated task would overwrite a human's work."""
        unlabeled, labeled, also_unlabeled = (
            MagicMock(is_labeled=False),
            MagicMock(is_labeled=True),
            MagicMock(is_labeled=False),
        )
        sdk_class.return_value.tasks.list.return_value = [
            unlabeled,
            labeled,
            also_unlabeled,
        ]
        client = LabelStudioClient("http://localhost:8080", "api-key")

        assert client.get_unlabeled_tasks(project_id=1) == [unlabeled, also_unlabeled]

    def test_an_empty_upload_is_not_sent(self, sdk_class: MagicMock) -> None:
        """A run that predicted nothing must not post an empty import."""
        client = LabelStudioClient("http://localhost:8080", "api-key")

        client.upload_predictions(project_id=1, predictions=[])

        sdk_class.return_value.projects.import_predictions.assert_not_called()

    def test_predictions_are_handed_to_the_sdk(self, sdk_class: MagicMock) -> None:
        client = LabelStudioClient("http://localhost:8080", "api-key")

        client.upload_predictions(
            project_id=1,
            predictions=[{"task": 1, "result": [], "model_version": "v1"}],
        )

        sdk_class.return_value.projects.import_predictions.assert_called_once()


class Deps(NamedTuple):
    """The two collaborators, stubbed at TaskPredictor's constructor seam."""

    predictor: MagicMock
    client: MagicMock


class TestTaskPredictor:
    @pytest.fixture
    def deps(self) -> Deps:
        return Deps(MagicMock(), MagicMock())

    @staticmethod
    def _predictor(
        deps: Deps | None = None, model_version: str = "v1"
    ) -> TaskPredictor:
        stubs = deps or Deps(MagicMock(), MagicMock())
        return TaskPredictor(
            cast("Any", stubs.predictor),
            cast("Any", stubs.client),
            model_version=model_version,
        )

    @staticmethod
    def _task(task_id: int, image: str) -> MagicMock:
        task = MagicMock(id=task_id)
        task.data = {"image": image}
        return task

    def test_pixels_are_converted_to_the_percent_space(self) -> None:
        """Label Studio stores points as percentages of the image size."""
        results = self._predictor()._to_ls_results(
            [Detection(label="question", polygon=QUAD)],
            img_width=100,
            img_height=100,
        )

        assert len(results) == 1
        assert results[0]["value"]["polygonlabels"] == ["question"]
        assert results[0]["value"]["points"] == [
            [10.0, 10.0],
            [50.0, 10.0],
            [50.0, 50.0],
            [10.0, 50.0],
        ]

    def test_a_run_uploads_one_prediction_per_task(
        self, deps: Deps, tmp_path: Path
    ) -> None:
        """The whole run: unlabeled tasks in, uploaded predictions out."""
        image_path = tmp_path / "page.png"
        Image.new("RGB", (100, 100), color="white").save(image_path)
        deps.client.get_unlabeled_tasks.return_value = [
            self._task(7, f"/data/local-files/?d={quote(str(image_path))}")
        ]
        deps.predictor.predict.return_value = [
            Detection(label="question", polygon=QUAD)
        ]

        predicted = self._predictor(deps, model_version="v3").predict_tasks(
            project_id=1
        )

        assert predicted == 1
        project_id, predictions = deps.client.upload_predictions.call_args.args
        assert project_id == 1
        assert predictions[0]["task"] == 7
        assert predictions[0]["model_version"] == "v3"
        assert predictions[0]["result"][0]["value"]["polygonlabels"] == ["question"]

    def test_a_task_whose_image_is_not_local_is_skipped(self, deps: Deps) -> None:
        deps.client.get_unlabeled_tasks.return_value = [
            self._task(1, "http://example.com/remote.png")
        ]

        assert self._predictor(deps).predict_tasks(project_id=1) == 0
        deps.client.upload_predictions.assert_not_called()

    @pytest.mark.parametrize(
        "image",
        [
            "http://example.com/remote.png",
            "/data/local-files/?d=nonexistent%5Cimage.png",
        ],
        ids=["not-a-local-uri", "file-is-gone"],
    )
    def test_a_task_that_cannot_be_read_predicts_nothing(self, image: str) -> None:
        assert self._predictor()._predict_task(self._task(1, image)) is None
