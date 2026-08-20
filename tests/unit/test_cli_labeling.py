"""Tests for the labeling CLI's contracts.

What matters here is not the work — :mod:`digitex.labeling.repair` and
:mod:`digitex.labeling.skipped` are tested on their own — but the shell around
it: that a run writes nothing until it is told to, that it archives what it is
about to destroy, and that a missing document root is refused before the server
is called.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from typer.testing import CliRunner

from digitex.studio.cli import labeling

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

SKIP: dict[str, Any] = {"was_cancelled": True, "result": []}


@pytest.fixture(autouse=True)
def _quiet_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the Typer callback from configuring real logging."""
    monkeypatch.setattr(labeling, "setup_logging", lambda _settings: None)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Settings rooted in tmp_path, so an archive lands there and not in var/."""
    fake = MagicMock()
    fake.paths.data_root = tmp_path
    fake.pipeline.label_studio.url = "http://localhost:8080"
    fake.pipeline.label_studio.api_key = "api-key"
    fake.pipeline.label_studio.local_files_document_root = tmp_path
    monkeypatch.setattr(labeling, "get_settings", lambda: fake)
    return fake


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The API adapter, stubbed at the seam every command builds it through."""
    stub = MagicMock()
    monkeypatch.setattr(labeling, "_client", lambda: stub)
    return stub


def _task(task_id: int, image: Path, *annotations: dict[str, Any]) -> Any:
    task = MagicMock(id=task_id)
    task.data = {"image": f"/data/local-files/?d={quote(str(image))}"}
    task.annotations = list(annotations)
    return task


def _archives(tmp_path: Path, stem: str) -> list[Path]:
    return sorted((tmp_path / "label-studio").glob(f"{stem}-*.json"))


class TestDeleteSkippedTasks:
    @pytest.fixture
    def image(self, tmp_path: Path) -> Path:
        path = tmp_path / "page.jpg"
        path.write_bytes(b"jpeg")
        return path

    def test_a_dry_run_is_the_default(
        self, settings: MagicMock, client: MagicMock, image: Path, tmp_path: Path
    ) -> None:
        """Nothing is destroyed by a command someone ran to see what it would do."""
        client.list_tasks.return_value = [_task(1, image, SKIP)]

        result = runner.invoke(
            labeling.app, ["delete-skipped-tasks", "--project-id", "1"]
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert image.exists()
        client.delete_task.assert_not_called()
        assert _archives(tmp_path, "skipped-tasks") == []

    def test_the_image_and_the_task_both_go(
        self, settings: MagicMock, client: MagicMock, image: Path
    ) -> None:
        client.list_tasks.return_value = [_task(1, image, SKIP)]

        result = runner.invoke(
            labeling.app,
            ["delete-skipped-tasks", "--project-id", "1", "--no-dry-run"],
        )

        assert result.exit_code == 0
        assert not image.exists()
        client.delete_task.assert_called_once_with(1)

    def test_the_annotation_is_archived_before_it_goes(
        self, settings: MagicMock, client: MagicMock, image: Path, tmp_path: Path
    ) -> None:
        """The skip is the record of a judgement, and deleting the task ends it.

        Neither ``unlink`` nor the API has an undo, so what the sweep is about to
        destroy is written down first — the annotation included.
        """
        client.list_tasks.return_value = [_task(1, image, SKIP)]

        result = runner.invoke(
            labeling.app,
            ["delete-skipped-tasks", "--project-id", "1", "--no-dry-run"],
        )

        assert result.exit_code == 0
        (archive,) = _archives(tmp_path, "skipped-tasks")
        (entry,) = json.loads(archive.read_text(encoding="utf-8"))
        assert entry["task_id"] == 1
        assert entry["path"] == str(image)
        assert entry["annotations"] == [SKIP]

    def test_a_project_with_no_skips_says_so_and_stops(
        self, settings: MagicMock, client: MagicMock, image: Path
    ) -> None:
        client.list_tasks.return_value = [_task(1, image)]

        result = runner.invoke(
            labeling.app,
            ["delete-skipped-tasks", "--project-id", "1", "--no-dry-run"],
        )

        assert result.exit_code == 0
        assert "no skipped tasks" in result.output
        assert image.exists()
        client.delete_task.assert_not_called()

    def test_a_skip_an_image_only_sweep_left_behind_still_goes(
        self, settings: MagicMock, client: MagicMock, tmp_path: Path
    ) -> None:
        """What the rename is for: the tasks the old command left in the project."""
        client.list_tasks.return_value = [_task(1, tmp_path / "gone.jpg", SKIP)]

        result = runner.invoke(
            labeling.app,
            ["delete-skipped-tasks", "--project-id", "1", "--no-dry-run"],
        )

        assert result.exit_code == 0
        assert "image already gone" in result.output
        client.delete_task.assert_called_once_with(1)
        assert len(_archives(tmp_path, "skipped-tasks")) == 1

    def test_the_command_it_replaces_is_gone(
        self, settings: MagicMock, client: MagicMock
    ) -> None:
        """It was renamed, not kept — a stale habit should fail loudly."""
        result = runner.invoke(
            labeling.app, ["delete-skipped-images", "--project-id", "1"]
        )

        assert result.exit_code != 0
        client.list_tasks.assert_not_called()


class TestFixTaskPaths:
    def test_an_unset_document_root_is_refused(
        self, settings: MagicMock, client: MagicMock
    ) -> None:
        """A task's path means nothing without the root it resolves against."""
        settings.pipeline.label_studio.local_files_document_root = None

        result = runner.invoke(labeling.app, ["fix-task-paths", "--project-id", "1"])

        assert result.exit_code != 0
        assert "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT" in result.output
        client.list_tasks.assert_not_called()

    def test_a_dry_run_is_the_default(
        self, settings: MagicMock, client: MagicMock, tmp_path: Path
    ) -> None:
        live = tmp_path / "page.jpg"
        live.write_bytes(b"jpeg")
        client.list_tasks.return_value = [
            _task(1, tmp_path / "moved" / "page.jpg", {"result": []}),
            _task(2, live),
        ]

        result = runner.invoke(labeling.app, ["fix-task-paths", "--project-id", "1"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        client.create_annotation.assert_not_called()
        client.delete_task.assert_not_called()
        assert _archives(tmp_path, "stranded-tasks") == []

    def test_an_empty_project_stops_before_planning(
        self, settings: MagicMock, client: MagicMock
    ) -> None:
        client.list_tasks.return_value = []

        result = runner.invoke(labeling.app, ["fix-task-paths", "--project-id", "1"])

        assert result.exit_code == 0
        assert "no tasks" in result.output


class TestPredict:
    def test_the_model_file_names_the_prediction_version(
        self, settings: MagicMock, client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uploaded predictions are grouped by version, so the weights name one."""
        captured: dict[str, Any] = {}

        class _Recorder:
            def __init__(self, predictor: Any, ls_client: Any, model_version: str):
                captured["model_version"] = model_version

            def predict_tasks(self, project_id: int) -> int:
                captured["project_id"] = project_id
                return 3

        monkeypatch.setattr(
            "digitex.labeling.predictor.TaskPredictor", _Recorder, raising=True
        )
        monkeypatch.setattr(
            "digitex.ml.predictors.YOLO_SegmentationPredictor",
            lambda *a, **k: MagicMock(),
            raising=True,
        )

        result = runner.invoke(
            labeling.app,
            [
                "predict",
                "--project-id",
                "7",
                "--model-path",
                "var/models/page.pt",
            ],
        )

        assert result.exit_code == 0
        assert captured == {"model_version": "page", "project_id": 7}
        assert "Predicted 3 tasks" in result.output
