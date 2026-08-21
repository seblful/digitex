"""Tests for the labeling CLI's contracts.

What matters here is not the work — :mod:`digitex.labeling.repair`,
:mod:`digitex.labeling.skipped` and :mod:`digitex.labeling.transfer` are tested
on their own — but the shell around it: that a run writes nothing until it is
told to, that it archives what it is about to destroy, and that a missing
document root is refused before the server is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from typer.testing import CliRunner

from digitex.studio.cli import labeling

runner = CliRunner()

SKIP: dict[str, Any] = {"was_cancelled": True, "result": []}

POLYGONS: list[dict[str, Any]] = [
    {
        "from_name": "label",
        "to_name": "image",
        "type": "polygonlabels",
        "value": {
            "points": [[10.0, 10.0], [90.0, 10.0], [90.0, 80.0], [10.0, 80.0]],
            "polygonlabels": ["question"],
        },
    }
]


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


class TestCopyAligned:
    @pytest.fixture
    def image(self, tmp_path: Path) -> Path:
        from PIL import Image

        path = tmp_path / "page.png"
        Image.new("L", (200, 120), 255).save(path)
        return path

    def _projects(
        self, client: MagicMock, source: list[Any], target: list[Any]
    ) -> None:
        """``list_tasks`` answers per project, as the command calls it twice."""
        client.list_tasks.side_effect = lambda project_id: (
            source if project_id == 1 else target
        )

    def test_a_dry_run_is_the_default(
        self, settings: MagicMock, client: MagicMock, image: Path
    ) -> None:
        """Nothing is written by a command someone ran to see what it would do."""
        self._projects(client, [_task(1, image, {"result": POLYGONS})], [])

        result = runner.invoke(
            labeling.app,
            ["copy-aligned", "--from-project", "1", "--to-project", "2"],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        client.create_task.assert_not_called()
        client.create_annotation.assert_not_called()

    def test_the_page_is_carried_once_told_to(
        self, settings: MagicMock, client: MagicMock, image: Path
    ) -> None:
        self._projects(client, [_task(1, image, {"result": POLYGONS})], [])
        client.create_task.return_value = 42

        result = runner.invoke(
            labeling.app,
            [
                "copy-aligned",
                "--from-project",
                "1",
                "--to-project",
                "2",
                "--no-dry-run",
            ],
        )

        assert result.exit_code == 0
        assert client.create_task.call_args.args[0] == 2
        assert client.create_annotation.call_args.args[0] == 42

    def test_a_second_run_over_a_finished_project_writes_nothing(
        self, settings: MagicMock, client: MagicMock, image: Path
    ) -> None:
        """The reason the command exists in this shape: rerunning is cheap."""
        carried = _task(9, image, {"result": POLYGONS})
        self._projects(client, [_task(1, image, {"result": POLYGONS})], [carried])

        result = runner.invoke(
            labeling.app,
            [
                "copy-aligned",
                "--from-project",
                "1",
                "--to-project",
                "2",
                "--no-dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Nothing new to carry across" in result.output
        client.create_task.assert_not_called()
        client.create_annotation.assert_not_called()

    def test_copying_a_project_onto_itself_is_refused(
        self, settings: MagicMock, client: MagicMock
    ) -> None:
        """It would stack a second annotation on every task it read."""
        result = runner.invoke(
            labeling.app,
            ["copy-aligned", "--from-project", "1", "--to-project", "1"],
        )

        assert result.exit_code != 0
        client.list_tasks.assert_not_called()

    def test_a_missing_document_root_is_refused_before_the_server_is_called(
        self, settings: MagicMock, client: MagicMock
    ) -> None:
        settings.pipeline.label_studio.local_files_document_root = None

        result = runner.invoke(
            labeling.app,
            ["copy-aligned", "--from-project", "1", "--to-project", "2"],
        )

        assert result.exit_code != 0
        client.list_tasks.assert_not_called()

    def test_a_limit_caps_what_a_trial_run_touches(
        self, settings: MagicMock, client: MagicMock, image: Path, tmp_path: Path
    ) -> None:
        from PIL import Image as Pillow

        second = tmp_path / "second.png"
        Pillow.new("L", (200, 120), 255).save(second)
        self._projects(
            client,
            [
                _task(1, image, {"result": POLYGONS}),
                _task(2, second, {"result": POLYGONS}),
            ],
            [],
        )
        client.create_task.return_value = 42

        result = runner.invoke(
            labeling.app,
            [
                "copy-aligned",
                "--from-project",
                "1",
                "--to-project",
                "2",
                "--limit",
                "1",
                "--no-dry-run",
            ],
        )

        assert result.exit_code == 0
        assert client.create_task.call_count == 1


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

    def test_a_run_with_every_skip_kept_stops_before_the_archive(
        self, settings: MagicMock, client: MagicMock, image: Path, tmp_path: Path
    ) -> None:
        """An empty plan ends the run at the report, even past the dry run."""
        done = {"was_cancelled": False, "result": [{"type": "polygonlabels"}]}
        client.list_tasks.return_value = [_task(1, image, SKIP, done)]

        result = runner.invoke(
            labeling.app,
            ["delete-skipped-tasks", "--project-id", "1", "--no-dry-run"],
        )

        assert result.exit_code == 0
        assert "Nothing to delete." in result.output
        assert image.exists()
        client.delete_task.assert_not_called()
        assert _archives(tmp_path, "skipped-tasks") == []

    def test_an_unset_document_root_is_refused(
        self, settings: MagicMock, client: MagicMock
    ) -> None:
        """The sweep unlinks what a URI resolves to, and that needs the root."""
        settings.pipeline.label_studio.local_files_document_root = None

        result = runner.invoke(
            labeling.app, ["delete-skipped-tasks", "--project-id", "1"]
        )

        assert result.exit_code != 0
        assert "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT" in result.output
        client.list_tasks.assert_not_called()

    def test_a_relative_uri_is_resolved_against_the_document_root(
        self, settings: MagicMock, client: MagicMock, tmp_path: Path
    ) -> None:
        """A local-files URI names its path relative to the server's root.

        Read as absolute, every such image counted as already gone: the task
        went, the file stayed, and the page synced straight back in as one
        nobody had ever judged.
        """
        image = tmp_path / "pages" / "page.jpg"
        image.parent.mkdir()
        image.write_bytes(b"jpeg")
        client.list_tasks.return_value = [_task(1, Path("pages") / "page.jpg", SKIP)]

        result = runner.invoke(
            labeling.app,
            ["delete-skipped-tasks", "--project-id", "1", "--no-dry-run"],
        )

        assert result.exit_code == 0
        assert not image.exists()
        client.delete_task.assert_called_once_with(1)


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

    def test_a_project_that_needs_nothing_stops_before_the_archive(
        self, settings: MagicMock, client: MagicMock, tmp_path: Path
    ) -> None:
        """An empty plan ends the run at the report, even past the dry run."""
        live = tmp_path / "page.jpg"
        live.write_bytes(b"jpeg")
        client.list_tasks.return_value = [_task(1, live)]

        result = runner.invoke(
            labeling.app, ["fix-task-paths", "--project-id", "1", "--no-dry-run"]
        )

        assert result.exit_code == 0
        assert "Nothing to repair." in result.output
        client.delete_task.assert_not_called()
        assert _archives(tmp_path, "stranded-tasks") == []

    def test_the_annotations_are_archived_before_they_go(
        self, settings: MagicMock, client: MagicMock, tmp_path: Path
    ) -> None:
        """The archive holds exactly the tasks apply deletes, work included.

        The annotations come back as new records and the stranded tasks go, so
        the dump is the only trace of what they were. It comes off the plan
        itself, not a CLI guess at what apply deletes.
        """
        for name in ("a.jpg", "b.jpg"):
            path = tmp_path / "var" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"jpeg")
        annotation = {"result": [{"type": "polygonlabels"}]}
        client.list_tasks.return_value = [
            _task(1, tmp_path / "old" / "a.jpg", annotation),  # moved, then deleted
            _task(2, tmp_path / "old" / "b.jpg"),  # deleted outright
            _task(10, tmp_path / "var" / "a.jpg"),
            _task(20, tmp_path / "var" / "b.jpg"),
        ]

        result = runner.invoke(
            labeling.app, ["fix-task-paths", "--project-id", "1", "--no-dry-run"]
        )

        assert result.exit_code == 0
        (archive,) = _archives(tmp_path, "stranded-tasks")
        entries = {e["id"]: e for e in json.loads(archive.read_text(encoding="utf-8"))}
        deleted = {call.args[0] for call in client.delete_task.call_args_list}
        assert set(entries) == deleted == {1, 2}
        assert entries[1]["annotations"] == [annotation]

    def test_the_archive_lands_before_the_first_delete(
        self, settings: MagicMock, client: MagicMock, tmp_path: Path
    ) -> None:
        """An apply that dies half-way must still leave the undo record behind."""
        live = tmp_path / "page.jpg"
        live.write_bytes(b"jpeg")
        client.list_tasks.return_value = [
            _task(1, tmp_path / "moved" / "page.jpg"),
            _task(2, live),
        ]
        client.delete_task.side_effect = RuntimeError("500")

        result = runner.invoke(
            labeling.app, ["fix-task-paths", "--project-id", "1", "--no-dry-run"]
        )

        assert result.exit_code != 0
        assert len(_archives(tmp_path, "stranded-tasks")) == 1

    def test_the_outcome_is_replanned_against_the_server(
        self, settings: MagicMock, client: MagicMock, tmp_path: Path
    ) -> None:
        """The 'still stranded' count comes off the server, not the plan.

        A partial failure leaves tasks stranded, and the operator needs the
        real number — the plan's own tally would always say zero remain.
        """
        live = tmp_path / "page.jpg"
        live.write_bytes(b"jpeg")
        stranded = _task(1, tmp_path / "moved" / "page.jpg", {"result": []})
        survivor = _task(2, live)
        client.list_tasks.side_effect = [[stranded, survivor], [stranded, survivor]]
        client.create_annotation.side_effect = RuntimeError("api down")

        result = runner.invoke(
            labeling.app, ["fix-task-paths", "--project-id", "1", "--no-dry-run"]
        )

        assert result.exit_code == 0
        assert client.list_tasks.call_count == 2
        assert "1 still stranded" in result.output


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
