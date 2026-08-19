"""Tests for the repair of a project whose images moved out from under it.

The subject is a project holding two tasks over one image: one synced before the
image moved, carrying the annotations and a path that no longer resolves, and one
synced after, resolving but empty. ``plan`` decides against the disk — the files
below are real, under a ``document_root`` of ``tmp_path`` — and ``apply`` is
handed a stubbed client, so what is asserted is the order it works in: the
annotations come back on the live task before the stranded one is deleted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from digitex.labeling import repair

if TYPE_CHECKING:
    from pathlib import Path

POLYGON = [{"from_name": "label", "type": "polygonlabels", "value": {"points": []}}]


def _task(
    task_id: int,
    served: str,
    *,
    annotations: list[dict[str, Any]] | None = None,
    key: str = "image",
) -> Any:
    """A task as ``list_tasks`` returns it, its image URI under ``key``."""
    task = MagicMock(id=task_id, is_labeled=bool(annotations))
    task.data = {key: f"/data/local-files/?d={quote(served)}"}
    task.annotations = annotations or []
    return task


@pytest.fixture
def image(tmp_path: Path) -> Path:
    """The image where it lives now: ``var/pages/page.jpg`` under the root."""
    path = tmp_path / "var" / "pages" / "page.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"jpeg")
    return path


class TestPlan:
    def test_annotations_move_to_the_task_that_resolves(
        self, tmp_path: Path, image: Path
    ) -> None:
        """The stranded task names the path the image had before it moved."""
        stranded = _task(1, "pages/page.jpg", annotations=[{"result": POLYGON}])
        live = _task(2, "var/pages/page.jpg")

        plan = repair.plan([stranded, live], document_root=tmp_path)

        assert plan.deletions == []
        assert plan.skipped == []
        assert len(plan.moves) == 1
        assert (plan.moves[0].stranded_id, plan.moves[0].live_id) == (1, 2)
        assert plan.annotations == 1

    def test_a_task_of_a_storage_sync_is_read_too(
        self, tmp_path: Path, image: Path
    ) -> None:
        """A sync of blob URLs files the image under ``$undefined$``.

        Reading only ``image`` would leave the whole project "nothing to do".
        """
        stranded = _task(
            1, "pages/page.jpg", annotations=[{"result": POLYGON}], key="$undefined$"
        )
        live = _task(2, "var/pages/page.jpg", key="$undefined$")

        plan = repair.plan([stranded, live], document_root=tmp_path)

        assert [(m.stranded_id, m.live_id) for m in plan.moves] == [(1, 2)]

    def test_a_stranded_task_with_no_work_is_deleted_outright(
        self, tmp_path: Path, image: Path
    ) -> None:
        plan = repair.plan(
            [_task(1, "pages/page.jpg"), _task(2, "var/pages/page.jpg")],
            document_root=tmp_path,
        )

        assert plan.moves == []
        assert plan.deletions == [1]

    def test_a_task_whose_image_is_nowhere_is_left_alone(self, tmp_path: Path) -> None:
        """Its annotations have no task to go to, and dropping them is not a fix."""
        plan = repair.plan(
            [_task(1, "pages/gone.jpg", annotations=[{"result": POLYGON}])],
            document_root=tmp_path,
        )

        assert (plan.moves, plan.deletions) == ([], [])
        assert [s.task_id for s in plan.skipped] == [1]

    def test_two_live_tasks_over_one_image_are_left_to_the_operator(
        self, tmp_path: Path, image: Path
    ) -> None:
        """Which annotator's task to keep is not this repair's call."""
        plan = repair.plan(
            [_task(2, "var/pages/page.jpg"), _task(3, "var/pages/page.jpg")],
            document_root=tmp_path,
        )

        assert (plan.moves, plan.deletions) == ([], [])
        assert {s.task_id for s in plan.skipped} == {2, 3}

    def test_a_task_with_no_local_file_uri_is_left_alone(self, tmp_path: Path) -> None:
        task = MagicMock(id=1)
        task.data = {"image": "https://example.com/page.jpg"}
        task.annotations = []

        plan = repair.plan([task], document_root=tmp_path)

        assert [s.task_id for s in plan.skipped] == [1]

    def test_a_project_that_never_moved_has_nothing_to_do(
        self, tmp_path: Path, image: Path
    ) -> None:
        plan = repair.plan([_task(2, "var/pages/page.jpg")], document_root=tmp_path)

        assert (plan.moves, plan.deletions, plan.skipped) == ([], [], [])


class TestApply:
    def test_the_work_lands_before_the_task_that_held_it_goes(self) -> None:
        client = MagicMock()
        calls: list[str] = []
        client.create_annotation.side_effect = lambda *a, **k: calls.append("create")
        client.delete_task.side_effect = lambda *a, **k: calls.append("delete")
        plan = repair.Plan(moves=[repair.Move(1, 2, [{"result": POLYGON}])])

        moved, deleted = repair.apply(cast("Any", client), plan)

        assert (moved, deleted) == (1, 1)
        assert calls == ["create", "delete"]
        client.create_annotation.assert_called_once_with(2, {"result": POLYGON})
        client.delete_task.assert_called_once_with(1)

    def test_a_failed_copy_leaves_the_annotations_where_they_are(self) -> None:
        """Deleting the task would be deleting the only copy of the work."""
        client = MagicMock()
        client.create_annotation.side_effect = RuntimeError("api down")
        plan = repair.Plan(moves=[repair.Move(1, 2, [{"result": POLYGON}])])

        moved, deleted = repair.apply(cast("Any", client), plan)

        assert (moved, deleted) == (0, 0)
        client.delete_task.assert_not_called()

    def test_a_task_with_nothing_on_it_is_deleted_without_a_copy(self) -> None:
        client = MagicMock()

        moved, deleted = repair.apply(cast("Any", client), repair.Plan(deletions=[7]))

        assert (moved, deleted) == (0, 1)
        client.create_annotation.assert_not_called()
        client.delete_task.assert_called_once_with(7)
