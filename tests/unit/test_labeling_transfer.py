"""Tests for carrying a project's annotations into another, snapped to the print.

The subject is two projects over one image pool. What matters and is asserted
here is which pages a second run picks up — the command's whole reason for
existing is that it can be rerun over a project it already half-populated — and
that the writes go in the order that makes that true: the task first, its
annotation last, because the annotation is what marks the page done.

The aligner is stubbed throughout. Whether an outline comes out snapped to the
print is :mod:`tests.unit.test_imaging_outlines`' subject; here it would only
mean opening real images to assert something already asserted elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from digitex.labeling import transfer

if TYPE_CHECKING:
    from pathlib import Path

POLYGON: dict[str, Any] = {
    "from_name": "label",
    "to_name": "image",
    "type": "polygonlabels",
    "value": {
        "points": [[10.0, 10.0], [50.0, 10.0], [50.0, 40.0]],
        "polygonlabels": ["question"],
    },
}


def _task(
    task_id: int,
    served: str,
    *,
    annotations: list[dict[str, Any]] | None = None,
    key: str = "image",
) -> Any:
    """A task as ``list_tasks`` returns it, its image URI under *key*."""
    task = MagicMock(id=task_id, is_labeled=bool(annotations))
    task.data = {key: f"/data/local-files/?d={quote(served)}"}
    task.annotations = annotations or []
    return task


def _annotated(task_id: int, served: str, **extra: Any) -> Any:
    return _task(task_id, served, annotations=[{"result": [POLYGON], **extra}])


@pytest.fixture
def page(tmp_path: Path) -> Path:
    """The image both projects point at: ``pages/page.jpg`` under the root."""
    path = tmp_path / "pages" / "page.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"jpeg")
    return path


@pytest.fixture
def aligner():
    """A stub aligner: reports one outline rebuilt, opens nothing."""

    def align(carry: transfer.Carry) -> tuple[list[dict[str, Any]], int, list[str]]:
        return carry.annotation["result"], 1, []

    return align


class TestPlan:
    def test_an_annotated_page_missing_from_the_target_is_carried(
        self, tmp_path: Path, page: Path
    ) -> None:
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")], [], document_root=tmp_path
        )

        assert plan.skipped == []
        assert len(plan.carries) == 1
        assert plan.carries[0].source_id == 1
        # Nothing in the target holds it, so the run has to import the page.
        assert plan.carries[0].target_id is None
        assert plan.creating == 1

    def test_a_page_the_target_already_has_an_annotation_for_is_left(
        self, tmp_path: Path, page: Path
    ) -> None:
        """The whole point: a second run does the pages the first one did not."""
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")],
            [_annotated(9, "pages/page.jpg")],
            document_root=tmp_path,
        )

        assert plan.carries == []
        assert plan.empty
        assert [left.reason for left in plan.skipped] == ["already carried across"]

    def test_a_target_task_without_an_annotation_is_used_rather_than_a_new_one(
        self, tmp_path: Path, page: Path
    ) -> None:
        """A storage sync got there first, so the page must not be imported twice.

        Importing it again is precisely the duplicate ``fix-task-paths`` exists
        to clean up afterwards.
        """
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")],
            [_task(9, "pages/page.jpg")],
            document_root=tmp_path,
        )

        assert len(plan.carries) == 1
        assert plan.carries[0].target_id == 9
        assert plan.creating == 0

    def test_the_pool_moving_between_projects_does_not_reprocess_the_page(
        self, tmp_path: Path, page: Path
    ) -> None:
        """Matched on filename, so two storages spelling one page agree."""
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")],
            [_annotated(9, "var/elsewhere/page.jpg")],
            document_root=tmp_path,
        )

        assert plan.carries == []

    def test_a_page_not_on_this_machine_is_left_alone(self, tmp_path: Path) -> None:
        plan = transfer.plan(
            [_annotated(1, "pages/absent.jpg")], [], document_root=tmp_path
        )

        assert plan.carries == []
        assert [left.reason for left in plan.skipped] == ["page is not on this machine"]

    def test_a_task_with_no_local_file_uri_is_left_alone(self, tmp_path: Path) -> None:
        task = MagicMock(id=1, is_labeled=True)
        task.data = {"image": "https://example.com/page.jpg"}
        task.annotations = [{"result": [POLYGON]}]

        plan = transfer.plan([task], [], document_root=tmp_path)

        assert [left.reason for left in plan.skipped] == [
            "no local-file URI in the task"
        ]

    @pytest.mark.parametrize(
        ("annotations", "reason"),
        [
            ([], "no usable annotation to carry"),
            ([{"result": [], "was_cancelled": False}], "no usable annotation to carry"),
            (
                [{"result": [POLYGON], "was_cancelled": True}],
                "no usable annotation to carry",
            ),
        ],
        ids=["none", "no polygon", "skipped by the annotator"],
    )
    def test_there_has_to_be_an_outline_worth_carrying(
        self, tmp_path: Path, page: Path, annotations: list[dict[str, Any]], reason: str
    ) -> None:
        plan = transfer.plan(
            [_task(1, "pages/page.jpg", annotations=annotations)],
            [],
            document_root=tmp_path,
        )

        assert plan.carries == []
        assert [left.reason for left in plan.skipped] == [reason]

    def test_the_freshest_annotation_is_the_one_carried(
        self, tmp_path: Path, page: Path
    ) -> None:
        """Label Studio appends, so the last is the annotator's latest word."""
        older = {"result": [POLYGON], "lead_time": 11.0}
        newer = {"result": [POLYGON], "lead_time": 22.0}
        plan = transfer.plan(
            [_task(1, "pages/page.jpg", annotations=[older, newer])],
            [],
            document_root=tmp_path,
        )

        assert plan.carries[0].annotation["lead_time"] == 22.0

    def test_the_carried_task_records_where_it_came_from(
        self, tmp_path: Path, page: Path
    ) -> None:
        plan = transfer.plan(
            [_annotated(7, "pages/page.jpg")], [], document_root=tmp_path
        )

        assert plan.carries[0].data[transfer.SOURCE_KEY] == 7
        # And still names the same image, or the target would show a broken page.
        assert "page.jpg" in plan.carries[0].data["image"]

    def test_the_report_says_what_it_would_do(self, tmp_path: Path, page: Path) -> None:
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg"), _annotated(2, "pages/absent.jpg")],
            [],
            document_root=tmp_path,
        )

        report = plan.report(total=2)

        assert "2 tasks in the source project" in report
        assert "1 pages" in report
        assert "page is not on this machine" in report


class TestApply:
    def test_the_task_is_created_before_its_annotation(
        self, tmp_path: Path, page: Path, aligner
    ) -> None:
        """Ordering is what makes an interrupted run resumable.

        A task written without its annotation is picked up again next run; an
        annotation cannot be written before there is a task to hang it on.
        """
        client = MagicMock()
        client.create_task.return_value = 42
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")], [], document_root=tmp_path
        )

        pages, regions = plan.apply(client, project_id=5, aligner=aligner)

        assert (pages, regions) == (1, 1)
        client.create_task.assert_called_once()
        assert client.create_task.call_args.args[0] == 5
        client.create_annotation.assert_called_once()
        assert client.create_annotation.call_args.args[0] == 42

    def test_an_existing_target_task_gets_the_annotation_and_no_new_task(
        self, tmp_path: Path, page: Path, aligner
    ) -> None:
        client = MagicMock()
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")],
            [_task(9, "pages/page.jpg")],
            document_root=tmp_path,
        )

        plan.apply(client, project_id=5, aligner=aligner)

        client.create_task.assert_not_called()
        assert client.create_annotation.call_args.args[0] == 9

    def test_the_source_project_is_never_written_to(
        self, tmp_path: Path, page: Path, aligner
    ) -> None:
        client = MagicMock()
        client.create_task.return_value = 42
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")], [], document_root=tmp_path
        )

        plan.apply(client, project_id=5, aligner=aligner)

        client.delete_task.assert_not_called()
        # Every write named the destination project, never the source.
        assert all(call.args[0] == 5 for call in client.create_task.call_args_list)

    def test_a_page_whose_alignment_fails_does_not_end_the_run(
        self, tmp_path: Path, page: Path
    ) -> None:
        other = tmp_path / "pages" / "second.jpg"
        other.write_bytes(b"jpeg")

        def explode(carry: transfer.Carry):
            if carry.source_id == 1:
                raise ValueError("unreadable page")
            return carry.annotation["result"], 2, []

        client = MagicMock()
        client.create_task.return_value = 42
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg"), _annotated(2, "pages/second.jpg")],
            [],
            document_root=tmp_path,
        )

        pages, regions = plan.apply(client, project_id=5, aligner=explode)

        assert (pages, regions) == (1, 2)
        # Nothing was written for the page that failed.
        assert client.create_task.call_count == 1

    def test_a_failed_write_leaves_the_page_for_the_next_run(
        self, tmp_path: Path, page: Path, aligner
    ) -> None:
        client = MagicMock()
        client.create_task.side_effect = RuntimeError("server said no")
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")], [], document_root=tmp_path
        )

        pages, regions = plan.apply(client, project_id=5, aligner=aligner)

        assert (pages, regions) == (0, 0)
        client.create_annotation.assert_not_called()

    def test_the_aligned_points_are_what_gets_written(
        self, tmp_path: Path, page: Path
    ) -> None:
        moved = [{**POLYGON, "value": {**POLYGON["value"], "points": [[1.0, 2.0]]}}]

        client = MagicMock()
        client.create_task.return_value = 42
        plan = transfer.plan(
            [_annotated(1, "pages/page.jpg")], [], document_root=tmp_path
        )

        plan.apply(client, project_id=5, aligner=lambda carry: (moved, 1, []))

        written = client.create_annotation.call_args.args[1]
        assert written["result"] == moved


class TestAligner:
    def test_only_the_points_move(self, tmp_path: Path) -> None:
        """Everything else on a region belongs to the destination's label config.

        Rewriting ``from_name`` or dropping the region's id would make the
        carried annotation something the target project cannot render.
        """
        from PIL import Image

        image = tmp_path / "page.png"
        Image.new("L", (200, 120), 255).save(image)

        original = {
            "id": "abc",
            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels",
            "value": {
                "points": [[10.0, 10.0], [90.0, 10.0], [90.0, 80.0], [10.0, 80.0]],
                "polygonlabels": ["question"],
            },
        }
        carry = transfer.Carry(
            source_id=1,
            image=image,
            data={},
            annotation={"result": [original]},
            target_id=None,
        )

        results, _, _ = transfer.Aligner()(carry)

        assert results[0]["id"] == "abc"
        assert results[0]["from_name"] == "label"
        assert results[0]["value"]["polygonlabels"] == ["question"]
        # The source annotation itself is untouched -- it is still in the project.
        assert original["value"]["points"][0] == [10.0, 10.0]

    def test_a_result_that_is_not_an_outline_travels_untouched(
        self, tmp_path: Path
    ) -> None:
        from PIL import Image

        image = tmp_path / "page.png"
        Image.new("L", (200, 120), 255).save(image)

        choice = {
            "from_name": "quality",
            "type": "choices",
            "value": {"choices": ["good"]},
        }
        carry = transfer.Carry(
            source_id=1,
            image=image,
            data={},
            annotation={"result": [choice]},
            target_id=None,
        )

        results, changed, unchanged = transfer.Aligner()(carry)

        assert results == [choice]
        assert (changed, unchanged) == (0, [])
