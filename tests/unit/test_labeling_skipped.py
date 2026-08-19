"""Tests for the sweep that retires the pages an annotator skipped.

``plan`` decides against the disk — the images below are real files under
``tmp_path`` — and ``apply`` is the only part that unlinks or calls the server,
so the two are tested apart. What the guards protect is somebody else's work: an
image is one annotator's skip away from deletion, and the same file can be behind
a second task that nobody skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from digitex.labeling import skipped

if TYPE_CHECKING:
    from pathlib import Path

SKIP: dict[str, Any] = {"was_cancelled": True, "result": []}
DONE: dict[str, Any] = {"was_cancelled": False, "result": [{"type": "polygonlabels"}]}


def _task(task_id: int, image: str | None, *annotations: dict[str, Any]) -> Any:
    """A task as ``list_tasks`` returns it, annotations included."""
    task = MagicMock(id=task_id)
    uri = "" if image is None else f"/data/local-files/?d={quote(image)}"
    task.data = {} if image is None else {"image": uri}
    task.annotations = list(annotations)
    return task


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "page.jpg"
    path.write_bytes(b"jpeg")
    return path


class TestPlan:
    def test_a_skipped_page_loses_its_image_and_its_task(self, image: Path) -> None:
        plan = skipped.plan([_task(1, str(image), SKIP)])

        assert plan.deletions == [skipped.Doomed(1, image)]
        assert plan.kept == []

    def test_a_page_nobody_skipped_is_not_touched(self, image: Path) -> None:
        plan = skipped.plan([_task(1, str(image), DONE)])

        assert plan.deletions == []
        assert plan.cancelled == 0

    def test_a_skip_does_not_delete_a_colleagues_finished_work(
        self, image: Path
    ) -> None:
        """Two completions per task can disagree; ``any`` cancelled used to win.

        One annotator skipping a page another had already annotated took the
        image out from under the annotation that survived.
        """
        plan = skipped.plan([_task(1, str(image), SKIP, DONE)])

        assert plan.deletions == []
        assert plan.kept == [skipped.Kept(1, "also holds a completed annotation")]
        assert image.exists()

    def test_an_image_a_second_task_holds_is_left_alone(self, image: Path) -> None:
        """A moved pool leaves two tasks over one file until repair has run.

        Deleting it for the skipped one blanks the twin that is still live.
        """
        plan = skipped.plan([_task(1, str(image), SKIP), _task(2, str(image))])

        assert plan.deletions == []
        assert "2 tasks hold page.jpg" in plan.kept[0].reason

    def test_a_task_with_no_local_uri_is_reported_not_guessed(self) -> None:
        plan = skipped.plan([_task(1, None, SKIP)])

        assert plan.deletions == []
        assert plan.kept == [skipped.Kept(1, "no local-file URI in the task")]

    def test_a_task_an_earlier_sweep_left_behind_still_goes(
        self, tmp_path: Path
    ) -> None:
        """The rerun that clears up after an image-only sweep.

        The image is already gone, so there is nothing to unlink — the task
        itself is what is left to delete.
        """
        missing = tmp_path / "gone.jpg"

        plan = skipped.plan([_task(1, str(missing), SKIP)])

        assert plan.deletions == [skipped.Doomed(1, None)]
        assert (plan.images, plan.kept) == (0, [])

    def test_the_count_covers_both_verdicts(self, tmp_path: Path, image: Path) -> None:
        other = tmp_path / "annotated.jpg"
        other.write_bytes(b"jpeg")

        plan = skipped.plan(
            [
                _task(1, str(image), SKIP),
                _task(2, str(tmp_path / "gone.jpg"), SKIP),
                _task(3, str(other), DONE),
                _task(4, None, SKIP),
            ]
        )

        assert (len(plan.deletions), plan.images, len(plan.kept)) == (2, 1, 1)
        assert plan.cancelled == 3


class TestApply:
    @pytest.fixture
    def client(self) -> MagicMock:
        return MagicMock()

    def test_it_unlinks_what_the_plan_named_and_deletes_the_task(
        self, client: MagicMock, image: Path
    ) -> None:
        plan = skipped.plan([_task(1, str(image), SKIP)])

        assert skipped.apply(client, plan) == (1, 1)
        assert not image.exists()
        client.delete_task.assert_called_once_with(1)

    def test_a_task_whose_image_is_already_gone_needs_no_unlink(
        self, client: MagicMock
    ) -> None:
        assert skipped.apply(client, skipped.Plan([skipped.Doomed(1, None)])) == (0, 1)
        client.delete_task.assert_called_once_with(1)

    def test_a_page_that_will_not_unlink_keeps_its_task(
        self, client: MagicMock, tmp_path: Path
    ) -> None:
        """The record of the skip outranks tidiness.

        A file that survives comes back on the next sync, and it should come back
        as a page somebody already refused — not as a fresh one.
        """
        locked = tmp_path / "locked"
        locked.mkdir()  # unlink() refuses a directory

        assert skipped.apply(client, skipped.Plan([skipped.Doomed(1, locked)])) == (
            0,
            0,
        )
        assert locked.exists()
        client.delete_task.assert_not_called()

    def test_one_file_that_will_not_go_does_not_strand_the_rest(
        self, client: MagicMock, tmp_path: Path, image: Path
    ) -> None:
        """A locked handle on one page must not keep the sweep from finishing."""
        locked = tmp_path / "locked"
        locked.mkdir()
        plan = skipped.Plan([skipped.Doomed(1, locked), skipped.Doomed(2, image)])

        assert skipped.apply(client, plan) == (1, 1)
        assert not image.exists()
        client.delete_task.assert_called_once_with(2)

    def test_one_task_the_server_will_not_delete_does_not_strand_the_rest(
        self, client: MagicMock, tmp_path: Path, image: Path
    ) -> None:
        other = tmp_path / "second.jpg"
        other.write_bytes(b"jpeg")
        client.delete_task.side_effect = [RuntimeError("500"), None]
        plan = skipped.Plan([skipped.Doomed(1, image), skipped.Doomed(2, other)])

        assert skipped.apply(client, plan) == (2, 1)
        assert not image.exists()
        assert not other.exists()

    def test_a_plan_that_deletes_nothing_touches_nothing(
        self, client: MagicMock, image: Path
    ) -> None:
        assert skipped.apply(client, skipped.Plan()) == (0, 0)
        assert image.exists()
        client.delete_task.assert_not_called()
