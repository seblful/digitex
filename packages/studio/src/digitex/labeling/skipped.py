"""Retiring the pages an annotator refused, task and image both.

Clicking "Skip" in Label Studio does not reject the page. It files a cancelled
annotation against the task and leaves both where they are: the image stays in
the pool the local-files storage syncs from, so the page returns as a task on
the next sync, and it stays in the pool a training dataset is built out of.
Deleting the file is what actually retires the page; deleting the task is what
keeps the project from filling up with skips nobody will look at again.

Two kinds of cancelled task are left alone — one that also carries an annotation
somebody completed, and one whose image a second task holds. Both are somebody
else's work standing behind the same file, and the second is
:mod:`digitex.labeling.repair`'s to resolve first.

Deciding is split from deleting, the same way it is in ``repair``: :func:`plan`
reads the project and says which tasks it would retire and which it would not
touch, and :meth:`Plan.apply` is the only part that unlinks or calls the server.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from digitex.labeling.sweeps import LIST_PREVIEW, LeftAlone, image_on_disk, preview
from digitex.labeling.uris import task_image_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from digitex.labeling.client import LabelStudioClient, LabelStudioTask

logger = structlog.get_logger()


@dataclass(frozen=True)
class Doomed:
    """A skipped task to delete, and the image that goes with it.

    ``path`` is ``None`` when there is nothing left to unlink — a sweep that has
    already taken the image leaves the task behind it, and the rerun is what
    clears that up.
    """

    task_id: int
    path: Path | None


@dataclass
class Plan:
    """What a run would delete, decided before any of it is deleted.

    The CLI drives it through :class:`digitex.labeling.sweeps.SweepPlan`, the
    same way it drives ``repair``'s.
    """

    deletions: list[Doomed] = field(default_factory=list)
    kept: list[LeftAlone] = field(default_factory=list)

    @property
    def cancelled(self) -> int:
        """How many cancelled tasks the plan looked at."""
        return len(self.deletions) + len(self.kept)

    @property
    def images(self) -> int:
        """How many of the doomed tasks still have an image to unlink."""
        return sum(1 for doomed in self.deletions if doomed.path is not None)

    @property
    def empty(self) -> bool:
        """Whether there is nothing to delete."""
        return not self.deletions

    def report(self, total: int) -> str:
        """The plan as the operator reads it before deciding to apply it."""

        def line(doomed: Doomed) -> str:
            gone = doomed.path if doomed.path is not None else "image already gone"
            return f"task {doomed.task_id}: {gone}"

        lines = [
            f"\n{total} tasks in the project, {self.cancelled} of them skipped.",
            f"  delete:      {len(self.deletions)} tasks, {self.images} with an image",
            f"  leave alone: {len(self.kept)} tasks",
            *preview(self.deletions, LIST_PREVIEW, line, "to delete"),
            *preview(
                self.kept,
                LIST_PREVIEW,
                lambda kept: f"task {kept.task_id}: {kept.reason}",
                "left alone",
            ),
        ]
        if self.empty:
            lines.append("\nNothing to delete.")
        return "\n".join(lines)

    def doomed(self, tasks: Sequence[LabelStudioTask]) -> list[dict[str, object]]:
        """Every task the sweep will delete, its image and annotations with it.

        The undo archive's payload. The cancelled annotation is the only record
        that anybody judged the page, and it goes when the task does, so it is
        written down first.
        """
        paths = {doomed.task_id: doomed.path for doomed in self.deletions}
        return [
            {
                "task_id": task.id,
                "path": paths[task.id],
                "data": task.data,
                "annotations": list(task.annotations or []),
            }
            for task in tasks
            if task.id in paths
        ]

    def apply(self, client: LabelStudioClient) -> tuple[int, int]:
        """Unlink every image the plan named, and delete the task behind it.

        The image goes first and the task only after it: a page whose file
        survives comes back on the next sync, and if its task went with the
        attempt, the record of the skip went too — so the page returns as one
        nobody has ever seen. One locked handle must not strand the rest of the
        sweep either, so a failure of either step is logged and passed over.

        Args:
            client: Label Studio API adapter.

        Returns:
            How many images were unlinked, and how many tasks were deleted.
        """
        unlinked = 0
        deleted = 0

        for doomed in self.deletions:
            if doomed.path is not None:
                try:
                    doomed.path.unlink()
                except Exception as e:
                    # The task stays: the record of the skip outranks tidiness.
                    logger.error(
                        "delete_failed",
                        task_id=doomed.task_id,
                        path=str(doomed.path),
                        error=str(e),
                    )
                    continue
                unlinked += 1
                logger.info(
                    "deleted_file", task_id=doomed.task_id, path=str(doomed.path)
                )

            try:
                client.delete_task(doomed.task_id)
            except Exception as e:
                logger.error("task_delete_failed", task_id=doomed.task_id, error=str(e))
                continue
            deleted += 1

        logger.info("skipped_sweep_complete", images=unlinked, tasks=deleted)
        return unlinked, deleted


def _was_cancelled(annotations: Iterable[dict[str, Any]]) -> bool:
    """Whether anybody skipped the page these annotations belong to."""
    return any(annotation.get("was_cancelled", False) for annotation in annotations)


def plan(tasks: Sequence[LabelStudioTask], *, document_root: Path) -> Plan:
    """Decide which skipped pages to retire.

    A cancelled task is kept in three cases: the task also carries an annotation
    somebody actually completed, another task in the project points at the same
    file, or the task names no local file at all. The first two are what stops
    an annotator's work being thrown out with a colleague's skip — a project
    with two completions per task can hold both verdicts at once, and a pool
    that has moved holds two tasks over one image until
    :mod:`digitex.labeling.repair` has run. The third is a task whose image
    cannot be found to check, so it is reported rather than guessed at.

    A cancelled task whose file is already gone is not kept: it is a page an
    earlier sweep retired and left a task behind for, so the task is deleted and
    there is no image to unlink.

    Args:
        tasks: Every task of the project, annotations included.
        document_root: Directory the server resolves a local-file URI against,
            which is what decides whether there is still an image to unlink.

    Returns:
        The tasks to delete with their images, and every cancelled task left
        alone with a reason.
    """
    # Parsed once and carried: the paths decide both who holds what and what
    # each cancelled task would lose.
    resolved = [(task, task_image_path(task.data)) for task in tasks]

    # Keyed casefolded, because the same file reached through two tasks can be
    # spelled with different case on the Windows host that indexed it.
    holders: dict[str, list[int]] = collections.defaultdict(list)
    for task, path in resolved:
        if path is not None:
            holders[str(path).casefold()].append(task.id)

    result = Plan()
    for task, path in resolved:
        annotations = list(task.annotations or [])
        if not _was_cancelled(annotations):
            continue

        if any(not a.get("was_cancelled", False) for a in annotations):
            result.kept.append(LeftAlone(task.id, "also holds a completed annotation"))
        elif path is None:
            result.kept.append(LeftAlone(task.id, "no local-file URI in the task"))
        elif len(others := holders[str(path).casefold()]) > 1:
            result.kept.append(
                LeftAlone(task.id, f"{len(others)} tasks hold {path.name}: {others}")
            )
        else:
            # A file that is already gone leaves only the task to delete.
            result.deletions.append(Doomed(task.id, image_on_disk(path, document_root)))

    logger.info(
        "planned_skipped_sweep",
        cancelled=result.cancelled,
        deletions=len(result.deletions),
        images=result.images,
        kept=len(result.kept),
    )
    return result
