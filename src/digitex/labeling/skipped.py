"""Retiring the pages an annotator refused, task and image both.

Clicking "Skip" in Label Studio does not reject the page — it files a cancelled
annotation against the task and leaves both where they are. The image stays in
the pool the local-files storage syncs from, so it comes back as a task on the
next sync, and stays in the pool a training dataset is built out of. Deleting
the file is what actually retires the page; deleting the task is what stops the
project filling up with skips nobody will look at again.

The image goes first and the task only after it, because a page whose file
survives comes back on the next sync — and if its task went, the record of the
skip went with it, so the page comes back as one nobody has ever seen.

Two kinds of cancelled task are left alone: one that also carries an annotation
somebody completed, and one whose image a second task holds. Both are somebody
else's work standing behind the same file, and the second is
:mod:`digitex.labeling.repair`'s to resolve first.

Deciding is split from deleting, the same way it is in ``repair``: :func:`plan`
reads the project and says which tasks it would retire and which it would not
touch, and :func:`apply` is the only part that unlinks or calls the server.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

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


@dataclass(frozen=True)
class Kept:
    """A cancelled task that is not deleted, and why."""

    task_id: int
    reason: str


@dataclass
class Plan:
    """What a run would delete, decided before any of it is deleted."""

    deletions: list[Doomed] = field(default_factory=list)
    kept: list[Kept] = field(default_factory=list)

    @property
    def cancelled(self) -> int:
        """How many cancelled tasks the plan looked at."""
        return len(self.deletions) + len(self.kept)

    @property
    def images(self) -> int:
        """How many of the doomed tasks still have an image to unlink."""
        return sum(1 for doomed in self.deletions if doomed.path is not None)


def _was_cancelled(annotations: Iterable[dict[str, Any]]) -> bool:
    return any(annotation.get("was_cancelled", False) for annotation in annotations)


def plan(tasks: Sequence[LabelStudioTask]) -> Plan:
    """Decide which skipped pages to retire.

    A cancelled task is kept in three cases: the task also carries an annotation
    somebody actually completed, another task in the project points at the same
    file, or the task names no local file at all. The first two are what stops
    an annotator's work being thrown out with a colleague's skip — a project with
    two completions per task can hold both verdicts at once, and a pool that has
    moved holds two tasks over one image until :mod:`digitex.labeling.repair` has
    run. The third is a task whose image cannot be found to check, so it is
    reported rather than guessed at.

    A cancelled task whose file is already gone is not kept: it is a page an
    earlier sweep retired and left a task behind for, so the task is deleted and
    there is no image to unlink.

    Args:
        tasks: Every task of the project, annotations included.

    Returns:
        The tasks to delete with their images, and every cancelled task left
        alone with a reason.
    """
    result = Plan()
    holders: dict[str, list[int]] = collections.defaultdict(list)
    for task in tasks:
        path = task_image_path(task.data)
        if path is not None:
            holders[str(path).casefold()].append(task.id)

    for task in tasks:
        annotations = list(task.annotations or [])
        if not _was_cancelled(annotations):
            continue

        if any(not a.get("was_cancelled", False) for a in annotations):
            result.kept.append(Kept(task.id, "also holds a completed annotation"))
            continue

        path = task_image_path(task.data)
        if path is None:
            result.kept.append(Kept(task.id, "no local-file URI in the task"))
        elif len(others := holders[str(path).casefold()]) > 1:
            result.kept.append(
                Kept(task.id, f"{len(others)} tasks hold {path.name}: {others}")
            )
        else:
            result.deletions.append(Doomed(task.id, path if path.exists() else None))

    logger.info(
        "planned_skipped_sweep",
        cancelled=result.cancelled,
        deletions=len(result.deletions),
        images=result.images,
        kept=len(result.kept),
    )
    return result


def apply(client: LabelStudioClient, plan: Plan) -> tuple[int, int]:
    """Unlink every image the plan named, and delete the task behind it.

    A task whose image will not delete is left where it is: the file surviving
    means the page returns on the next sync, and it should return as one that has
    already been skipped rather than as a fresh page. One locked handle must not
    strand the rest of the sweep either, so a failure of either step is logged
    and passed over.

    Args:
        client: Label Studio API adapter.
        plan: What :func:`plan` decided.

    Returns:
        How many images were unlinked, and how many tasks were deleted.
    """
    images = 0
    tasks = 0

    for doomed in plan.deletions:
        if doomed.path is not None:
            try:
                doomed.path.unlink()
            except Exception as e:
                logger.error(
                    "delete_failed",
                    task_id=doomed.task_id,
                    path=str(doomed.path),
                    error=str(e),
                )
                continue
            images += 1
            logger.info("deleted_file", task_id=doomed.task_id, path=str(doomed.path))

        try:
            client.delete_task(doomed.task_id)
        except Exception as e:
            logger.error("task_delete_failed", task_id=doomed.task_id, error=str(e))
            continue
        tasks += 1

    logger.info("skipped_sweep_complete", images=images, tasks=tasks)
    return images, tasks
