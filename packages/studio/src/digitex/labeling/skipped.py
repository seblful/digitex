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
    """Whether anybody skipped the page these annotations belong to."""
    return any(annotation.get("was_cancelled", False) for annotation in annotations)


def plan(tasks: Sequence[LabelStudioTask]) -> Plan:
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
            result.kept.append(Kept(task.id, "also holds a completed annotation"))
        elif path is None:
            result.kept.append(Kept(task.id, "no local-file URI in the task"))
        elif len(others := holders[str(path).casefold()]) > 1:
            result.kept.append(
                Kept(task.id, f"{len(others)} tasks hold {path.name}: {others}")
            )
        else:
            # A file that is already gone leaves only the task to delete.
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

    The image goes first and the task only after it: a page whose file survives
    comes back on the next sync, and if its task went with the attempt, the
    record of the skip went too — so the page returns as one nobody has ever
    seen. One locked handle must not strand the rest of the sweep either, so a
    failure of either step is logged and passed over.

    Args:
        client: Label Studio API adapter.
        plan: What :func:`plan` decided.

    Returns:
        How many images were unlinked, and how many tasks were deleted.
    """
    unlinked = 0
    deleted = 0

    for doomed in plan.deletions:
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
            logger.info("deleted_file", task_id=doomed.task_id, path=str(doomed.path))

        try:
            client.delete_task(doomed.task_id)
        except Exception as e:
            logger.error("task_delete_failed", task_id=doomed.task_id, error=str(e))
            continue
        deleted += 1

    logger.info("skipped_sweep_complete", images=unlinked, tasks=deleted)
    return unlinked, deleted
