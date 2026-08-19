"""Clearing out the local images of pages an annotator refused.

Clicking "Skip" in Label Studio does not reject the page — it files a cancelled
annotation against the task and leaves both where they are. The image stays in
the pool the local-files storage syncs from, so it comes back as a task on the
next sync, and stays in the pool a training dataset is built out of. Deleting
the file is what actually retires the page.

The task itself is left alone. Its cancelled annotation is the record of an
annotator's judgement, and the image being gone is enough to keep the page from
returning. What that leaves behind is a task whose image 404s in the editor,
which is the deliberate trade: this module removes pages from the pool, and
:mod:`digitex.labeling.repair` is what removes tasks.

Deciding is split from deleting, the same way it is in ``repair``: :func:`plan`
reads the project and says which files it would unlink and which it would not
touch, and :func:`apply` is the only part that reaches the filesystem.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from digitex.domain.geometry import task_image_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from digitex.labeling.client import LabelStudioTask

logger = structlog.get_logger()


@dataclass(frozen=True)
class Kept:
    """A cancelled task whose image is not deleted, and why."""

    task_id: int
    reason: str


@dataclass
class Plan:
    """What a run would delete, decided before any of it is deleted."""

    deletions: list[tuple[int, Path]] = field(default_factory=list)
    kept: list[Kept] = field(default_factory=list)

    @property
    def cancelled(self) -> int:
        """How many cancelled tasks the plan looked at."""
        return len(self.deletions) + len(self.kept)


def _was_cancelled(annotations: Iterable[dict[str, Any]]) -> bool:
    return any(annotation.get("was_cancelled", False) for annotation in annotations)


def plan(tasks: Sequence[LabelStudioTask]) -> Plan:
    """Decide which skipped pages' images to unlink.

    A cancelled task keeps its image in three cases: the task also carries an
    annotation somebody actually completed, another task in the project points
    at the same file, or there is no local file to delete. The first two are
    what stops an annotator's work being thrown out with a colleague's skip —
    a project with two completions per task can hold both verdicts at once, and
    a pool that has moved holds two tasks over one image until
    :mod:`digitex.labeling.repair` has run.

    Args:
        tasks: Every task of the project, annotations included.

    Returns:
        The images to unlink, and every cancelled task left alone with a reason.
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
        elif not path.exists():
            result.kept.append(Kept(task.id, f"no file at {path}"))
        else:
            result.deletions.append((task.id, path))

    logger.info(
        "planned_skipped_sweep",
        cancelled=result.cancelled,
        deletions=len(result.deletions),
        kept=len(result.kept),
    )
    return result


def apply(plan: Plan) -> int:
    """Unlink every image the plan named. Returns how many went.

    A file that will not delete is logged and passed over — one locked handle
    must not strand the rest of the sweep.
    """
    deleted = 0
    for task_id, path in plan.deletions:
        try:
            path.unlink()
        except Exception as e:
            logger.error("delete_failed", task_id=task_id, path=str(path), error=str(e))
            continue
        deleted += 1
        logger.info("deleted_file", task_id=task_id, path=str(path))

    logger.info("skipped_sweep_complete", deleted=deleted)
    return deleted
