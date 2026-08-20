"""Moving a project's annotations onto the tasks that hold their images now.

A local-files import storage keys every task by the absolute path the file had
at sync time. Move the pool and point the storage at its new directory — the
training images moving under ``var/`` is what happened here — and the tasks
synced before the move are stranded: the URI in their ``data`` names a path the
server no longer serves, so the image 404s in the editor, and their storage link
still names the old key, so the next sync reads every moved file as one it has
never seen and imports a second, unannotated task for it.

The repair works with the grain of that: the freshly imported task is the one
Label Studio's own bookkeeping agrees with, so the annotations move onto it and
the stranded task goes. That keeps the fix inside the public API — a storage
link has no endpoint, and hand-editing one in the database is what this module
exists to avoid.

What does not survive the move is an annotation's identity: the server assigns
new ids and timestamps, and the annotation history of the original task goes
with it. The result, the labels, who made it and how long it took do survive.
Predictions on a stranded task are dropped rather than copied — ``digitex-label
predict`` writes them and can write them again, and a task an annotator has
finished has no use for the model's guess.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from digitex.labeling.uris import task_image_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from digitex.labeling.client import LabelStudioClient, LabelStudioTask

logger = structlog.get_logger()


@dataclass(frozen=True)
class Move:
    """A stranded task's annotations, and the task that holds its image now."""

    stranded_id: int
    live_id: int
    annotations: list[dict[str, Any]]


@dataclass(frozen=True)
class Skipped:
    """A task the plan will not touch, and why."""

    task_id: int
    reason: str


@dataclass
class Plan:
    """What a run would do, decided before it does any of it."""

    moves: list[Move] = field(default_factory=list)
    deletions: list[int] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)

    @property
    def annotations(self) -> int:
        """How many annotations the run would recreate."""
        return sum(len(move.annotations) for move in self.moves)


def plan(tasks: Sequence[LabelStudioTask], *, document_root: Path) -> Plan:
    """Decide which tasks are stranded and where their work belongs.

    Args:
        tasks: Every task of the project, annotations included.
        document_root: Directory the server resolves a local-file URI against,
            which is what decides whether a task's image is reachable at all.

    Returns:
        The moves and deletions to make, and every task left alone with a reason.
    """
    result = Plan()
    live: dict[str, list[LabelStudioTask]] = defaultdict(list)
    stranded: list[tuple[LabelStudioTask, Path]] = []

    for task in tasks:
        path = task_image_path(task.data)
        if path is None:
            result.skipped.append(Skipped(task.id, "no local-file URI in the task"))
        elif (document_root / path).exists():
            live[path.name].append(task)
        else:
            stranded.append((task, path))

    for task, path in stranded:
        twins = live.get(path.name, [])
        if len(twins) != 1:
            result.skipped.append(
                Skipped(task.id, f"{len(twins)} live tasks hold {path.name}")
            )
        elif task.annotations:
            result.moves.append(Move(task.id, twins[0].id, list(task.annotations)))
        else:
            result.deletions.append(task.id)

    # Two live tasks over one image are a duplicate this repair did not create
    # and cannot tell apart; deciding which annotator's task to keep is not its
    # call to make.
    for name, group in live.items():
        if len(group) > 1:
            result.skipped.extend(
                Skipped(task.id, f"{len(group)} live tasks hold {name}")
                for task in group
            )

    logger.info(
        "planned_repair",
        moves=len(result.moves),
        annotations=result.annotations,
        deletions=len(result.deletions),
        skipped=len(result.skipped),
    )
    return result


def apply(client: LabelStudioClient, plan: Plan) -> tuple[int, int]:
    """Recreate each stranded task's annotations on its twin, then delete it.

    A stranded task is deleted only once every one of its annotations has come
    back on the live task, so a failure leaves the work where it was — at the
    cost that a rerun after a partial failure would copy the ones that did land
    a second time. The log names both tasks when that happens.

    Args:
        client: Label Studio API adapter.
        plan: What :func:`plan` decided.

    Returns:
        How many annotations were recreated, and how many tasks were deleted.
    """
    moved = 0
    deleted: list[int] = []

    for move in plan.moves:
        try:
            for annotation in move.annotations:
                client.create_annotation(move.live_id, annotation)
                moved += 1
        except Exception as e:
            logger.error(
                "annotation_move_failed",
                stranded_task=move.stranded_id,
                live_task=move.live_id,
                error=str(e),
            )
            continue
        client.delete_task(move.stranded_id)
        deleted.append(move.stranded_id)

    for task_id in plan.deletions:
        client.delete_task(task_id)
        deleted.append(task_id)

    logger.info("repair_complete", annotations=moved, deleted=len(deleted))
    return moved, len(deleted)
