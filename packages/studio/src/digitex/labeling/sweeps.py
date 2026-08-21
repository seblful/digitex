"""What the two destructive sweeps are to each other.

:mod:`digitex.labeling.repair` and :mod:`digitex.labeling.skipped` are the same
kind of job run for different reasons: read every task of a project, reach a
verdict on each, and only then touch the server or the disk. This module holds
what that sameness is made of — the seam the CLI drives either sweep through,
and the one decision neither may make on its own.

:class:`SweepPlan` is the seam. A plan describes itself to the operator, says
whether it would change anything, hands over the record of what it is about to
destroy, and applies itself. The CLI never reads a plan's insides, so what a
sweep deletes and what its undo archive holds cannot drift apart — they are the
same plan answering twice.

:func:`image_on_disk` is the shared decision. :mod:`digitex.labeling.uris`
deliberately stops at reading the path out of a URI; what that path means on
this machine is resolved here, once. Two sweeps resolving it differently is how
the skipped sweep once read every relative path as a file already gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from digitex.labeling.client import LabelStudioClient, LabelStudioTask

# How many of a plan's entries a report names before summarising the rest.
# Deletions and tasks left alone get the longer list, because those are the two
# an operator reads before deciding to pass --no-dry-run.
MOVE_PREVIEW = 5
LIST_PREVIEW = 10


@dataclass(frozen=True)
class LeftAlone:
    """A task the sweep will not touch, and why."""

    task_id: int
    reason: str


def image_on_disk(path: Path, document_root: Path) -> Path | None:
    """Where the path a task's URI names is on this machine, if it is here.

    A local-files URI usually names its path relative to the server's document
    root; a host that indexed absolute paths wrote the whole path into the URI
    instead. ``/`` serves both spellings — joining an absolute path onto the
    root yields the absolute path unchanged.

    Args:
        path: The path as :func:`digitex.labeling.uris.task_image_path` read it.
        document_root: Directory the server resolves a local-files URI against.

    Returns:
        The resolved path when the file exists on this machine, else None.
    """
    resolved = document_root / path
    return resolved if resolved.exists() else None


def preview[T](
    items: Sequence[T], limit: int, render: Callable[[T], str], more: str
) -> list[str]:
    """Name the first *limit* entries, then say how many were not named.

    A plan can cover thousands of tasks and the operator is deciding whether to
    apply it, so a report has to be readable and honest about what it elided.
    """
    lines = [f"    {render(item)}" for item in items[:limit]]
    if len(items) > limit:
        lines.append(f"    ... and {len(items) - limit} more {more}")
    return lines


class SweepPlan(Protocol):
    """What the CLI needs of a plan, whichever sweep produced it.

    Everything the drive-a-sweep ritual reads comes off the plan itself, so a
    sweep that satisfies this protocol is driven by the same shell as the
    other one.
    """

    @property
    def empty(self) -> bool:
        """Whether applying the plan would change nothing."""
        ...

    def report(self, total: int) -> str:
        """The plan as the operator reads it before deciding to apply it."""
        ...

    def doomed(self, tasks: Sequence[LabelStudioTask]) -> list[dict[str, object]]:
        """Everything the plan is about to destroy, written down to archive."""
        ...

    def apply(self, client: LabelStudioClient) -> tuple[int, int]:
        """Do what the plan decided.

        Returns:
            The sweep's own two counts, for the command's success line.
        """
        ...
