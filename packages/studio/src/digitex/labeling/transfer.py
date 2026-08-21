"""Carrying a project's annotations into another project, snapped to the print.

One direction, one page at a time: read a task out of the source project, rebuild
its outlines against the image on disk with
:func:`~digitex.imaging.outlines.align_outlines`, and write the result into the
destination project. The source project is never written to — the original
hand-drawn work stays exactly where it is, which is what makes the whole thing
repeatable.

Rerunnable is the point. A run over a few hundred pages is long enough to be
interrupted, so the destination project itself is the record of what has been
done: a page whose image is already there carrying an annotation is a page this
has already carried across, and the next run passes over it. Nothing is stored
on the side and nothing has to be cleaned up after a failure.

Which is why the work is split the way it is, and not the way
:mod:`digitex.labeling.repair` splits it. That sweep decides every verdict up
front because it *deletes* things, and its dry run has to show exactly what will
go. This one destroys nothing, so :func:`plan` only settles which pages are new —
an index comparison, no images opened — and :meth:`Plan.apply` aligns and writes
one page at a time. An interrupted run loses one page's work rather than all of
it, and the dry run stays fast enough to be worth running.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import structlog
from PIL import Image

from digitex.domain.geometry import percent_to_pixel, pixel_to_percent
from digitex.imaging import outlines
from digitex.labeling.sweeps import LIST_PREVIEW, MOVE_PREVIEW, LeftAlone, preview
from digitex.labeling.uris import task_image_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from digitex.domain.entities import PercentPolygon
    from digitex.labeling.client import LabelStudioClient, LabelStudioTask

logger = structlog.get_logger()

# The result type a region outline arrives as, and the key its class sits under.
_POLYGON_TYPE = "polygonlabels"

# Where a carried task records the task it came from. Nothing reads it to decide
# anything — the destination's own images are what :func:`plan` compares — but a
# project assembled by a script should say so somewhere, and this is the only
# place it can say it.
SOURCE_KEY = "aligned_from_task"


class Realign(Protocol):
    """Whatever turns one carried annotation into the one to write.

    A protocol rather than the concrete :class:`Aligner`, because the only thing
    :meth:`Plan.apply` needs of it is the call — and because that lets a test
    hand in something that opens no images.
    """

    def __call__(self, carry: Carry) -> tuple[list[dict[str, Any]], int, list[str]]:
        """The result to write, how many outlines moved, and why the rest did not."""
        ...


@dataclass(frozen=True)
class Carry:
    """One source task to take across, decided before any image is opened."""

    source_id: int
    """The task in the source project."""

    image: Path
    """Where its page is on this machine."""

    data: dict[str, Any]
    """The ``data`` the destination task carries, provenance included."""

    annotation: dict[str, Any]
    """The source annotation whose outlines get rebuilt."""

    target_id: int | None
    """The destination task to hang it on, or None to create one."""


@dataclass
class Plan:
    """Which pages are new, and what to do about each.

    Deciding this needs no images: a page is new when the destination has no
    annotated task holding it. Everything expensive happens in :meth:`apply`.
    """

    carries: list[Carry] = field(default_factory=list)
    skipped: list[LeftAlone] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """Whether there is nothing new to carry across."""
        return not self.carries

    @property
    def creating(self) -> int:
        """How many destination tasks the run would import."""
        return sum(1 for carry in self.carries if carry.target_id is None)

    def report(self, total: int) -> str:
        """The plan as the operator reads it before deciding to apply it."""
        reasons: dict[str, int] = {}
        for left in self.skipped:
            reasons[left.reason] = reasons.get(left.reason, 0) + 1

        lines = [
            f"\n{total} tasks in the source project.",
            f"  carry across, aligned:  {len(self.carries)} pages",
            f"    of those, needing a new task: {self.creating}",
            f"  leave alone:            {len(self.skipped)} pages",
            *preview(
                self.carries,
                MOVE_PREVIEW,
                lambda carry: (
                    f"task {carry.source_id} -> "
                    + (
                        f"task {carry.target_id}"
                        if carry.target_id is not None
                        else "a new task"
                    )
                    + f" ({carry.image.name})"
                ),
                "to carry",
            ),
            *preview(
                sorted(reasons.items(), key=lambda item: -item[1]),
                LIST_PREVIEW,
                lambda item: f"{item[1]} pages: {item[0]}",
                "reasons",
            ),
        ]
        if self.empty:
            lines.append("\nNothing new to carry across.")
        return "\n".join(lines)

    def apply(
        self, client: LabelStudioClient, project_id: int, *, aligner: Realign
    ) -> tuple[int, int]:
        """Align and write one page at a time, into *project_id*.

        A page that fails is logged and passed over rather than ending the run,
        and because the destination is its own record, the next run picks that
        page up again with nothing to undo first.

        Returns:
            How many pages were carried across, and how many regions were
            rebuilt across them.
        """
        pages = 0
        regions = 0
        for carry in self.carries:
            try:
                results, aligned, unchanged = aligner(carry)
            except Exception as error:
                logger.error(
                    "align_failed", source_task=carry.source_id, error=str(error)
                )
                continue

            try:
                target = carry.target_id
                if target is None:
                    target = client.create_task(project_id, carry.data)
                # The annotation goes on last, and it is what marks the page
                # done. A task created without one is picked up again next run.
                client.create_annotation(
                    target, {**carry.annotation, "result": results}
                )
            except Exception as error:
                logger.error(
                    "carry_failed", source_task=carry.source_id, error=str(error)
                )
                continue

            pages += 1
            regions += aligned
            logger.info(
                "carried",
                source_task=carry.source_id,
                target_task=target,
                aligned=aligned,
                unchanged=len(unchanged),
                progress=f"{pages}/{len(self.carries)}",
            )
        logger.info("transfer_complete", pages=pages, regions=regions)
        return pages, regions


def _usable_annotation(task: LabelStudioTask) -> dict[str, Any] | None:
    """The annotation whose outlines are worth carrying, if there is one.

    A cancelled annotation is a page the annotator skipped, and one holding no
    region outline has nothing for the aligner to work on. Where a task carries
    several, the last wins: Label Studio appends, so that is the freshest.
    """
    usable = [
        annotation
        for annotation in task.annotations or []
        if not annotation.get("was_cancelled")
        and any(
            item.get("type") == _POLYGON_TYPE for item in annotation.get("result", [])
        )
    ]
    return usable[-1] if usable else None


def _annotated_images(tasks: Sequence[LabelStudioTask]) -> dict[str, int]:
    """Which images the destination already holds an annotation for.

    Keyed by filename rather than by full path: the two projects can be synced
    through different storages, or the pool can have moved between them, and the
    page is the same page either way.
    """
    held: dict[str, int] = {}
    for task in tasks:
        path = task_image_path(task.data)
        if path is None or not task.annotations:
            continue
        held[path.name] = task.id
    return held


def _unannotated_images(tasks: Sequence[LabelStudioTask]) -> dict[str, int]:
    """Which images the destination holds a task for but no annotation on.

    These are the tasks a storage sync put there. The annotation goes onto one
    of them rather than importing the page a second time — which is exactly the
    duplicate :mod:`digitex.labeling.repair` exists to clean up.
    """
    held: dict[str, int] = {}
    for task in tasks:
        path = task_image_path(task.data)
        if path is None or task.annotations:
            continue
        held.setdefault(path.name, task.id)
    return held


def plan(
    source: Sequence[LabelStudioTask],
    target: Sequence[LabelStudioTask],
    *,
    document_root: Path,
) -> Plan:
    """Decide which of *source*'s pages are not yet in *target*.

    Args:
        source: Every task of the source project, annotations included.
        target: Every task of the destination project, annotations included.
        document_root: Directory the server resolves a local-files URI against,
            which is what decides whether the page can be read at all.

    Returns:
        The pages to carry across, and every page left alone with a reason.
    """
    result = Plan()
    done = _annotated_images(target)
    waiting = _unannotated_images(target)

    for task in source:
        path = task_image_path(task.data)
        if path is None:
            result.skipped.append(LeftAlone(task.id, "no local-file URI in the task"))
            continue
        if path.name in done:
            result.skipped.append(LeftAlone(task.id, "already carried across"))
            continue

        annotation = _usable_annotation(task)
        if annotation is None:
            result.skipped.append(LeftAlone(task.id, "no usable annotation to carry"))
            continue

        resolved = document_root / path
        if not resolved.exists():
            result.skipped.append(LeftAlone(task.id, "page is not on this machine"))
            continue

        result.carries.append(
            Carry(
                source_id=task.id,
                image=resolved,
                data={**task.data, SOURCE_KEY: task.id},
                annotation=annotation,
                target_id=waiting.get(path.name),
            )
        )

    logger.info(
        "planned_transfer",
        carries=len(result.carries),
        creating=result.creating,
        skipped=len(result.skipped),
    )
    return result


class Aligner:
    """Rebuilds one carried annotation's outlines against its page.

    A class rather than a closure so the CLI can hand the plan something whose
    knobs are visible, and so a test can hand :meth:`Plan.apply` a stub that
    opens no images at all.

    Args:
        margin: Target clearance between an outline and its print, in line
            heights.
        snap: How far apart two lines' ends may be and still share one edge, in
            line heights.
        grow: How far outside its original an outline may travel, in line
            heights.
        budget: Vertex budget for the thinning.
    """

    def __init__(
        self,
        *,
        margin: float = outlines.MARGIN,
        snap: float = outlines.SNAP,
        grow: float = outlines.GROW,
        budget: int = outlines.BUDGET,
    ) -> None:
        self._margin = margin
        self._snap = snap
        self._grow = grow
        self._budget = budget

    def __call__(self, carry: Carry) -> tuple[list[dict[str, Any]], int, list[str]]:
        """The carried annotation's result, with every outline snapped to print.

        Returns:
            The result to write, how many outlines were rebuilt, and a reason for
            each one that was not.
        """
        with Image.open(carry.image) as page:
            page.load()
            width, height = page.size
            results = copy.deepcopy(carry.annotation.get("result", []))
            spots = [
                index
                for index, item in enumerate(results)
                if item.get("type") == _POLYGON_TYPE
                and item.get("value", {}).get("points")
            ]
            if not spots:
                return results, 0, []

            traced = [
                outlines.Outline(
                    label=(results[index]["value"].get(_POLYGON_TYPE) or [""])[0],
                    polygon=percent_to_pixel(
                        cast("PercentPolygon", results[index]["value"]["points"]),
                        width,
                        height,
                    ),
                )
                for index in spots
            ]
            aligned = outlines.align_outlines(
                page,
                traced,
                margin=self._margin,
                snap=self._snap,
                grow=self._grow,
                budget=self._budget,
            )

        changed = 0
        unchanged: list[str] = []
        for index, outcome in zip(spots, aligned, strict=True):
            if not outcome.changed:
                unchanged.append(outcome.reason)
                continue
            # Only the points move. Everything else Label Studio put on the
            # region — which tag it belongs to, its id, its class — is the
            # destination project's business and travels untouched.
            results[index]["value"]["points"] = pixel_to_percent(
                outcome.polygon, width, height
            )
            changed += 1
        return results, changed, unchanged
