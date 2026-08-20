"""Label Studio CLI commands.

Everything that talks to the annotation server: pre-annotating a project's tasks,
repairing the ones a moved image pool stranded, and retiring the pages an
annotator skipped.

Settings are resolved per command rather than at import, and the SDK and the model
are imported inside the command that needs them, so ``--help`` reads no files and
loads neither.

Both commands that change something default to a dry run and print the plan
first; ``--no-dry-run`` is what applies it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import typer

from digitex.config import get_settings
from digitex.logging import setup_logging

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from digitex.labeling import repair, skipped
    from digitex.labeling.client import LabelStudioClient, LabelStudioTask

app = typer.Typer(help="Label Studio pre-annotation and project repair.")

# How many of a plan's entries to name before summarising the rest. Deletions and
# tasks left alone get the longer list, because those are the two an operator
# reads before deciding to pass --no-dry-run.
_MOVE_PREVIEW = 5
_LIST_PREVIEW = 10


@app.callback()
def configure() -> None:
    """Set up logging before any command runs."""
    setup_logging(get_settings())


def _client() -> LabelStudioClient:
    """The API adapter, pointed at the configured server."""
    from digitex.labeling.client import LabelStudioClient

    label_studio = get_settings().pipeline.label_studio
    return LabelStudioClient(url=label_studio.url, api_key=label_studio.api_key)


def _archive(payload: list[dict[str, object]], stem: str) -> Path:
    """Write *payload* to a timestamped JSON file under the data root.

    Both destructive commands write one before they touch anything: the Label
    Studio API has no undo, and neither has ``unlink``.
    """
    directory = get_settings().paths.data_root / "label-studio"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dump = directory / f"{stem}-{stamp}.json"
    dump.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return dump


def _preview[T](
    items: Sequence[T], limit: int, render: Callable[[T], str], more: str
) -> None:
    """Name the first *limit* entries, then say how many were not named.

    A plan can cover thousands of tasks and the operator is deciding whether to
    apply it, so the report has to be readable and honest about what it elided.
    """
    for item in items[:limit]:
        typer.echo(f"    {render(item)}")
    if len(items) > limit:
        typer.echo(f"    ... and {len(items) - limit} more {more}")


@app.command()
def predict(
    project_id: int = typer.Option(..., "--project-id", help="Label Studio project ID"),
    model_path: str = typer.Option(
        ..., "--model-path", help="Path to trained YOLO model (.pt file)"
    ),
) -> None:
    """Run model predictions on a project's unannotated tasks."""
    from pathlib import Path

    from digitex.labeling.predictor import TaskPredictor
    from digitex.ml.predictors import YOLO_SegmentationPredictor

    predictor = TaskPredictor(
        YOLO_SegmentationPredictor(model_path, simplify=True),
        _client(),
        # Uploaded predictions are grouped by version, so the model file names
        # one.
        model_version=Path(model_path).stem,
    )

    count = predictor.predict_tasks(project_id)
    typer.echo(
        typer.style(f"✓ Predicted {count} tasks in project {project_id}", fg="green")
    )


def _report_repair(plan: repair.Plan, total: int) -> None:
    typer.echo(f"\n{total} tasks in the project.")
    typer.echo(f"  move {plan.annotations} annotations off: {len(plan.moves)} tasks")
    typer.echo(f"  delete, nothing on them:                {len(plan.deletions)} tasks")
    typer.echo(f"  leave alone:                            {len(plan.skipped)} tasks")

    _preview(
        plan.moves,
        _MOVE_PREVIEW,
        lambda move: (
            f"task {move.stranded_id} -> task {move.live_id}"
            f" ({len(move.annotations)} annotations)"
        ),
        "to move",
    )
    _preview(
        plan.skipped,
        _LIST_PREVIEW,
        lambda left: f"task {left.task_id}: {left.reason}",
        "left alone",
    )


@app.command(name="fix-task-paths")
def fix_task_paths(
    project_id: int = typer.Option(..., help="Label Studio project ID"),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Print the plan without writing it",
    ),
) -> None:
    """Repair a project whose images moved out from under its tasks.

    Run it after the image pool a local-files storage points at has moved. The
    tasks synced before the move still name the old path, so their image 404s in
    the editor, and the sync that followed the move imported every file a second
    time as a fresh, unannotated task. This moves each stranded task's
    annotations onto its freshly imported twin and deletes the stranded task,
    leaving one task per image with the annotations intact.
    """
    from digitex.labeling import repair

    settings = get_settings()
    document_root = settings.pipeline.label_studio.local_files_document_root
    if document_root is None:
        raise typer.BadParameter(
            "set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT — a task's path means"
            " nothing without the root the server serves it from",
            param_hint="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
        )

    client = _client()
    tasks = client.list_tasks(project_id)
    if not tasks:
        typer.echo(f"Project {project_id} has no tasks.")
        return

    plan = repair.plan(tasks, document_root=document_root)
    _report_repair(plan, total=len(tasks))

    if not plan.moves and not plan.deletions:
        typer.echo("\nNothing to repair.")
        return
    if dry_run:
        typer.echo("\n--- DRY RUN: nothing written. Pass --no-dry-run to apply. ---")
        return

    dump = _archive(_doomed_tasks(plan, tasks), "stranded-tasks")
    typer.echo(f"\nWrote the tasks about to be deleted to {dump}")

    moved, deleted = repair.apply(client, plan)
    typer.echo(
        typer.style(
            f"✓ Moved {moved} annotations, deleted {deleted} stranded tasks",
            fg="green",
        )
    )

    # Re-planned against the server rather than deduced from the plan: a partial
    # failure leaves tasks behind, and the operator needs the real number.
    left = repair.plan(client.list_tasks(project_id), document_root=document_root)
    outstanding = len(left.moves) + len(left.deletions)
    typer.echo(
        typer.style(
            f"✓ {len(left.skipped)} tasks left alone, {outstanding} still stranded",
            fg="green" if not outstanding else "yellow",
        )
    )


def _doomed_tasks(
    plan: repair.Plan, tasks: list[LabelStudioTask]
) -> list[dict[str, object]]:
    """Every task the repair would delete, annotations and all."""
    doomed = {move.stranded_id for move in plan.moves} | set(plan.deletions)
    return [
        {
            "id": task.id,
            "data": task.data,
            "annotations": list(task.annotations or []),
        }
        for task in tasks
        if task.id in doomed
    ]


def _report_skipped(plan: skipped.Plan, total: int) -> None:
    typer.echo(f"\n{total} tasks in the project, {plan.cancelled} of them skipped.")
    typer.echo(
        f"  delete:      {len(plan.deletions)} tasks, {plan.images} with an image"
    )
    typer.echo(f"  leave alone: {len(plan.kept)} tasks")

    _preview(
        plan.deletions,
        _LIST_PREVIEW,
        lambda doomed: (
            f"task {doomed.task_id}:"
            f" {doomed.path if doomed.path is not None else 'image already gone'}"
        ),
        "to delete",
    )
    _preview(
        plan.kept,
        _LIST_PREVIEW,
        lambda kept: f"task {kept.task_id}: {kept.reason}",
        "left alone",
    )


@app.command(name="delete-skipped-tasks")
def delete_skipped_tasks(
    project_id: int = typer.Option(..., help="Label Studio project ID"),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Print what would be deleted without deleting it",
    ),
) -> None:
    """Retire the pages an annotator skipped: image unlinked, task deleted.

    A task is cancelled when an annotator clicks "Skip" in Label Studio. Its
    image going is what keeps the page from syncing back into the pool and out of
    the next training set; its task going is what keeps the project from filling
    up with skips. A cancelled task an earlier sweep already took the image of is
    deleted here too.
    """
    from digitex.labeling import skipped

    client = _client()
    plan = skipped.plan(tasks := client.list_tasks(project_id))
    if not plan.cancelled:
        typer.echo(f"Project {project_id} has no skipped tasks.")
        return

    _report_skipped(plan, total=len(tasks))

    if not plan.deletions:
        typer.echo("\nNothing to delete.")
        return
    if dry_run:
        typer.echo("\n--- DRY RUN: nothing deleted. Pass --no-dry-run to apply. ---")
        return

    dump = _archive(_doomed_skips(plan, tasks), "skipped-tasks")
    typer.echo(f"\nWrote the tasks about to be deleted to {dump}")

    images, deleted = skipped.apply(client, plan)
    typer.echo(
        typer.style(f"✓ Deleted {images} images and {deleted} tasks", fg="green")
    )


def _doomed_skips(
    plan: skipped.Plan, tasks: list[LabelStudioTask]
) -> list[dict[str, object]]:
    """Every task the sweep would delete, its image and annotations with it.

    The cancelled annotation is the only record that anybody judged the page, and
    it goes when the task does, so it is written down first.
    """
    paths = {doomed.task_id: doomed.path for doomed in plan.deletions}
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


if __name__ == "__main__":
    app()
