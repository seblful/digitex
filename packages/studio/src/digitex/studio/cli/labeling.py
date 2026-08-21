"""Label Studio CLI commands.

Everything that talks to the annotation server: pre-annotating a project's tasks,
repairing the ones a moved image pool stranded, and retiring the pages an
annotator skipped.

Settings are resolved per command rather than at import, and the SDK and the model
are imported inside the command that needs them, so ``--help`` reads no files and
loads neither.

Both commands that change something default to a dry run and print the plan
first; ``--no-dry-run`` is what applies it. The two share one shell,
:func:`_run_sweep`, and everything it reports, archives or applies comes off
the plan it is handed — the seam :class:`digitex.labeling.sweeps.SweepPlan`
names.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import typer

from digitex.config import get_settings
from digitex.logging import setup_logging

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.labeling.client import LabelStudioClient, LabelStudioTask
    from digitex.labeling.sweeps import SweepPlan

app = typer.Typer(help="Label Studio pre-annotation and project repair.")


@app.callback()
def configure() -> None:
    """Set up logging before any command runs."""
    setup_logging(get_settings())


def _client() -> LabelStudioClient:
    """The API adapter, pointed at the configured server."""
    from digitex.labeling.client import LabelStudioClient

    label_studio = get_settings().pipeline.label_studio
    return LabelStudioClient(url=label_studio.url, api_key=label_studio.api_key)


def _document_root() -> Path:
    """The directory the server resolves a local-files URI against.

    Both destructive commands need it before they touch the server: the repair
    to decide which side of the move a task fell on, the skipped sweep to find
    the file it is about to unlink.
    """
    root = get_settings().pipeline.label_studio.local_files_document_root
    if root is None:
        raise typer.BadParameter(
            "set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT — a task's path means"
            " nothing without the root the server serves it from",
            param_hint="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
        )
    return root


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


def _run_sweep(
    client: LabelStudioClient,
    tasks: list[LabelStudioTask],
    plan: SweepPlan,
    *,
    stem: str,
    dry_run: bool,
) -> tuple[int, int] | None:
    """Report *plan*, and apply it only past every guard.

    The shell both destructive commands share. Everything it prints, archives
    and applies comes off the plan itself, so the undo archive can never hold
    more or less than what ``apply`` is about to destroy — they are one plan
    answering twice.

    Returns:
        The two counts ``apply`` reports, or None when nothing was applied.
    """
    typer.echo(plan.report(len(tasks)))
    if plan.empty:
        return None
    if dry_run:
        typer.echo("\n--- DRY RUN: nothing changed. Pass --no-dry-run to apply. ---")
        return None

    dump = _archive(plan.doomed(tasks), stem)
    typer.echo(f"\nWrote the tasks about to be deleted to {dump}")
    return plan.apply(client)


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

    document_root = _document_root()
    client = _client()
    tasks = client.list_tasks(project_id)
    if not tasks:
        typer.echo(f"Project {project_id} has no tasks.")
        return

    counts = _run_sweep(
        client,
        tasks,
        repair.plan(tasks, document_root=document_root),
        stem="stranded-tasks",
        dry_run=dry_run,
    )
    if counts is None:
        return

    moved, deleted = counts
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

    document_root = _document_root()
    client = _client()
    tasks = client.list_tasks(project_id)
    plan = skipped.plan(tasks, document_root=document_root)
    if not plan.cancelled:
        typer.echo(f"Project {project_id} has no skipped tasks.")
        return

    counts = _run_sweep(client, tasks, plan, stem="skipped-tasks", dry_run=dry_run)
    if counts is None:
        return

    images, deleted = counts
    typer.echo(
        typer.style(f"✓ Deleted {images} images and {deleted} tasks", fg="green")
    )


if __name__ == "__main__":
    app()
