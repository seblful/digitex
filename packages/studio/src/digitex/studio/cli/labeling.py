"""Label Studio CLI commands.

Everything that talks to the annotation server: pre-annotating a project's tasks,
copying one project's annotations into another with their outlines snapped to the
print, repairing the ones a moved image pool stranded, and retiring the pages an
annotator skipped.

Settings are resolved per command rather than at import, and the SDK and the model
are imported inside the command that needs them, so ``--help`` reads no files and
loads neither.

Every command that changes something defaults to a dry run and prints the plan
first; ``--no-dry-run`` is what applies it. The two *destructive* ones share one
shell, :func:`_run_sweep`, and everything it reports, archives or applies comes
off the plan it is handed — the seam :class:`digitex.labeling.sweeps.SweepPlan`
names. ``copy-aligned`` does not use that shell: it only ever adds, so there is
nothing to archive, and it aligns page by page as it writes so that an
interrupted run is resumed rather than restarted.
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

app = typer.Typer(
    help="Label Studio pre-annotation, outline alignment and project repair."
)


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

    Three commands need it before they touch the server: the repair to decide
    which side of the move a task fell on, the skipped sweep to find the file it
    is about to unlink, and the aligned copy to read the page it is measuring a
    margin against.
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
        YOLO_SegmentationPredictor(model_path),
        _client(),
        # Uploaded predictions are grouped by version, so the model file names
        # one.
        model_version=Path(model_path).stem,
    )

    count = predictor.predict_tasks(project_id)
    typer.echo(
        typer.style(f"✓ Predicted {count} tasks in project {project_id}", fg="green")
    )


@app.command(name="copy-aligned")
def copy_aligned(
    from_project: int = typer.Option(
        ..., "--from-project", help="Label Studio project to read annotations from"
    ),
    to_project: int = typer.Option(
        ..., "--to-project", help="Label Studio project to write them into"
    ),
    margin: float = typer.Option(
        None,
        "--margin",
        help="Clearance to leave around the print, in line heights"
        " (default: the tuned 0.25)",
    ),
    limit: int = typer.Option(
        None, "--limit", help="Carry at most this many pages, for a trial run"
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Print what would be carried without writing it",
    ),
) -> None:
    """Copy a project's annotations into another project, snapped to the print.

    Reads each annotated page of the source project, rebuilds its region outlines
    against the image on disk so every region sits the same distance from its own
    text, and writes the result into the destination project. The source project
    is never written to.

    Safe to rerun and safe to interrupt: the destination project is its own
    record of what has been done, so a page already carried across is passed
    over, and a run that stops halfway is resumed by running it again. Pages the
    destination already holds a task for — a storage sync got there first — have
    the annotation attached rather than the page imported twice.
    """
    from digitex.labeling import transfer

    if from_project == to_project:
        raise typer.BadParameter(
            "the source and the destination have to be two different projects",
            param_hint="--to-project",
        )

    document_root = _document_root()
    client = _client()
    source = client.list_tasks(from_project)
    if not source:
        typer.echo(f"Project {from_project} has no tasks.")
        return

    plan = transfer.plan(
        source, client.list_tasks(to_project), document_root=document_root
    )
    if limit:
        plan.carries = plan.carries[:limit]
    typer.echo(plan.report(len(source)))
    if plan.empty:
        return
    if dry_run:
        typer.echo("\n--- DRY RUN: nothing written. Pass --no-dry-run to apply. ---")
        return

    aligner = (
        transfer.Aligner(margin=margin) if margin is not None else transfer.Aligner()
    )
    pages, regions = plan.apply(client, to_project, aligner=aligner)
    typer.echo(
        typer.style(
            f"✓ Carried {pages} pages into project {to_project},"
            f" {regions} outlines rebuilt",
            fg="green",
        )
    )
    outstanding = len(plan.carries) - pages
    if outstanding:
        typer.echo(
            typer.style(
                f"! {outstanding} pages did not make it — rerun to pick them up",
                fg="yellow",
            )
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
