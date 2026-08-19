"""Move a Label Studio project's annotations onto the tasks that hold them now.

Run it after the image pool a local-files storage points at has moved. The tasks
synced before the move still name the old path, so their image 404s in the
editor, and the sync that followed the move imported every file a second time as
a fresh, unannotated task. This moves each stranded task's annotations onto its
freshly imported twin and deletes the stranded task, leaving one task per image
with the annotations intact.

Everything goes through the Label Studio API, so the server has to be running.
The run writes nothing unless ``--no-dry-run`` is passed, and dumps the tasks it
is about to delete to JSON first.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from digitex.config import get_settings
from digitex.labeling import repair
from digitex.labeling.client import LabelStudioClient, LabelStudioTask
from digitex.logging import setup_logging

app = typer.Typer(help="Move annotations off tasks whose image path went stale")


@app.callback()
def configure() -> None:
    """Set up logging before the command runs."""
    setup_logging(get_settings())


def _report(plan: repair.Plan, total: int) -> None:
    typer.echo(f"\n{total} tasks in the project.")
    typer.echo(f"  move {plan.annotations} annotations off: {len(plan.moves)} tasks")
    typer.echo(f"  delete, nothing on them:                {len(plan.deletions)} tasks")
    typer.echo(f"  leave alone:                            {len(plan.skipped)} tasks")

    for move in plan.moves[:5]:
        typer.echo(
            f"    task {move.stranded_id} -> task {move.live_id}"
            f" ({len(move.annotations)} annotations)"
        )
    if len(plan.moves) > 5:
        typer.echo(f"    ... and {len(plan.moves) - 5} more to move")

    for skipped in plan.skipped[:10]:
        typer.echo(f"    task {skipped.task_id}: {skipped.reason}")
    if len(plan.skipped) > 10:
        typer.echo(f"    ... and {len(plan.skipped) - 10} more left alone")


def _dump(plan: repair.Plan, tasks: list[LabelStudioTask], directory: Path) -> Path:
    """Write every task the run would delete, annotations and all, to JSON.

    The API has no undo, and a task carrying an annotator's work is about to go.
    """
    doomed = {move.stranded_id for move in plan.moves} | set(plan.deletions)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dump = directory / f"stranded-tasks-{stamp}.json"
    dump.write_text(
        json.dumps(
            [
                {
                    "id": task.id,
                    "data": task.data,
                    "annotations": list(task.annotations or []),
                }
                for task in tasks
                if task.id in doomed
            ],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return dump


@app.command()
def fix_task_paths(
    project_id: int = typer.Option(..., help="Label Studio project ID"),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Print the plan without writing it",
    ),
) -> None:
    """Repair a project whose images moved out from under its tasks."""
    settings = get_settings()
    document_root = settings.pipeline.label_studio.local_files_document_root
    if document_root is None:
        raise typer.BadParameter(
            "set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT — a task's path means"
            " nothing without the root the server serves it from",
            param_hint="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
        )

    client = LabelStudioClient(
        url=settings.pipeline.label_studio.url,
        api_key=settings.pipeline.label_studio.api_key,
    )
    tasks = client.list_tasks(project_id)
    if not tasks:
        typer.echo(f"Project {project_id} has no tasks.")
        return

    plan = repair.plan(tasks, document_root=document_root)
    _report(plan, total=len(tasks))

    if not plan.moves and not plan.deletions:
        typer.echo("\nNothing to repair.")
        return
    if dry_run:
        typer.echo("\n--- DRY RUN: nothing written. Pass --no-dry-run to apply. ---")
        return

    dump = _dump(plan, tasks, settings.paths.data_root / "label-studio")
    typer.echo(f"\nWrote the tasks about to be deleted to {dump}")

    moved, deleted = repair.apply(client, plan)
    typer.echo(
        typer.style(
            f"✓ Moved {moved} annotations, deleted {deleted} stranded tasks",
            fg="green",
        )
    )

    left = repair.plan(client.list_tasks(project_id), document_root=document_root)
    outstanding = len(left.moves) + len(left.deletions)
    typer.echo(
        typer.style(
            f"✓ {len(left.skipped)} tasks left alone, {outstanding} still stranded",
            fg="green" if not outstanding else "yellow",
        )
    )


if __name__ == "__main__":
    app()
