"""Delete local images for cancelled Label Studio tasks.

A task is cancelled when an annotator clicks "Skip" or "Cancel" in Label Studio.
This script finds such tasks and deletes their local image files.
"""

from pathlib import Path

import structlog
import typer

from digitex.config import get_settings
from digitex.domain.geometry import task_image_path
from digitex.labeling.client import LabelStudioClient
from digitex.logging import setup_logging

logger = structlog.get_logger()

app = typer.Typer(help="Delete local images for cancelled Label Studio tasks")


@app.callback()
def configure() -> None:
    """Set up logging before the command runs."""
    setup_logging(get_settings())


def _collect_cancelled(
    client: LabelStudioClient, project_id: int
) -> list[tuple[int, Path | None]]:
    """Cancelled tasks as ``(task_id, local image path)``.

    The path stays ``None`` when the task's image URI carries no local-file
    parameter — collapsing that into a placeholder string would make it
    indistinguishable from a real path, and it gets fed to ``unlink`` below.
    """
    tasks = client.list_tasks(project_id)

    cancelled: list[tuple[int, Path | None]] = []
    for task in tasks:
        if not any(ann.get("was_cancelled", False) for ann in task.annotations):
            continue
        image_path = task_image_path(task.data)
        cancelled.append((task.id, image_path))
        logger.debug("found_cancelled_task", task_id=task.id, path=str(image_path))

    logger.info("scan_complete", total_tasks=len(tasks), cancelled=len(cancelled))
    return cancelled


def _partition_paths(
    cancelled: list[tuple[int, Path | None]],
) -> tuple[list[tuple[int, Path]], list[tuple[int, Path | None]]]:
    """Split into files that are there to delete, and everything else."""
    existing: list[tuple[int, Path]] = []
    missing: list[tuple[int, Path | None]] = []
    for task_id, path in cancelled:
        if path is not None and path.exists():
            existing.append((task_id, path))
        else:
            missing.append((task_id, path))
    return existing, missing


def _delete(existing: list[tuple[int, Path]]) -> int:
    deleted = 0
    for task_id, path in existing:
        try:
            path.unlink()
            deleted += 1
            logger.info("deleted_file", task_id=task_id, path=str(path))
        except Exception as e:
            logger.error("delete_failed", task_id=task_id, error=str(e))
    return deleted


@app.command()
def delete_skipped_images(
    project_id: int = typer.Option(..., help="Label Studio project ID"),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Print what would be deleted without actually deleting",
    ),
) -> None:
    """Find cancelled tasks (was_cancelled=true) and delete their local image files.

    Args:
        project_id: Label Studio project ID to scan.
        dry_run: If True, only log what would be deleted.
    """
    settings = get_settings()
    client = LabelStudioClient(
        url=settings.pipeline.label_studio.url,
        api_key=settings.pipeline.label_studio.api_key,
    )

    cancelled = _collect_cancelled(client, project_id)
    if not cancelled:
        typer.echo("No cancelled tasks found.")
        return

    existing, missing = _partition_paths(cancelled)

    if dry_run:
        typer.echo(f"\n--- DRY RUN: Would delete {len(existing)} files ---\n")
        for task_id, path in existing:
            typer.echo(f"  Task {task_id}: {path}")
    else:
        deleted = _delete(existing)
        typer.echo(f"\nDeleted {deleted} files.")

    if missing:
        typer.echo(f"\n--- Cancelled tasks with no local file ({len(missing)}) ---")
        for task_id, path in missing:
            typer.echo(f"  Task {task_id}: {path or 'no local path'}")


if __name__ == "__main__":
    app()
