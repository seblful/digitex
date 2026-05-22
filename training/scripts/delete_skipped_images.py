"""Delete local images for cancelled Label Studio tasks.

A task is cancelled when an annotator clicks "Skip" or "Cancel" in Label Studio.
This script finds such tasks and deletes their local image files.
"""

from pathlib import Path

import structlog
import typer

from digitex.config import get_settings
from digitex.label_studio import LabelStudioClient
from digitex.logging import setup_logging

_settings = get_settings()
setup_logging(_settings)
logger = structlog.get_logger()

app = typer.Typer(help="Delete local images for cancelled Label Studio tasks")


def _collect_cancelled(
    client: LabelStudioClient, project_id: int
) -> list[tuple[int, str]]:
    tasks = client.get_tasks(project_id)
    logger.info("fetched_tasks", project_id=project_id, count=len(tasks))

    cancelled: list[tuple[int, str]] = []
    for task in tasks:
        if not any(ann.get("was_cancelled", False) for ann in task.annotations):
            continue
        image_path = client.get_local_path(task)
        path_str = str(image_path) if image_path else "unknown"
        cancelled.append((task.id, path_str))
        logger.debug("found_cancelled_task", task_id=task.id, path=path_str)

    logger.info("scan_complete", total_tasks=len(tasks), cancelled=len(cancelled))
    return cancelled


def _partition_paths(
    cancelled: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    existing: list[tuple[int, str]] = []
    missing: list[tuple[int, str]] = []
    for task_id, path_str in cancelled:
        (existing if Path(path_str).exists() else missing).append((task_id, path_str))
    return existing, missing


def _delete(existing: list[tuple[int, str]]) -> int:
    deleted = 0
    for task_id, path_str in existing:
        try:
            Path(path_str).unlink()
            deleted += 1
            logger.info("deleted_file", task_id=task_id, path=path_str)
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
    client = LabelStudioClient(
        url=_settings.label_studio.url,
        api_key=_settings.label_studio.api_key,
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
        typer.echo(
            f"\n--- Cancelled tasks with missing local files ({len(missing)}) ---"
        )
        for task_id, path in missing:
            typer.echo(f"  Task {task_id}: {path}")


if __name__ == "__main__":
    app()
