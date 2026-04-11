"""Delete local images for cancelled Label Studio tasks.

A task is cancelled when an annotator clicks "Skip" or "Cancel" in Label Studio.
This script finds such tasks and deletes their local image files.
"""

import structlog
import typer

from digitex.config import get_settings
from digitex.label_studio import LabelStudioClient
from digitex.logging import setup_logging

setup_logging()
logger = structlog.get_logger()

app = typer.Typer(help="Delete local images for cancelled Label Studio tasks")


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
    s = get_settings()

    client = LabelStudioClient(
        url=s.label_studio.url,
        api_key=s.label_studio.api_key,
    )

    tasks = client.get_tasks(project_id)
    logger.info("fetched_tasks", project_id=project_id, count=len(tasks))

    skipped_tasks: list[tuple[int, str]] = []
    for task in tasks:
        is_cancelled = any(ann.get("was_cancelled", False) for ann in task.annotations)
        if is_cancelled:
            image_path = client.get_local_path(task)
            path_str = str(image_path) if image_path else "unknown"
            skipped_tasks.append((task.id, path_str))
            logger.debug("found_cancelled_task", task_id=task.id, path=path_str)

    logger.info("scan_complete", total_tasks=len(tasks), cancelled=len(skipped_tasks))

    if not skipped_tasks:
        typer.echo("No cancelled tasks found.")
        return

    existing_files: list[tuple[int, str]] = []
    missing_files: list[tuple[int, str]] = []

    for task_id, path_str in skipped_tasks:
        from pathlib import Path

        path = Path(path_str)
        if path.exists():
            existing_files.append((task_id, path_str))
        else:
            missing_files.append((task_id, path_str))

    if dry_run:
        typer.echo(f"\n--- DRY RUN: Would delete {len(existing_files)} files ---\n")
        for task_id, path in existing_files:
            typer.echo(f"  Task {task_id}: {path}")
    else:
        deleted_count = 0
        for task_id, path_str in existing_files:
            try:
                Path(path_str).unlink()
                deleted_count += 1
                logger.info("deleted_file", task_id=task_id, path=path_str)
            except Exception as e:
                logger.error("delete_failed", task_id=task_id, error=str(e))
        typer.echo(f"\nDeleted {deleted_count} files.")

    if missing_files:
        typer.echo(
            f"\n--- Cancelled tasks with missing local files ({len(missing_files)}) ---"
        )
        for task_id, path in missing_files:
            typer.echo(f"  Task {task_id}: {path}")


if __name__ == "__main__":
    app()
