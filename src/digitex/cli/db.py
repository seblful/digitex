"""Database CLI — migrations, and loading the corpus behind them.

Examples:
--------
    uv run digitex-db upgrade            # upgrade to head
    uv run digitex-db downgrade -1       # revert one revision
    uv run digitex-db current            # show current revision
    uv run digitex-db history            # list all revisions
    uv run digitex-db revision "msg"     # new hand-written revision
    uv run digitex-db populate           # load extraction output (migrates first)
    uv run digitex-db populate biology   # ... one subject
    uv run digitex-db check-images       # rows vs. image files on this machine
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Annotated

import typer
from alembic import command
from alembic.config import Config

from digitex.config import BASE_DIR, get_settings

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(help="Alembic-backed migrations and corpus loading.")


def _cfg() -> Config:
    # BASE_DIR, not PathsSettings.root_dir: alembic.ini and migrations/ ship
    # with the package, wherever the process happens to be running from.
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    return cfg


@app.command()
def upgrade(revision: str = "head") -> None:
    """Upgrade schema to the given revision (default: head)."""
    command.upgrade(_cfg(), revision)


@app.command()
def downgrade(revision: str = "-1") -> None:
    """Downgrade schema by one revision (or to a given target)."""
    command.downgrade(_cfg(), revision)


@app.command()
def current() -> None:
    """Print the current revision applied to the database."""
    command.current(_cfg(), verbose=True)


@app.command()
def history() -> None:
    """List the full revision history."""
    command.history(_cfg(), verbose=True)


@app.command()
def revision(message: str) -> None:
    """Create a new (empty, hand-written) revision file."""
    command.revision(_cfg(), message=message, autogenerate=False)


def _require_dir(path: Path, what: str) -> None:
    """Exit with a message rather than a traceback when the corpus is absent."""
    if not path.is_dir():
        typer.echo(
            typer.style(f"{what} not found: {path}", fg="red", bold=True), err=True
        )
        raise typer.Exit(code=1)


@app.command()
def populate(
    subject: Annotated[
        str | None,
        typer.Argument(help="Subject to load; omit to load every subject"),
    ] = None,
) -> None:
    """Load extraction output into the database, migrating the schema first.

    Idempotent — every write is a ``get_or_create``, so re-running after a new
    extraction adds what is new and leaves the rest alone.
    """
    # Imported here, not at module scope: `upgrade` runs in a container that
    # has no corpus, and this pulls the whole data-access layer in with it.
    from digitex.db import null_pool_lifespan
    from digitex.db.seed import populate as populate_db

    command.upgrade(_cfg(), "head")

    settings = get_settings()
    output_dir = settings.paths.extraction_output_dir
    _require_dir(output_dir, "Extraction output")

    async def _run() -> None:
        # The null pool, not the app's: this is a one-shot command, and
        # AsyncConnectionPool's background workers stall on Windows.
        async with null_pool_lifespan(settings.database) as pool:
            await populate_db(pool, output_dir, subject)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run())
    typer.echo("\nDone.")


@app.command("check-images")
def check_images() -> None:
    """Reconcile the ``images`` rows against the image files on this machine.

    Question images live on disk, so a row and its file can drift apart — the
    sync ran without a re-seed, or the other way round. Exits non-zero when
    anything is off, so a deploy can gate on it.
    """
    from digitex.db import null_pool_lifespan
    from digitex.db.seed import ImageCheck
    from digitex.db.seed import check_images as run_check

    settings = get_settings()
    questions_dir = settings.paths.question_images_dir
    _require_dir(questions_dir, "Question images directory")

    async def _run() -> ImageCheck:
        async with null_pool_lifespan(settings.database) as pool:
            return await run_check(pool, questions_dir)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    result = asyncio.run(_run())

    for label, keys in (
        ("missing on disk (sync the corpus)", result.missing),
        ("changed since seeding (run populate)", result.stale),
        ("on disk but unreferenced", result.orphaned),
    ):
        if not keys:
            continue
        typer.echo(typer.style(f"\n{len(keys)} {label}:", fg="yellow", bold=True))
        # The full list is what makes the report actionable, and 4k keys is a
        # page of scrollback, not a problem.
        for key in keys:
            typer.echo(f"  {key}")

    if result.ok:
        typer.echo(typer.style("\nImages and rows agree.", fg="green"))
        return
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
