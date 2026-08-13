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
"""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated

import typer
from alembic import command
from alembic.config import Config

from digitex.config import BASE_DIR, get_settings

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
    from digitex.core.db import null_pool_lifespan
    from digitex.core.seed import populate as populate_db

    command.upgrade(_cfg(), "head")

    settings = get_settings()
    output_dir = settings.paths.extraction_output_dir
    if not output_dir.exists():
        typer.echo(
            typer.style(
                f"Extraction output not found: {output_dir}", fg="red", bold=True
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        # The null pool, not the app's: this is a one-shot command, and
        # AsyncConnectionPool's background workers stall on Windows.
        async with null_pool_lifespan(settings.database) as pool:
            await populate_db(pool, output_dir, subject)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run())
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
