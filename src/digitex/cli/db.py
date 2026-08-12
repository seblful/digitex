"""Database CLI — thin wrapper over Alembic.

Examples:
--------
    uv run digitex-db upgrade            # upgrade to head
    uv run digitex-db downgrade -1       # revert one revision
    uv run digitex-db current            # show current revision
    uv run digitex-db history            # list all revisions
    uv run digitex-db revision "msg"     # new hand-written revision
"""

from __future__ import annotations

import typer
from alembic import command
from alembic.config import Config

from digitex.config import BASE_DIR

app = typer.Typer(help="Alembic-backed database migrations.")


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


if __name__ == "__main__":
    app()
