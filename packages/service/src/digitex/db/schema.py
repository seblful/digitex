"""Finding the migrations, wherever the package ended up installed.

``alembic.ini`` and ``migrations/`` are package data, not repository files, and
they are located through :mod:`importlib.resources` rather than by walking up
from some module's ``__file__``. That is what lets the production image install
``digitex-service`` as an ordinary wheel: nothing has to guess where a checkout
was, and the container carries no ``src/`` tree to guess about.

One function, used by both the ``digitex-db`` CLI and the integration suite, so
"where are the migrations" has a single answer rather than two that can drift.
Applying them is Alembic's job and lives in the CLI; this only points at them.
"""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alembic.config import Config

PACKAGE = "digitex.db"


def alembic_config() -> Config:
    """Build an Alembic config aimed at the packaged migration scripts.

    The resource paths are stringified directly rather than borrowed through
    :func:`importlib.resources.as_file`, because Alembic keeps
    ``script_location`` well past the point a context manager would have closed.
    Installing a wheel unpacks it, so these are real files on disk; only running
    from a zipapp would break the assumption, and nothing does.
    """
    # Deferred on purpose: Alembic pulls in SQLAlchemy, and the bot imports
    # digitex.db for its pool without ever migrating anything.
    from alembic.config import Config

    package = files(PACKAGE)
    config = Config(str(package / "alembic.ini"))
    config.set_main_option("script_location", str(package / "migrations"))
    return config
