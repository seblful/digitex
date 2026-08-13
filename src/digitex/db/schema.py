"""Locating the migrations, wherever the package was installed.

``alembic.ini`` and ``migrations/`` ship inside this package, so they are found
through :mod:`importlib.resources` rather than by walking up from a source
file's ``__file__``. That is what lets the production image install the project
as an ordinary wheel: nothing has to guess where a checkout was, and the
container needs no copy of ``src/``.

Both the ``digitex-db`` CLI and the integration test suite build their Alembic
config here, so there is one answer to "where are the migrations" rather than
two that can drift apart.
"""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alembic.config import Config

PACKAGE = "digitex.db"


def alembic_config() -> Config:
    """Build an Alembic config pointed at the packaged migration scripts.

    The paths are used directly rather than through
    :func:`importlib.resources.as_file`, because Alembic needs a
    ``script_location`` that outlives a context manager. Wheels are always
    unpacked on install, so the resources are real files — this would only
    break if the project were ever run from a zipapp.
    """
    # Imported here rather than at module scope: the bot imports digitex.db for
    # its connection pool and never migrates, and Alembic drags in SQLAlchemy.
    from alembic.config import Config

    package = files(PACKAGE)
    config = Config(str(package / "alembic.ini"))
    config.set_main_option("script_location", str(package / "migrations"))
    return config
