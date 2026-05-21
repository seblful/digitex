"""Async PostgreSQL connection pool factory and lifespan helper."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from digitex.config.settings import DatabaseSettings


def build_pool(settings: DatabaseSettings) -> AsyncConnectionPool:
    """Build an *unopened* async connection pool.

    Callers must use it as a context manager (``async with build_pool(...) as
    pool: ...``) or call ``await pool.open()`` / ``await pool.close()``
    explicitly. We pass ``open=False`` so pool creation does not perform I/O at
    import time.
    """
    return AsyncConnectionPool(
        conninfo=settings.conninfo,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        timeout=settings.pool_timeout,
        kwargs={
            "autocommit": False,
            "row_factory": dict_row,
            "options": settings.server_options,
        },
        open=False,
    )


@asynccontextmanager
async def pool_lifespan(
    settings: DatabaseSettings,
) -> AsyncIterator[AsyncConnectionPool]:
    """Open the pool, yield it, and close it on exit.

    Used by ``cli/bot.py`` to scope the pool to the application lifetime.
    """
    pool = build_pool(settings)
    await pool.open()
    try:
        # Verify connectivity early so misconfiguration fails fast.
        await pool.wait()
        yield pool
    finally:
        await pool.close()
