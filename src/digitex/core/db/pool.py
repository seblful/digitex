"""Async PostgreSQL connection pool factory and lifespan helper."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, AsyncNullConnectionPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

    from digitex.config import DatabaseSettings


def _connection_kwargs(settings: DatabaseSettings) -> dict[str, Any]:
    """The psycopg connection arguments every pool here must be built with.

    ``row_factory=dict_row`` is not a preference: the repositories index rows by
    column name and :class:`~digitex.core.db.unit_of_work.UnitOfWork` casts on
    the strength of it. A pool built without it fails deep inside a query.
    """
    return {
        "autocommit": False,
        "row_factory": dict_row,
        "options": settings.server_options,
    }


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
        kwargs=_connection_kwargs(settings),
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


@asynccontextmanager
async def null_pool_lifespan(
    settings: DatabaseSettings,
) -> AsyncIterator[AsyncNullConnectionPool]:
    """Open a null pool (one connection per acquire, no background workers).

    Use this for short-lived scripts and migration tools — anywhere that
    ``AsyncConnectionPool``'s background worker tasks are problematic (e.g.
    Windows SelectorEventLoop). The bot uses ``pool_lifespan`` instead.

    ``pool_max_size`` and ``pool_timeout`` are honoured; ``pool_min_size`` is
    not, and cannot be — a null pool holds no idle connections by definition.
    """
    pool = AsyncNullConnectionPool(
        conninfo=settings.conninfo,
        max_size=settings.pool_max_size,
        timeout=settings.pool_timeout,
        kwargs=_connection_kwargs(settings),
        open=False,
    )
    await pool.open()
    try:
        yield pool
    finally:
        await pool.close()
