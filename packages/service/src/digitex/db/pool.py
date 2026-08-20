"""Building the connection pool the whole layer borrows connections from.

Two kinds of pool, because two kinds of process use one. A long-running bot
wants the real :class:`AsyncConnectionPool`, which keeps connections warm in
background worker tasks. A one-shot command — a migration, a seed, the test
suite on Windows — wants :class:`AsyncNullConnectionPool`, because those same
workers stall on a Windows event loop even a selector one.

Transactions are not here: opening one and handing out the repositories inside
it is :class:`~digitex.db.unit_of_work.UnitOfWork`'s job. This module only
answers "where do connections come from".
"""

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
    """How psycopg must be told to build a connection for this layer.

    ``row_factory=dict_row`` is load-bearing rather than a preference: every
    repository indexes rows by column name, and the unit of work casts to
    ``DictConn`` on the strength of it. Built without this, a pool type-checks
    and then fails inside the first query. Both pools below go through here so
    neither can be the one that forgets.

    ``autocommit=False`` leaves commit boundaries to the transaction block the
    unit of work opens.
    """
    return {
        "autocommit": False,
        "row_factory": dict_row,
        "options": settings.server_options,
    }


def build_pool(settings: DatabaseSettings) -> AsyncConnectionPool:
    """Build the real pool, *unopened*.

    ``open=False`` keeps construction free of I/O, so a module may hold a pool
    without a database having to be reachable when it is imported. The caller
    opens it — through :func:`pool_lifespan`, as an ``async with``, or with
    explicit ``open()`` / ``close()`` calls.
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
    """Scope a real pool to a block — ``cli/bot.py`` scopes it to the process.

    ``wait()`` blocks until ``min_size`` connections are established, which
    turns a bad DSN or an unreachable server into a failure at startup instead
    of one inside the first handler a student triggers.
    """
    pool = build_pool(settings)
    await pool.open()
    try:
        await pool.wait()
        yield pool
    finally:
        await pool.close()


@asynccontextmanager
async def null_pool_lifespan(
    settings: DatabaseSettings,
) -> AsyncIterator[AsyncNullConnectionPool]:
    """Scope a null pool to a block: a connection per acquire, no workers.

    For anything short-lived — the ``digitex-db`` commands, the integration
    suite on Windows — and for any process whose event loop the real pool's
    background tasks do not survive. There is no ``wait()`` to make: a null pool
    holds nothing open, so the first acquire is the first connection attempt.

    ``pool_max_size`` and ``pool_timeout`` still apply. ``pool_min_size`` cannot
    and is not passed — a pool that keeps no idle connections has no floor.
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
