"""Fixtures for integration tests — a real Postgres via testcontainers.

Tests here are skipped automatically when Docker or testcontainers are
missing; run the unit suite alone with ``pytest tests/unit``.

Set ``DIGITEX_TEST_DSN`` to run against an already-running Postgres instead of
a container. **Every table in that database is truncated after each test**, so
point it at a throwaway database and nothing else.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from psycopg_pool import AsyncConnectionPool


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Windows' default ProactorEventLoop is rejected by psycopg."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    """Yield a migrated Postgres DSN, from ``DIGITEX_TEST_DSN`` or a container.

    The container is started once per test session. Tests that depend on
    Postgres are skipped automatically if Docker or testcontainers are missing.
    """
    external = os.environ.get("DIGITEX_TEST_DSN")
    if external:
        yield from _external_dsn(external)
        return

    testcontainers = pytest.importorskip("testcontainers.postgres")
    try:
        # Constructing the container already reaches for the Docker daemon, so
        # both steps have to be guarded or a stopped daemon errors every test
        # instead of skipping it.
        container = testcontainers.PostgresContainer("postgres:17-alpine")
        container.start()
    except Exception as e:
        pytest.skip(f"Cannot start Postgres container (is Docker running?): {e}")

    # testcontainers default URL uses psycopg2 driver; strip the driver suffix.
    url = container.get_connection_url()
    dsn = url.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgres+psycopg2://", "postgresql://"
    )
    prev_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = dsn

    # Clear cached settings so the new DSN is picked up.
    from digitex.config import reset_settings_cache

    reset_settings_cache()

    try:
        _run_migrations()
        yield dsn
    finally:
        if prev_db_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev_db_url
        reset_settings_cache()
        container.stop()


def _external_dsn(dsn: str) -> Iterator[str]:
    """Migrate an already-running Postgres and yield its DSN."""
    prev_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = dsn

    from digitex.config import reset_settings_cache

    reset_settings_cache()
    try:
        _run_migrations()
        yield dsn
    finally:
        if prev_db_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev_db_url
        reset_settings_cache()


def _run_migrations() -> None:
    from alembic import command

    from digitex.db.schema import alembic_config

    # The same config the digitex-db CLI uses, so the suite migrates through
    # the path production does rather than a copy of it.
    command.upgrade(alembic_config(), "head")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_pool(pg_dsn: str) -> AsyncIterator[AsyncConnectionPool]:
    """Open a connection pool against the test container.

    On Windows ``AsyncConnectionPool``'s background workers stall even on the
    SelectorEventLoop, so the null pool stands in there — the same split
    ``cli/bot.py`` makes for the same reason.
    """
    from digitex.config import get_settings
    from digitex.db import build_pool, null_pool_lifespan

    settings = get_settings().database

    if sys.platform == "win32":
        async with null_pool_lifespan(settings) as pool:
            yield pool
        return

    pool = build_pool(settings)
    await pool.open()
    await pool.wait()
    try:
        yield pool
    finally:
        await pool.close()


_TABLES = (
    "session_answers",
    "test_sessions",
    "question_topics",
    "topics",
    "images",
    "questions",
    "options",
    "books",
    "subjects",
    "students",
)


@pytest_asyncio.fixture
async def clean_db(pg_pool: AsyncConnectionPool) -> AsyncIterator[None]:
    """Truncate every table after each test to give per-test isolation.

    Cheaper than dropping/re-creating the schema; ``RESTART IDENTITY`` resets
    sequences so id assignments are deterministic per test.
    """
    yield
    async with pg_pool.connection() as conn, conn.transaction():
        await conn.execute(
            "TRUNCATE TABLE " + ", ".join(_TABLES) + " RESTART IDENTITY CASCADE"
        )
