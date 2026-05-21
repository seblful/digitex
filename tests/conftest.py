"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
import structlog
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from psycopg_pool import AsyncConnectionPool

logger = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# PostgreSQL fixtures (require Docker; tests are skipped if testcontainers
# or Docker aren't available).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    """Start a Postgres container and yield its DSN.

    The container is started once per test session. Tests that depend on
    Postgres are skipped automatically if Docker or testcontainers are missing.
    """
    testcontainers = pytest.importorskip("testcontainers.postgres")
    container = testcontainers.PostgresContainer("postgres:17-alpine")
    try:
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
    from digitex.config import settings as _settings_mod

    _settings_mod._settings = None

    try:
        _run_migrations()
        yield dsn
    finally:
        if prev_db_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev_db_url
        _settings_mod._settings = None
        container.stop()


def _run_migrations() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_pool(pg_dsn: str) -> AsyncIterator[AsyncConnectionPool]:
    """Open an :class:`AsyncConnectionPool` against the test container."""
    from digitex.config import get_settings
    from digitex.core.db import build_pool

    pool = build_pool(get_settings().database)
    await pool.open()
    await pool.wait()
    try:
        yield pool
    finally:
        await pool.close()


_TABLES = (
    "session_answers",
    "test_sessions",
    "authorized_users",
    "question_topics",
    "images",
    "part_b_questions",
    "part_a_questions",
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


@pytest.fixture
def sample_image(tmp_path: Path) -> Image.Image:
    """Create a sample RGB image for testing.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        A sample PIL Image.
    """
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)
    return img


@pytest.fixture
def sample_label_file(tmp_path: Path) -> Path:
    """Create a sample YOLO label file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Path to the label file.
    """
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True)

    label_path = label_dir / "image_0.txt"
    label_content = """0 0.5 0.5 0.8 0.8
1 0.2 0.2 0.3 0.3
"""
    label_path.write_text(label_content)

    return label_path


@pytest.fixture
def sample_classes_file(tmp_path: Path) -> Path:
    """Create a sample classes.txt file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Path to the classes file.
    """
    classes_path = tmp_path / "classes.txt"
    classes_content = """question
answer
"""
    classes_path.write_text(classes_content)

    return classes_path


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a sample dataset directory structure.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Path to the dataset directory.
    """
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    classes_path = data_dir / "classes.txt"
    classes_path.write_text("question\nanswer\n")

    return data_dir
