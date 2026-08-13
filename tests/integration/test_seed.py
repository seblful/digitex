"""Integration tests for loading the corpus and reconciling it afterwards.

Question images are files, not rows, so the database can disagree with the disk
it was seeded from. These drive ``populate`` over a small real corpus and then
make each way it can drift actually happen.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from digitex.db.seed import check_images, populate

if TYPE_CHECKING:
    from pathlib import Path

    from psycopg_pool import AsyncConnectionPool

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("clean_db")]

_A1 = "biology/2016/1/A/1.jpg"
_B1 = "biology/2016/1/B/1.jpg"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """One subject, one year, one option, two questions."""
    year_dir = tmp_path / "biology" / "2016"
    (year_dir / "1" / "A").mkdir(parents=True)
    (year_dir / "1" / "B").mkdir()
    (tmp_path / _A1).write_bytes(b"part-a-image")
    (tmp_path / _B1).write_bytes(b"part-b-image")
    (year_dir / "answers.json").write_text(
        json.dumps({"1": {"A1": "3", "B1": "protein"}}), encoding="utf-8"
    )
    return tmp_path


class TestPopulateAndCheck:
    async def test_a_freshly_seeded_corpus_reconciles_clean(
        self, pg_pool: AsyncConnectionPool, corpus: Path
    ) -> None:
        await populate(pg_pool, corpus, "biology")

        result = await check_images(pg_pool, corpus)

        assert result.ok
        assert (result.missing, result.stale, result.orphaned) == ([], [], [])

    async def test_keys_are_stored_relative_to_the_corpus_root(
        self, pg_pool: AsyncConnectionPool, corpus: Path
    ) -> None:
        """Nothing absolute, or the rows would only resolve on this machine."""
        from digitex.db import UnitOfWork

        await populate(pg_pool, corpus, "biology")

        async with UnitOfWork(pg_pool) as uow:
            images = await uow.questions.list_images()

        assert [key for key, _hash in images] == [_A1, _B1]

    async def test_a_deleted_file_is_reported_missing(
        self, pg_pool: AsyncConnectionPool, corpus: Path
    ) -> None:
        await populate(pg_pool, corpus, "biology")
        (corpus / _A1).unlink()

        result = await check_images(pg_pool, corpus)

        assert result.missing == [_A1]
        assert not result.ok

    async def test_an_edited_file_is_reported_stale(
        self, pg_pool: AsyncConnectionPool, corpus: Path
    ) -> None:
        """Synced without re-seeding: same path, different bytes."""
        await populate(pg_pool, corpus, "biology")
        (corpus / _A1).write_bytes(b"re-extracted")

        result = await check_images(pg_pool, corpus)

        assert result.stale == [_A1]
        assert result.missing == []

    async def test_an_unseeded_file_is_reported_orphaned(
        self, pg_pool: AsyncConnectionPool, corpus: Path
    ) -> None:
        await populate(pg_pool, corpus, "biology")
        (corpus / "biology" / "2016" / "1" / "A" / "2.jpg").write_bytes(b"new")

        result = await check_images(pg_pool, corpus)

        assert result.orphaned == ["biology/2016/1/A/2.jpg"]

    async def test_re_seeding_an_edited_file_clears_the_drift(
        self, pg_pool: AsyncConnectionPool, corpus: Path
    ) -> None:
        """The fix the report names actually fixes it."""
        await populate(pg_pool, corpus, "biology")
        (corpus / _A1).write_bytes(b"re-extracted")
        await populate(pg_pool, corpus, "biology")

        assert (await check_images(pg_pool, corpus)).ok
