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
    output_dir = tmp_path / "output"
    year_dir = output_dir / "biology" / "2016"
    (year_dir / "1" / "A").mkdir(parents=True)
    (year_dir / "1" / "B").mkdir()
    (output_dir / _A1).write_bytes(b"part-a-image")
    (output_dir / _B1).write_bytes(b"part-b-image")
    (year_dir / "answers.json").write_text(
        json.dumps({"1": {"A1": "3", "B1": "protein"}}), encoding="utf-8"
    )
    return output_dir


@pytest.fixture
def books(tmp_path: Path) -> Path:
    """The book archive root, which a corpus need not have a topic map in."""
    books_dir = tmp_path / "books"
    books_dir.mkdir()
    return books_dir


class TestPopulateAndCheck:
    async def test_a_freshly_seeded_corpus_reconciles_clean(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        await populate(pg_pool, corpus, books, "biology")

        result = await check_images(pg_pool, corpus)

        assert result.ok
        assert (result.missing, result.stale, result.orphaned) == ([], [], [])

    async def test_keys_are_stored_relative_to_the_corpus_root(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        """Nothing absolute, or the rows would only resolve on this machine."""
        from digitex.db import UnitOfWork

        await populate(pg_pool, corpus, books, "biology")

        async with UnitOfWork(pg_pool) as uow:
            images = await uow.questions.list_images()

        assert [key for key, _hash in images] == [_A1, _B1]

    async def test_a_deleted_file_is_reported_missing(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        await populate(pg_pool, corpus, books, "biology")
        (corpus / _A1).unlink()

        result = await check_images(pg_pool, corpus)

        assert result.missing == [_A1]
        assert not result.ok

    async def test_an_edited_file_is_reported_stale(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        """Synced without re-seeding: same path, different bytes."""
        await populate(pg_pool, corpus, books, "biology")
        (corpus / _A1).write_bytes(b"re-extracted")

        result = await check_images(pg_pool, corpus)

        assert result.stale == [_A1]
        assert result.missing == []

    async def test_an_unseeded_file_is_reported_orphaned(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        await populate(pg_pool, corpus, books, "biology")
        (corpus / "biology" / "2016" / "1" / "A" / "2.jpg").write_bytes(b"new")

        result = await check_images(pg_pool, corpus)

        assert result.orphaned == ["biology/2016/1/A/2.jpg"]

    async def test_re_seeding_an_edited_file_clears_the_drift(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        """The fix the report names actually fixes it."""
        await populate(pg_pool, corpus, books, "biology")
        (corpus / _A1).write_bytes(b"re-extracted")
        await populate(pg_pool, corpus, books, "biology")

        assert (await check_images(pg_pool, corpus)).ok


class TestPopulateTopics:
    """The topic map is hand-written, so it lives in the book archive."""

    async def test_the_topic_map_is_read_from_the_book_archive(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        from digitex.db import UnitOfWork
        from digitex.db.seed import get_subject_name

        (books / "biology").mkdir()
        (books / "biology" / "topics.json").write_text(
            json.dumps({"Клетка": {"2016": {"CT": ["A1"]}}}, ensure_ascii=False),
            encoding="utf-8",
        )

        await populate(pg_pool, corpus, books, "biology")

        async with UnitOfWork(pg_pool) as uow:
            subject_id = await uow.books.get_or_create_subject(
                get_subject_name("biology")
            )
            assert await uow.questions.get_topics_for_subject(subject_id) == ["Клетка"]
            assert await uow.questions.count_topics() == 1

    async def test_a_subject_without_a_topic_map_still_seeds(
        self, pg_pool: AsyncConnectionPool, corpus: Path, books: Path
    ) -> None:
        """Topics are optional — a missing file must not lose the questions."""
        from digitex.db import UnitOfWork

        await populate(pg_pool, corpus, books, "biology")

        async with UnitOfWork(pg_pool) as uow:
            assert len(await uow.questions.list_images()) == 2
            assert await uow.questions.count_topics() == 0
