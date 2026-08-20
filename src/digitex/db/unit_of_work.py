"""Async Unit of Work — one pool connection, one transaction, the repos.

Usage::

    async with UnitOfWork(pool) as uow:
        subject_id = await uow.books.get_or_create_subject("biology")
        book_id = await uow.books.create_book(subject_id, 2016)
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, cast

from digitex.db.repositories import (
    BookRepository,
    FileIdCache,
    QuestionCatalog,
    QuestionCorpus,
    QuestionDraw,
    SessionRepository,
    StudentRepository,
    TopicIndex,
)

if TYPE_CHECKING:
    from types import TracebackType

    from psycopg_pool import AsyncConnectionPool

    from digitex.db.mapping import DictConn


class UnitOfWork:
    """Async context manager that wraps a single transaction.

    psycopg's ``conn.transaction()`` block commits on clean exit and rolls back
    on exception — we delegate transaction lifecycle to it rather than calling
    ``commit()`` / ``rollback()`` manually.
    """

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool
        self._stack = AsyncExitStack()
        self._conn: DictConn | None = None

    async def __aenter__(self) -> UnitOfWork:
        async with AsyncExitStack() as stack:
            raw_conn = await stack.enter_async_context(self._pool.connection())
            await stack.enter_async_context(raw_conn.transaction())
            self._stack = stack.pop_all()
        # The pool is configured with ``row_factory=dict_row`` in
        # ``build_pool``, but psycopg's type stubs default the row type to
        # ``tuple``. Cast at this single boundary so every repository sees
        # ``dict[str, Any]`` rows without per-call ``cast`` noise.
        conn = cast("DictConn", raw_conn)
        self._conn = conn
        self.books = BookRepository(conn)
        self.questions = QuestionCatalog(conn)
        self.draw = QuestionDraw(conn)
        self.topics = TopicIndex(conn)
        self.file_ids = FileIdCache(conn)
        self.corpus = QuestionCorpus(conn)
        self.students = StudentRepository(conn)
        self.sessions = SessionRepository(conn)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return bool(await self._stack.__aexit__(exc_type, exc_val, exc_tb))
