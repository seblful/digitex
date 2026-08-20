"""One borrowed connection, one transaction, and the repositories inside it.

Every write in the application goes through here, so the boundary of a
transaction is always the boundary of an ``async with``::

    async with UnitOfWork(pool) as uow:
        subject_id = await uow.books.get_or_create_subject("Биология")
        book_id = await uow.books.create_book(subject_id, 2016)

The repositories are attributes rather than arguments, and they are built on
entry — an unentered unit of work has none, because there is no connection for
them to share yet. Nothing here mentions the protocols the bot is written
against; the fit is structural on purpose.
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
    """Async context manager wrapping a single transaction.

    Commit and rollback are psycopg's ``conn.transaction()`` block, not manual
    ``commit()`` / ``rollback()`` calls: a clean exit commits, an exception
    rolls back, and there is no third path for a bug to take.
    """

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool
        # Replaced on entry by the stack that owns the connection and the
        # transaction. Present from construction so that exiting a unit of work
        # which never entered is a no-op rather than an AttributeError.
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> UnitOfWork:
        """Borrow a connection, begin its transaction, hand out the roles."""
        async with AsyncExitStack() as stack:
            raw_conn = await stack.enter_async_context(self._pool.connection())
            await stack.enter_async_context(raw_conn.transaction())
            # Ownership moves out of the local stack only once both succeeded,
            # so a transaction that fails to begin returns its connection to the
            # pool on the way out instead of leaking it.
            self._stack = stack.pop_all()

        # ``build_pool`` configures ``row_factory=dict_row``, but psycopg's stubs
        # type a connection's rows as tuples. Casting once, here, is what lets
        # every repository write ``row["column"]`` without a cast of its own.
        conn = cast("DictConn", raw_conn)

        self.books = BookRepository(conn)
        self.students = StudentRepository(conn)
        self.sessions = SessionRepository(conn)
        # The five roles a question is addressed through.
        self.questions = QuestionCatalog(conn)
        self.draw = QuestionDraw(conn)
        self.topics = TopicIndex(conn)
        self.file_ids = FileIdCache(conn)
        self.corpus = QuestionCorpus(conn)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Close the transaction, reporting whether it swallowed the exception.

        ``bool`` rather than None, and that is not pedantry: psycopg's
        transaction manager suppresses a ``Rollback`` raised inside the block, so
        an ``async with`` over a unit of work genuinely can complete without its
        body having. Code after the block that reports what was written has to be
        able to tell.
        """
        return bool(await self._stack.__aexit__(exc_type, exc_val, exc_tb))
