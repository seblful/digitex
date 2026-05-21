"""Async Unit of Work — one pool connection, one transaction, five repositories.

Usage::

    async with UnitOfWork(pool) as uow:
        subject_id = await uow.books.get_or_create_subject("biology")
        book_id = await uow.books.get_or_create_book(subject_id, 2016)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from digitex.core.db.repositories import (
    AuthorizedUserRepository,
    BookRepository,
    QuestionRepository,
    SessionRepository,
    StudentRepository,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool


class UnitOfWork:
    """Async context manager that wraps a single transaction.

    psycopg's ``conn.transaction()`` block commits on clean exit and rolls back
    on exception — we delegate transaction lifecycle to it rather than calling
    ``commit()`` / ``rollback()`` manually.
    """

    books: BookRepository
    questions: QuestionRepository
    students: StudentRepository
    sessions: SessionRepository
    authorized_users: AuthorizedUserRepository

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool
        self._conn_cm: Any = None
        self._tx_cm: Any = None
        self._conn: AsyncConnection | None = None

    async def __aenter__(self) -> UnitOfWork:
        conn_cm = self._pool.connection()
        conn = await conn_cm.__aenter__()
        try:
            tx_cm = conn.transaction()
            await tx_cm.__aenter__()
        except BaseException:
            await conn_cm.__aexit__(None, None, None)
            raise
        self._conn_cm = conn_cm
        self._tx_cm = tx_cm
        self._conn = conn
        self.books = BookRepository(conn)
        self.questions = QuestionRepository(conn)
        self.students = StudentRepository(conn)
        self.sessions = SessionRepository(conn)
        self.authorized_users = AuthorizedUserRepository(conn)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            await self._tx_cm.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            await self._conn_cm.__aexit__(exc_type, exc_val, exc_tb)
