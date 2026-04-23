"""Unit of Work — manages a single DB connection and transaction."""

from __future__ import annotations

import sqlite3

from digitex.core.db.repositories import (
    BookRepository,
    QuestionRepository,
    SessionRepository,
    StudentRepository,
)


class UnitOfWork:
    """Context manager that wraps a DB transaction.

    Commits on clean exit, rolls back on exception.

    Usage::

        with UnitOfWork(db_path) as uow:
            subject_id = uow.books.get_or_create_subject("biology")
            book_id = uow.books.get_or_create_book(subject_id, 2016)
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def __enter__(self) -> UnitOfWork:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.books = BookRepository(self._conn)
        self.questions = QuestionRepository(self._conn)
        self.students = StudentRepository(self._conn)
        self.sessions = SessionRepository(self._conn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type:
                self._conn.rollback()
            else:
                self._conn.commit()
        finally:
            self._conn.close()
