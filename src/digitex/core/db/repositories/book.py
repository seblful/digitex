"""Reads/writes for subjects, books, and options."""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.core.db.repositories._common import (
    SubjectRow,
    _get_or_create,
)

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn


class BookRepository:
    """Reads/writes for subjects, books, and options."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_or_create_subject(self, name: str) -> int:
        return await _get_or_create(
            self._conn, "subjects", "subject_id", {"name": name}
        )

    async def get_book(self, subject_id: int, year: int) -> int | None:
        cur = await self._conn.execute(
            "SELECT book_id FROM books WHERE subject_id = %s AND year_value = %s",
            (subject_id, year),
        )
        row = await cur.fetchone()
        return row["book_id"] if row else None

    async def create_book(self, subject_id: int, year: int) -> int:
        cur = await self._conn.execute(
            "INSERT INTO books (subject_id, year_value) VALUES (%s, %s)"
            " RETURNING book_id",
            (subject_id, year),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["book_id"]

    async def get_or_create_option(
        self,
        book_id: int,
        option_number: int,
        exam_type: str = "CT",
    ) -> int:
        cur = await self._conn.execute(
            "INSERT INTO options (book_id, option_number, exam_type)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (book_id, option_number)"
            " DO UPDATE SET exam_type = EXCLUDED.exam_type"
            " RETURNING option_id",
            (book_id, option_number, exam_type),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["option_id"]

    async def list_subjects(self) -> list[SubjectRow]:
        cur = await self._conn.execute(
            "SELECT subject_id, name FROM subjects ORDER BY name"
        )
        rows = await cur.fetchall()
        return [SubjectRow(r["subject_id"], r["name"]) for r in rows]

    async def list_years(self, subject_id: int) -> list[int]:
        cur = await self._conn.execute(
            "SELECT year_value FROM books"
            " WHERE subject_id = %s ORDER BY year_value DESC",
            (subject_id,),
        )
        rows = await cur.fetchall()
        return [r["year_value"] for r in rows]

    async def list_options(self, book_id: int, exam_type: str) -> list[int]:
        cur = await self._conn.execute(
            "SELECT option_number FROM options"
            " WHERE book_id = %s AND exam_type = %s ORDER BY option_number",
            (book_id, exam_type),
        )
        rows = await cur.fetchall()
        return [r["option_number"] for r in rows]

    async def get_option_id(self, book_id: int, option_number: int) -> int:
        cur = await self._conn.execute(
            "SELECT option_id FROM options WHERE book_id = %s AND option_number = %s",
            (book_id, option_number),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Option {option_number} not found for book {book_id}")
        return row["option_id"]


__all__ = ["BookRepository"]
