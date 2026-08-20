"""Subjects, books and options — the three levels above a question.

One class, not three: these rows are reference data written together by the
seeder and walked together by the bot's navigation keyboards, so splitting them
by table would only spread one role across three files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.domain.entities import SubjectRow

if TYPE_CHECKING:
    from digitex.db.mapping import DictConn
    from digitex.domain.entities import ExamType


class BookRepository:
    """The corpus above a question: which subjects, years and options exist."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_or_create_subject(self, name: str) -> int:
        """Return the id of the subject called *name*, creating it if new."""
        # DO UPDATE rather than DO NOTHING, and the update is deliberately a
        # no-op: RETURNING yields no row for a conflict that changed nothing,
        # and the id is needed whether or not this call was the one that
        # inserted.
        cur = await self._conn.execute(
            """
            INSERT INTO subjects (name) VALUES (%s)
                 ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
              RETURNING subject_id
            """,
            (name,),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["subject_id"]

    async def get_book(self, subject_id: int, year: int) -> int | None:
        """The id of a subject's book for *year*, or None if there is none."""
        cur = await self._conn.execute(
            """
            SELECT book_id
              FROM books
             WHERE subject_id = %s AND year_value = %s
            """,
            (subject_id, year),
        )
        row = await cur.fetchone()
        return row["book_id"] if row else None

    async def create_book(self, subject_id: int, year: int) -> int:
        """Create a book for one subject-year.

        Plain INSERT: ``(subject_id, year_value)`` is unique, so calling this
        for a year that already has a book is a bug the constraint should
        report. Callers that do not know check :meth:`get_book` first.
        """
        cur = await self._conn.execute(
            """
            INSERT INTO books (subject_id, year_value) VALUES (%s, %s)
              RETURNING book_id
            """,
            (subject_id, year),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["book_id"]

    async def get_or_create_option(
        self,
        book_id: int,
        option_number: int,
        exam_type: ExamType = "CT",
    ) -> int:
        """Return the id of a book's option, creating it if new.

        A book identifies an option by its number alone, so the exam type is
        settled on conflict rather than treated as part of the identity: this is
        the one upsert here whose UPDATE does real work. Which type an option
        belongs to is derived from its year and number by
        ``domain.entities.exam_type_for``, and a year that gains the CE/CT split
        must be able to correct rows seeded before it did.
        """
        cur = await self._conn.execute(
            """
            INSERT INTO options (book_id, option_number, exam_type)
                 VALUES (%s, %s, %s)
                 ON CONFLICT (book_id, option_number)
                 DO UPDATE SET exam_type = EXCLUDED.exam_type
              RETURNING option_id
            """,
            (book_id, option_number, exam_type),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["option_id"]

    async def list_subjects(self) -> list[SubjectRow]:
        """Every subject, alphabetically — the order the keyboard shows."""
        cur = await self._conn.execute(
            """
            SELECT subject_id, name
              FROM subjects
             ORDER BY name
            """
        )
        rows = await cur.fetchall()
        return [SubjectRow(row["subject_id"], row["name"]) for row in rows]

    async def list_years(self, subject_id: int) -> list[int]:
        """A subject's years, newest first — the recent exam is the likely pick."""
        cur = await self._conn.execute(
            """
            SELECT year_value
              FROM books
             WHERE subject_id = %s
             ORDER BY year_value DESC
            """,
            (subject_id,),
        )
        rows = await cur.fetchall()
        return [row["year_value"] for row in rows]

    async def list_options(self, book_id: int, exam_type: ExamType) -> list[int]:
        """The option numbers of one book that belong to *exam_type*.

        Numbers, not ids: the student picks a number off a keyboard, and
        :meth:`get_option_id` turns the choice back into a row.
        """
        cur = await self._conn.execute(
            """
            SELECT option_number
              FROM options
             WHERE book_id = %s AND exam_type = %s
             ORDER BY option_number
            """,
            (book_id, exam_type),
        )
        rows = await cur.fetchall()
        return [row["option_number"] for row in rows]

    async def get_option_id(self, book_id: int, option_number: int) -> int:
        """Resolve a book and an option number to the option's id.

        No exam type: the number is unique within the book, and the type is a
        property of the row rather than part of how it is addressed.

        Raises:
            KeyError: If the book has no option with that number.
        """
        cur = await self._conn.execute(
            """
            SELECT option_id
              FROM options
             WHERE book_id = %s AND option_number = %s
            """,
            (book_id, option_number),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Option {option_number} not found for book {book_id}")
        return row["option_id"]


__all__ = ["BookRepository"]
