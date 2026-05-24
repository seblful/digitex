"""Repository for Telegram users (students)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.core.db.mapping import row_to_model
from digitex.core.domain import Student

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn


class StudentRepository:
    """Repository for Telegram users (students)."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_or_create(
        self, telegram_id: int, name: str, username: str | None = None
    ) -> Student:
        # One round-trip get-or-create: if the row exists, RETURNING fires via
        # the no-op DO UPDATE; if it doesn't, the INSERT fires. The row's
        # original ``created_at`` is preserved because we don't touch it.
        cur = await self._conn.execute(
            "INSERT INTO students (telegram_id, name, username)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (telegram_id)"
            " DO UPDATE SET name = EXCLUDED.name, username = EXCLUDED.username"
            " RETURNING student_id, telegram_id, name, username",
            (telegram_id, name, username),
        )
        row = await cur.fetchone()
        assert row is not None
        return row_to_model(row, Student)


__all__ = ["StudentRepository"]
