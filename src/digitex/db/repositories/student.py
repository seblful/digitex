"""Repository for Telegram users and their authorization to use the bot.

Identity and registration are one aggregate because they are one row: a student
has exactly one status, so the authorization check every callback passes through
is a single primary-key lookup rather than a join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.db.mapping import row_to_model
from digitex.domain.entities import Student

if TYPE_CHECKING:
    from digitex.db.mapping import DictConn
    from digitex.domain.entities import RegistrationStatus

# Every Student field, spelled once — a field added to the model but not here
# comes back as a ValidationError at runtime rather than a type error.
_COLUMNS = (
    "telegram_id, telegram_name, telegram_username, full_name,"
    " status, created_at, handled_at, handled_by"
)


class StudentRepository:
    """Repository for Telegram users and their authorization to use the bot."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_or_create(
        self,
        telegram_id: int,
        telegram_name: str,
        telegram_username: str | None = None,
    ) -> Student:
        """Refresh what Telegram tells us about a user, creating the row if new.

        Touches the identity fields only: an approved student who runs /start
        again must not be handed back to the registration queue. One round-trip —
        if the row exists, RETURNING fires via the DO UPDATE; if it doesn't, the
        INSERT does. ``created_at`` survives because nothing writes it.
        """
        cur = await self._conn.execute(
            "INSERT INTO students (telegram_id, telegram_name, telegram_username)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (telegram_id) DO UPDATE SET"
            "   telegram_name = EXCLUDED.telegram_name,"
            "   telegram_username = EXCLUDED.telegram_username"
            f" RETURNING {_COLUMNS}",
            (telegram_id, telegram_name, telegram_username),
        )
        row = await cur.fetchone()
        assert row is not None
        return row_to_model(row, Student)

    async def create_request(
        self,
        telegram_id: int,
        full_name: str,
        telegram_name: str,
        telegram_username: str | None = None,
    ) -> Student:
        """File (or re-file) a registration request.

        Re-applying preserves the original ``created_at`` and clears any previous
        decision, which is how a rejected student applies again. The rejection is
        overwritten rather than deleted — the row is the person, and they still
        have sessions hanging off it.
        """
        cur = await self._conn.execute(
            "INSERT INTO students"
            " (telegram_id, telegram_name, telegram_username, full_name, status)"
            " VALUES (%s, %s, %s, %s, 'pending')"
            " ON CONFLICT (telegram_id) DO UPDATE SET"
            "   telegram_name = EXCLUDED.telegram_name,"
            "   telegram_username = EXCLUDED.telegram_username,"
            "   full_name = EXCLUDED.full_name,"
            "   status = 'pending',"
            "   handled_at = NULL,"
            "   handled_by = NULL"
            f" RETURNING {_COLUMNS}",
            (telegram_id, telegram_name, telegram_username, full_name),
        )
        row = await cur.fetchone()
        assert row is not None
        return row_to_model(row, Student)

    async def approve(self, telegram_id: int, admin_id: int) -> Student:
        return await self._set_status(telegram_id, admin_id, "approved")

    async def reject(self, telegram_id: int, admin_id: int) -> Student:
        return await self._set_status(telegram_id, admin_id, "rejected")

    async def _set_status(
        self, telegram_id: int, admin_id: int, status: RegistrationStatus
    ) -> Student:
        """Record a decision.

        ``handled_by`` is a real reference, so *admin_id* must already have a
        students row — callers upsert the admin first.
        """
        cur = await self._conn.execute(
            "UPDATE students"
            " SET status = %s, handled_at = NOW(), handled_by = %s"
            " WHERE telegram_id = %s"
            f" RETURNING {_COLUMNS}",
            (status, admin_id, telegram_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No student found for {telegram_id}")
        return row_to_model(row, Student)

    async def get(self, telegram_id: int) -> Student | None:
        cur = await self._conn.execute(
            f"SELECT {_COLUMNS} FROM students WHERE telegram_id = %s",
            (telegram_id,),
        )
        row = await cur.fetchone()
        return row_to_model(row, Student) if row else None

    async def is_authorized(self, telegram_id: int) -> bool:
        cur = await self._conn.execute(
            "SELECT 1 FROM students WHERE telegram_id = %s AND status = 'approved'",
            (telegram_id,),
        )
        return await cur.fetchone() is not None


__all__ = ["StudentRepository"]
