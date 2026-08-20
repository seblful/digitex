"""The person behind a Telegram id, and whether they may take a test.

Identity and registration are one role because they are one row. A student has
exactly one status, so the authorization check every callback passes through is a
primary-key lookup and not a join, and there is no state in which someone is
registered twice or approved without existing.

Nothing here deletes. A rejection is overwritten rather than removed — the row
is the person, and their sessions hang off it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.db.mapping import row_to_model
from digitex.domain.entities import Student

if TYPE_CHECKING:
    from digitex.db.mapping import DictConn
    from digitex.domain.entities import RegistrationStatus

# Every column ``Student`` reads, spelled once. Each write below returns the
# whole row, so a caller never has to follow one with a read — and a field added
# to the model but not to this list fails as a ValidationError on the first
# query rather than as a missing attribute somewhere downstream.
_COLUMNS = (
    "telegram_id, telegram_name, telegram_username, full_name,"
    " status, created_at, handled_at, handled_by"
)


class StudentRepository:
    """Telegram users and their authorization to use the bot."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_or_create(
        self,
        telegram_id: int,
        telegram_name: str,
        telegram_username: str | None = None,
    ) -> Student:
        """Refresh what Telegram says about a user, creating the row if new.

        Touches the two identity fields and nothing else: an approved student
        who runs /start again must not be handed back to the registration queue,
        and ``created_at`` survives because no statement here writes it.

        One round-trip either way — a new user is served by the INSERT, an
        existing one by the DO UPDATE, and RETURNING fires on both.
        """
        cur = await self._conn.execute(
            f"""
            INSERT INTO students (telegram_id, telegram_name, telegram_username)
                 VALUES (%s, %s, %s)
                 ON CONFLICT (telegram_id) DO UPDATE
                    SET telegram_name = EXCLUDED.telegram_name,
                        telegram_username = EXCLUDED.telegram_username
              RETURNING {_COLUMNS}
            """,
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
        """File — or re-file — a registration request.

        Re-applying clears the previous decision and keeps the original
        ``created_at``, which is the whole of how a rejected student applies
        again. ``full_name`` is what they typed, and the schema requires it
        before any decision can be recorded, so the application is what supplies
        it.
        """
        cur = await self._conn.execute(
            f"""
            INSERT INTO students
                 (telegram_id, telegram_name, telegram_username, full_name, status)
                 VALUES (%s, %s, %s, %s, 'pending')
                 ON CONFLICT (telegram_id) DO UPDATE
                    SET telegram_name = EXCLUDED.telegram_name,
                        telegram_username = EXCLUDED.telegram_username,
                        full_name = EXCLUDED.full_name,
                        status = 'pending',
                        handled_at = NULL,
                        handled_by = NULL
              RETURNING {_COLUMNS}
            """,
            (telegram_id, telegram_name, telegram_username, full_name),
        )
        row = await cur.fetchone()
        assert row is not None
        return row_to_model(row, Student)

    async def approve(self, telegram_id: int, admin_id: int) -> Student:
        """Let *telegram_id* take tests, on *admin_id*'s decision."""
        return await self._set_status(telegram_id, admin_id, "approved")

    async def reject(self, telegram_id: int, admin_id: int) -> Student:
        """Turn *telegram_id* down, on *admin_id*'s decision."""
        return await self._set_status(telegram_id, admin_id, "rejected")

    async def _set_status(
        self, telegram_id: int, admin_id: int, status: RegistrationStatus
    ) -> Student:
        """Record one decision: the status, the moment, and who made it.

        The three move together because the schema will not have them apart — a
        status other than pending must carry a ``handled_at``, and vice versa.

        ``handled_by`` is a real reference to a students row, so the admin has to
        have one; the callers upsert themselves first.

        Raises:
            KeyError: If no student has that Telegram id — a decision on
                somebody the bot has never seen, which is a caller bug rather
                than a rejection.
        """
        cur = await self._conn.execute(
            f"""
            UPDATE students
               SET status = %s, handled_at = NOW(), handled_by = %s
             WHERE telegram_id = %s
         RETURNING {_COLUMNS}
            """,
            (status, admin_id, telegram_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No student found for {telegram_id}")
        return row_to_model(row, Student)

    async def get(self, telegram_id: int) -> Student | None:
        """The student with that Telegram id, or None if they are unknown."""
        cur = await self._conn.execute(
            f"""
            SELECT {_COLUMNS}
              FROM students
             WHERE telegram_id = %s
            """,
            (telegram_id,),
        )
        row = await cur.fetchone()
        return row_to_model(row, Student) if row else None

    async def is_authorized(self, telegram_id: int) -> bool:
        """Whether this user may take a test.

        Asked on every callback, so it selects a constant and reads no columns:
        the question is whether the row exists in that state, not what is in it.
        """
        cur = await self._conn.execute(
            """
            SELECT 1
              FROM students
             WHERE telegram_id = %s AND status = 'approved'
            """,
            (telegram_id,),
        )
        return await cur.fetchone() is not None


__all__ = ["StudentRepository"]
