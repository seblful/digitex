"""Repository for the registration / approval workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.core.db.mapping import row_to_model
from digitex.core.domain import AuthorizedUser

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn


class AuthorizedUserRepository:
    """Repository for the registration / approval workflow."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_status(self, telegram_id: int) -> str | None:
        cur = await self._conn.execute(
            "SELECT status FROM authorized_users WHERE telegram_id = %s",
            (telegram_id,),
        )
        row = await cur.fetchone()
        return row["status"] if row else None

    async def create_request(
        self,
        telegram_id: int,
        full_name: str,
        telegram_username: str | None = None,
    ) -> AuthorizedUser:
        # Re-applying preserves the original ``created_at`` and clears the
        # handled_at / handled_by fields from any previous decision.
        cur = await self._conn.execute(
            "INSERT INTO authorized_users"
            " (telegram_id, full_name, telegram_username, status)"
            " VALUES (%s, %s, %s, 'pending')"
            " ON CONFLICT (telegram_id) DO UPDATE SET"
            "   full_name = EXCLUDED.full_name,"
            "   telegram_username = EXCLUDED.telegram_username,"
            "   status = 'pending',"
            "   handled_at = NULL,"
            "   handled_by = NULL"
            " RETURNING telegram_id, full_name, telegram_username,"
            "          status, created_at, handled_at, handled_by",
            (telegram_id, full_name, telegram_username),
        )
        row = await cur.fetchone()
        assert row is not None
        return row_to_model(row, AuthorizedUser)

    async def approve(self, telegram_id: int, admin_id: int) -> AuthorizedUser:
        return await self._set_status(telegram_id, admin_id, "approved")

    async def reject(self, telegram_id: int, admin_id: int) -> AuthorizedUser:
        return await self._set_status(telegram_id, admin_id, "rejected")

    async def _set_status(
        self, telegram_id: int, admin_id: int, status: str
    ) -> AuthorizedUser:
        cur = await self._conn.execute(
            "UPDATE authorized_users"
            " SET status = %s, handled_at = NOW(), handled_by = %s"
            " WHERE telegram_id = %s"
            " RETURNING telegram_id, full_name, telegram_username,"
            "          status, created_at, handled_at, handled_by",
            (status, admin_id, telegram_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No registration request found for {telegram_id}")
        return row_to_model(row, AuthorizedUser)

    async def delete_request(self, telegram_id: int) -> None:
        await self._conn.execute(
            "DELETE FROM authorized_users WHERE telegram_id = %s",
            (telegram_id,),
        )

    async def get_request(self, telegram_id: int) -> AuthorizedUser | None:
        cur = await self._conn.execute(
            "SELECT telegram_id, full_name, telegram_username,"
            "       status, created_at, handled_at, handled_by"
            "  FROM authorized_users WHERE telegram_id = %s",
            (telegram_id,),
        )
        row = await cur.fetchone()
        return row_to_model(row, AuthorizedUser) if row else None

    async def is_authorized(self, telegram_id: int) -> bool:
        cur = await self._conn.execute(
            "SELECT 1 FROM authorized_users"
            " WHERE telegram_id = %s AND status = 'approved'",
            (telegram_id,),
        )
        return await cur.fetchone() is not None


__all__ = ["AuthorizedUserRepository"]
