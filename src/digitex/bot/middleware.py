"""Auth middleware — blocks unauthorized callback queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, TelegramObject

from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool


class AuthMiddleware(BaseMiddleware):
    """Outer middleware that blocks non-authorized users from using inline keyboards.

    Unauthorized users can still send text messages (needed for registration),
    but their callback queries are silently dropped so they can't interact
    with inline keyboards (subject selection, answers, etc.).
    """

    def __init__(self, admin_user_id: int, pool: AsyncConnectionPool) -> None:
        self._admin_user_id = admin_user_id
        self._pool = pool

    async def __call__(
        self,
        handler,
        event: TelegramObject,
        data: dict,
    ) -> None:
        # Text messages (/start, /help, registration flow) always pass through —
        # their own handlers decide what to do with unauthorized users.
        if not isinstance(event, CallbackQuery):
            await handler(event, data)
            return

        user = data.get("event_from_user")
        if user is None:
            await handler(event, data)
            return

        telegram_id = user.id

        if telegram_id == self._admin_user_id:
            await handler(event, data)
            return

        async with UnitOfWork(self._pool) as uow:
            authorized = await uow.authorized_users.is_authorized(telegram_id)
        if not authorized:
            return

        await handler(event, data)
