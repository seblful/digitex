"""Middleware — what every callback query passes through before a handler.

``AuthMiddleware`` drops taps from users who are not authorized;
``AccessibleMessageMiddleware`` narrows the optional ``CallbackQuery.message``
once, so handlers can declare a real ``Message`` and stop re-checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from aiogram import BaseMiddleware
from aiogram.dispatcher.event.bases import UNHANDLED
from aiogram.types import CallbackQuery, Message, TelegramObject

from digitex.db import UnitOfWork

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # The handler's result rides back to the dispatcher — it is how
        # UNHANDLED propagates, and how a webhook-mode reply would be sent.

        # Text messages (/start, /help, registration flow) always pass through —
        # their own handlers decide what to do with unauthorized users.
        if not isinstance(event, CallbackQuery):
            return await handler(event, data)

        user = data.get("event_from_user")
        if user is None:
            return await handler(event, data)

        telegram_id = user.id

        if telegram_id == self._admin_user_id:
            return await handler(event, data)

        async with UnitOfWork(self._pool) as uow:
            authorized = await uow.students.is_authorized(telegram_id)
        if not authorized:
            return UNHANDLED

        return await handler(event, data)


class AccessibleMessageMiddleware(BaseMiddleware):
    """Give callback handlers the message their keyboard is attached to, as ``msg``.

    ``CallbackQuery.message`` is optional: Telegram sends an
    ``InaccessibleMessage`` once the original is older than 48 hours or has been
    deleted, and there is nothing a handler can do with one. Acknowledging those
    taps here — instead of in every callback handler — is what lets a handler
    declare ``msg: Message`` and get one.

    Register on the ``callback_query`` observer only; every event it sees is a
    :class:`CallbackQuery`.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        callback = cast("CallbackQuery", event)

        if not isinstance(callback.message, Message):
            # Ack so the client stops spinning; there is nothing to edit here.
            await callback.answer()
            return None

        data["msg"] = callback.message
        return await handler(event, data)
