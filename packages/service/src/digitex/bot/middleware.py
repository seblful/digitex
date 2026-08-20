"""What an update passes through before a handler ever sees it.

Two narrowings, each done once here instead of at the top of every handler:
:class:`AuthMiddleware` decides whether a tap is allowed to act at all, and
:class:`AccessibleMessageMiddleware` turns the optional message behind a
callback into one a handler can declare as ``Message``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from aiogram import BaseMiddleware
from aiogram.dispatcher.event.bases import UNHANDLED
from aiogram.types import CallbackQuery, Message, TelegramObject

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from digitex.domain.ports import OpenUow


class AuthMiddleware(BaseMiddleware):
    """Outer middleware that blocks non-authorized users from using inline keyboards.

    Text messages always pass — registration is a conversation an unauthorized
    user has to be able to hold — so the gate is on callback queries alone: an
    unapproved student can talk to the bot but cannot tap a subject, a mode or
    an answer.

    The admin is let through on their configured id rather than on a row, so a
    fresh deployment has someone who can approve the first student.
    """

    def __init__(self, admin_user_id: int, open_uow: OpenUow) -> None:
        self._admin_user_id = admin_user_id
        self._open_uow = open_uow

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # The handler's own result is returned untouched: it is how UNHANDLED
        # propagates back to the dispatcher, and how a webhook-mode reply is
        # sent. Returning None instead would read as "handled".
        if isinstance(event, CallbackQuery):
            user = data.get("event_from_user")
            if user is not None and user.id != self._admin_user_id:
                async with self._open_uow() as uow:
                    if not await uow.students.is_authorized(user.id):
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
