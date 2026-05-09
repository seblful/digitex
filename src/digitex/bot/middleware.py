"""Auth middleware — blocks unauthorized callback queries."""

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, TelegramObject

from digitex.bot.database import with_uow


class AuthMiddleware(BaseMiddleware):
    """Outer middleware that blocks non-authorized users from using inline keyboards.

    Unauthorized users can still send text messages (needed for registration),
    but their callback queries are silently dropped so they can't interact
    with inline keyboards (subject selection, answers, etc.).
    """

    def __init__(self, admin_user_id: int, db_path: str) -> None:
        self._admin_user_id = admin_user_id
        self._db_path = db_path

    async def __call__(
        self,
        handler,
        event: TelegramObject,
        data: dict,
    ) -> None:
        user = data.get("event_from_user")
        if user is None:
            await handler(event, data)
            return

        telegram_id = user.id

        # Admin always passes
        if telegram_id == self._admin_user_id:
            await handler(event, data)
            return

        # Check DB authorization
        authorized = await with_uow(
            self._db_path, lambda uow: uow.authorized_users.is_authorized(telegram_id)
        )
        if authorized:
            await handler(event, data)
            return

        # Block callbacks from unauthorized users (inline keyboard interactions)
        if isinstance(event, CallbackQuery):
            return

        # Everything else (text messages, /start, etc.) — let through
        await handler(event, data)
