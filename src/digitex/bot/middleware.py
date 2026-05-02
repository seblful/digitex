"""Auth middleware — blocks unauthorized users."""

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from digitex.bot.database import with_uow
from digitex.config import get_settings


class AuthMiddleware(BaseMiddleware):
    """Outer middleware that blocks non-authorized users.

    Only lets through:
    - The bot admin (configured via BOT__ADMIN_USER_ID)
    - Authorized users (status = 'approved' in DB)
    - /start and /help messages (so unregistered users can initiate registration)
    """

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

        settings = get_settings()
        telegram_id = user.id

        # Admin always passes
        if telegram_id == settings.bot.admin_user_id:
            await handler(event, data)
            return

        # Check DB authorization
        db_path = settings.database.path
        authorized = await with_uow(db_path, lambda uow: uow.authorized_users.is_authorized(telegram_id))
        if authorized:
            await handler(event, data)
            return

        # Non-authorized — only allow /start (and /help) text messages
        if isinstance(event, Message):
            if event.text and event.text.startswith("/"):
                await handler(event, data)
                return

        # Everything else (callbacks, other messages, etc.) — block
