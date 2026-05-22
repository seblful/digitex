"""Bot entrypoint."""

from __future__ import annotations

import asyncio

import structlog

from digitex.bot import create_dispatcher
from digitex.config import get_settings
from digitex.core.db import pool_lifespan
from digitex.logging import setup_logging

# Per ADR 0001 — resolve settings once at the CLI boundary.
_settings = get_settings()
setup_logging(_settings)
logger = structlog.get_logger()


def main() -> None:
    """Start the Telegram bot in polling mode."""
    settings = _settings
    token = settings.bot.token

    if not token:
        logger.error("BOT_TOKEN is not set")
        return

    admin_user_id = settings.bot.admin_user_id

    async def _main() -> None:
        from aiogram import Bot
        from aiogram.types import BotCommand

        from digitex.bot.messages import CMD_HELP_DESC, CMD_START_DESC

        # Log only the safe parts of the DSN — never the full string.
        logger.info(
            "Opening DB pool",
            host=settings.database.dsn.host,
            db=settings.database.dsn.path,
        )

        async with pool_lifespan(settings.database) as pool:
            bot = Bot(token=token)
            await bot.set_my_commands(
                [
                    BotCommand(command="start", description=CMD_START_DESC),
                    BotCommand(command="help", description=CMD_HELP_DESC),
                ]
            )
            dispatcher = create_dispatcher(admin_user_id=admin_user_id, pool=pool)
            logger.info("Starting bot polling...")
            await dispatcher.start_polling(bot, pool=pool, admin_user_id=admin_user_id)

    asyncio.run(_main())


# typer-compatible app object for the project script entry point.
app = main


if __name__ == "__main__":
    main()
