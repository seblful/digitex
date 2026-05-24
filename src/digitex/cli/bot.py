"""Bot entrypoint."""

from __future__ import annotations

import asyncio
import sys

import structlog

from digitex.bot import create_dispatcher
from digitex.config import get_settings
from digitex.core.db import null_pool_lifespan, pool_lifespan
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

    # Local Windows dev only: psycopg rejects ProactorEventLoop, and
    # AsyncConnectionPool's background workers stall on SelectorEventLoop too —
    # so use the SelectorEventLoop policy AND NullConnectionPool (which has no
    # background workers). Linux production uses the real pool.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    _pool_lifespan = null_pool_lifespan if sys.platform == "win32" else pool_lifespan

    async def _main() -> None:
        from aiogram import Bot
        from aiogram.types import BotCommand

        from digitex.bot.messages import CMD_HELP_DESC, CMD_START_DESC

        hosts = settings.database.dsn.hosts()
        logger.info(
            "Opening DB pool",
            host=hosts[0].get("host") if hosts else "unknown",
            db=settings.database.dsn.path,
        )

        async with _pool_lifespan(settings.database) as pool:
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
