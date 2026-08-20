"""Bot entrypoint."""

from __future__ import annotations

import sys
from zoneinfo import ZoneInfo

import structlog

from digitex.bot.dispatcher import create_dispatcher
from digitex.cli._shared import run_async
from digitex.config import get_settings
from digitex.db import UnitOfWork, null_pool_lifespan, pool_lifespan
from digitex.logging import setup_logging

logger = structlog.get_logger()


def main() -> None:
    """Start the Telegram bot in polling mode."""
    # Settings are resolved here, at the entry point, and passed down — so
    # importing this module reads no files and installs no log handlers.
    settings = get_settings()
    setup_logging(settings)
    token = settings.bot.token

    if not token:
        logger.error("BOT_TOKEN is not set")
        return

    admin_user_id = settings.bot.admin_user_id
    tz = ZoneInfo(settings.timezone.name)
    questions_dir = settings.paths.question_images_dir

    # Local Windows dev only: AsyncConnectionPool's background workers stall
    # even on the SelectorEventLoop run_async installs, so use the
    # NullConnectionPool (which has none). Linux production uses the real pool.
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
        # A missing corpus only surfaces on the first uncached question
        # otherwise — one broken render per student, minutes into a session.
        if not questions_dir.is_dir():
            logger.error("Question images directory not found", path=str(questions_dir))
            return

        async with _pool_lifespan(settings.database) as pool:
            # The composition root: the one place a Postgres pool becomes the
            # transaction factory the bot is written against. Nothing under
            # `digitex.bot` names a pool, a unit of work, or psycopg.
            def open_uow() -> UnitOfWork:
                return UnitOfWork(pool)

            bot = Bot(token=token)
            await bot.set_my_commands(
                [
                    BotCommand(command="start", description=CMD_START_DESC),
                    BotCommand(command="help", description=CMD_HELP_DESC),
                ]
            )
            dispatcher = create_dispatcher(
                admin_user_id=admin_user_id, open_uow=open_uow
            )
            logger.info("Starting bot polling...")
            await dispatcher.start_polling(
                bot,
                open_uow=open_uow,
                admin_user_id=admin_user_id,
                tz=tz,
                questions_dir=questions_dir,
            )

    run_async(_main())


# typer-compatible app object for the project script entry point.
app = main


if __name__ == "__main__":
    main()
