"""Bot entrypoint — the one place a Postgres pool becomes a transaction seam.

`digitex.bot` is written against the protocols in `digitex.domain.ports` and
`digitex.db` provides classes that answer to them; neither imports the other. The
wiring is here, which is what lets the boundary be stated as a direction rather
than maintained as a habit.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

from digitex.bot.dispatcher import create_dispatcher
from digitex.config import get_settings
from digitex.console import run_async
from digitex.db import UnitOfWork, null_pool_lifespan, pool_lifespan
from digitex.logging import setup_logging

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from psycopg_pool import AsyncConnectionPool

    from digitex.config import DatabaseSettings, Settings

logger = structlog.get_logger()


def main() -> None:
    """Start the Telegram bot in polling mode."""
    # Settings are resolved here, at the entry point, and passed down — so
    # importing this module reads no files and installs no log handlers.
    settings = get_settings()
    setup_logging(settings)

    if not settings.bot.token:
        logger.error("BOT_TOKEN is not set")
        return

    run_async(_serve(settings))


def _pool_lifespan_for_platform() -> Callable[
    [DatabaseSettings], AbstractAsyncContextManager[AsyncConnectionPool]
]:
    """The pool this platform can actually run.

    Local Windows dev only: `AsyncConnectionPool`'s background workers stall even
    on the selector loop `run_async` installs, so the null pool — which has none
    — stands in. Linux production uses the real pool.
    """
    return null_pool_lifespan if sys.platform == "win32" else pool_lifespan


async def _serve(settings: Settings) -> None:
    """Open the pool, wire the bot to it, and poll until stopped."""
    # aiogram is imported here rather than at module scope so that importing
    # this module stays cheap — `tests/contracts` walks every deployed module.
    from aiogram import Bot
    from aiogram.types import BotCommand

    from digitex.bot.messages import CMD_HELP_DESC, CMD_START_DESC

    hosts = settings.database.dsn.hosts()
    logger.info(
        "Opening DB pool",
        host=hosts[0].get("host") if hosts else "unknown",
        db=settings.database.dsn.path,
    )

    # A missing corpus only surfaces on the first uncached question otherwise —
    # one broken render per student, minutes into a session.
    questions_dir = settings.paths.question_images_dir
    if not questions_dir.is_dir():
        logger.error("Question images directory not found", path=str(questions_dir))
        return

    async with _pool_lifespan_for_platform()(settings.database) as pool:
        # The composition root: the one place a Postgres pool becomes the
        # transaction factory the bot is written against. Nothing under
        # `digitex.bot` names a pool, a unit of work, or psycopg.
        def open_uow() -> UnitOfWork:
            return UnitOfWork(pool)

        bot = Bot(token=settings.bot.token)
        await bot.set_my_commands(
            [
                BotCommand(command="start", description=CMD_START_DESC),
                BotCommand(command="help", description=CMD_HELP_DESC),
            ]
        )

        admin_user_id = settings.bot.admin_user_id
        dispatcher = create_dispatcher(admin_user_id=admin_user_id, open_uow=open_uow)
        logger.info("Starting bot polling...")
        await dispatcher.start_polling(
            bot,
            open_uow=open_uow,
            admin_user_id=admin_user_id,
            tz=ZoneInfo(settings.timezone.name),
            questions_dir=questions_dir,
        )


if __name__ == "__main__":
    main()
