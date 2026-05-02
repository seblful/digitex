"""Bot entrypoint."""

import asyncio

import structlog

from digitex.bot import create_dispatcher
from digitex.config import get_settings
from digitex.logging import setup_logging

setup_logging()
logger = structlog.get_logger()


def main() -> None:
    """Start the Telegram bot in polling mode."""
    settings = get_settings()
    token = settings.bot.token

    if not token:
        logger.error("BOT__TOKEN is not set")
        return

    dispatcher = create_dispatcher()

    async def _main() -> None:
        from aiogram import Bot
        from aiogram.types import BotCommand

        from digitex.bot.messages import CMD_HELP_DESC, CMD_START_DESC

        bot = Bot(token=token)
        await bot.set_my_commands([
            BotCommand(command="start", description=CMD_START_DESC),
            BotCommand(command="help", description=CMD_HELP_DESC),
        ])
        logger.info("Starting bot polling...")
        await dispatcher.start_polling(bot)

    asyncio.run(_main())


if __name__ == "__main__":
    main()
