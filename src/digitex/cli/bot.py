"""Bot CLI commands."""

import structlog
import typer

from digitex.bot import create_dispatcher
from digitex.config import get_settings
from digitex.logging import setup_logging

setup_logging()
logger = structlog.get_logger()

app = typer.Typer(help="Telegram bot commands.")


@app.command()
def run() -> None:
    """Start the Telegram bot in polling mode."""
    settings = get_settings()
    token = settings.bot.token

    if not token:
        logger.error("BOT__TOKEN is not set")
        raise typer.Exit(1)

    dispatcher = create_dispatcher()

    async def _main() -> None:
        from aiogram import Bot

        bot = Bot(token=token)
        logger.info("Starting bot polling...")
        await dispatcher.start_polling(bot)

    import asyncio

    asyncio.run(_main())
