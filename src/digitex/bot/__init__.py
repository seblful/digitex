"""Telegram bot package."""

from aiogram import Dispatcher

from digitex.bot.handlers.navigation import router as navigation_router
from digitex.bot.handlers.results import router as results_router
from digitex.bot.handlers.start import router as start_router
from digitex.bot.handlers.testing import router as testing_router


def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_routers(
        start_router,
        navigation_router,
        testing_router,
        results_router,
    )
    return dp
