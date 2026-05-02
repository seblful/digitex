"""Telegram bot package."""

from aiogram import Dispatcher

from digitex.bot.handlers.commands import router as commands_router
from digitex.bot.handlers.navigation import router as navigation_router
from digitex.bot.handlers.random import router as random_router
from digitex.bot.handlers.results import router as results_router
from digitex.bot.handlers.start import router as start_router
from digitex.bot.handlers.testing import router as testing_router
from digitex.bot.middleware import AuthMiddleware


def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.message.outer_middleware(AuthMiddleware())
    dp.callback_query.outer_middleware(AuthMiddleware())
    dp.include_routers(
        start_router,
        commands_router,
        navigation_router,
        testing_router,
        random_router,
        results_router,
    )
    return dp

