"""Telegram bot package."""

from aiogram import Dispatcher

from digitex.bot.handlers.commands import router as commands_router
from digitex.bot.handlers.navigation import router as navigation_router
from digitex.bot.handlers.random import router as random_router
from digitex.bot.handlers.results import router as results_router
from digitex.bot.handlers.start import router as start_router
from digitex.bot.handlers.testing import router as testing_router
from digitex.bot.middleware import AuthMiddleware


def create_dispatcher(admin_user_id: int, db_path: str) -> Dispatcher:
    dp = Dispatcher()
    auth = AuthMiddleware(admin_user_id=admin_user_id, db_path=db_path)
    dp.message.outer_middleware(auth)
    dp.callback_query.outer_middleware(auth)
    dp.include_routers(
        start_router,
        commands_router,
        navigation_router,
        testing_router,
        random_router,
        results_router,
    )
    return dp
