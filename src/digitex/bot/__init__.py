"""Telegram bot package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram import Dispatcher

from digitex.bot.handlers.navigation import router as navigation_router
from digitex.bot.handlers.random import router as random_router
from digitex.bot.handlers.start import router as start_router
from digitex.bot.handlers.testing import router as testing_router
from digitex.bot.middleware import AccessibleMessageMiddleware, AuthMiddleware

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool


def create_dispatcher(admin_user_id: int, pool: AsyncConnectionPool) -> Dispatcher:
    dp = Dispatcher()
    auth = AuthMiddleware(admin_user_id=admin_user_id, pool=pool)
    dp.message.outer_middleware(auth)
    dp.callback_query.outer_middleware(auth)
    # After auth: an unauthorized tap is dropped before it is acknowledged.
    dp.callback_query.outer_middleware(AccessibleMessageMiddleware())
    # Order matters: /start is registered first, so it wins over an in-progress
    # test. ``results`` has no router — its screen is drawn by the testing
    # handlers, never by a tap of its own.
    dp.include_routers(
        start_router,
        navigation_router,
        testing_router,
        random_router,
    )
    return dp
