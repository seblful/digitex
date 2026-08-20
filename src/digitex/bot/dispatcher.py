"""Dispatcher assembly — middleware and router wiring for the bot.

This lives beside the handlers rather than in ``bot/__init__.py`` because the
handlers import ``digitex.bot.fsm_data``, which runs the package's ``__init__``
first. With the wiring in there, importing any handler imported every handler,
and the package cycled with its own contents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram import Dispatcher

from digitex.bot.handlers.navigation import router as navigation_router
from digitex.bot.handlers.random import router as random_router
from digitex.bot.handlers.start import router as start_router
from digitex.bot.handlers.testing import router as testing_router
from digitex.bot.middleware import AccessibleMessageMiddleware, AuthMiddleware

if TYPE_CHECKING:
    from digitex.domain.ports import OpenUow


def create_dispatcher(admin_user_id: int, open_uow: OpenUow) -> Dispatcher:
    dp = Dispatcher()
    auth = AuthMiddleware(admin_user_id=admin_user_id, open_uow=open_uow)
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
