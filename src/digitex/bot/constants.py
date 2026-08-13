"""Shared bot-level constants and the Student-identity fallback they belong to."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiogram import types

FALLBACK_NAME = "Пользователь"


def student_identity(
    event: types.Message | types.CallbackQuery,
) -> tuple[int, str, str | None]:
    """``(telegram_id, name, username)`` for the user behind *event*.

    Both the /start flow and the option screen create Students, and they must
    agree on what to do when Telegram gives us no user or no name — otherwise
    the two paths write different rows for the same person.
    """
    user = event.from_user
    if user:
        return user.id, user.full_name or FALLBACK_NAME, user.username
    return 0, FALLBACK_NAME, None
