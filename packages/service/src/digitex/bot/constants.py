"""Bot-level constants, and the one rule that needs them: who an update is from."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiogram import types

FALLBACK_NAME = "Пользователь"


def student_identity(
    event: types.Message | types.CallbackQuery,
) -> tuple[int, str, str | None]:
    """``(telegram_id, name, username)`` for the user behind *event*.

    Students are created down two separate paths — the /start flow and the
    option screen — and both write the same row. Reading the identity through
    one function is what stops them disagreeing about a user Telegram gave us
    without a name, or without a user at all.
    """
    user = event.from_user
    if user is None:
        return 0, FALLBACK_NAME, None
    return user.id, user.full_name or FALLBACK_NAME, user.username
