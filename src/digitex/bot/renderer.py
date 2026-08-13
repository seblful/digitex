"""Question image renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from aiogram.types import FSInputFile

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram import Bot
    from aiogram.types import InlineKeyboardMarkup

    from digitex.core.domain import Question

logger = structlog.get_logger()


async def send_question(
    bot: Bot,
    chat_id: int,
    question: Question,
    questions_dir: Path,
    reply_markup: InlineKeyboardMarkup | None = None,
    caption: str | None = None,
    parse_mode: str | None = None,
) -> str | None:
    """Send a question image; return the new Telegram file_id when uploaded fresh.

    *questions_dir* is the corpus root this process serves from — the question
    carries only the key beneath it, so the same row renders from the extraction
    tree on a laptop and from the synced mount in production.

    Returns None when the cached file_id was reused or when no photo appeared
    in the response. The caller is responsible for persisting any returned
    file_id.
    """
    if question.telegram_file_id:
        await bot.send_photo(
            chat_id=chat_id,
            photo=question.telegram_file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
        return None

    if question.image_key is None:
        raise KeyError(f"No image stored for question {question.question_id}")

    msg = await bot.send_photo(
        chat_id=chat_id,
        photo=FSInputFile(
            questions_dir / question.image_key,
            filename=f"q{question.question_number}.jpg",
        ),
        caption=caption,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )
    if msg.photo:
        return msg.photo[-1].file_id
    logger.warning(
        "No photo in response for question",
        question_id=question.question_id,
        part=question.part,
    )
    return None
