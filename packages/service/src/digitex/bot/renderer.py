"""Putting a question's image on screen — the primitive, not the protocol.

One function, and the only place in the bot that touches the corpus on disk. It
sends and reports; deciding what the render *means* — the keyboard a Part gets,
the ``file_id`` a fresh upload leaves owed — belongs to
:mod:`digitex.bot.answer_flow`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from aiogram.types import FSInputFile

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram import Bot
    from aiogram.types import InlineKeyboardMarkup

    from digitex.domain.entities import Question

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
    tree on a laptop and from the synced mount in production. A question with a
    cached ``file_id`` needs no root at all, which is what keeps the bot serving
    when the mount goes missing.

    Returns None when the cached file_id was reused, or when Telegram's reply
    carried no photo to take an id from. Persisting a returned id is the
    caller's business.

    Raises:
        KeyError: If the question has neither a cached file_id nor an image.
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

    # Named for the student, not for the storage layout: the key is a corpus
    # path and Telegram shows the filename under the photo.
    sent = await bot.send_photo(
        chat_id=chat_id,
        photo=FSInputFile(
            questions_dir / question.image_key,
            filename=f"q{question.question_number}.jpg",
        ),
        caption=caption,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )
    if sent.photo:
        return sent.photo[-1].file_id

    # Not an error the student sees — the image went out, it just cannot be
    # cached, so every later render of this question re-uploads it.
    logger.warning(
        "No photo in response for question",
        question_id=question.question_id,
        part=question.part,
    )
    return None
