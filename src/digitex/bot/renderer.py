"""Question image renderer."""

import structlog
from aiogram import Bot
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup

from digitex.core.domain import Question

logger = structlog.get_logger()


async def send_question(
    bot: Bot,
    chat_id: int,
    question: Question,
    reply_markup: InlineKeyboardMarkup | None = None,
    caption: str | None = None,
    parse_mode: str | None = None,
) -> str | None:
    """Send a question image; return the new Telegram file_id when uploaded fresh.

    Returns None when the cached file_id was reused or when no photo appeared
    in the response. The caller is responsible for persisting any returned file_id.
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

    msg = await bot.send_photo(
        chat_id=chat_id,
        photo=BufferedInputFile(
            question.image_data,
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
