"""Question image renderer with Telegram file_id caching."""

import structlog

from aiogram import Bot
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup

from digitex.bot.database import with_uow
from digitex.bot.schemas import Question

logger = structlog.get_logger()


async def send_question(
    bot: Bot,
    chat_id: int,
    question: Question,
    db_path: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    caption: str | None = None,
    parse_mode: str | None = None,
) -> None:
    """Send a question image with optional inline keyboard, caching the Telegram file_id."""
    if question.telegram_file_id:
        await bot.send_photo(
            chat_id=chat_id,
            photo=question.telegram_file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
        return

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
        file_id = msg.photo[-1].file_id

        def cache(uow):
            uow.questions.cache_file_id(question.question_id, question.part, file_id)

        await with_uow(db_path, cache)
    else:
        logger.warning(
            "No photo in response for question",
            question_id=question.question_id,
            part=question.part,
        )
