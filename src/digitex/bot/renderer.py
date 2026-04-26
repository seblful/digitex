"""Question image renderer with Telegram file_id caching."""

from aiogram import Bot
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup

from digitex.bot.schemas import Question


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
        from digitex.core.db import UnitOfWork
        with UnitOfWork(db_path) as uow:
            uow.questions.cache_file_id(question.question_id, question.part, file_id)
