"""Presenting a question in the bot conversation and caching its file_id.

`ask_question` owns the shared "render → prompt → cache" recipe that both the
standard testing mode and the random-question mode would otherwise duplicate.
Handler modules stay focused on their FSM transitions; the rendering recipe
lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram import Bot, types
    from psycopg_pool import AsyncConnectionPool

    from digitex.core.schemas import Question


async def ask_question(
    bot: Bot,
    message: types.Message,
    question: Question,
    pool: AsyncConnectionPool,
    *,
    caption: str | None = None,
    parse_mode: str | None = None,
) -> None:
    """Send a question to the chat and cache the resulting Telegram file_id.

    Part A questions go out with the option-picker keyboard. Part B questions
    get a follow-up "enter your answer" prompt. The new file_id, if any, is
    cached so future renders skip the upload.
    """
    if question.part == "A":
        new_file_id = await send_question(
            bot,
            message.chat.id,
            question,
            reply_markup=part_a_kb(question.num_options),
            caption=caption,
            parse_mode=parse_mode,
        )
    else:
        new_file_id = await send_question(
            bot,
            message.chat.id,
            question,
            caption=caption,
            parse_mode=parse_mode,
        )
        await message.answer(MSG_ENTER_ANSWER)

    if new_file_id:
        async with UnitOfWork(pool) as uow:
            await uow.questions.cache_file_id(
                question.question_id, question.part, new_file_id
            )
