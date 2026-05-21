"""Presenting a question in the bot conversation and caching its file_id.

`ask_question` owns the shared "render → prompt → cache" recipe that both the
standard testing mode and the random-question mode would otherwise duplicate.
Handler modules stay focused on their FSM transitions; the rendering recipe
lives here.
"""

from aiogram import Bot, types

from digitex.bot.database import with_uow
from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.core.schemas import Question


async def ask_question(
    bot: Bot,
    message: types.Message,
    question: Question,
    db_path: str,
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
        qid, qpart = question.question_id, question.part

        def cache(uow):
            uow.questions.cache_file_id(qid, qpart, new_file_id)

        await with_uow(db_path, cache)
