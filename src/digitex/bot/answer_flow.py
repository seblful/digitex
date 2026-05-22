"""Bot answer-flow helpers — render and evaluate one question.

Two recipes shared across the standard and random testing modes:

- ``ask_question`` (per ADR 0003) — send the question image, attach the right
  keyboard or follow-up prompt, and cache any new Telegram ``file_id``.
- ``evaluate_answer`` — fetch the correct answer and check correctness in
  one round-trip. The mode-specific "what happens next" (record to a Session
  vs. send immediate feedback) stays in the handler files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.core.answer import check_answer
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram import Bot, types
    from psycopg_pool import AsyncConnectionPool

    from digitex.core.domain import Part, Question


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


async def evaluate_answer(
    pool: AsyncConnectionPool,
    question_id: int,
    part: Part,
    answer: str,
) -> tuple[bool, int | str]:
    """Fetch the correct answer for a question and check the student's reply.

    Returns ``(is_correct, correct_answer)``. The caller decides what to do
    with the result — record it to a Session, send immediate feedback, etc.
    """
    async with UnitOfWork(pool) as uow:
        correct = await uow.questions.get_correct_answer(question_id, part)
    return check_answer(part, answer, correct), correct
