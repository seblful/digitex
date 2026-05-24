"""Bot answer-flow helpers — render and evaluate one question.

Two recipes shared across the standard and random testing modes:

- ``ask_question`` (per ADR 0003) — send the question image and attach the
  right keyboard or follow-up prompt. Returns any new Telegram ``file_id``
  so the caller can fold the cache write into its next UoW.
- ``evaluate_answer_in_uow`` — fetch the correct answer and check correctness
  inside a UoW already owned by the caller. The mode-specific "what happens
  next" (record to a Session vs. send immediate feedback) stays in the handler
  files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.core.answer import check_answer

if TYPE_CHECKING:
    from aiogram import Bot, types

    from digitex.core.db import UnitOfWork
    from digitex.core.domain import Part, Question


async def ask_question(
    bot: Bot,
    message: types.Message,
    question: Question,
    *,
    caption: str | None = None,
    parse_mode: str | None = None,
) -> str | None:
    """Send a question to the chat and return the new Telegram ``file_id``.

    Part A goes out with the option-picker keyboard; Part B gets a follow-up
    "enter your answer" prompt. The caller is responsible for persisting any
    returned ``file_id`` (typically folded into the next UoW via the
    ``pending_file_id_cache`` FSM field).
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
    return new_file_id


async def evaluate_answer_in_uow(
    uow: UnitOfWork,
    question_id: int,
    part: Part,
    answer: str,
) -> tuple[bool, int | str]:
    """Fetch the correct answer and check the student's reply inside a UoW.

    Returns ``(is_correct, correct_answer)``. Lets callers fold the correctness
    lookup into the same transaction as ``record_answer`` and the next-question
    fetch.
    """
    correct = await uow.questions.get_correct_answer(question_id, part)
    return check_answer(part, answer, correct), correct
