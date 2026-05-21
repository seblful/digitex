"""Question answering loop — Part A (callbacks) and Part B (text)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, F, Router, types

from digitex.bot.answer_flow import ask_question
from digitex.bot.handlers.results import show_results
from digitex.bot.states import Testing
from digitex.core.answer import check_answer
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

router = Router()


async def send_current_question(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    pool: AsyncConnectionPool,
) -> None:
    data = await state.get_data()
    question_ids: list[tuple[int, str]] = data["question_ids"]
    current_index: int = data["current_index"]

    if current_index >= len(question_ids):
        await show_results(message, state, bot, pool)
        return

    question_id, part = question_ids[current_index]

    async with UnitOfWork(pool) as uow:
        question = await uow.questions.get(question_id, part)

    await state.update_data(
        question_start_time=time.time(),
        current_part=part,
        waiting_for_answer=True,
    )

    await ask_question(bot, message, question, pool)


async def _record_and_advance(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    answer: str,
    pool: AsyncConnectionPool,
) -> None:
    data = await state.get_data()
    session_id: int = data["session_id"]
    question_ids: list[tuple[int, str]] = data["question_ids"]
    current_index: int = data["current_index"]
    question_id, part = question_ids[current_index]
    question_start_time: float = data.get("question_start_time", time.time())

    time_spent = time.time() - question_start_time

    async with UnitOfWork(pool) as uow:
        correct = await uow.questions.get_correct_answer(question_id, part)
        is_correct = check_answer(part, answer, correct)
        await uow.sessions.record_answer(
            session_id=session_id,
            question_id=question_id,
            student_answer=answer.strip(),
            is_correct=is_correct,
            time_spent=time_spent,
        )

    await state.update_data(current_index=current_index + 1)
    await send_current_question(message, state, bot, pool)


@router.callback_query(Testing.answering, F.data.startswith("ans:"))
async def on_part_a_answer(
    callback: types.CallbackQuery, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    answer = callback.data.split(":")[1]
    await _record_and_advance(callback.message, state, callback.bot, answer, pool)
    await callback.answer()


@router.message(Testing.answering)
async def on_part_b_answer(
    message: types.Message, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    if not message.text:
        return

    data = await state.get_data()
    current_part = data.get("current_part")
    waiting = data.get("waiting_for_answer", False)

    if current_part != "B" or not waiting:
        return

    await state.update_data(waiting_for_answer=False)
    await _record_and_advance(message, state, message.bot, message.text, pool)
