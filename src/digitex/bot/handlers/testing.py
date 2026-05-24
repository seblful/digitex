"""Question answering loop — Part A (callbacks) and Part B (text)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, Router, types

from digitex.bot import fsm_data
from digitex.bot.answer_flow import ask_question, evaluate_answer
from digitex.bot.callbacks import AnswerCB
from digitex.bot.fsm_data import TestingState
from digitex.bot.handlers.results import show_results
from digitex.bot.states import Testing
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
    testing = await fsm_data.load(state, TestingState)

    if testing.current_index >= len(testing.question_ids):
        await show_results(message, state, bot, pool)
        return

    question_id, part = testing.question_ids[testing.current_index]

    async with UnitOfWork(pool) as uow:
        question = await uow.questions.get(question_id, part)

    await fsm_data.merge(
        state,
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
    testing = await fsm_data.load(state, TestingState)
    question_id, part = testing.question_ids[testing.current_index]
    question_start_time = testing.question_start_time or time.time()
    time_spent = time.time() - question_start_time

    is_correct, _ = await evaluate_answer(pool, question_id, part, answer)
    async with UnitOfWork(pool) as uow:
        await uow.sessions.record_answer(
            session_id=testing.session_id,
            question_id=question_id,
            part=part,
            student_answer=answer.strip(),
            is_correct=is_correct,
            time_spent=time_spent,
        )

    await fsm_data.merge(state, current_index=testing.current_index + 1)
    await send_current_question(message, state, bot, pool)


@router.callback_query(Testing.answering, AnswerCB.filter())
async def on_part_a_answer(
    callback: types.CallbackQuery,
    callback_data: AnswerCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    await _record_and_advance(
        callback.message, state, callback.bot, str(callback_data.value), pool
    )
    await callback.answer()


@router.message(Testing.answering)
async def on_part_b_answer(
    message: types.Message, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    if not message.text:
        return

    testing = await fsm_data.load(state, TestingState)
    if testing.current_part != "B" or not testing.waiting_for_answer:
        return

    await fsm_data.merge(state, waiting_for_answer=False)
    await _record_and_advance(message, state, message.bot, message.text, pool)
