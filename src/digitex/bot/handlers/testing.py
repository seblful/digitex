"""Question answering loop — Part A (callbacks) and Part B (text)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, Router, types

from digitex.bot import fsm_data
from digitex.bot.answer_flow import (
    RoundFinished,
    ask_question,
    file_id_debt,
    load_renderable,
    run_testing_round,
)
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
    """Render the question at ``current_index`` (used to start the loop)."""
    testing = await fsm_data.load(state, TestingState)

    if testing.current_index >= len(testing.question_ids):
        await show_results(message, state, bot, pool)
        return

    question_id, part = testing.question_ids[testing.current_index]
    async with UnitOfWork(pool) as uow:
        question = await load_renderable(uow, question_id, part)

    await fsm_data.merge(
        state,
        question_start_time=time.time(),
        current_part=part,
        waiting_for_answer=True,
    )

    debt = file_id_debt(question, await ask_question(bot, message, question))
    if debt:
        await fsm_data.merge(state, pending_file_id_cache=debt)


async def _record_and_advance(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    answer: str,
    pool: AsyncConnectionPool,
) -> None:
    testing = await fsm_data.load(state, TestingState)

    async with UnitOfWork(pool) as uow:
        outcome = await run_testing_round(uow, testing, answer, now=time.time())

    if isinstance(outcome, RoundFinished):
        await fsm_data.merge(
            state, current_index=outcome.next_index, pending_file_id_cache=None
        )
        await show_results(message, state, bot, pool)
        return

    await fsm_data.merge(
        state,
        current_index=outcome.next_index,
        question_start_time=time.time(),
        current_part=outcome.question.part,
        waiting_for_answer=True,
        pending_file_id_cache=None,
    )

    debt = file_id_debt(
        outcome.question, await ask_question(bot, message, outcome.question)
    )
    if debt:
        await fsm_data.merge(state, pending_file_id_cache=debt)


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

    assert callback.bot is not None
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

    assert message.bot is not None
    await fsm_data.merge(state, waiting_for_answer=False)
    await _record_and_advance(message, state, message.bot, message.text, pool)
