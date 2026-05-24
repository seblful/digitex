"""Question answering loop — Part A (callbacks) and Part B (text)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, Router, types

from digitex.bot import fsm_data
from digitex.bot.answer_flow import ask_question, evaluate_answer_in_uow
from digitex.bot.callbacks import AnswerCB
from digitex.bot.fsm_data import TestingState
from digitex.bot.handlers.results import show_results
from digitex.bot.states import Testing
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

    from digitex.core.domain import Question

router = Router()


async def _load_renderable(uow: UnitOfWork, question_id: int, part: str) -> Question:
    """Fetch a question's metadata and the image bytes only on a cache miss."""
    question = await uow.questions.get(question_id, part)
    if not question.telegram_file_id:
        image_data = await uow.questions.get_image(question_id, part)
        question = question.model_copy(update={"image_data": image_data})
    return question


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
        question = await _load_renderable(uow, question_id, part)

    await fsm_data.merge(
        state,
        question_start_time=time.time(),
        current_part=part,
        waiting_for_answer=True,
    )

    new_file_id = await ask_question(bot, message, question)
    if new_file_id:
        await fsm_data.merge(
            state, pending_file_id_cache=(question_id, part, new_file_id)
        )


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
    next_index = testing.current_index + 1
    pending = testing.pending_file_id_cache

    next_question: Question | None = None
    async with UnitOfWork(pool) as uow:
        if pending is not None:
            await uow.questions.cache_file_id(*pending)
        is_correct, _ = await evaluate_answer_in_uow(uow, question_id, part, answer)
        await uow.sessions.record_answer(
            session_id=testing.session_id,
            question_id=question_id,
            part=part,
            student_answer=answer.strip(),
            is_correct=is_correct,
            time_spent=time_spent,
        )
        if next_index < len(testing.question_ids):
            next_qid, next_part = testing.question_ids[next_index]
            next_question = await _load_renderable(uow, next_qid, next_part)

    if next_question is None:
        await fsm_data.merge(
            state, current_index=next_index, pending_file_id_cache=None
        )
        await show_results(message, state, bot, pool)
        return

    await fsm_data.merge(
        state,
        current_index=next_index,
        question_start_time=time.time(),
        current_part=next_question.part,
        waiting_for_answer=True,
        pending_file_id_cache=None,
    )

    new_file_id = await ask_question(bot, message, next_question)
    if new_file_id:
        await fsm_data.merge(
            state,
            pending_file_id_cache=(
                next_question.question_id,
                next_question.part,
                new_file_id,
            ),
        )


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
