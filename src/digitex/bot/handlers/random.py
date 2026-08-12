"""Handler for random question mode."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, Router, types

from digitex.bot import fsm_data
from digitex.bot.answer_flow import (
    end_round,
    evaluate_random_answer,
    pick_random_question,
    show_question,
)
from digitex.bot.callbacks import AnswerCB, RandomFeedbackCB
from digitex.bot.fsm_data import RandomState
from digitex.bot.keyboards import random_feedback_kb
from digitex.bot.messages import (
    EXAM_LABELS,
    MSG_CORRECT_ANSWER,
    MSG_NO_RANDOM_QUESTION,
    MSG_NO_TOPIC_QUESTION,
    MSG_RANDOM_FINISH,
    MSG_RANDOM_ORIGIN,
    MSG_RANDOM_TOPIC,
    MSG_WRONG_ANSWER,
)
from digitex.bot.states import RandomTesting
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

    from digitex.core.domain import QuestionOrigin

router = Router()


def _build_caption(origin: QuestionOrigin, topic_name: str | None) -> str:
    origin_line = MSG_RANDOM_ORIGIN.format(
        exam_label=EXAM_LABELS[origin.exam_type],
        year=origin.year,
        option_number=origin.option_number,
    )
    if topic_name:
        return MSG_RANDOM_TOPIC.format(topic_name=topic_name, origin=origin_line)
    return origin_line


async def start_random_question(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    pool: AsyncConnectionPool,
) -> None:
    rnd = await fsm_data.load(state, RandomState)

    async with UnitOfWork(pool) as uow:
        picked = await pick_random_question(uow, rnd)

    if picked is None:
        if rnd.topic_name:
            await message.answer(MSG_NO_TOPIC_QUESTION)
        else:
            await message.answer(MSG_NO_RANDOM_QUESTION)
        return
    question, origin = picked

    await show_question(
        bot,
        message,
        state,
        question,
        started_at=time.time(),
        caption=_build_caption(origin, rnd.topic_name),
        parse_mode="HTML",
    )
    await state.set_state(RandomTesting.answering)


@router.callback_query(RandomTesting.answering, AnswerCB.filter())
async def on_random_part_a_answer(
    callback: types.CallbackQuery,
    callback_data: AnswerCB,
    state: FSMContext,
    msg: types.Message,
    pool: AsyncConnectionPool,
) -> None:
    # Old keyboards stay live in the chat, so a tap can arrive while a Part B
    # question is on screen — it would otherwise be scored against that
    # question and disclose its answer. Mirrors the Part B guard below.
    rnd = await fsm_data.load(state, RandomState)
    if rnd.current_part != "A":
        await callback.answer()
        return

    await process_random_answer(msg, state, str(callback_data.value), pool)
    await callback.answer()


@router.message(RandomTesting.answering)
async def on_random_part_b_answer(
    message: types.Message, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    if not message.text:
        return

    rnd = await fsm_data.load(state, RandomState)
    if rnd.current_part != "B":
        return

    await process_random_answer(message, state, message.text, pool)


async def process_random_answer(
    message: types.Message,
    state: FSMContext,
    answer: str,
    pool: AsyncConnectionPool,
) -> None:
    rnd = await fsm_data.load(state, RandomState)

    async with UnitOfWork(pool) as uow:
        verdict = await evaluate_random_answer(uow, rnd, answer)
    if verdict is None:
        return
    is_correct, correct_answer = verdict

    if is_correct:
        await message.answer(MSG_CORRECT_ANSWER, reply_markup=random_feedback_kb())
    else:
        await message.answer(
            MSG_WRONG_ANSWER.format(correct_answer=correct_answer),
            reply_markup=random_feedback_kb(),
            parse_mode="HTML",
        )
    await state.set_state(RandomTesting.feedback)


@router.callback_query(RandomTesting.feedback, RandomFeedbackCB.filter())
async def on_random_feedback(
    callback: types.CallbackQuery,
    callback_data: RandomFeedbackCB,
    state: FSMContext,
    msg: types.Message,
    bot: Bot,
    pool: AsyncConnectionPool,
) -> None:
    if callback_data.action == "next":
        await start_random_question(msg, state, bot, pool)
    else:
        await msg.answer(MSG_RANDOM_FINISH)
        await end_round(pool, state)
    await callback.answer()
