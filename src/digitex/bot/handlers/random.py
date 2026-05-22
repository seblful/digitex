"""Handler for random question mode."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, Router, types

from digitex.bot import fsm_data
from digitex.bot.answer_flow import ask_question, evaluate_answer
from digitex.bot.callbacks import AnswerCB, RandomFeedbackCB
from digitex.bot.fsm_data import RandomState
from digitex.bot.keyboards import random_feedback_kb
from digitex.bot.messages import (
    MSG_CORRECT_ANSWER,
    MSG_EXAM_CE,
    MSG_EXAM_CT,
    MSG_NO_RANDOM_QUESTION,
    MSG_NO_TOPIC_QUESTION,
    MSG_RANDOM_FINISH,
    MSG_WRONG_ANSWER,
)
from digitex.bot.states import RandomTesting
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

router = Router()


def _build_caption(origin, topic_name: str | None) -> str:
    exam_label = MSG_EXAM_CE if origin.exam_type == "CE" else MSG_EXAM_CT
    spoiler = (
        f"<tg-spoiler>{exam_label} {origin.year} год,"
        f" вариант {origin.option_number}</tg-spoiler>"
    )
    if topic_name:
        return f"Тема: {topic_name}\n{spoiler}"
    return spoiler


async def start_random_question(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    pool: AsyncConnectionPool,
) -> None:
    rnd = await fsm_data.load(state, RandomState)

    try:
        async with UnitOfWork(pool) as uow:
            if rnd.topic_name:
                qid, part = await uow.questions.get_random_question_id_by_topic(
                    rnd.subject_id, rnd.topic_name
                )
            else:
                if rnd.random_part is None:
                    raise KeyError("random_part not set")
                part = rnd.random_part
                qid = await uow.questions.get_random_question_id(
                    rnd.subject_id, part, rnd.exam_type
                )
            question, origin = await uow.questions.get_full(qid, part)
    except KeyError:
        if rnd.topic_name:
            await message.answer(MSG_NO_TOPIC_QUESTION)
        else:
            await message.answer(MSG_NO_RANDOM_QUESTION)
        return

    await fsm_data.merge(
        state,
        current_question_id=question.question_id,
        current_part=question.part,
        question_start_time=time.time(),
    )

    caption = _build_caption(origin, rnd.topic_name)
    await ask_question(bot, message, question, pool, caption=caption, parse_mode="HTML")
    await state.set_state(RandomTesting.answering)


@router.callback_query(RandomTesting.answering, AnswerCB.filter())
async def on_random_part_a_answer(
    callback: types.CallbackQuery,
    callback_data: AnswerCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    await process_random_answer(callback.message, state, str(callback_data.value), pool)
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
    if rnd.current_question_id is None or rnd.current_part is None:
        return

    is_correct, correct_answer = await evaluate_answer(
        pool, rnd.current_question_id, rnd.current_part, answer
    )

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
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer("Ошибка: сообщение недоступно")
        return

    if callback_data.action == "next":
        await start_random_question(callback.message, state, callback.bot, pool)
    else:
        await callback.message.answer(MSG_RANDOM_FINISH)
        await state.clear()
    await callback.answer()
