"""Handler for random question mode."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, F, Router, types

from digitex.bot.answer_flow import ask_question
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
from digitex.core.answer import check_answer
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
    data = await state.get_data()
    subject_id = data["subject_id"]
    topic_name = data.get("topic_name")

    try:
        async with UnitOfWork(pool) as uow:
            if topic_name:
                qid, part = await uow.questions.get_random_question_id_by_topic(
                    subject_id, topic_name
                )
            else:
                part = data["random_part"]
                exam_type = data.get("exam_type")
                qid = await uow.questions.get_random_question_id(
                    subject_id, part, exam_type
                )
            question, origin = await uow.questions.get_full(qid, part)
    except KeyError:
        if topic_name:
            await message.answer(MSG_NO_TOPIC_QUESTION)
        else:
            await message.answer(MSG_NO_RANDOM_QUESTION)
        return

    await state.update_data(
        current_question_id=question.question_id,
        current_part=question.part,
        question_start_time=time.time(),
    )

    caption = _build_caption(origin, topic_name)
    await ask_question(bot, message, question, pool, caption=caption, parse_mode="HTML")
    await state.set_state(RandomTesting.answering)


@router.callback_query(RandomTesting.answering, F.data.startswith("ans:"))
async def on_random_part_a_answer(
    callback: types.CallbackQuery, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    answer = callback.data.split(":")[1]
    await process_random_answer(callback.message, state, answer, pool)
    await callback.answer()


@router.message(RandomTesting.answering)
async def on_random_part_b_answer(
    message: types.Message, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    if not message.text:
        return

    data = await state.get_data()
    if data.get("current_part") != "B":
        return

    await process_random_answer(message, state, message.text, pool)


async def process_random_answer(
    message: types.Message,
    state: FSMContext,
    answer: str,
    pool: AsyncConnectionPool,
) -> None:
    data = await state.get_data()
    question_id = data["current_question_id"]
    current_part = data["current_part"]

    async with UnitOfWork(pool) as uow:
        correct_answer = await uow.questions.get_correct_answer(
            question_id, current_part
        )
    is_correct = check_answer(current_part, answer, correct_answer)

    if is_correct:
        await message.answer(MSG_CORRECT_ANSWER, reply_markup=random_feedback_kb())
    else:
        await message.answer(
            MSG_WRONG_ANSWER.format(correct_answer=correct_answer),
            reply_markup=random_feedback_kb(),
            parse_mode="HTML",
        )
    await state.set_state(RandomTesting.feedback)


@router.callback_query(RandomTesting.feedback, F.data == "random:next")
async def on_random_next(
    callback: types.CallbackQuery, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    if not callback.message or not isinstance(callback.message, types.Message):
        await callback.answer("Ошибка: сообщение недоступно")
        return
    await start_random_question(callback.message, state, callback.bot, pool)
    await callback.answer()


@router.callback_query(RandomTesting.feedback, F.data == "random:finish")
async def on_random_finish(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.message or not isinstance(callback.message, types.Message):
        await callback.answer("Ошибка: сообщение недоступно")
        return
    await callback.message.answer(MSG_RANDOM_FINISH)
    await state.clear()
    await callback.answer()
