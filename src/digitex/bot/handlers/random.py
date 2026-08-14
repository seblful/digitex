"""Handler for random question mode."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, Router, types
from aiogram.utils.text_decorations import html_decoration

from digitex.bot import fsm_data
from digitex.bot.answer_flow import (
    Round,
    evaluate_random_answer,
    pick_random_question,
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
    format_answer,
)
from digitex.bot.states import RandomTesting

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

    from digitex.domain.entities import QuestionOrigin

router = Router()


def _build_caption(origin: QuestionOrigin, topic_name: str | None) -> str:
    origin_line = MSG_RANDOM_ORIGIN.format(
        exam_label=EXAM_LABELS[origin.exam_type],
        year=origin.year,
        option_number=origin.option_number,
    )
    if topic_name:
        return MSG_RANDOM_TOPIC.format(
            topic_name=html_decoration.quote(topic_name), origin=origin_line
        )
    return origin_line


async def start_random_question(message: types.Message, round: Round) -> None:
    rnd = await fsm_data.load(round.state, RandomState)

    async with round.open_uow() as uow:
        picked = await pick_random_question(uow, rnd)

    if picked is None:
        if rnd.topic_name:
            await message.answer(MSG_NO_TOPIC_QUESTION)
        else:
            await message.answer(MSG_NO_RANDOM_QUESTION)
        return
    question, origin = picked

    await round.show_random_question(
        message,
        question,
        started_at=time.time(),
        caption=_build_caption(origin, rnd.topic_name),
        parse_mode="HTML",
    )
    await round.state.set_state(RandomTesting.answering)


@router.callback_query(RandomTesting.answering, AnswerCB.filter())
async def on_random_part_a_answer(
    callback: types.CallbackQuery,
    callback_data: AnswerCB,
    state: FSMContext,
    msg: types.Message,
    bot: Bot,
    pool: AsyncConnectionPool,
    questions_dir: Path,
) -> None:
    # Old keyboards stay live in the chat, so a tap can arrive while a Part B
    # question is on screen — it would otherwise be scored against that
    # question and disclose its answer. Mirrors the Part B guard below.
    rnd = await fsm_data.load(state, RandomState)
    if rnd.current_part != "A" or not rnd.waiting_for_answer:
        await callback.answer()
        return

    await fsm_data.merge(state, RandomState, waiting_for_answer=False)
    await process_random_answer(
        msg, Round(bot, state, pool, questions_dir), str(callback_data.value)
    )
    await callback.answer()


@router.message(RandomTesting.answering)
async def on_random_part_b_answer(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    pool: AsyncConnectionPool,
    questions_dir: Path,
) -> None:
    if not message.text:
        return

    rnd = await fsm_data.load(state, RandomState)
    if rnd.current_part != "B" or not rnd.waiting_for_answer:
        return

    await fsm_data.merge(state, RandomState, waiting_for_answer=False)
    await process_random_answer(
        message, Round(bot, state, pool, questions_dir), message.text
    )


async def process_random_answer(
    message: types.Message, round: Round, answer: str
) -> None:
    rnd = await fsm_data.load(round.state, RandomState)

    async with round.open_uow() as uow:
        verdict = await evaluate_random_answer(uow, rnd, answer)
    if verdict is None:
        return
    is_correct, correct_answer = verdict

    if is_correct:
        await message.answer(MSG_CORRECT_ANSWER, reply_markup=random_feedback_kb())
    else:
        # Part B answers are free text and the message goes out as HTML, so an
        # unescaped "<" would make Telegram reject it — and the raised error
        # would skip the state transition below, stranding the round.
        await message.answer(
            MSG_WRONG_ANSWER.format(
                correct_answer=html_decoration.quote(
                    format_answer(correct_answer.stored)
                )
            ),
            reply_markup=random_feedback_kb(),
            parse_mode="HTML",
        )
    await round.state.set_state(RandomTesting.feedback)


@router.callback_query(RandomTesting.feedback, RandomFeedbackCB.filter())
async def on_random_feedback(
    callback: types.CallbackQuery,
    callback_data: RandomFeedbackCB,
    state: FSMContext,
    msg: types.Message,
    bot: Bot,
    pool: AsyncConnectionPool,
    questions_dir: Path,
) -> None:
    round = Round(bot, state, pool, questions_dir)
    if callback_data.action == "next":
        await start_random_question(msg, round)
    else:
        await msg.answer(MSG_RANDOM_FINISH)
        await round.end()
    await callback.answer()
