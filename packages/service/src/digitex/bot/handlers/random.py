"""Random and topic mode — one question at a time, scored on the spot.

No Session is recorded: the student gets a verdict and a choice between another
question and stopping. Topic mode is the same loop with the draw restricted to
one topic name, which is why both arrive here.

The two answer handlers deliberately mirror the pair in ``testing.py``. Topic
mode draws from both Parts, so a Part A keyboard left in the chat can be tapped
while a Part B question is showing — each handler claims the reply through the
round, which refuses one that does not match the Part on screen.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Router, types
from aiogram.utils.text_decorations import html_decoration

from digitex.bot.answer_flow import (
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
    from digitex.bot.answer_flow import Round
    from digitex.domain.entities import QuestionOrigin

router = Router()


def _build_caption(origin: QuestionOrigin, topic_name: str | None) -> str:
    """Where the question came from, behind a spoiler so it gives nothing away."""
    origin_line = MSG_RANDOM_ORIGIN.format(
        exam_label=EXAM_LABELS[origin.exam_type],
        year=origin.year,
        option_number=origin.option_number,
    )
    if topic_name:
        # Topic names come from the corpus and the caption is sent as HTML.
        return MSG_RANDOM_TOPIC.format(
            topic_name=html_decoration.quote(topic_name), origin=origin_line
        )
    return origin_line


async def start_random_question(message: types.Message, round: Round) -> None:
    """Draw a question and show it, or say there was none to draw."""
    rnd = await round.load(RandomState)

    async with round.transaction() as uow:
        picked = await pick_random_question(uow, rnd)

    if picked is None:
        # The two dead ends read differently to the student: an empty topic is
        # a gap in the corpus, an empty draw is a gap in the whole subject.
        await message.answer(
            MSG_NO_TOPIC_QUESTION if rnd.topic_name else MSG_NO_RANDOM_QUESTION
        )
        return

    question, origin = picked
    await round.show_random_question(
        message,
        question,
        started_at=time.time(),
        caption=_build_caption(origin, rnd.topic_name),
        parse_mode="HTML",
    )
    await round.set_state(RandomTesting.answering)


@router.callback_query(RandomTesting.answering, AnswerCB.filter())
async def on_random_part_a_answer(
    callback: types.CallbackQuery,
    callback_data: AnswerCB,
    msg: types.Message,
    round: Round,
) -> None:
    # A stale tap would otherwise be scored against the Part B question on
    # screen and disclose its answer.
    if not await round.claim_reply("A", callback):
        return

    await process_random_answer(msg, round, str(callback_data.value))
    await callback.answer()


@router.message(RandomTesting.answering)
async def on_random_part_b_answer(message: types.Message, round: Round) -> None:
    if not message.text:
        return

    if not await round.claim_reply("B"):
        return

    await process_random_answer(message, round, message.text)


async def process_random_answer(
    message: types.Message, round: Round, answer: str
) -> None:
    """Score the reply, show the verdict, and offer the next question."""
    rnd = await round.load(RandomState)

    async with round.transaction() as uow:
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
    await round.set_state(RandomTesting.feedback)


@router.callback_query(RandomTesting.feedback, RandomFeedbackCB.filter())
async def on_random_feedback(
    callback: types.CallbackQuery,
    callback_data: RandomFeedbackCB,
    msg: types.Message,
    round: Round,
) -> None:
    if callback_data.action == "next":
        await start_random_question(msg, round)
    else:
        await msg.answer(MSG_RANDOM_FINISH)
        await round.end()
    await callback.answer()
