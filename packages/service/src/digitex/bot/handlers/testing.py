"""The standard testing loop — Part A arrives as a tap, Part B as text.

Both handlers do the same two things: claim the reply for the question on
screen, and hand the answer to the round. What the answer *means* is decided
in :mod:`digitex.bot.answer_flow`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Router, types

from digitex.bot.answer_flow import RoundFinished, run_testing_round
from digitex.bot.callbacks import AnswerCB
from digitex.bot.fsm_data import TestingState
from digitex.bot.handlers.results import show_results
from digitex.bot.states import Testing

if TYPE_CHECKING:
    from digitex.bot.answer_flow import Round

router = Router()


async def send_current_question(message: types.Message, round: Round) -> None:
    """Render the question at ``current_index`` (used to start the loop)."""
    testing = await round.load(TestingState)

    # An option holding no questions finishes the moment it starts.
    if testing.current_index >= len(testing.question_ids):
        await show_results(message, round)
        return

    question_id, _part = testing.question_ids[testing.current_index]
    async with round.transaction() as uow:
        question = await uow.questions.get(question_id)

    await round.show_testing_question(
        message, question, index=testing.current_index, started_at=time.time()
    )


async def _record_and_advance(
    message: types.Message, round: Round, answer: str
) -> None:
    """Score the reply, then either show the next question or the results."""
    testing = await round.load(TestingState)

    async with round.transaction() as uow:
        outcome = await run_testing_round(uow, testing, answer, now=time.time())

    if isinstance(outcome, RoundFinished):
        # The round already settled the debt, and show_results ends it.
        await round.merge(TestingState, current_index=outcome.next_index)
        await show_results(message, round)
        return

    await round.show_testing_question(
        message, outcome.question, index=outcome.next_index, started_at=time.time()
    )


@router.callback_query(Testing.answering, AnswerCB.filter())
async def on_part_a_answer(
    callback: types.CallbackQuery,
    callback_data: AnswerCB,
    msg: types.Message,
    round: Round,
) -> None:
    # A stale tap would otherwise be recorded against the next, unseen question.
    if not await round.claim_reply("A", callback):
        return

    await _record_and_advance(msg, round, str(callback_data.value))
    await callback.answer()


@router.message(Testing.answering)
async def on_part_b_answer(message: types.Message, round: Round) -> None:
    if not message.text:
        return

    if not await round.claim_reply("B"):
        return

    await _record_and_advance(message, round, message.text)
