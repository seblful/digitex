"""Question answering loop — Part A (callbacks) and Part B (text)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram import Bot, Router, types

from digitex.bot import fsm_data
from digitex.bot.answer_flow import Round, RoundFinished, run_testing_round
from digitex.bot.callbacks import AnswerCB
from digitex.bot.fsm_data import TestingState
from digitex.bot.handlers.results import show_results
from digitex.bot.states import Testing

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.fsm.context import FSMContext

    from digitex.domain.ports import OpenUow

router = Router()


async def send_current_question(message: types.Message, round: Round) -> None:
    """Render the question at ``current_index`` (used to start the loop)."""
    testing = await fsm_data.load(round.state, TestingState)

    if testing.current_index >= len(testing.question_ids):
        await show_results(message, round)
        return

    question_id, _part = testing.question_ids[testing.current_index]
    async with round.open_uow() as uow:
        question = await uow.questions.get(question_id)

    await round.show_testing_question(
        message, question, index=testing.current_index, started_at=time.time()
    )


async def _record_and_advance(
    message: types.Message, round: Round, answer: str
) -> None:
    testing = await fsm_data.load(round.state, TestingState)

    async with round.open_uow() as uow:
        outcome = await run_testing_round(uow, testing, answer, now=time.time())

    if isinstance(outcome, RoundFinished):
        # The round already settled the debt, and show_results ends it.
        await fsm_data.merge(
            round.state, TestingState, current_index=outcome.next_index
        )
        await show_results(message, round)
        return

    await round.show_testing_question(
        message, outcome.question, index=outcome.next_index, started_at=time.time()
    )


@router.callback_query(Testing.answering, AnswerCB.filter())
async def on_part_a_answer(
    callback: types.CallbackQuery,
    callback_data: AnswerCB,
    state: FSMContext,
    msg: types.Message,
    bot: Bot,
    open_uow: OpenUow,
    questions_dir: Path,
) -> None:
    # Old keyboards stay live in the chat, so a tap can arrive for a question
    # that is already answered — it would otherwise be recorded against the
    # next, unseen one. Mirrors the Part B guard below.
    testing = await fsm_data.load(state, TestingState)
    if testing.current_part != "A" or not testing.waiting_for_answer:
        await callback.answer()
        return

    await fsm_data.merge(state, TestingState, waiting_for_answer=False)
    await _record_and_advance(
        msg, Round(bot, state, questions_dir, open_uow), str(callback_data.value)
    )
    await callback.answer()


@router.message(Testing.answering)
async def on_part_b_answer(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    open_uow: OpenUow,
    questions_dir: Path,
) -> None:
    if not message.text:
        return

    testing = await fsm_data.load(state, TestingState)
    if testing.current_part != "B" or not testing.waiting_for_answer:
        return

    await fsm_data.merge(state, TestingState, waiting_for_answer=False)
    await _record_and_advance(
        message, Round(bot, state, questions_dir, open_uow), message.text
    )
