"""The question round — every decision between two Telegram messages.

The handlers in ``handlers/testing.py`` and ``handlers/random.py`` are thin
adapters: they build a :class:`Round` from the injected dependencies, load the
typed FSM state, open the round's transaction, call one function here, then
perform the returned outcome. Everything that *decides* — scoring, recording,
what question comes next, and the deferred ``file_id`` write owed after each
render — lives here.

The file_id debt protocol: rendering a question with no cached Telegram
``file_id`` uploads the image and yields a fresh ``file_id``. Writing it back
would cost a dedicated round-trip, so the debt is parked in the FSM
(``pending_file_id_cache``) and settled inside the *next* round's transaction.
The round's ``show_*`` methods incur the debt, the two round functions pay it
off on the way in, and :meth:`Round.end` pays whatever is left and clears the
conversation state together.

Clearing the state on its own would drop the parked write, and Telegram would
re-upload that image the next time it was shown — which is why ``end`` holds
the bot's only ``state.clear()``, and no handler names
``pending_file_id_cache``. That key belongs to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from digitex.bot import fsm_data
from digitex.bot.fsm_data import RandomState, RoundDebt, TestingState
from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.db import UnitOfWork
from digitex.domain.entities import PART_A_OPTION_COUNT

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from aiogram import Bot, types
    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

    from digitex.domain.answer import AnswerKey
    from digitex.domain.entities import Question, QuestionOrigin


@dataclass(frozen=True)
class NextQuestion:
    """Continue the testing loop by rendering this question."""

    question: Question
    next_index: int


@dataclass(frozen=True)
class RoundFinished:
    """Every question is answered — show the session results."""

    next_index: int


class Round:
    """Handle on one question round: its dependencies and its exits.

    Handlers build one per update from the injected dependencies and speak to
    the round through it: render a question (``show_testing_question`` /
    ``show_random_question``), open the round's transaction (``open_uow``),
    and leave (``end``).

    ``open_uow`` is the transaction seam: production opens a
    :class:`~digitex.db.UnitOfWork` on the pool, tests hand in a factory
    yielding their fake.
    """

    def __init__(
        self,
        bot: Bot,
        state: FSMContext,
        pool: AsyncConnectionPool,
        questions_dir: Path,
        *,
        open_uow: Callable[[], AbstractAsyncContextManager[UnitOfWork]] | None = None,
    ) -> None:
        self.bot = bot
        self.state = state
        self.questions_dir = questions_dir
        self.open_uow = open_uow or (lambda: UnitOfWork(pool))

    async def show_testing_question(
        self,
        message: types.Message,
        question: Question,
        *,
        index: int,
        started_at: float,
    ) -> None:
        """Put the playlist question at *index* on screen and record it."""
        await fsm_data.merge(
            self.state,
            TestingState,
            current_index=index,
            current_part=question.part,
            question_start_time=started_at,
            waiting_for_answer=True,
            pending_file_id_cache=None,
        )
        await self._send(message, question)

    async def show_random_question(
        self,
        message: types.Message,
        question: Question,
        *,
        started_at: float,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> None:
        """Put a random / topic question on screen and record it.

        Random mode has no playlist, so the question's own id is recorded —
        it is what scoring looks up when the reply arrives.
        """
        await fsm_data.merge(
            self.state,
            RandomState,
            current_question_id=question.question_id,
            current_part=question.part,
            question_start_time=started_at,
            waiting_for_answer=True,
            pending_file_id_cache=None,
        )
        await self._send(message, question, caption=caption, parse_mode=parse_mode)

    async def end(self) -> None:
        """Leave the round: pay whatever ``file_id`` is owed, then clear the state.

        The only way out of a question round, whichever mode it was. A
        transaction is opened only when something is actually owed, so ending
        a round that rendered from cache costs no round-trip.
        """
        debt = await fsm_data.load(self.state, RoundDebt)
        if debt.pending_file_id_cache is not None:
            async with self.open_uow() as uow:
                await uow.file_ids.cache_file_id(*debt.pending_file_id_cache)
        await self.state.clear()

    async def _send(
        self,
        message: types.Message,
        question: Question,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> None:
        """Send the question, then park the debt if the upload produced one.

        Part A goes out with the option-picker keyboard; Part B gets a
        follow-up "enter your answer" prompt.
        """
        reply_markup = part_a_kb(PART_A_OPTION_COUNT) if question.part == "A" else None
        new_file_id = await send_question(
            self.bot,
            message.chat.id,
            question,
            self.questions_dir,
            reply_markup=reply_markup,
            caption=caption,
            parse_mode=parse_mode,
        )
        if question.part == "B":
            await message.answer(MSG_ENTER_ANSWER)
        if new_file_id is not None:
            await fsm_data.merge(
                self.state,
                RoundDebt,
                pending_file_id_cache=(question.question_id, new_file_id),
            )


# ---------------------------------------------------------------------------
# The rounds
# ---------------------------------------------------------------------------


async def _settle_file_id_debt(
    uow: UnitOfWork, state: TestingState | RandomState
) -> None:
    """Pay off the ``file_id`` debt parked by the last render, if any.

    Called at the top of a round so the write rides along in that round's
    transaction rather than costing one of its own.
    """
    if state.pending_file_id_cache is not None:
        await uow.file_ids.cache_file_id(*state.pending_file_id_cache)


async def run_testing_round(
    uow: UnitOfWork,
    testing: TestingState,
    answer: str,
    *,
    now: float,
) -> NextQuestion | RoundFinished:
    """Settle the file_id debt, score and record the answer, fetch what's next.

    One transaction: the pending file_id write, the correctness lookup, the
    answer row, and the next question's metadata commit together.
    """
    question_id, _part = testing.question_ids[testing.current_index]
    started = testing.question_start_time or now
    next_index = testing.current_index + 1

    await _settle_file_id_debt(uow, testing)

    key = await uow.questions.get_correct_answer(question_id)
    await uow.sessions.record_answer(
        session_id=testing.session_id,
        question_id=question_id,
        student_answer=answer.strip(),
        correct_answer=key,
        is_correct=key.matches(answer),
        time_spent_seconds=now - started,
    )

    if next_index >= len(testing.question_ids):
        return RoundFinished(next_index=next_index)

    next_qid, _next_part = testing.question_ids[next_index]
    return NextQuestion(
        question=await uow.questions.get(next_qid),
        next_index=next_index,
    )


async def pick_random_question(
    uow: UnitOfWork, rnd: RandomState
) -> tuple[Question, QuestionOrigin] | None:
    """Settle the file_id debt and draw the next random / topic question.

    Returns None when no question matches the student's filters (or the
    filters are incomplete).
    """
    await _settle_file_id_debt(uow, rnd)

    try:
        if rnd.topic_name:
            qid = await uow.draw.get_random_question_id_by_topic(
                rnd.subject_id, rnd.topic_name
            )
        elif rnd.random_part is not None:
            qid = await uow.draw.get_random_question_id(
                rnd.subject_id, rnd.random_part, rnd.exam_type
            )
        else:
            return None
    except KeyError:
        return None

    return await uow.questions.get_full(qid)


async def evaluate_random_answer(
    uow: UnitOfWork, rnd: RandomState, answer: str
) -> tuple[bool, AnswerKey] | None:
    """Score a random-mode reply. Returns (is_correct, key).

    None when no question is active in the FSM state. A key whose value is
    None matches nothing — the feedback says the key is unknown rather than
    naming a value.
    """
    if rnd.current_question_id is None:
        return None
    key = await uow.questions.get_correct_answer(rnd.current_question_id)
    return key.matches(answer), key
