"""The question round — everything decided between two Telegram messages.

The handlers in ``handlers/testing.py`` and ``handlers/random.py`` are thin
adapters: build a :class:`Round` from the injected dependencies, load the typed
FSM state, open the round's transaction, call one function here, perform the
outcome it returns. Scoring, recording, which question comes next and the
deferred ``file_id`` write each render leaves behind are all decided here.

The ``file_id`` debt protocol. Rendering a question with no cached Telegram
``file_id`` uploads the image, and Telegram answers with an id worth keeping —
but writing it back inside the render would extend that transaction across a
network round-trip to Telegram. So the id is parked in the FSM
(``pending_file_id_cache``) and written inside the *next* round's transaction,
where a statement is being sent anyway. The ``show_*`` methods incur the debt,
the round functions pay it on the way in, and :meth:`Round.end` pays whatever
is still outstanding as it clears the state.

Those last two happen together for a reason: clearing the state alone would
drop the parked id, and Telegram would re-upload that image the next time the
question came up. Which is why ``end`` holds the bot's only ``state.clear()``,
and why no handler names ``pending_file_id_cache`` — the key belongs to this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from digitex.bot import fsm_data
from digitex.bot.fsm_data import RandomState, RoundDebt, TestingState
from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.domain.entities import PART_A_OPTION_COUNT

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram import Bot, types
    from aiogram.fsm.context import FSMContext

    from digitex.domain.answer import AnswerKey
    from digitex.domain.entities import Question, QuestionOrigin
    from digitex.domain.ports import OpenUow, Repositories


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

    Handlers build one per update and speak to the round through it: render a
    question (``show_testing_question`` / ``show_random_question``), open the
    round's transaction (``open_uow``), and leave (``end``).

    ``open_uow`` is the transaction seam, and it is required rather than
    defaulted: a round does not know what a database is, so there is nothing
    for it to fall back to. Production hands in a factory opening a unit of
    work on the pool; tests hand in one yielding their fakes.
    """

    def __init__(
        self,
        bot: Bot,
        state: FSMContext,
        questions_dir: Path,
        open_uow: OpenUow,
    ) -> None:
        self.bot = bot
        self.state = state
        self.questions_dir = questions_dir
        self.open_uow = open_uow

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
            # The round that led here has already paid the previous render's
            # debt; leaving it in place would write it a second time.
            pending_file_id_cache=None,
        )
        await self._send_and_park(message, question)

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
        await self._send_and_park(
            message, question, caption=caption, parse_mode=parse_mode
        )

    async def end(self) -> None:
        """Leave the round: pay whatever ``file_id`` is owed, then clear the state.

        The only way out of a question round, whichever mode it was. A
        transaction is opened only when something is actually owed, so ending
        a round that rendered from cache costs no round-trip.
        """
        debt = (await fsm_data.load(self.state, RoundDebt)).pending_file_id_cache
        if debt is not None:
            async with self.open_uow() as uow:
                await uow.file_ids.cache_file_id(*debt)
        await self.state.clear()

    async def _send_and_park(
        self,
        message: types.Message,
        question: Question,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> None:
        """Render the question, then park the debt if the upload produced one.

        Part A carries the option-picker keyboard, so tapping a number is the
        answer. Part B is typed, and gets a follow-up prompt saying so.
        """
        reply_markup = part_a_kb(PART_A_OPTION_COUNT) if question.part == "A" else None
        fresh_file_id = await send_question(
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
        if fresh_file_id is not None:
            await fsm_data.merge(
                self.state,
                RoundDebt,
                pending_file_id_cache=(question.question_id, fresh_file_id),
            )


# ---------------------------------------------------------------------------
# The rounds
# ---------------------------------------------------------------------------


async def _settle_file_id_debt(uow: Repositories, debt: tuple[int, str] | None) -> None:
    """Write off the ``file_id`` parked by the last render, if there was one.

    Called at the top of a round so the write rides along in that round's
    transaction rather than costing one of its own.
    """
    if debt is not None:
        await uow.file_ids.cache_file_id(*debt)


async def run_testing_round(
    uow: Repositories,
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
    # A round whose render never recorded a start time scores as instant rather
    # than as the whole time since the epoch.
    started = testing.question_start_time or now
    next_index = testing.current_index + 1

    await _settle_file_id_debt(uow, testing.pending_file_id_cache)

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

    next_question_id, _next_part = testing.question_ids[next_index]
    return NextQuestion(
        question=await uow.questions.get(next_question_id),
        next_index=next_index,
    )


async def pick_random_question(
    uow: Repositories, rnd: RandomState
) -> tuple[Question, QuestionOrigin] | None:
    """Settle the file_id debt and draw the next random / topic question.

    Returns None when no question matches the student's filters, and equally
    when the filters are incomplete — a state that says nothing about what to
    draw is the same dead end as a corpus with nothing in it.
    """
    await _settle_file_id_debt(uow, rnd.pending_file_id_cache)

    try:
        if rnd.topic_name:
            question_id = await uow.draw.get_random_question_id_by_topic(
                rnd.subject_id, rnd.topic_name
            )
        elif rnd.random_part is not None:
            question_id = await uow.draw.get_random_question_id(
                rnd.subject_id, rnd.random_part, rnd.exam_type
            )
        else:
            return None
    except KeyError:
        return None

    # get_full, not get: the caption names the year and option the question came
    # from, and that costs no second round-trip.
    return await uow.questions.get_full(question_id)


async def evaluate_random_answer(
    uow: Repositories, rnd: RandomState, answer: str
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
