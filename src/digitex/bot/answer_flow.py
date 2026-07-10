"""The question round — every decision between two Telegram messages.

The handlers in ``handlers/testing.py`` and ``handlers/random.py`` are thin
adapters: they load the typed FSM state, open a UnitOfWork, call one function
here, then perform the returned outcome (send a message, merge FSM fields).
Everything that *decides* — scoring, recording, what question comes next,
and the deferred ``file_id`` write owed after each render — lives here.

The file_id debt protocol: rendering a question with no cached Telegram
``file_id`` uploads the image and yields a fresh ``file_id``. Writing it back
would cost a dedicated round-trip, so the debt is parked in the FSM
(``pending_file_id_cache``) and settled inside the *next* round's
transaction. ``file_id_debt`` creates the debt; ``run_testing_round`` and
``pick_random_question`` settle it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.core.answer import check_answer

if TYPE_CHECKING:
    from aiogram import Bot, types

    from digitex.bot.fsm_data import RandomState, TestingState
    from digitex.core.db import UnitOfWork
    from digitex.core.db.repositories._common import QuestionOrigin
    from digitex.core.domain import Part, Question


@dataclass(frozen=True)
class NextQuestion:
    """Continue the testing loop by rendering this question."""

    question: Question
    next_index: int


@dataclass(frozen=True)
class RoundFinished:
    """Every question is answered — show the session results."""

    next_index: int


def file_id_debt(
    question: Question, new_file_id: str | None
) -> tuple[int, Part, str] | None:
    """The deferred images-table write owed after rendering a question.

    Park the returned tuple in the FSM's ``pending_file_id_cache``; the next
    round settles it. None means the cached ``file_id`` was reused.
    """
    if new_file_id is None:
        return None
    return (question.question_id, question.part, new_file_id)


async def load_renderable(uow: UnitOfWork, question_id: int, part: Part) -> Question:
    """Fetch a question's metadata, plus image bytes only on a cache miss."""
    question = await uow.questions.get(question_id, part)
    if not question.telegram_file_id:
        image_data = await uow.questions.get_image(question_id, part)
        question = question.model_copy(update={"image_data": image_data})
    return question


async def run_testing_round(
    uow: UnitOfWork,
    testing: TestingState,
    answer: str,
    *,
    now: float,
) -> NextQuestion | RoundFinished:
    """Settle the file_id debt, score and record the answer, fetch what's next.

    One transaction: the pending file_id write, the correctness lookup, the
    answer row, and the next question's metadata/bytes commit together.
    """
    question_id, part = testing.question_ids[testing.current_index]
    started = testing.question_start_time or now
    next_index = testing.current_index + 1

    if testing.pending_file_id_cache is not None:
        await uow.questions.cache_file_id(*testing.pending_file_id_cache)

    correct = await uow.questions.get_correct_answer(question_id, part)
    await uow.sessions.record_answer(
        session_id=testing.session_id,
        question_id=question_id,
        part=part,
        student_answer=answer.strip(),
        is_correct=check_answer(part, answer, correct),
        time_spent=now - started,
    )

    if next_index >= len(testing.question_ids):
        return RoundFinished(next_index=next_index)

    next_qid, next_part = testing.question_ids[next_index]
    return NextQuestion(
        question=await load_renderable(uow, next_qid, next_part),
        next_index=next_index,
    )


async def pick_random_question(
    uow: UnitOfWork, rnd: RandomState
) -> tuple[Question, QuestionOrigin] | None:
    """Settle the file_id debt and draw the next random / topic question.

    Returns None when no question matches the student's filters (or the
    filters are incomplete).
    """
    if rnd.pending_file_id_cache is not None:
        await uow.questions.cache_file_id(*rnd.pending_file_id_cache)

    try:
        if rnd.topic_name:
            qid, part = await uow.questions.get_random_question_id_by_topic(
                rnd.subject_id, rnd.topic_name
            )
        elif rnd.random_part is not None:
            part = rnd.random_part
            qid = await uow.questions.get_random_question_id(
                rnd.subject_id, part, rnd.exam_type
            )
        else:
            return None
    except KeyError:
        return None

    question, origin = await uow.questions.get_full(qid, part)
    if not question.telegram_file_id:
        image_data = await uow.questions.get_image(qid, part)
        question = question.model_copy(update={"image_data": image_data})
    return question, origin


async def evaluate_random_answer(
    uow: UnitOfWork, rnd: RandomState, answer: str
) -> tuple[bool, int | str] | None:
    """Score a random-mode reply. Returns (is_correct, correct_answer).

    None when no question is active in the FSM state.
    """
    if rnd.current_question_id is None or rnd.current_part is None:
        return None
    correct = await uow.questions.get_correct_answer(
        rnd.current_question_id, rnd.current_part
    )
    return check_answer(rnd.current_part, answer, correct), correct


async def ask_question(
    bot: Bot,
    message: types.Message,
    question: Question,
    *,
    caption: str | None = None,
    parse_mode: str | None = None,
) -> str | None:
    """Send a question to the chat and return the new Telegram ``file_id``.

    Part A goes out with the option-picker keyboard; Part B gets a follow-up
    "enter your answer" prompt. Pass the result to ``file_id_debt``.
    """
    if question.part == "A":
        new_file_id = await send_question(
            bot,
            message.chat.id,
            question,
            reply_markup=part_a_kb(question.num_options),
            caption=caption,
            parse_mode=parse_mode,
        )
    else:
        new_file_id = await send_question(
            bot,
            message.chat.id,
            question,
            caption=caption,
            parse_mode=parse_mode,
        )
        await message.answer(MSG_ENTER_ANSWER)
    return new_file_id
