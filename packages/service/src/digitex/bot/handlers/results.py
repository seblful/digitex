"""The screen that closes a session: the score, and what was missed.

No router. This screen is reached only by finishing the last question, so the
testing loop draws it on its way out — there is no tap that leads here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiogram.utils.text_decorations import html_decoration

from digitex.bot import fsm_data
from digitex.bot.fsm_data import TestingState
from digitex.bot.keyboards import subjects_kb
from digitex.bot.messages import (
    EXAM_LABELS,
    MSG_RESULTS_ERROR_ITEM,
    MSG_RESULTS_ERRORS,
    MSG_RESULTS_HEADER,
    MSG_RESULTS_OPTION,
    MSG_RESULTS_PART_A,
    MSG_RESULTS_PART_A_H,
    MSG_RESULTS_PART_B,
    MSG_RESULTS_PART_B_H,
    MSG_RESULTS_RETRY,
    MSG_RESULTS_SCORE,
    MSG_RESULTS_SUBJECT,
    MSG_RESULTS_TIME,
    MSG_RESULTS_TYPE,
    MSG_RESULTS_YEAR,
    format_answer,
)
from digitex.bot.states import Navigation

if TYPE_CHECKING:
    from aiogram import types

    from digitex.bot.answer_flow import Round
    from digitex.domain.entities import (
        SessionInfo,
        SubjectRow,
        TestResult,
        WrongAnswer,
    )
    from digitex.domain.ports import Repositories


@dataclass(frozen=True)
class SessionOutcome:
    """Everything the results screen shows, read in one transaction."""

    result: TestResult
    wrong_answers: list[WrongAnswer]
    info: SessionInfo
    subjects: list[SubjectRow]


async def finish_session(uow: Repositories, session_id: int) -> SessionOutcome:
    """Close the session and read back everything the results screen needs.

    One transaction for the whole screen — including the subject list the
    "try again" prompt is built from, which used to cost a second one.
    """
    return SessionOutcome(
        result=await uow.sessions.complete(session_id),
        wrong_answers=await uow.sessions.get_wrong_answers(session_id),
        info=await uow.sessions.get_session_info(session_id),
        subjects=await uow.books.list_subjects(),
    )


def _summary_lines(result: TestResult, info: SessionInfo) -> list[str]:
    """What was taken, what it scored, and how long it took."""
    return [
        MSG_RESULTS_HEADER,
        "",
        MSG_RESULTS_SUBJECT.format(subject_name=info.subject_name),
        MSG_RESULTS_TYPE.format(exam_type=EXAM_LABELS[result.exam_type]),
        MSG_RESULTS_YEAR.format(year=info.year),
        MSG_RESULTS_OPTION.format(option_number=info.option_number),
        "",
        MSG_RESULTS_SCORE.format(
            total_score=result.total_score, max_score=result.max_score
        ),
        MSG_RESULTS_PART_A.format(part_a_score=result.part_a_score),
        MSG_RESULTS_PART_B.format(part_b_score=result.part_b_score),
        "",
        MSG_RESULTS_TIME.format(time_spent=result.time_spent),
    ]


def _mistake_lines(wrong_rows: list[WrongAnswer]) -> list[str]:
    """The review block, grouped by Part. Empty when nothing was missed.

    A Part B line carries free text on both sides and the message is sent with
    ``parse_mode="HTML"``, so both sides are escaped.
    """
    by_part = [
        (MSG_RESULTS_PART_A_H, [row for row in wrong_rows if row.part == "A"]),
        (MSG_RESULTS_PART_B_H, [row for row in wrong_rows if row.part == "B"]),
    ]
    if not any(rows for _, rows in by_part):
        return []

    lines: list[str] = ["", MSG_RESULTS_ERRORS]
    for header, rows in by_part:
        if not rows:
            continue
        lines.append("")
        lines.append(header)
        lines.extend(
            MSG_RESULTS_ERROR_ITEM.format(
                qnum=row.question_number,
                user_ans=html_decoration.quote(row.student_answer),
                correct_ans=html_decoration.quote(format_answer(row.correct_answer)),
            )
            for row in rows
        )
    return lines


def _format_result_lines(
    result: TestResult,
    wrong_rows: list[WrongAnswer],
    info: SessionInfo,
) -> list[str]:
    """Every line of the results screen, in the order it is read."""
    return _summary_lines(result, info) + _mistake_lines(wrong_rows)


async def show_results(message: types.Message, round: Round) -> None:
    """Send the results, end the round, and offer a new subject."""
    testing = await fsm_data.load(round.state, TestingState)

    async with round.open_uow() as uow:
        outcome = await finish_session(uow, testing.session_id)

    text = "\n".join(
        _format_result_lines(outcome.result, outcome.wrong_answers, outcome.info)
    )
    await round.bot.send_message(message.chat.id, text, parse_mode="HTML")
    # Ending the round pays the last render's file_id debt and clears the state,
    # so the state below is set on an otherwise empty conversation.
    await round.end()
    await round.state.set_state(Navigation.select_subject)

    await round.bot.send_message(
        message.chat.id,
        MSG_RESULTS_RETRY,
        reply_markup=subjects_kb(outcome.subjects),
    )
