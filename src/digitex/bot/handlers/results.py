"""Test results and mistake review."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram import Bot, Router, types

from digitex.bot.keyboards import subjects_kb
from digitex.bot.messages import (
    MSG_EXAM_CE,
    MSG_EXAM_CT,
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
)
from digitex.bot.states import Navigation
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

    from digitex.core.db.repositories import SessionInfo, WrongAnswer
    from digitex.core.domain import TestResult

router = Router()


def _format_result_lines(
    result: TestResult,
    wrong_rows: list[WrongAnswer],
    info: SessionInfo,
) -> list[str]:
    exam_type_label = MSG_EXAM_CE if result.exam_type == "CE" else MSG_EXAM_CT
    wrong_a = [r for r in wrong_rows if r.part == "A"]
    wrong_b = [r for r in wrong_rows if r.part == "B"]

    lines = [
        MSG_RESULTS_HEADER,
        "",
        MSG_RESULTS_SUBJECT.format(subject_name=info.subject_name),
        MSG_RESULTS_TYPE.format(exam_type=exam_type_label),
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

    if wrong_a or wrong_b:
        lines.append("")
        lines.append(MSG_RESULTS_ERRORS)

        if wrong_a:
            lines.append("")
            lines.append(MSG_RESULTS_PART_A_H)
            for row in wrong_a:
                lines.append(
                    MSG_RESULTS_ERROR_ITEM.format(
                        qnum=row.question_number,
                        user_ans=row.student_answer,
                        correct_ans=row.correct_answer,
                    )
                )

        if wrong_b:
            lines.append("")
            lines.append(MSG_RESULTS_PART_B_H)
            for row in wrong_b:
                lines.append(
                    MSG_RESULTS_ERROR_ITEM.format(
                        qnum=row.question_number,
                        user_ans=row.student_answer,
                        correct_ans=row.correct_answer,
                    )
                )

    return lines


async def show_results(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    pool: AsyncConnectionPool,
) -> None:
    data = await state.get_data()
    session_id: int = data["session_id"]

    async with UnitOfWork(pool) as uow:
        result = await uow.sessions.complete(session_id)
        wrong_rows = await uow.sessions.get_wrong_answers(session_id)
        info = await uow.sessions.get_session_info(session_id)

    text = "\n".join(_format_result_lines(result, wrong_rows, info))
    await bot.send_message(message.chat.id, text, parse_mode="HTML")
    await state.clear()
    await state.set_state(Navigation.select_subject)

    async with UnitOfWork(pool) as uow:
        subjects = await uow.books.list_subjects()
    await bot.send_message(
        message.chat.id,
        MSG_RESULTS_RETRY,
        reply_markup=subjects_kb(subjects),
    )
