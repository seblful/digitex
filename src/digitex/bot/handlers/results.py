"""Test results and mistake review."""

from aiogram import Router, types
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.keyboards import subjects_kb
from digitex.bot.states import Navigation
from digitex.config import get_settings

router = Router()


async def show_results(
    message: types.Message,
    state: FSMContext,
    bot,
) -> None:
    data = await state.get_data()
    session_id: int = data["session_id"]
    db_path = get_settings().database.path

    def get_results(uow):
        result = uow.sessions.complete(session_id)
        wrong_rows = uow._conn.execute(
            "SELECT q.question_number, q.part, sa.student_answer,"
            "       COALESCE(pa.correct_order, pb.answer_text) AS correct_answer"
            "  FROM session_answers sa"
            "  JOIN questions q ON q.question_id = sa.question_id"
            "  LEFT JOIN part_a_answers pa ON pa.question_id = sa.question_id"
            "  LEFT JOIN part_b_answers pb ON pb.question_id = sa.question_id"
            " WHERE sa.session_id = ? AND sa.is_correct = 0"
            " ORDER BY q.part, q.question_number",
            (session_id,),
        ).fetchall()

        session_row = uow._conn.execute(
            "SELECT b.subject_id, b.year_value, s.name, ts.option_number"
            "  FROM test_sessions ts"
            "  JOIN books b ON b.book_id = ts.book_id"
            "  JOIN subjects s ON s.subject_id = b.subject_id"
            " WHERE ts.session_id = ?",
            (session_id,),
        ).fetchone()

        return result, wrong_rows, session_row

    result, wrong_rows, session_row = await with_uow(db_path, get_results)
    subject_name = session_row[2]
    year = session_row[1]
    option_number = session_row[3]

    lines = [
        f"Test Complete: {subject_name} {year} — Option {option_number}",
        "",
        f"Correct: {result.total_score}/{result.max_score}",
        f"Part A: {result.part_a_score} | Part B: {result.part_b_score}",
        f"Time: {result.time_spent:.0f}s",
    ]

    if wrong_rows:
        lines.append("")
        lines.append("Mistakes:")
        for row in wrong_rows:
            qnum, part, user_ans, correct_ans = row
            lines.append(f"  Q{qnum} ({part}): yours '{user_ans}', correct '{correct_ans}'")

    await bot.send_message(message.chat.id, "\n".join(lines))
    await state.clear()
    await state.set_state(Navigation.select_subject)

    def list_subjects(uow):
        return uow._conn.execute(
            "SELECT subject_id, name FROM subjects ORDER BY name"
        ).fetchall()

    subjects = await with_uow(db_path, list_subjects)
    await bot.send_message(
        message.chat.id,
        "Select a subject to start a new test:",
        reply_markup=subjects_kb(subjects),
    )
