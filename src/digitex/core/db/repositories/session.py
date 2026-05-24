"""Repository for test sessions and per-question answers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.core.db.mapping import row_to_model
from digitex.core.db.repositories._common import (
    SessionInfo,
    WrongAnswer,
)
from digitex.core.domain import Session, TestResult

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn


class SessionRepository:
    """Repository for test sessions and per-question answers."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def create(self, student_id: int, option_id: int) -> Session:
        cur = await self._conn.execute(
            "INSERT INTO test_sessions (student_id, option_id)"
            " VALUES (%s, %s)"
            " RETURNING session_id, student_id, option_id, started_at, completed_at",
            (student_id, option_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return row_to_model(row, Session)

    async def record_answer(
        self,
        session_id: int,
        question_id: int,
        part: str,
        student_answer: str,
        is_correct: bool,
        time_spent: float,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO session_answers"
            "  (session_id, question_id, part, student_answer, is_correct, time_spent)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (session_id, question_id) DO NOTHING",
            (session_id, question_id, part, student_answer, is_correct, time_spent),
        )

    async def complete(self, session_id: int) -> TestResult:
        await self._conn.execute(
            "UPDATE test_sessions SET completed_at = NOW() WHERE session_id = %s",
            (session_id,),
        )
        return await self.get_result(session_id)

    async def get_session_info(self, session_id: int) -> SessionInfo:
        cur = await self._conn.execute(
            "SELECT s.name AS subject_name, b.year_value, o.option_number"
            "  FROM test_sessions ts"
            "  JOIN options o ON o.option_id = ts.option_id"
            "  JOIN books b ON b.book_id = o.book_id"
            "  JOIN subjects s ON s.subject_id = b.subject_id"
            " WHERE ts.session_id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Session {session_id} not found")
        return SessionInfo(row["subject_name"], row["year_value"], row["option_number"])

    async def get_wrong_answers(self, session_id: int) -> list[WrongAnswer]:
        cur_a = await self._conn.execute(
            "SELECT q.question_number, 'A' AS part,"
            "       sa.student_answer, q.answer::text AS correct_answer"
            "  FROM session_answers sa"
            "  JOIN part_a_questions q ON q.question_id = sa.question_id"
            " WHERE sa.session_id = %s AND sa.part = 'A' AND sa.is_correct = FALSE"
            " ORDER BY q.question_number",
            (session_id,),
        )
        cur_b = await self._conn.execute(
            "SELECT q.question_number, 'B' AS part,"
            "       sa.student_answer, q.answer AS correct_answer"
            "  FROM session_answers sa"
            "  JOIN part_b_questions q ON q.question_id = sa.question_id"
            " WHERE sa.session_id = %s AND sa.part = 'B' AND sa.is_correct = FALSE"
            " ORDER BY q.question_number",
            (session_id,),
        )
        rows_a = await cur_a.fetchall()
        rows_b = await cur_b.fetchall()
        return [
            WrongAnswer(
                question_number=r["question_number"],
                part=r["part"],
                student_answer=r["student_answer"],
                correct_answer=r["correct_answer"],
            )
            for r in rows_a + rows_b
        ]

    async def get_result(self, session_id: int) -> TestResult:
        cur = await self._conn.execute(
            "SELECT o.exam_type, o.option_number, ts.started_at, ts.completed_at"
            "  FROM test_sessions ts"
            "  JOIN options o ON o.option_id = ts.option_id"
            " WHERE ts.session_id = %s",
            (session_id,),
        )
        session_row = await cur.fetchone()
        if session_row is None:
            raise KeyError(f"Session {session_id} not found")

        cur_a = await self._conn.execute(
            "SELECT COUNT(*) FILTER (WHERE sa.is_correct) AS correct,"
            "       COUNT(*) AS total"
            "  FROM session_answers sa"
            "  JOIN part_a_questions q ON sa.question_id = q.question_id"
            " WHERE sa.session_id = %s AND sa.part = 'A'",
            (session_id,),
        )
        part_a = await cur_a.fetchone()

        cur_b = await self._conn.execute(
            "SELECT COUNT(*) FILTER (WHERE sa.is_correct) AS correct,"
            "       COUNT(*) AS total"
            "  FROM session_answers sa"
            "  JOIN part_b_questions q ON sa.question_id = q.question_id"
            " WHERE sa.session_id = %s AND sa.part = 'B'",
            (session_id,),
        )
        part_b = await cur_b.fetchone()
        assert part_a is not None
        assert part_b is not None

        started = session_row["started_at"]
        completed = session_row["completed_at"]
        return TestResult(
            session_id=session_id,
            exam_type=session_row["exam_type"],
            part_a_score=part_a["correct"],
            part_b_score=part_b["correct"],
            total_score=part_a["correct"] + part_b["correct"],
            max_score=part_a["total"] + part_b["total"],
            time_spent=(completed - started).total_seconds(),
            completed_at=completed,
        )


__all__ = ["SessionRepository"]
