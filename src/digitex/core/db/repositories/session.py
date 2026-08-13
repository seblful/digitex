"""Repository for test sessions and per-question answers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.core.db.mapping import row_to_model
from digitex.core.domain import Session, SessionInfo, TestResult, WrongAnswer

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn
    from digitex.core.domain import Part


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
        part: Part,
        student_answer: str,
        is_correct: bool,
        time_spent: float,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO session_answers"
            "  (session_id, question_id, part, student_answer, is_correct, time_spent)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (session_id, question_id, part) DO NOTHING",
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
        """Every question the student got wrong, Part A first."""
        cur = await self._conn.execute(
            "SELECT q.part, q.question_number,"
            "       sa.student_answer, q.answer AS correct_answer"
            "  FROM session_answers sa"
            "  JOIN questions q"
            "    ON q.question_id = sa.question_id AND q.part = sa.part"
            " WHERE sa.session_id = %s AND sa.is_correct = FALSE"
            " ORDER BY q.part, q.question_number",
            (session_id,),
        )
        rows = await cur.fetchall()
        return [
            WrongAnswer(
                question_number=r["question_number"],
                part=r["part"],
                student_answer=r["student_answer"],
                correct_answer=r["correct_answer"],
            )
            for r in rows
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

        scores = await self._scores_by_part(session_id)
        a_correct, a_total = scores.get("A", (0, 0))
        b_correct, b_total = scores.get("B", (0, 0))

        started = session_row["started_at"]
        completed = session_row["completed_at"]
        return TestResult(
            session_id=session_id,
            exam_type=session_row["exam_type"],
            part_a_score=a_correct,
            part_b_score=b_correct,
            total_score=a_correct + b_correct,
            max_score=a_total + b_total,
            time_spent=(completed - started).total_seconds(),
            completed_at=completed,
        )

    async def _scores_by_part(self, session_id: int) -> dict[Part, tuple[int, int]]:
        """``{part: (correct, total)}`` over the answers recorded for a session."""
        cur = await self._conn.execute(
            "SELECT sa.part,"
            "       COUNT(*) FILTER (WHERE sa.is_correct) AS correct,"
            "       COUNT(*) AS total"
            "  FROM session_answers sa"
            " WHERE sa.session_id = %s"
            " GROUP BY sa.part",
            (session_id,),
        )
        rows = await cur.fetchall()
        return {r["part"]: (r["correct"], r["total"]) for r in rows}


__all__ = ["SessionRepository"]
