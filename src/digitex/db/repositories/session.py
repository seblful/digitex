"""Repository for test sessions and per-question answers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.db.mapping import row_to_model
from digitex.domain.entities import Session, SessionInfo, TestResult, WrongAnswer

if TYPE_CHECKING:
    from digitex.db.mapping import DictConn
    from digitex.domain.entities import Part


class SessionRepository:
    """Repository for test sessions and per-question answers."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def create(self, student_telegram_id: int, option_id: int) -> Session:
        cur = await self._conn.execute(
            "INSERT INTO test_sessions (student_telegram_id, option_id)"
            " VALUES (%s, %s)"
            " RETURNING session_id, student_telegram_id, option_id,"
            "           started_at, completed_at",
            (student_telegram_id, option_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return row_to_model(row, Session)

    async def record_answer(
        self,
        session_id: int,
        question_id: int,
        student_answer: str,
        correct_answer: int | str | None,
        is_correct: bool,
        time_spent_seconds: float,
    ) -> None:
        """Record one answer, together with the key it was judged against.

        The key is stored rather than looked up on read: a correction to the
        corpus afterwards must not change what a finished test reported, and
        ``is_correct`` must not come to disagree with the answer shown beside it.
        """
        await self._conn.execute(
            "INSERT INTO session_answers"
            "  (session_id, question_id, student_answer, correct_answer,"
            "   is_correct, time_spent_seconds)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (session_id, question_id) DO NOTHING",
            (
                session_id,
                question_id,
                student_answer,
                None if correct_answer is None else str(correct_answer),
                is_correct,
                time_spent_seconds,
            ),
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
            "       sa.student_answer, sa.correct_answer"
            "  FROM session_answers sa"
            "  JOIN questions q ON q.question_id = sa.question_id"
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
        """``{part: (correct, total)}`` over the answers recorded for a session.

        The part comes from the question, which is the only place it is stored.
        A session holds at most a couple of dozen answers, so the join costs
        nothing worth denormalizing for.
        """
        cur = await self._conn.execute(
            "SELECT q.part,"
            "       COUNT(*) FILTER (WHERE sa.is_correct) AS correct,"
            "       COUNT(*) AS total"
            "  FROM session_answers sa"
            "  JOIN questions q ON q.question_id = sa.question_id"
            " WHERE sa.session_id = %s"
            " GROUP BY q.part",
            (session_id,),
        )
        rows = await cur.fetchall()
        return {r["part"]: (r["correct"], r["total"]) for r in rows}


__all__ = ["SessionRepository"]
