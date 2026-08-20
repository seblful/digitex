"""A student's run through one option, and what it scored.

Recorded answers are history rather than a view over the corpus. Each row keeps
the answer key it was judged against and points at its question ``ON DELETE
RESTRICT``, so re-seeding or correcting the corpus can neither rewrite nor erase
a finished test. Nothing here updates or deletes an answer — the only write is
the insert that records one.

The part a question belongs to is not copied here either. It lives on
``questions``, and the two reads that need it join for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.db.mapping import row_to_model
from digitex.domain.entities import Session, SessionInfo, TestResult, WrongAnswer

if TYPE_CHECKING:
    from digitex.db.mapping import DictConn
    from digitex.domain.answer import AnswerKey
    from digitex.domain.entities import Part


class SessionRepository:
    """Test sessions and the answers recorded against them."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def create(self, student_telegram_id: int, option_id: int) -> Session:
        """Open a session. ``started_at`` is the server's clock, not the bot's."""
        cur = await self._conn.execute(
            """
            INSERT INTO test_sessions (student_telegram_id, option_id)
                 VALUES (%s, %s)
              RETURNING session_id, student_telegram_id, option_id,
                        started_at, completed_at
            """,
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
        correct_answer: AnswerKey,
        is_correct: bool,
        time_spent_seconds: float,
    ) -> None:
        """Record one answer together with the key it was judged against.

        The key is stored, not looked up on read: a later correction to the
        corpus must not change what a finished test reported, and ``is_correct``
        must never come to contradict the answer displayed beside it. A question
        that had no key at all stores NULL.

        DO NOTHING on conflict, because the first answer is the answer — a
        duplicate update from a double-tapped keyboard must not overwrite it. The
        key is ``(session_id, question_id)``, and that is only sound because an
        id names one question across both parts: A1 and B1 are different rows.
        """
        await self._conn.execute(
            """
            INSERT INTO session_answers
                 (session_id, question_id, student_answer, correct_answer,
                  is_correct, time_spent_seconds)
                 VALUES (%s, %s, %s, %s, %s, %s)
                 ON CONFLICT (session_id, question_id) DO NOTHING
            """,
            (
                session_id,
                question_id,
                student_answer,
                correct_answer.stored,
                is_correct,
                time_spent_seconds,
            ),
        )

    async def complete(self, session_id: int) -> TestResult:
        """Close the session and score it.

        Stamping ``completed_at`` is what makes the elapsed time in the result
        well defined, so the two happen in this order and in one transaction.
        """
        await self._conn.execute(
            """
            UPDATE test_sessions
               SET completed_at = NOW()
             WHERE session_id = %s
            """,
            (session_id,),
        )
        return await self.get_result(session_id)

    async def get_session_info(self, session_id: int) -> SessionInfo:
        """Name the test a session was: subject, year and option number.

        Three joins up from the session, because a session records the option it
        was sat on and everything above that is reference data.

        Raises:
            KeyError: If no session has that id.
        """
        cur = await self._conn.execute(
            """
            SELECT s.name AS subject_name, b.year_value, o.option_number
              FROM test_sessions ts
              JOIN options o ON o.option_id = ts.option_id
              JOIN books b ON b.book_id = o.book_id
              JOIN subjects s ON s.subject_id = b.subject_id
             WHERE ts.session_id = %s
            """,
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Session {session_id} not found")
        return SessionInfo(
            subject_name=row["subject_name"],
            year=row["year_value"],
            option_number=row["option_number"],
        )

    async def get_wrong_answers(self, session_id: int) -> list[WrongAnswer]:
        """Every question the student got wrong, Part A first then by number.

        The key comes from the answer row, not from the question: what the
        results screen shows is what the answer was judged against at the time.
        """
        cur = await self._conn.execute(
            """
            SELECT q.part, q.question_number,
                   sa.student_answer, sa.correct_answer
              FROM session_answers sa
              JOIN questions q ON q.question_id = sa.question_id
             WHERE sa.session_id = %s AND sa.is_correct = FALSE
             ORDER BY q.part, q.question_number
            """,
            (session_id,),
        )
        rows = await cur.fetchall()
        return [
            WrongAnswer(
                question_number=row["question_number"],
                part=row["part"],
                student_answer=row["student_answer"],
                correct_answer=row["correct_answer"],
            )
            for row in rows
        ]

    async def get_result(self, session_id: int) -> TestResult:
        """Score a session: per-part marks, the exam type, and the clock.

        ``max_score`` counts the answers recorded rather than the questions the
        option holds, so an abandoned test scores out of what it actually
        answered.

        Raises:
            KeyError: If no session has that id.
        """
        cur = await self._conn.execute(
            """
            SELECT o.exam_type, o.option_number, ts.started_at, ts.completed_at
              FROM test_sessions ts
              JOIN options o ON o.option_id = ts.option_id
             WHERE ts.session_id = %s
            """,
            (session_id,),
        )
        session_row = await cur.fetchone()
        if session_row is None:
            raise KeyError(f"Session {session_id} not found")

        scores = await self._scores_by_part(session_id)
        part_a_correct, part_a_total = scores.get("A", (0, 0))
        part_b_correct, part_b_total = scores.get("B", (0, 0))

        started = session_row["started_at"]
        completed = session_row["completed_at"]
        return TestResult(
            session_id=session_id,
            exam_type=session_row["exam_type"],
            part_a_score=part_a_correct,
            part_b_score=part_b_correct,
            total_score=part_a_correct + part_b_correct,
            max_score=part_a_total + part_b_total,
            # Both are TIMESTAMPTZ, so this cannot go negative across a DST
            # boundary the way two naive timestamps could.
            time_spent=(completed - started).total_seconds(),
            completed_at=completed,
        )

    async def _scores_by_part(self, session_id: int) -> dict[Part, tuple[int, int]]:
        """``{part: (correct, total)}`` over the answers a session recorded.

        A part the student answered nothing in produces no group at all, so the
        caller supplies the zero rather than this inventing an empty row for it.

        The part is joined for because ``questions`` is the only place it is
        stored, and a session holds a couple of dozen answers at most — nothing
        worth denormalizing a column for.
        """
        cur = await self._conn.execute(
            """
            SELECT q.part,
                   COUNT(*) FILTER (WHERE sa.is_correct) AS correct,
                   COUNT(*) AS total
              FROM session_answers sa
              JOIN questions q ON q.question_id = sa.question_id
             WHERE sa.session_id = %s
             GROUP BY q.part
            """,
            (session_id,),
        )
        rows = await cur.fetchall()
        return {row["part"]: (row["correct"], row["total"]) for row in rows}


__all__ = ["SessionRepository"]
