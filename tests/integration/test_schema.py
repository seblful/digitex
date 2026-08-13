"""The invariants the schema itself refuses to break.

These are the guarantees the repositories are allowed to assume, so they are
asserted against a real Postgres rather than inferred from the SQL: a reference
that has to resolve, a decision that cannot be half-recorded, and history that
reference data cannot delete out from under a student.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import psycopg
import pytest
import pytest_asyncio

from digitex.db import UnitOfWork
from digitex.domain.entities import QuestionKey

if TYPE_CHECKING:
    from typing import LiteralString

    from psycopg_pool import AsyncConnectionPool

    from digitex.db.mapping import DictConn

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("clean_db")]


@dataclass(frozen=True)
class Seeded:
    """One of everything, so a probe only has to add the row under test."""

    subject_id: int
    option_id: int
    question_id: int
    student_id: int
    session_id: int


@pytest_asyncio.fixture
async def seeded(pg_pool: AsyncConnectionPool) -> Seeded:
    async with UnitOfWork(pg_pool) as uow:
        subject_id = await uow.books.get_or_create_subject("Physics")
        book_id = await uow.books.create_book(subject_id, 2024)
        option_id = await uow.books.get_or_create_option(book_id, 1, "CT")
        question_id = await uow.questions.get_or_create(
            option_id, QuestionKey(part="A", number=1), "3"
        )
        student = await uow.students.get_or_create(500, "Ada")
        session = await uow.sessions.create(student.telegram_id, option_id)
    return Seeded(
        subject_id=subject_id,
        option_id=option_id,
        question_id=question_id,
        student_id=student.telegram_id,
        session_id=session.session_id,
    )


async def _execute(
    pool: AsyncConnectionPool, sql: LiteralString, params: tuple[Any, ...] = ()
) -> None:
    """Run one statement in its own transaction, so a failure rolls back alone."""
    async with pool.connection() as conn:
        await conn.execute(sql, params)


async def _count(
    pool: AsyncConnectionPool, sql: LiteralString, params: tuple[Any, ...] = ()
) -> int:
    async with pool.connection() as raw_conn:
        # The pool is built with ``row_factory=dict_row``; psycopg's stubs
        # default to tuple rows. Same cast the UnitOfWork makes.
        conn = cast("DictConn", raw_conn)
        cur = await conn.execute(sql, params)
        row = await cur.fetchone()
    assert row is not None
    return row["n"]


async def _record_answer(pool: AsyncConnectionPool, seeded: Seeded) -> None:
    await _execute(
        pool,
        "INSERT INTO session_answers"
        " (session_id, question_id, student_answer, correct_answer,"
        "  is_correct, time_spent_seconds)"
        " VALUES (%s, %s, '1', '3', FALSE, 1.0)",
        (seeded.session_id, seeded.question_id),
    )


class TestQuestionReferences:
    async def test_an_image_for_an_unknown_question_is_rejected(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await _execute(
                pg_pool,
                "INSERT INTO images (question_id, object_key, content_hash)"
                " VALUES (999999, %s, %s)",
                ("biology/2016/1/A/1.jpg", "hash"),
            )

    async def test_a_topic_mapping_for_an_unknown_question_is_rejected(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        await _execute(
            pg_pool,
            "INSERT INTO topics (topic_id, subject_id, name)"
            " OVERRIDING SYSTEM VALUE VALUES (1, %s, 'optics')",
            (seeded.subject_id,),
        )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await _execute(
                pg_pool,
                "INSERT INTO question_topics (question_id, topic_id)"
                " VALUES (999999, 1)",
            )

    async def test_a_question_has_at_most_one_image(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        await _execute(
            pg_pool,
            "INSERT INTO images (question_id, object_key, content_hash)"
            " VALUES (%s, %s, %s)",
            (seeded.question_id, "biology/2016/1/A/1.jpg", "hash"),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            await _execute(
                pg_pool,
                "INSERT INTO images (question_id, object_key, content_hash)"
                " VALUES (%s, %s, %s)",
                (seeded.question_id, "biology/2016/1/A/2.jpg", "hash"),
            )


class TestHistoryIsProtected:
    async def test_a_question_with_recorded_answers_cannot_be_deleted(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        """Editing the corpus must not erase what a student was scored on."""
        await _record_answer(pg_pool, seeded)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await _execute(
                pg_pool,
                "DELETE FROM questions WHERE question_id = %s",
                (seeded.question_id,),
            )

    async def test_an_option_a_session_was_sat_on_cannot_be_deleted(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        """A session records which option it was, and that reference has to hold."""
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await _execute(
                pg_pool,
                "DELETE FROM options WHERE option_id = %s",
                (seeded.option_id,),
            )

    async def test_an_unreferenced_question_goes_with_its_option(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        """With nothing pinning it, the cascade reaches the questions."""
        await _execute(
            pg_pool,
            "DELETE FROM test_sessions WHERE session_id = %s",
            (seeded.session_id,),
        )
        await _execute(
            pg_pool, "DELETE FROM options WHERE option_id = %s", (seeded.option_id,)
        )
        assert await _count(pg_pool, "SELECT COUNT(*) AS n FROM questions") == 0

    async def test_one_answer_per_question_per_session(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        await _record_answer(pg_pool, seeded)
        with pytest.raises(psycopg.errors.UniqueViolation):
            await _record_answer(pg_pool, seeded)

    async def test_deleting_a_student_takes_their_sessions(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        """A person asking to be forgotten is not a corpus edit."""
        await _record_answer(pg_pool, seeded)
        await _execute(
            pg_pool, "DELETE FROM students WHERE telegram_id = %s", (seeded.student_id,)
        )
        assert await _count(pg_pool, "SELECT COUNT(*) AS n FROM test_sessions") == 0
        assert await _count(pg_pool, "SELECT COUNT(*) AS n FROM session_answers") == 0

    async def test_a_negative_duration_is_rejected(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.CheckViolation):
            await _execute(
                pg_pool,
                "INSERT INTO session_answers"
                " (session_id, question_id, student_answer, correct_answer,"
                "  is_correct, time_spent_seconds)"
                " VALUES (%s, %s, '1', '3', FALSE, -1.0)",
                (seeded.session_id, seeded.question_id),
            )

    async def test_a_session_cannot_finish_before_it_started(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.CheckViolation):
            await _execute(
                pg_pool,
                "UPDATE test_sessions SET completed_at = started_at - interval '1 hour'"
                " WHERE session_id = %s",
                (seeded.session_id,),
            )


class TestAnswerKeys:
    async def test_a_part_a_answer_must_be_an_option_index(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        for answer in ("abc", "0", "01"):
            with pytest.raises(psycopg.errors.CheckViolation):
                await _execute(
                    pg_pool,
                    "INSERT INTO questions"
                    " (option_id, part, question_number, answer)"
                    " VALUES (%s, 'A', 90, %s)",
                    (seeded.option_id, answer),
                )

    async def test_a_question_may_have_no_answer_key(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        """A missing key is spelled NULL, in both parts."""
        await _execute(
            pg_pool,
            "INSERT INTO questions (option_id, part, question_number, answer)"
            " VALUES (%s, 'A', 91, NULL), (%s, 'B', 91, NULL)",
            (seeded.option_id, seeded.option_id),
        )
        assert (
            await _count(
                pg_pool, "SELECT COUNT(*) AS n FROM questions WHERE answer IS NULL"
            )
            == 2
        )

    async def test_a_question_number_starts_at_one(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.CheckViolation):
            await _execute(
                pg_pool,
                "INSERT INTO questions (option_id, part, question_number, answer)"
                " VALUES (%s, 'A', 0, '1')",
                (seeded.option_id,),
            )


class TestTopicNames:
    async def test_a_subject_cannot_name_the_same_topic_twice(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        await _execute(
            pg_pool,
            "INSERT INTO topics (subject_id, name) VALUES (%s, 'optics')",
            (seeded.subject_id,),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            await _execute(
                pg_pool,
                "INSERT INTO topics (subject_id, name) VALUES (%s, 'optics')",
                (seeded.subject_id,),
            )


class TestRegistrationStates:
    async def test_a_pending_student_cannot_carry_a_decision(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.CheckViolation):
            await _execute(
                pg_pool,
                "UPDATE students SET handled_at = NOW(), handled_by = %s"
                " WHERE telegram_id = %s",
                (seeded.student_id, seeded.student_id),
            )

    async def test_a_decided_student_must_say_when(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.CheckViolation):
            await _execute(
                pg_pool,
                "UPDATE students SET status = 'approved', full_name = 'Ada L.'"
                " WHERE telegram_id = %s",
                (seeded.student_id,),
            )

    async def test_a_student_cannot_be_approved_without_applying(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        """``full_name`` is what the application carried, so a decision needs it."""
        with pytest.raises(psycopg.errors.CheckViolation):
            await _execute(
                pg_pool,
                "UPDATE students"
                " SET status = 'approved', handled_at = NOW(), handled_by = %s"
                " WHERE telegram_id = %s",
                (seeded.student_id, seeded.student_id),
            )

    async def test_a_decision_names_a_real_student(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await _execute(
                pg_pool,
                "UPDATE students SET status = 'approved', full_name = 'Ada L.',"
                "                    handled_at = NOW(), handled_by = 999999"
                " WHERE telegram_id = %s",
                (seeded.student_id,),
            )

    async def test_an_unknown_status_is_rejected(
        self, pg_pool: AsyncConnectionPool, seeded: Seeded
    ) -> None:
        with pytest.raises(psycopg.errors.CheckViolation):
            await _execute(
                pg_pool,
                "UPDATE students SET status = 'maybe' WHERE telegram_id = %s",
                (seeded.student_id,),
            )


class TestDowngrade:
    """Declared last on purpose — it drops the schema the session shares.

    ``docs/database-reference.md`` requires every revision to provide a working
    ``downgrade()``, and nothing else exercises this one.
    """

    async def test_the_schema_can_be_dropped_and_rebuilt(self, pg_dsn: str) -> None:
        from alembic import command
        from alembic.config import Config

        project_root = Path(__file__).resolve().parent.parent.parent
        cfg = Config(str(project_root / "alembic.ini"))
        cfg.set_main_option("script_location", str(project_root / "migrations"))

        command.downgrade(cfg, "base")
        try:
            async with await psycopg.AsyncConnection.connect(pg_dsn) as conn:
                cur = await conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables"
                    " WHERE table_schema = 'public' AND table_name = 'questions'"
                )
                row = await cur.fetchone()
            assert row is not None
            assert row[0] == 0
        finally:
            command.upgrade(cfg, "head")
