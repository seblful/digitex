"""Integration tests for the async PostgreSQL repositories.

These tests run against a real Postgres instance launched via testcontainers
(see :func:`tests.conftest.pg_dsn`). They are skipped automatically when
Docker is not available.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from digitex.core.db import UnitOfWork
from digitex.core.domain import QuestionKey

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

pytestmark = [pytest.mark.usefixtures("clean_db")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_option(uow, subject_name: str = "Physics") -> tuple[int, int, int]:
    """Create subject → book → option and return their ids."""
    subject_id = await uow.books.get_or_create_subject(subject_name)
    book_id = await uow.books.create_book(subject_id, 2024)
    option_id = await uow.books.get_or_create_option(book_id, 1, "CT")
    return subject_id, book_id, option_id


# ---------------------------------------------------------------------------
# BookRepository
# ---------------------------------------------------------------------------


class TestBookRepository:
    async def test_get_or_create_subject_is_idempotent(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            first = await uow.books.get_or_create_subject("Math")
            second = await uow.books.get_or_create_subject("Math")
        assert first == second

    async def test_list_subjects_sorted_by_name(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            await uow.books.get_or_create_subject("Zoology")
            await uow.books.get_or_create_subject("Biology")
            subjects = await uow.books.list_subjects()
        assert [s.name for s in subjects] == ["Biology", "Zoology"]

    async def test_get_option_id_raises_keyerror_for_missing(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow, pytest.raises(KeyError):
            await uow.books.get_option_id(book_id=999, option_number=1)


# ---------------------------------------------------------------------------
# QuestionRepository
# ---------------------------------------------------------------------------


class TestQuestionRepository:
    async def test_get_or_create_part_a_rejects_non_digit(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            with pytest.raises(ValueError, match="Part A answer must be a digit"):
                await uow.questions.get_or_create(
                    option_id, QuestionKey(part="A", number=1), "abc"
                )

    async def test_get_or_create_updates_answer_on_conflict(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            key = QuestionKey(part="A", number=1)
            qid1 = await uow.questions.get_or_create(option_id, key, "3")
            qid2 = await uow.questions.get_or_create(option_id, key, "5")
            assert qid1 == qid2
            answer = await uow.questions.get_correct_answer(qid2, "A")
        assert answer == 5

    async def test_insert_image_idempotent_for_unchanged_payload(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "1"
            )
            await uow.questions.insert_image(qid, "A", b"payload")
            await uow.questions.insert_image(qid, "A", b"payload")
            await uow.questions.insert_image(qid, "A", b"new-payload")
            image = await uow.questions.get_image(qid, "A")
        assert image == b"new-payload"

    async def test_get_random_question_id_raises_when_empty(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id = await uow.books.get_or_create_subject("Empty")
            with pytest.raises(KeyError):
                await uow.questions.get_random_question_id(subject_id, "A")

    async def test_topic_upsert_then_query(self, pg_pool: AsyncConnectionPool) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, option_id = await _seed_option(uow)
            await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "1"
            )
            await uow.questions.upsert_topic(option_id, 1, "A", "kinematics")
            await uow.questions.upsert_topic(option_id, 1, "A", "kinematics")
            count = await uow.questions.count_topics()
            topics = await uow.questions.get_topics_for_subject(subject_id)
        assert count == 1
        assert topics == ["kinematics"]

    async def test_delete_topic(self, pg_pool: AsyncConnectionPool) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "1"
            )
            await uow.questions.upsert_topic(option_id, 1, "A", "kinematics")
            await uow.questions.delete_topic(option_id, 1, "A", "kinematics")
            count = await uow.questions.count_topics()
        assert count == 0


# ---------------------------------------------------------------------------
# StudentRepository
# ---------------------------------------------------------------------------


class TestStudentRepository:
    async def test_get_or_create_returns_existing_row(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            s1 = await uow.students.get_or_create(1000, "Ada", "@ada")
            s2 = await uow.students.get_or_create(1000, "Ada Renamed", "@ada2")
        assert s1.student_id == s2.student_id
        assert s2.name == "Ada Renamed"
        assert s2.username == "@ada2"


# ---------------------------------------------------------------------------
# SessionRepository — full lifecycle
# ---------------------------------------------------------------------------


class TestSessionRepository:
    async def test_full_session_lifecycle(self, pg_pool: AsyncConnectionPool) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qa = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "3"
            )
            qb = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=1), "neutron"
            )
            student = await uow.students.get_or_create(42, "Bob")
            session = await uow.sessions.create(student.student_id, option_id)

            await uow.sessions.record_answer(
                session.session_id, qa, "3", is_correct=True, time_spent=5.0
            )
            await uow.sessions.record_answer(
                session.session_id, qb, "wrong", is_correct=False, time_spent=10.0
            )
            result = await uow.sessions.complete(session.session_id)
            wrong = await uow.sessions.get_wrong_answers(session.session_id)
            info = await uow.sessions.get_session_info(session.session_id)

        assert result.part_a_score == 1
        assert result.part_b_score == 0
        assert result.total_score == 1
        assert result.max_score == 2
        assert result.time_spent >= 0  # tz-aware datetimes => non-negative
        assert isinstance(result.completed_at, datetime)
        assert result.completed_at.tzinfo is not None
        assert [w.part for w in wrong] == ["B"]
        assert info.subject_name == "Physics"
        assert info.year == 2024
        assert info.option_number == 1


# ---------------------------------------------------------------------------
# AuthorizedUserRepository
# ---------------------------------------------------------------------------


class TestAuthorizedUserRepository:
    async def test_request_approve_then_authorized(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            req = await uow.authorized_users.create_request(7, "Alice", "@alice")
            assert req.status == "pending"
            assert req.created_at.tzinfo is not None
            approved = await uow.authorized_users.approve(7, admin_id=999)
            assert approved.status == "approved"
            assert await uow.authorized_users.is_authorized(7) is True

    async def test_request_reject_then_not_authorized(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            await uow.authorized_users.create_request(8, "Eve", None)
            rejected = await uow.authorized_users.reject(8, admin_id=999)
            assert rejected.status == "rejected"
            assert await uow.authorized_users.is_authorized(8) is False

    async def test_re_request_preserves_created_at_clears_handled(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            first = await uow.authorized_users.create_request(9, "X", None)
            await uow.authorized_users.reject(9, admin_id=999)
            second = await uow.authorized_users.create_request(9, "X Renamed", None)
        assert second.created_at == first.created_at
        assert second.handled_at is None
        assert second.handled_by is None
        assert second.full_name == "X Renamed"

    async def test_delete_request(self, pg_pool: AsyncConnectionPool) -> None:
        async with UnitOfWork(pg_pool) as uow:
            await uow.authorized_users.create_request(10, "Tmp")
            await uow.authorized_users.delete_request(10)
            assert await uow.authorized_users.get_request(10) is None
            assert await uow.authorized_users.get_status(10) is None
