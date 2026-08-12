"""Integration tests for the async PostgreSQL repositories.

These tests run against a real Postgres instance launched via testcontainers
(see ``conftest.pg_dsn`` in this directory). They are skipped automatically
when Docker is not available; deselect them with ``-m "not integration"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from digitex.core.db import UnitOfWork
from digitex.core.domain import QuestionKey

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("clean_db")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_option(
    uow,
    subject_name: str = "Physics",
    year: int = 2024,
    option_number: int = 1,
    exam_type: str = "CT",
) -> tuple[int, int, int]:
    """Create subject → book → option and return their ids."""
    subject_id = await uow.books.get_or_create_subject(subject_name)
    book_id = await uow.books.get_book(subject_id, year) or await uow.books.create_book(
        subject_id, year
    )
    option_id = await uow.books.get_or_create_option(book_id, option_number, exam_type)
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
        async with UnitOfWork(pg_pool) as uow:
            with pytest.raises(KeyError):
                await uow.books.get_option_id(book_id=999, option_number=1)

    async def test_book_lookup_round_trips(self, pg_pool: AsyncConnectionPool) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, book_id, option_id = await _seed_option(uow)
            found = await uow.books.get_book(subject_id, 2024)
            missing = await uow.books.get_book(subject_id, 1999)
            assert await uow.books.get_option_id(book_id, 1) == option_id
        assert found == book_id
        assert missing is None

    async def test_years_come_back_newest_first(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            for year in (2019, 2024, 2021):
                await _seed_option(uow, year=year)
            subject_id = await uow.books.get_or_create_subject("Physics")
            years = await uow.books.list_years(subject_id)
        assert years == [2024, 2021, 2019]

    async def test_options_are_listed_per_exam_type(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, book_id, _ = await _seed_option(uow, option_number=3, exam_type="CT")
            await uow.books.get_or_create_option(book_id, 1, "CE")
            await uow.books.get_or_create_option(book_id, 2, "CT")
            ce = await uow.books.list_options(book_id, "CE")
            ct = await uow.books.list_options(book_id, "CT")
        assert ce == [1]
        assert ct == [2, 3]

    async def test_option_conflict_updates_the_exam_type(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, book_id, option_id = await _seed_option(uow, exam_type="CT")
            again = await uow.books.get_or_create_option(book_id, 1, "CE")
            assert again == option_id
            assert await uow.books.list_options(book_id, "CE") == [1]


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

    async def test_same_number_in_both_parts_are_distinct_questions(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """A1 and B1 are different questions with different ids.

        Under the old two-table split they could share a ``question_id``, which
        is what made a Part B answer collide with a Part A one in the same
        session and be silently discarded.
        """
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qa = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "3"
            )
            qb = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=1), "neutron"
            )
        assert qa != qb

    async def test_part_b_answer_comes_back_as_text(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=1), "ВЕРНАДСКИЙ"
            )
            answer = await uow.questions.get_correct_answer(qid, "B")
        assert answer == "ВЕРНАДСКИЙ"

    async def test_the_unmatchable_part_a_placeholder_is_storable(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """``populate_db`` writes '0' when a year has no answer key.

        The option buttons start at 1, so 0 can never be matched — but the old
        ``CHECK (answer BETWEEN 1 AND 5)`` rejected the write and rolled back
        the whole year's load.
        """
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "0"
            )
            answer = await uow.questions.get_correct_answer(qid, "A")
        assert answer == 0

    async def test_get_reads_metadata_and_cached_file_id(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=4), "2"
            )
            await uow.questions.insert_image(qid, "A", b"payload")

            before = await uow.questions.get(qid, "A")
            await uow.questions.cache_file_id(qid, "A", "tg-file-1")
            after = await uow.questions.get(qid, "A")

        assert before.question_number == 4
        assert before.part == "A"
        assert before.telegram_file_id is None
        assert before.image_data == b""  # metadata only — no BYTEA payload
        assert after.telegram_file_id == "tg-file-1"

    async def test_get_full_carries_the_question_origin(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(
                uow, year=2023, option_number=2, exam_type="CE"
            )
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=7), "photon"
            )
            question, origin = await uow.questions.get_full(qid, "B")

        assert question.question_number == 7
        assert origin.year == 2023
        assert origin.option_number == 2
        assert origin.exam_type == "CE"

    async def test_get_raises_for_the_wrong_part(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "1"
            )
            with pytest.raises(KeyError):
                await uow.questions.get(qid, "B")

    async def test_playlist_is_ordered_part_a_then_b_by_number(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """This order decides the sequence a student answers in."""
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            for number in (2, 1, 10):
                await uow.questions.get_or_create(
                    option_id, QuestionKey(part="A", number=number), "1"
                )
            for number in (2, 1):
                await uow.questions.get_or_create(
                    option_id, QuestionKey(part="B", number=number), "x"
                )
            playlist = await uow.questions.list_ids_for_option(option_id)

            numbers = [
                (await uow.questions.get(qid, part)).question_number
                for qid, part in playlist
            ]
            parts = [part for _, part in playlist]

        assert parts == ["A", "A", "A", "B", "B"]
        assert numbers == [1, 2, 10, 1, 2]

    async def test_random_question_is_drawn_from_the_requested_part(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, option_id = await _seed_option(uow)
            await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "1"
            )
            qb = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=1), "x"
            )
            drawn = await uow.questions.get_random_question_id(subject_id, "B")
        assert drawn == qb

    async def test_random_question_honours_the_exam_type_filter(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, ct_option = await _seed_option(
                uow, year=2023, option_number=8, exam_type="CT"
            )
            _, _, ce_option = await _seed_option(
                uow, year=2023, option_number=1, exam_type="CE"
            )
            await uow.questions.get_or_create(
                ct_option, QuestionKey(part="A", number=1), "1"
            )
            expected = await uow.questions.get_or_create(
                ce_option, QuestionKey(part="A", number=1), "2"
            )
            drawn = await uow.questions.get_random_question_id(subject_id, "A", "CE")
        assert drawn == expected

    async def test_random_topic_question_returns_its_part(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, option_id = await _seed_option(uow)
            expected = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=3), "x"
            )
            await uow.questions.upsert_topic(option_id, 3, "B", "optics")
            qid, part = await uow.questions.get_random_question_id_by_topic(
                subject_id, "optics"
            )
        assert (qid, part) == (expected, "B")

    async def test_topic_lookup_for_an_unknown_topic_raises(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, _ = await _seed_option(uow)
            with pytest.raises(KeyError):
                await uow.questions.get_random_question_id_by_topic(subject_id, "nope")


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
                session.session_id, qa, "A", "3", is_correct=True, time_spent=5.0
            )
            await uow.sessions.record_answer(
                session.session_id, qb, "B", "wrong", is_correct=False, time_spent=10.0
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

    async def test_both_parts_are_recorded_even_with_the_same_number(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """The bug migration 0002 was written for, now unrepresentable.

        The two part tables had separate identity sequences, so A1 and B1 could
        share a ``question_id``; keyed on ``(session_id, question_id)`` alone,
        the Part B row collided with the Part A row already recorded and
        ``ON CONFLICT DO NOTHING`` discarded it — unscored and unreviewable.
        One table means one sequence, so the two ids simply differ.
        """
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qa = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "3"
            )
            qb = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=1), "neutron"
            )
            student = await uow.students.get_or_create(43, "Cleo")
            session = await uow.sessions.create(student.student_id, option_id)

            await uow.sessions.record_answer(
                session.session_id, qa, "A", "1", is_correct=False, time_spent=1.0
            )
            await uow.sessions.record_answer(
                session.session_id, qb, "B", "proton", is_correct=False, time_spent=1.0
            )
            result = await uow.sessions.complete(session.session_id)
            wrong = await uow.sessions.get_wrong_answers(session.session_id)

        assert qa != qb
        assert result.max_score == 2  # both answers landed, neither was dropped
        assert result.total_score == 0
        assert [(w.part, w.question_number) for w in wrong] == [("A", 1), ("B", 1)]
        # Part A answers are stored as text now; the results screen shows them
        # as-is rather than through a per-part cast.
        assert [w.correct_answer for w in wrong] == ["3", "neutron"]

    async def test_a_session_with_no_answers_scores_zero_of_zero(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            student = await uow.students.get_or_create(44, "Dee")
            session = await uow.sessions.create(student.student_id, option_id)
            result = await uow.sessions.complete(session.session_id)

        assert (result.part_a_score, result.part_b_score) == (0, 0)
        assert result.max_score == 0


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
