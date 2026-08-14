"""Integration tests for the async PostgreSQL repositories.

These tests run against a real Postgres instance launched via testcontainers
(see ``conftest.pg_dsn`` in this directory). They are skipped automatically
when Docker is not available; deselect them with ``-m "not integration"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from digitex.db import UnitOfWork
from digitex.domain.answer import AnswerKey
from digitex.domain.entities import QuestionKey

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


async def _seed_admin(uow, telegram_id: int = 999) -> int:
    """Create the student row a decision's ``handled_by`` refers to."""
    admin = await uow.students.get_or_create(telegram_id, "Admin")
    return admin.telegram_id


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
            answer = await uow.questions.get_correct_answer(qid2)
        assert answer == AnswerKey(part="A", value=5)

    async def test_a_question_with_no_answer_key_has_no_correct_answer(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """``populate_db`` loads a question whose year shipped no answer key.

        The question is stored so its image is servable; the key is NULL rather
        than a value picked for being unreachable, so nothing can match it and
        callers can see that there is nothing to match.
        """
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), None
            )
            answer = await uow.questions.get_correct_answer(qid)
        assert answer == AnswerKey(part="A", value=None)

    async def test_set_image_is_idempotent_and_records_the_latest_hash(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "1"
            )
            await uow.questions.set_image(qid, "s/2016/1/A/1.jpg", "hash-1")
            await uow.questions.set_image(qid, "s/2016/1/A/1.jpg", "hash-1")
            await uow.questions.set_image(qid, "s/2016/1/A/1.jpg", "hash-2")
            images = await uow.questions.list_images()
        assert images == [("s/2016/1/A/1.jpg", "hash-2")]

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
            topic_id = await uow.questions.get_or_create_topic(subject_id, "kinematics")
            await uow.questions.upsert_topic(option_id, 1, "A", topic_id)
            await uow.questions.upsert_topic(option_id, 1, "A", topic_id)
            count = await uow.questions.count_topics()
            topics = await uow.questions.get_topics_for_subject(subject_id)
        assert count == 1
        assert topics == ["kinematics"]

    async def test_naming_a_topic_twice_returns_the_same_id(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, _ = await _seed_option(uow)
            first = await uow.questions.get_or_create_topic(subject_id, "optics")
            second = await uow.questions.get_or_create_topic(subject_id, "optics")
        assert first == second

    async def test_a_topic_belongs_to_its_subject(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """Two subjects can share a topic name without sharing questions."""
        async with UnitOfWork(pg_pool) as uow:
            physics, _, physics_option = await _seed_option(uow)
            chemistry, _, _ = await _seed_option(uow, subject_name="Chemistry")

            physics_topic = await uow.questions.get_or_create_topic(physics, "Атом")
            chemistry_topic = await uow.questions.get_or_create_topic(chemistry, "Атом")
            await uow.questions.get_or_create(
                physics_option, QuestionKey(part="A", number=1), "1"
            )
            await uow.questions.upsert_topic(physics_option, 1, "A", physics_topic)

            assert physics_topic != chemistry_topic
            assert await uow.questions.get_topics_for_subject(physics) == ["Атом"]
            # Named but unmapped, so it is not offered as a round to play.
            assert await uow.questions.get_topics_for_subject(chemistry) == []

    async def test_same_number_in_both_parts_are_distinct_questions(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """A1 and B1 are different questions with different ids.

        One table and one identity sequence, so an id names exactly one
        question — which is what lets everything referencing a question carry
        the id alone.
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
            answer = await uow.questions.get_correct_answer(qid)
        assert answer == AnswerKey(part="B", value="ВЕРНАДСКИЙ")

    async def test_part_a_answer_comes_back_as_an_integer(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """The part is read off the row, so no caller has to say which it is."""
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=1), "4"
            )
            answer = await uow.questions.get_correct_answer(qid)
        assert answer == AnswerKey(part="A", value=4)

    async def test_get_reads_metadata_and_cached_file_id(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=4), "2"
            )
            await uow.questions.set_image(qid, "s/2016/1/A/4.jpg", "hash-1")

            before = await uow.questions.get(qid)
            await uow.questions.cache_file_id(qid, "tg-file-1")
            after = await uow.questions.get(qid)

        assert before.question_number == 4
        assert before.part == "A"
        assert before.telegram_file_id is None
        # The key rides along with the metadata — one query renders a question.
        assert before.image_key == "s/2016/1/A/4.jpg"
        assert after.telegram_file_id == "tg-file-1"

    async def test_a_question_with_no_image_row_reads_back_with_no_key(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """The join is LEFT: a question loads before its image is seeded."""
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=9), "1"
            )
            question = await uow.questions.get(qid)

        assert question.image_key is None

    async def test_reseeding_the_same_image_keeps_the_cached_file_id(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """An idempotent re-run must not throw away a working cache.

        Re-seeding is routine, and discarding the file_id would make the bot
        re-upload every image it had already cached.
        """
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=5), "1"
            )
            await uow.questions.set_image(qid, "s/2016/1/A/5.jpg", "hash-1")
            await uow.questions.cache_file_id(qid, "tg-file-1")

            await uow.questions.set_image(qid, "s/2016/1/A/5.jpg", "hash-1")
            question = await uow.questions.get(qid)

        assert question.telegram_file_id == "tg-file-1"

    async def test_reseeding_changed_bytes_drops_the_stale_file_id(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """A corrected image invalidates the id naming the old upload.

        Re-extracting a question rewrites the same path, so the key is unchanged
        and only the hash shows that anything happened. send_question prefers the
        cached id over the file, so leaving it in place would serve the
        superseded image forever.
        """
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=6), "1"
            )
            await uow.questions.set_image(qid, "s/2016/1/A/6.jpg", "hash-1")
            await uow.questions.cache_file_id(qid, "tg-file-1")

            await uow.questions.set_image(qid, "s/2016/1/A/6.jpg", "hash-2")
            question = await uow.questions.get(qid)

        assert question.telegram_file_id is None
        assert question.image_key == "s/2016/1/A/6.jpg"

    async def test_a_moved_image_drops_the_stale_file_id(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """Same bytes at a new path is still a change the cache must not survive."""
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="A", number=7), "1"
            )
            await uow.questions.set_image(qid, "s/2016/1/A/7.jpg", "hash-1")
            await uow.questions.cache_file_id(qid, "tg-file-1")

            await uow.questions.set_image(qid, "s/2017/1/A/7.jpg", "hash-1")
            question = await uow.questions.get(qid)

        assert question.telegram_file_id is None
        assert question.image_key == "s/2017/1/A/7.jpg"

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
            question, origin = await uow.questions.get_full(qid)

        assert question.question_number == 7
        assert question.part == "B"
        assert origin.year == 2023
        assert origin.option_number == 2
        assert origin.exam_type == "CE"

    async def test_get_raises_for_an_unknown_question(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            with pytest.raises(KeyError):
                await uow.questions.get(999_999)

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
                (await uow.questions.get(qid)).question_number for qid, _ in playlist
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

    async def test_random_topic_question_is_drawn_from_the_topic(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, option_id = await _seed_option(uow)
            expected = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=3), "x"
            )
            topic_id = await uow.questions.get_or_create_topic(subject_id, "optics")
            await uow.questions.upsert_topic(option_id, 3, "B", topic_id)
            drawn = await uow.questions.get_random_question_id_by_topic(
                subject_id, "optics"
            )
        assert drawn == expected

    async def test_topic_lookup_for_an_unknown_topic_raises(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            subject_id, _, _ = await _seed_option(uow)
            with pytest.raises(KeyError):
                await uow.questions.get_random_question_id_by_topic(subject_id, "nope")


# ---------------------------------------------------------------------------
# StudentRepository — identity and the registration workflow
# ---------------------------------------------------------------------------


class TestStudentRepository:
    async def test_get_or_create_returns_the_existing_row(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            first = await uow.students.get_or_create(1000, "Ada", "@ada")
            second = await uow.students.get_or_create(1000, "Ada Renamed", "@ada2")
        assert first.telegram_id == second.telegram_id
        assert first.created_at == second.created_at
        assert second.telegram_name == "Ada Renamed"
        assert second.telegram_username == "@ada2"

    async def test_get_or_create_leaves_authorization_alone(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """An approved student running /start again stays approved."""
        async with UnitOfWork(pg_pool) as uow:
            admin_id = await _seed_admin(uow)
            await uow.students.create_request(11, "Ann", "ann")
            await uow.students.approve(11, admin_id)

            refreshed = await uow.students.get_or_create(11, "Ann Renamed")

        assert refreshed.status == "approved"
        assert refreshed.full_name == "Ann"

    async def test_get_returns_none_for_an_unknown_user(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            assert await uow.students.get(4321) is None

    async def test_request_approve_then_authorized(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            admin_id = await _seed_admin(uow)
            request = await uow.students.create_request(7, "Alice", "Alice", "@alice")
            assert request.status == "pending"
            assert request.created_at.tzinfo is not None

            approved = await uow.students.approve(7, admin_id)
            assert approved.status == "approved"
            assert approved.handled_by == admin_id
            assert approved.handled_at is not None
            assert await uow.students.is_authorized(7) is True

    async def test_request_reject_then_not_authorized(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            admin_id = await _seed_admin(uow)
            await uow.students.create_request(8, "Eve", "Eve")
            rejected = await uow.students.reject(8, admin_id)
            assert rejected.status == "rejected"
            assert await uow.students.is_authorized(8) is False

    async def test_re_request_preserves_created_at_and_clears_the_decision(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """How a rejected student applies again — no row is deleted to do it."""
        async with UnitOfWork(pg_pool) as uow:
            admin_id = await _seed_admin(uow)
            first = await uow.students.create_request(9, "X", "X")
            await uow.students.reject(9, admin_id)
            second = await uow.students.create_request(9, "X Renamed", "X")

        assert second.created_at == first.created_at
        assert second.status == "pending"
        assert second.handled_at is None
        assert second.handled_by is None
        assert second.full_name == "X Renamed"

    async def test_a_decision_on_an_unknown_student_raises(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            admin_id = await _seed_admin(uow)
            with pytest.raises(KeyError):
                await uow.students.approve(12_345, admin_id)


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
            session = await uow.sessions.create(student.telegram_id, option_id)

            await uow.sessions.record_answer(
                session.session_id,
                qa,
                student_answer="3",
                correct_answer=AnswerKey(part="A", value=3),
                is_correct=True,
                time_spent_seconds=5.0,
            )
            await uow.sessions.record_answer(
                session.session_id,
                qb,
                student_answer="wrong",
                correct_answer=AnswerKey(part="B", value="neutron"),
                is_correct=False,
                time_spent_seconds=10.0,
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
        """A1 and B1 both land in the same session.

        The answer key is ``(session_id, question_id)``, which is only sound
        because an id names one question across both parts. Were that not so,
        the second answer would collide and ``ON CONFLICT DO NOTHING`` would
        drop it — unscored and unreviewable.
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
            session = await uow.sessions.create(student.telegram_id, option_id)

            await uow.sessions.record_answer(
                session.session_id,
                qa,
                student_answer="1",
                correct_answer=AnswerKey(part="A", value=3),
                is_correct=False,
                time_spent_seconds=1.0,
            )
            await uow.sessions.record_answer(
                session.session_id,
                qb,
                student_answer="proton",
                correct_answer=AnswerKey(part="B", value="neutron"),
                is_correct=False,
                time_spent_seconds=1.0,
            )
            result = await uow.sessions.complete(session.session_id)
            wrong = await uow.sessions.get_wrong_answers(session.session_id)

        assert qa != qb
        assert result.max_score == 2  # both answers landed, neither was dropped
        assert result.total_score == 0
        assert [(w.part, w.question_number) for w in wrong] == [("A", 1), ("B", 1)]
        assert [w.correct_answer for w in wrong] == ["3", "neutron"]

    async def test_a_recorded_answer_keeps_the_key_it_was_judged_against(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        """Correcting the corpus does not rewrite a finished test.

        Without the snapshot, the results screen would read the current key and
        could contradict the stored verdict — showing a student's answer as
        wrong beside the very value they gave.
        """
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            key = QuestionKey(part="B", number=1)
            qid = await uow.questions.get_or_create(option_id, key, "neutron")
            student = await uow.students.get_or_create(45, "Eli")
            session = await uow.sessions.create(student.telegram_id, option_id)
            await uow.sessions.record_answer(
                session.session_id,
                qid,
                student_answer="proton",
                correct_answer=AnswerKey(part="B", value="neutron"),
                is_correct=False,
                time_spent_seconds=1.0,
            )

            # The answer key is corrected after the test was taken.
            await uow.questions.get_or_create(option_id, key, "neutrino")
            wrong = await uow.sessions.get_wrong_answers(session.session_id)

        assert [w.correct_answer for w in wrong] == ["neutron"]

    async def test_an_answer_to_an_unkeyed_question_records_no_key(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            qid = await uow.questions.get_or_create(
                option_id, QuestionKey(part="B", number=1), None
            )
            student = await uow.students.get_or_create(46, "Fay")
            session = await uow.sessions.create(student.telegram_id, option_id)
            await uow.sessions.record_answer(
                session.session_id,
                qid,
                student_answer="anything",
                correct_answer=AnswerKey(part="B", value=None),
                is_correct=False,
                time_spent_seconds=1.0,
            )
            wrong = await uow.sessions.get_wrong_answers(session.session_id)

        assert [w.correct_answer for w in wrong] == [None]

    async def test_a_session_with_no_answers_scores_zero_of_zero(
        self, pg_pool: AsyncConnectionPool
    ) -> None:
        async with UnitOfWork(pg_pool) as uow:
            _, _, option_id = await _seed_option(uow)
            student = await uow.students.get_or_create(44, "Dee")
            session = await uow.sessions.create(student.telegram_id, option_id)
            result = await uow.sessions.complete(session.session_id)

        assert (result.part_a_score, result.part_b_score) == (0, 0)
        assert result.max_score == 0
