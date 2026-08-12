"""Tests for the bot's result formatting, screen reads, and question rendering."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from digitex.bot.handlers.results import _format_result_lines, finish_session
from digitex.bot.handlers.start import open_registration_gate
from digitex.bot.renderer import send_question
from digitex.core.db import UnitOfWork
from digitex.core.domain import (
    AuthorizedUser,
    ExamType,
    Question,
    SessionInfo,
    SubjectRow,
    TestResult,
    WrongAnswer,
)


class TestFormatResultLines:
    def _make_result(self, exam_type: ExamType = "CT") -> TestResult:
        return TestResult(
            session_id=1,
            exam_type=exam_type,
            part_a_score=8,
            part_b_score=6,
            total_score=14,
            max_score=20,
            time_spent=600.0,
            completed_at=datetime.now(UTC),
        )

    def _make_info(self) -> SessionInfo:
        return SessionInfo(subject_name="Физика", year=2023, option_number=3)

    def test_no_wrong_answers(self) -> None:
        lines = _format_result_lines(self._make_result(), [], self._make_info())
        text = "\n".join(lines)
        assert "Физика" in text
        assert "2023" in text
        assert "3" in text
        assert "14" in text

    def test_wrong_answers_appear(self) -> None:
        wrong = [
            WrongAnswer(
                question_number=2, part="A", student_answer="3", correct_answer="4"
            ),
            WrongAnswer(
                question_number=1, part="B", student_answer="xyz", correct_answer="abc"
            ),
        ]
        lines = _format_result_lines(self._make_result(), wrong, self._make_info())
        text = "\n".join(lines)
        assert "xyz" in text
        assert "abc" in text

    def test_exam_type_ce_label(self) -> None:
        lines = _format_result_lines(
            self._make_result(exam_type="CE"), [], self._make_info()
        )
        text = "\n".join(lines)
        assert "ЦЭ" in text or "CE" in text

    def test_exam_type_ct_label(self) -> None:
        lines = _format_result_lines(
            self._make_result(exam_type="CT"), [], self._make_info()
        )
        text = "\n".join(lines)
        assert "ЦТ" in text or "CT" in text


@dataclass
class FakeSessions:
    """Records the calls a results screen makes, in order."""

    result: TestResult | None = None
    wrong: list[WrongAnswer] = field(default_factory=list)
    info: SessionInfo | None = None
    calls: list[str] = field(default_factory=list)

    async def complete(self, session_id: int) -> TestResult:
        self.calls.append("complete")
        assert self.result is not None
        return self.result

    async def get_wrong_answers(self, session_id: int) -> list[WrongAnswer]:
        self.calls.append("get_wrong_answers")
        return self.wrong

    async def get_session_info(self, session_id: int) -> SessionInfo:
        self.calls.append("get_session_info")
        assert self.info is not None
        return self.info


@dataclass
class FakeBooks:
    subjects: list[SubjectRow] = field(default_factory=list)

    async def list_subjects(self) -> list[SubjectRow]:
        return self.subjects


@dataclass
class FakeAuthorizedUsers:
    request: AuthorizedUser | None = None
    deleted: list[int] = field(default_factory=list)
    lookups: int = 0

    async def get_request(self, telegram_id: int) -> AuthorizedUser | None:
        self.lookups += 1
        return self.request

    async def delete_request(self, telegram_id: int) -> None:
        self.deleted.append(telegram_id)


@dataclass
class FakeUow:
    sessions: FakeSessions = field(default_factory=FakeSessions)
    books: FakeBooks = field(default_factory=FakeBooks)
    authorized_users: FakeAuthorizedUsers = field(default_factory=FakeAuthorizedUsers)


def as_uow(fake: FakeUow) -> UnitOfWork:
    """The fakes satisfy UnitOfWork's contract structurally; cast for the checker."""
    return cast("UnitOfWork", fake)


def _test_result() -> TestResult:
    return TestResult(
        session_id=1,
        exam_type="CT",
        part_a_score=8,
        part_b_score=6,
        total_score=14,
        max_score=20,
        time_spent=600.0,
        completed_at=datetime.now(UTC),
    )


def _authorized_user(status: str, **overrides: Any) -> AuthorizedUser:
    defaults: dict[str, Any] = {
        "telegram_id": 42,
        "full_name": "Иван Иванов",
        "status": status,
        "created_at": datetime(2026, 3, 4, 9, 30, tzinfo=UTC),
    }
    defaults.update(overrides)
    return AuthorizedUser(**defaults)


class TestFinishSession:
    """The results screen's whole read set, through one interface."""

    def _uow(self) -> FakeUow:
        uow = FakeUow()
        uow.sessions.result = _test_result()
        uow.sessions.info = SessionInfo(
            subject_name="Физика", year=2023, option_number=3
        )
        uow.books.subjects = [SubjectRow(1, "Физика")]
        return uow

    async def test_returns_everything_the_screen_renders(self) -> None:
        uow = self._uow()

        outcome = await finish_session(as_uow(uow), session_id=7)

        assert outcome.result.total_score == 14
        assert outcome.info.subject_name == "Физика"
        assert outcome.subjects == [SubjectRow(1, "Физика")]
        assert outcome.wrong_answers == []

    async def test_completes_the_session_before_reading_it_back(self) -> None:
        uow = self._uow()

        await finish_session(as_uow(uow), session_id=7)

        assert uow.sessions.calls == [
            "complete",
            "get_wrong_answers",
            "get_session_info",
        ]

    async def test_carries_wrong_answers_through(self) -> None:
        uow = self._uow()
        uow.sessions.wrong = [
            WrongAnswer(
                question_number=2, part="A", student_answer="3", correct_answer="4"
            )
        ]

        outcome = await finish_session(as_uow(uow), session_id=7)

        assert outcome.wrong_answers[0].question_number == 2


class TestOpenRegistrationGate:
    async def test_unknown_user_is_new(self) -> None:
        uow = FakeUow()

        gate = await open_registration_gate(as_uow(uow), telegram_id=42)

        assert gate.status == "new"
        assert gate.requested_at is None

    async def test_pending_user_carries_the_submission_date(self) -> None:
        uow = FakeUow()
        uow.authorized_users.request = _authorized_user("pending")

        gate = await open_registration_gate(as_uow(uow), telegram_id=42)

        assert gate.status == "pending"
        assert gate.requested_at == datetime(2026, 3, 4, 9, 30, tzinfo=UTC)

    async def test_approved_user_passes_the_gate(self) -> None:
        uow = FakeUow()
        uow.authorized_users.request = _authorized_user("approved")

        gate = await open_registration_gate(as_uow(uow), telegram_id=42)

        assert gate.status == "approved"

    async def test_rejected_user_is_reset_so_they_can_reapply(self) -> None:
        uow = FakeUow()
        uow.authorized_users.request = _authorized_user("rejected")

        gate = await open_registration_gate(as_uow(uow), telegram_id=42)

        assert gate.status == "new"
        assert uow.authorized_users.deleted == [42]

    async def test_reads_the_registration_record_once(self) -> None:
        uow = FakeUow()
        uow.authorized_users.request = _authorized_user("pending")

        await open_registration_gate(as_uow(uow), telegram_id=42)

        assert uow.authorized_users.lookups == 1


class TestSendQuestion:
    async def test_sends_with_file_id_when_cached(self) -> None:
        bot = AsyncMock()
        question = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake",
            telegram_file_id="cached_file_id",
        )
        result = await send_question(bot, 1, question)
        bot.send_photo.assert_awaited_once()
        call_kwargs = bot.send_photo.call_args.kwargs
        assert call_kwargs["photo"] == "cached_file_id"
        assert result is None

    async def test_uploads_and_returns_file_id_when_not_cached(self) -> None:
        bot = AsyncMock()
        photo_msg = MagicMock()
        photo_msg.photo = [MagicMock(file_id="new_file_id")]
        bot.send_photo.return_value = photo_msg

        question = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake",
            telegram_file_id=None,
        )

        result = await send_question(bot, 1, question)
        assert bot.send_photo.await_count == 1
        assert result == "new_file_id"

    async def test_logs_warning_when_no_photo_in_response(self) -> None:
        bot = AsyncMock()
        photo_msg = MagicMock()
        photo_msg.photo = []
        bot.send_photo.return_value = photo_msg

        question = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake",
            telegram_file_id=None,
        )

        with patch("digitex.bot.renderer.logger") as mock_logger:
            result = await send_question(bot, 1, question)
        mock_logger.warning.assert_called_once()
        assert result is None
