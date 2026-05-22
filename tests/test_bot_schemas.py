"""Tests for bot Pydantic schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from digitex.core.domain import Question, Session, Student, TestResult


class TestStudent:
    def test_valid_student(self) -> None:
        student = Student(student_id=1, telegram_id=123, name="Test")
        assert student.student_id == 1
        assert student.telegram_id == 123
        assert student.name == "Test"
        assert student.username is None

    def test_student_with_username(self) -> None:
        student = Student(student_id=1, telegram_id=123, name="Test", username="@test")
        assert student.username == "@test"

    def test_student_requires_fields(self) -> None:
        with pytest.raises(ValidationError):
            Student()  # type: ignore


class TestQuestion:
    def test_valid_question(self) -> None:
        q = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake_image_data",
        )
        assert q.question_id == 1
        assert q.part == "A"
        assert q.question_number == 1
        assert q.image_data == b"fake_image_data"
        assert q.telegram_file_id is None
        assert q.num_options == 5

    def test_question_with_file_id(self) -> None:
        q = Question(
            question_id=2,
            part="B",
            question_number=10,
            image_data=b"data",
            telegram_file_id="file_123",
            num_options=3,
        )
        assert q.telegram_file_id == "file_123"
        assert q.num_options == 3

    def test_invalid_part(self) -> None:
        with pytest.raises(ValidationError):
            Question(
                question_id=1,
                part="C",  # type: ignore
                question_number=1,
                image_data=b"data",
            )


class TestSession:
    def test_valid_session(self) -> None:
        now = datetime.now(UTC)
        session = Session(
            session_id=1,
            student_id=1,
            option_id=5,
            started_at=now,
        )
        assert session.session_id == 1
        assert session.option_id == 5
        assert session.completed_at is None

    def test_session_with_completed_at(self) -> None:
        now = datetime.now(UTC)
        session = Session(
            session_id=1,
            student_id=1,
            option_id=5,
            started_at=now,
            completed_at=now,
        )
        assert session.completed_at == now


class TestTestResult:
    def test_valid_result(self) -> None:
        now = datetime.now(UTC)
        result = TestResult(
            session_id=1,
            part_a_score=5,
            part_b_score=3,
            total_score=8,
            max_score=10,
            time_spent=120.5,
            completed_at=now,
        )
        assert result.part_a_score == 5
        assert result.part_b_score == 3
        assert result.total_score == 8
        assert result.max_score == 10
        assert result.time_spent == 120.5
