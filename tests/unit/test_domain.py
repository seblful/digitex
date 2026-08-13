"""Tests for the domain models and value objects in ``digitex.core.domain``."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from digitex.core.domain import (
    Question,
    QuestionKey,
    Session,
    Student,
    TestResult,
    exam_type_for,
    year_has_exam_types,
)


class TestStudent:
    def test_a_student_who_has_not_applied_yet(self) -> None:
        student = Student(
            telegram_id=12345,
            telegram_name="John",
            status="pending",
            created_at=datetime.now(UTC),
        )
        assert student.telegram_id == 12345
        assert student.telegram_name == "John"
        assert student.telegram_username is None
        assert student.full_name is None
        assert student.handled_at is None
        assert student.handled_by is None

    def test_a_student_carries_their_application_and_its_decision(self) -> None:
        handled_at = datetime.now(UTC)
        student = Student(
            telegram_id=67890,
            telegram_name="Jane",
            telegram_username="jane_doe",
            full_name="Jane Doe",
            status="approved",
            created_at=datetime.now(UTC),
            handled_at=handled_at,
            handled_by=1,
        )
        assert student.telegram_username == "jane_doe"
        assert student.full_name == "Jane Doe"
        assert student.status == "approved"
        assert student.handled_at == handled_at
        assert student.handled_by == 1

    def test_student_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Student()  # type: ignore
        with pytest.raises(ValidationError):
            Student(telegram_id=1, telegram_name="John")  # type: ignore

    def test_student_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            Student(
                telegram_id=1,
                telegram_name="John",
                status="maybe",  # type: ignore
                created_at=datetime.now(UTC),
            )


class TestQuestion:
    def test_valid_question(self) -> None:
        q = Question(
            question_id=1,
            part="A",
            question_number=5,
            image_data=b"fake_image_bytes",
        )
        assert q.question_id == 1
        assert q.part == "A"
        assert q.question_number == 5
        assert q.image_data == b"fake_image_bytes"
        assert q.telegram_file_id is None

    def test_question_with_optional_fields(self) -> None:
        q = Question(
            question_id=2,
            part="B",
            question_number=10,
            image_data=b"more_bytes",
            telegram_file_id="AgAC...",
        )
        assert q.telegram_file_id == "AgAC..."

    def test_question_invalid_part(self) -> None:
        with pytest.raises(ValidationError):
            Question(
                question_id=1,
                part="C",  # type: ignore
                question_number=1,
                image_data=b"bytes",
            )


class TestSession:
    def test_valid_session(self) -> None:
        now = datetime.now(UTC)
        session = Session(
            session_id=1,
            student_telegram_id=42,
            option_id=3,
            started_at=now,
        )
        assert session.session_id == 1
        assert session.student_telegram_id == 42
        assert session.option_id == 3
        assert session.started_at == now
        assert session.completed_at is None

    def test_session_with_completed_at(self) -> None:
        now = datetime.now(UTC)
        later = datetime.now(UTC)
        session = Session(
            session_id=2,
            student_telegram_id=99,
            option_id=1,
            started_at=now,
            completed_at=later,
        )
        assert session.completed_at == later

    def test_session_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Session()  # type: ignore
        with pytest.raises(ValidationError):
            Session(session_id=1)  # type: ignore


class TestTestResult:
    def test_valid_test_result(self) -> None:
        now = datetime.now(UTC)
        result = TestResult(
            session_id=1,
            part_a_score=8,
            part_b_score=7,
            total_score=15,
            max_score=20,
            time_spent=1200.0,
            completed_at=now,
        )
        assert result.session_id == 1
        assert result.exam_type == "CT"
        assert result.part_a_score == 8
        assert result.part_b_score == 7
        assert result.total_score == 15
        assert result.max_score == 20
        assert result.time_spent == 1200.0
        assert result.completed_at == now

    def test_test_result_custom_exam_type(self) -> None:
        now = datetime.now(UTC)
        result = TestResult(
            session_id=2,
            exam_type="CE",
            part_a_score=10,
            part_b_score=10,
            total_score=20,
            max_score=20,
            time_spent=600.0,
            completed_at=now,
        )
        assert result.exam_type == "CE"

    def test_test_result_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            TestResult(session_id=1)  # type: ignore


class TestQuestionKey:
    def test_valid_question_key(self) -> None:
        key = QuestionKey(part="A", number=1)
        assert key.part == "A"
        assert key.number == 1

    def test_question_key_string_representation(self) -> None:
        assert str(QuestionKey(part="A", number=1)) == "A1"
        assert str(QuestionKey(part="B", number=12)) == "B12"

    def test_question_key_parse(self) -> None:
        key = QuestionKey.parse("B5")
        assert key.part == "B"
        assert key.number == 5

        key = QuestionKey.parse(" a3 ")
        assert key.part == "A"
        assert key.number == 3

    @pytest.mark.parametrize(
        ("raw", "part"),
        [("А1", "A"), ("В2", "B"), ("а3", "A")],
        ids=["cyrillic-a", "cyrillic-ve", "lowercase-cyrillic-a"],
    )
    def test_parse_accepts_cyrillic_part_letters(self, raw: str, part: str) -> None:
        """Hand-typed keys in a Russian corpus carry Cyrillic А/В."""
        assert QuestionKey.parse(raw).part == part

    @pytest.mark.parametrize(
        "raw", ["", "C1", "A"], ids=["empty", "bad-part", "no-number"]
    )
    def test_question_key_parse_invalid(self, raw: str) -> None:
        with pytest.raises(ValueError, match="Invalid question key"):
            QuestionKey.parse(raw)


class TestExamTypeRule:
    """The CE/CT exam-type domain rule."""

    def test_years_before_2023_are_ct_only(self) -> None:
        assert year_has_exam_types(2022) is False
        assert exam_type_for(2022, 1) == "CT"
        assert exam_type_for(2022, 10) == "CT"

    def test_years_from_2023_split_into_variants(self) -> None:
        assert year_has_exam_types(2023) is True
        assert year_has_exam_types(2026) is True

    def test_options_one_to_five_are_ce(self) -> None:
        assert exam_type_for(2023, 1) == "CE"
        assert exam_type_for(2023, 5) == "CE"

    def test_options_above_five_are_ct(self) -> None:
        assert exam_type_for(2023, 6) == "CT"
        assert exam_type_for(2023, 10) == "CT"
