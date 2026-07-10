"""Tests for core schemas and value objects."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from digitex.core.answer import check_answer
from digitex.core.domain import (
    ExamType,
    Question,
    QuestionKey,
    Session,
    Student,
    TestResult,
    exam_type_for,
    year_has_exam_types,
)


class TestStudent:
    """Test Student schema."""

    def test_valid_student(self) -> None:
        student = Student(student_id=1, telegram_id=12345, name="John")
        assert student.student_id == 1
        assert student.telegram_id == 12345
        assert student.name == "John"
        assert student.username is None

    def test_student_with_username(self) -> None:
        student = Student(
            student_id=2, telegram_id=67890, name="Jane", username="jane_doe"
        )
        assert student.username == "jane_doe"

    def test_student_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Student()  # type: ignore
        with pytest.raises(ValidationError):
            Student(student_id=1, name="John")  # type: ignore
        with pytest.raises(ValidationError):
            Student(student_id=1, telegram_id=12345)  # type: ignore


class TestQuestion:
    """Test Question schema."""

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
        assert q.num_options == 5

    def test_question_with_optional_fields(self) -> None:
        q = Question(
            question_id=2,
            part="B",
            question_number=10,
            image_data=b"more_bytes",
            telegram_file_id="AgAC...",
            num_options=4,
        )
        assert q.telegram_file_id == "AgAC..."
        assert q.num_options == 4

    def test_question_invalid_part(self) -> None:
        with pytest.raises(ValidationError):
            Question(
                question_id=1,
                part="C",  # type: ignore
                question_number=1,
                image_data=b"bytes",
            )

    def test_question_negative_number(self) -> None:
        q = Question(
            question_id=1,
            part="A",
            question_number=-1,
            image_data=b"bytes",
        )
        assert q.question_number == -1


class TestSession:
    """Test Session schema."""

    def test_valid_session(self) -> None:
        now = datetime.now(UTC)
        session = Session(
            session_id=1,
            student_id=42,
            option_id=3,
            started_at=now,
        )
        assert session.session_id == 1
        assert session.student_id == 42
        assert session.option_id == 3
        assert session.started_at == now
        assert session.completed_at is None

    def test_session_with_completed_at(self) -> None:
        now = datetime.now(UTC)
        later = datetime.now(UTC)
        session = Session(
            session_id=2,
            student_id=99,
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
    """Test TestResult schema."""

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
    """Test QuestionKey value object."""

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

    def test_question_key_parse_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid question key"):
            QuestionKey.parse("")
        with pytest.raises(ValueError, match="Invalid question key"):
            QuestionKey.parse("C1")
        with pytest.raises(ValueError, match="Invalid question key"):
            QuestionKey.parse("A")


class TestExamType:
    """Test ExamType literal."""

    def test_exam_type_values(self) -> None:
        valid: ExamType = "CE"
        assert valid == "CE"
        valid = "CT"
        assert valid == "CT"

    def test_exam_type_usage(self) -> None:
        now = datetime.now(UTC)
        result = TestResult(
            session_id=1,
            exam_type="CT",
            part_a_score=5,
            part_b_score=5,
            total_score=10,
            max_score=20,
            time_spent=300.0,
            completed_at=now,
        )
        assert result.exam_type in ("CE", "CT")


class TestExamTypeRule:
    """Test the CE/CT exam-type domain rule."""

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


class TestCheckAnswer:
    """Test check_answer pure function."""

    def test_part_a_correct(self) -> None:
        assert check_answer("A", "3", 3) is True

    def test_part_a_wrong(self) -> None:
        assert check_answer("A", "2", 3) is False

    def test_part_a_strips_whitespace(self) -> None:
        assert check_answer("A", " 3 ", 3) is True

    def test_part_b_single_correct(self) -> None:
        assert check_answer("B", "ANS", "ANS") is True

    def test_part_b_multi_correct_first(self) -> None:
        assert check_answer("B", "ANS1", "ANS1/ANS2") is True

    def test_part_b_multi_correct_second(self) -> None:
        assert check_answer("B", "ANS2", "ANS1/ANS2") is True

    def test_part_b_wrong(self) -> None:
        assert check_answer("B", "WRONG", "ANS1/ANS2") is False

    def test_part_b_strips_whitespace(self) -> None:
        assert check_answer("B", " ANS1 ", "ANS1 / ANS2") is True
