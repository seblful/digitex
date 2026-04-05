"""Tests for core schemas module."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from digitex.core.schemas import (
    AnswerRecord,
    Book,
    Option,
    PartA,
    PartB,
    QuestionA,
    QuestionB,
    QuestionPart,
    QuestionRef,
    Student,
    StudentProgress,
    SubjectProgress,
    TestResult,
)


class TestQuestionA:
    """Test QuestionA schema."""

    def test_valid_question_a(self) -> None:
        """Test creating a valid QuestionA."""
        q = QuestionA(number=1, image=Path("q1.png"), answer=3)
        assert q.number == 1
        assert q.image == Path("q1.png")
        assert q.answer == 3

    def test_answer_must_be_1_to_5(self) -> None:
        """Test that answer must be between 1 and 5."""
        with pytest.raises(ValidationError):
            QuestionA(number=1, image=Path("q1.png"), answer=0)
        with pytest.raises(ValidationError):
            QuestionA(number=1, image=Path("q1.png"), answer=6)


class TestQuestionB:
    """Test QuestionB schema."""

    def test_valid_question_b(self) -> None:
        """Test creating a valid QuestionB."""
        q = QuestionB(number=1, image=Path("q1.png"), answer="42")
        assert q.number == 1
        assert q.image == Path("q1.png")
        assert q.answer == "42"


class TestPartA:
    """Test PartA schema."""

    def test_valid_part_a(self) -> None:
        """Test creating a valid PartA."""
        questions = [
            QuestionA(number=1, image=Path("q1.png"), answer=1),
            QuestionA(number=2, image=Path("q2.png"), answer=3),
        ]
        part = PartA(questions=questions)
        assert len(part.questions) == 2


class TestPartB:
    """Test PartB schema."""

    def test_valid_part_b(self) -> None:
        """Test creating a valid PartB."""
        questions = [
            QuestionB(number=1, image=Path("q1.png"), answer="10"),
            QuestionB(number=2, image=Path("q2.png"), answer="20"),
        ]
        part = PartB(questions=questions)
        assert len(part.questions) == 2


class TestOption:
    """Test Option schema."""

    def test_valid_option(self) -> None:
        """Test creating a valid Option."""
        option = Option(
            option_number=1,
            part_a=PartA(
                questions=[QuestionA(number=1, image=Path("q1.png"), answer=1)]
            ),
            part_b=PartB(
                questions=[QuestionB(number=1, image=Path("q1.png"), answer="10")]
            ),
        )
        assert option.option_number == 1
        assert len(option.part_a.questions) == 1
        assert len(option.part_b.questions) == 1


class TestBook:
    """Test Book schema."""

    def test_valid_book(self) -> None:
        """Test creating a valid Book."""
        book = Book(
            id=1,
            subject="math",
            year=2024,
            options=[
                Option(
                    option_number=1,
                    part_a=PartA(
                        questions=[QuestionA(number=1, image=Path("q1.png"), answer=1)]
                    ),
                    part_b=PartB(
                        questions=[
                            QuestionB(number=1, image=Path("q1.png"), answer="10")
                        ]
                    ),
                )
            ],
        )
        assert book.id == 1
        assert book.subject == "math"
        assert book.year == 2024
        assert len(book.options) == 1


class TestStudent:
    """Test Student schema."""

    def test_valid_student(self) -> None:
        """Test creating a valid Student."""
        student = Student(id=1, name="John")
        assert student.id == 1
        assert student.name == "John"


class TestQuestionPart:
    """Test QuestionPart enum."""

    def test_question_part_values(self) -> None:
        """Test QuestionPart enum values."""
        assert QuestionPart.A.value == "A"
        assert QuestionPart.B.value == "B"


class TestQuestionRef:
    """Test QuestionRef schema."""

    def test_valid_question_ref(self) -> None:
        """Test creating a valid QuestionRef."""
        ref = QuestionRef(
            book_id=1,
            option_number=2,
            part=QuestionPart.A,
            question_number=5,
        )
        assert ref.book_id == 1
        assert ref.option_number == 2
        assert ref.part == QuestionPart.A
        assert ref.question_number == 5


class TestAnswerRecord:
    """Test AnswerRecord schema."""

    def test_valid_answer_record(self) -> None:
        """Test creating a valid AnswerRecord."""
        record = AnswerRecord(
            question_ref=QuestionRef(
                book_id=1, option_number=1, part=QuestionPart.A, question_number=1
            ),
            student_answer=3,
            correct_answer=3,
            is_correct=True,
            time_spent=10.5,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert record.is_correct is True
        assert record.time_spent == 10.5


class TestTestResult:
    """Test TestResult schema."""

    def test_valid_test_result(self) -> None:
        """Test creating a valid TestResult."""
        result = TestResult(
            book_id=1,
            option_number=1,
            part_a_score=8,
            part_b_score=7,
            total_score=15,
            max_score=20,
            time_spent=1200.0,
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            answers=[],
        )
        assert result.total_score == 15
        assert result.max_score == 20


class TestSubjectProgress:
    """Test SubjectProgress schema."""

    def test_valid_subject_progress(self) -> None:
        """Test creating a valid SubjectProgress."""
        progress = SubjectProgress(
            subject="math",
            tests_completed=5,
            total_tests=10,
            average_score=75.5,
            total_time_spent=3600.0,
            results=[],
        )
        assert progress.subject == "math"
        assert progress.average_score == 75.5


class TestStudentProgress:
    """Test StudentProgress schema."""

    def test_valid_student_progress(self) -> None:
        """Test creating a valid StudentProgress."""
        progress = StudentProgress(
            student=Student(id=1, name="John"),
            subjects={},
            total_tests_completed=10,
            total_time_spent=7200.0,
        )
        assert progress.student.name == "John"
        assert progress.total_tests_completed == 10
