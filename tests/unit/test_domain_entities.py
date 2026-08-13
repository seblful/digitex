"""Tests for the domain types in ``digitex.domain.entities``.

What is worth pinning here is the domain's own rules: which values a field is
allowed to take, what a record looks like before anything has happened to it,
and the CE/CT rule that decides which exam an option belongs to. Field
round-trips through Pydantic are not — those check the library, not the domain.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from digitex.domain.entities import (
    OPTIONS_PER_BOOK,
    Question,
    QuestionKey,
    Student,
    TestResult,
    exam_type_for,
    normalize_option_number,
    parse_exam_type,
    year_has_exam_types,
)


class TestStudent:
    def test_a_student_who_has_not_applied_yet_carries_no_application(self) -> None:
        """``full_name`` is what they typed when applying, so it is None until then.

        The schema mirrors this as a CHECK: a decision without an application
        is a state the database refuses too.
        """
        student = Student(
            telegram_id=12345,
            telegram_name="John",
            status="pending",
            created_at=datetime.now(UTC),
        )

        assert student.full_name is None
        assert student.telegram_username is None
        assert student.handled_at is None
        assert student.handled_by is None

    @pytest.mark.parametrize(
        "status", ["maybe", "APPROVED", ""], ids=["unknown", "wrong-case", "empty"]
    )
    def test_only_the_three_registration_states_are_accepted(self, status: str) -> None:
        with pytest.raises(ValidationError):
            Student(
                telegram_id=1,
                telegram_name="John",
                status=status,  # ty: ignore[invalid-argument-type]
                created_at=datetime.now(UTC),
            )


class TestQuestion:
    @pytest.mark.parametrize(
        "part", ["C", "a", ""], ids=["unknown", "lowercase", "empty"]
    )
    def test_a_question_belongs_to_part_a_or_part_b(self, part: str) -> None:
        with pytest.raises(ValidationError):
            Question(
                question_id=1,
                part=part,  # ty: ignore[invalid-argument-type]
                question_number=1,
                image_key="biology/2016/1/A/1.jpg",
            )

    def test_a_question_starts_with_nothing_cached(self) -> None:
        """The file_id is filled in by the first upload, not by the corpus."""
        question = Question(question_id=1, part="A", question_number=5)

        assert question.telegram_file_id is None
        assert question.image_key is None


class TestTestResult:
    def test_a_result_is_a_ct_result_unless_it_says_otherwise(self) -> None:
        """Most of the corpus predates the CE/CT split, so CT is the default."""
        result = TestResult(
            session_id=1,
            part_a_score=8,
            part_b_score=7,
            total_score=15,
            max_score=20,
            time_spent=1200.0,
            completed_at=datetime.now(UTC),
        )

        assert result.exam_type == "CT"


class TestQuestionKey:
    def test_a_key_renders_as_it_is_written_in_answers_json(self) -> None:
        assert str(QuestionKey(part="A", number=1)) == "A1"
        assert str(QuestionKey(part="B", number=12)) == "B12"

    @pytest.mark.parametrize(
        "raw", ["B5", " b5 ", "b5"], ids=["plain", "padded", "lowercase"]
    )
    def test_parsing_tolerates_case_and_padding(self, raw: str) -> None:
        assert QuestionKey.parse(raw) == QuestionKey(part="B", number=5)

    @pytest.mark.parametrize(
        ("raw", "part"),
        [("А1", "A"), ("В2", "B"), ("а3", "A")],
        ids=["cyrillic-a", "cyrillic-ve", "lowercase-cyrillic-a"],
    )
    def test_parsing_accepts_cyrillic_part_letters(self, raw: str, part: str) -> None:
        """Hand-typed keys in a Russian corpus carry Cyrillic А/В.

        On screen they are indistinguishable from Latin A/B, so without the
        fold a hand-corrected answers.json rejects every key in it.
        """
        assert QuestionKey.parse(raw).part == part

    @pytest.mark.parametrize(
        "raw",
        ["", "C1", "A", "1A", "A1B"],
        ids=["empty", "bad-part", "no-number", "reversed", "trailing-letter"],
    )
    def test_anything_but_part_plus_number_is_refused(self, raw: str) -> None:
        with pytest.raises(ValueError, match="Invalid question key"):
            QuestionKey.parse(raw)


class TestExamTypeRule:
    """Which exam an option belongs to — the CE/CT split introduced in 2023."""

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


class TestParseExamType:
    @pytest.mark.parametrize("raw", ["CE", "CT"])
    def test_the_two_exam_types_narrow_to_themselves(self, raw: str) -> None:
        assert parse_exam_type(raw) == raw

    @pytest.mark.parametrize(
        "raw", ["ce", "ЦТ", "", "CX"], ids=["lowercase", "cyrillic", "empty", "unknown"]
    )
    def test_anything_else_is_refused(self, raw: str) -> None:
        """The narrowing is what lets callers treat the result as an ExamType."""
        with pytest.raises(ValueError, match="Unknown exam type"):
            parse_exam_type(raw)


class TestNormalizeOptionNumber:
    @pytest.mark.parametrize(
        ("raw", "folded"),
        [(1, 1), (10, 10), (11, 1), (20, 10), (31, 1), (40, 10)],
        ids=["1", "10", "11-to-1", "20-to-10", "31-to-1", "40-to-10"],
    )
    def test_every_block_of_ten_is_the_same_ten_options(
        self, raw: int, folded: int
    ) -> None:
        """Sheets number options 1-10, then 11-20 — 11 and 21 both mean Option 1."""
        assert normalize_option_number(raw) == folded

    def test_the_fold_never_leaves_the_books_option_range(self) -> None:
        assert {normalize_option_number(n) for n in range(1, 101)} == set(
            range(1, OPTIONS_PER_BOOK + 1)
        )
