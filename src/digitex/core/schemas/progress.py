"""Schemas for student progress tracking."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Student(BaseModel):
    """Student entity."""

    id: int
    telegram_id: int
    name: str
    username: str | None = None


class QuestionPart(str, Enum):
    """Question part identifier."""

    A = "A"
    B = "B"


class QuestionRef(BaseModel):
    """Reference to a specific question in a test."""

    book_id: int
    option_number: int
    part: QuestionPart
    question_number: int


class AnswerRecord(BaseModel):
    """Record of a student's answer to a question."""

    question_ref: QuestionRef
    student_answer: str | int
    is_correct: bool
    time_spent: float = Field(description="Time in seconds")
    timestamp: datetime


class TestResult(BaseModel):
    """Result of a completed test option."""

    book_id: int
    option_number: int
    part_a_score: int
    part_b_score: int
    total_score: int
    max_score: int
    time_spent: float = Field(description="Time in seconds")
    completed_at: datetime
    answers: list[AnswerRecord]


class SubjectProgress(BaseModel):
    """Progress tracking for a single subject."""

    subject: str
    tests_completed: int
    total_tests: int
    average_score: float
    total_time_spent: float = Field(description="Time in seconds")
    results: list[TestResult]


class StudentProgress(BaseModel):
    """Overall progress tracking for a student."""

    student: Student
    subjects: dict[str, SubjectProgress]
    total_tests_completed: int
    total_time_spent: float = Field(description="Time in seconds")
