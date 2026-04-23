"""Schemas for student progress tracking."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Student(BaseModel):
    id: int
    telegram_id: int
    name: str
    username: str | None = None


class QuestionRef(BaseModel):
    """Identifies a question by its natural key matching the DB unique constraint."""

    option_id: int
    part: Literal["A", "B"]
    question_number: int


class AnswerRecord(BaseModel):
    question_ref: QuestionRef
    student_answer: str
    is_correct: bool
    time_spent: float = Field(description="Time in seconds")
    timestamp: datetime


class TestResult(BaseModel):
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
    subject: str
    tests_completed: int
    total_tests: int
    average_score: float
    total_time_spent: float = Field(description="Time in seconds")
    results: list[TestResult]


class StudentProgress(BaseModel):
    student: Student
    subjects: dict[str, SubjectProgress]
    total_tests_completed: int
    total_time_spent: float = Field(description="Time in seconds")
