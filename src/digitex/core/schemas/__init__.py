"""Core domain schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from digitex.core.value_objects import ExamType

__all__ = ["Question", "Session", "Student", "TestResult"]


class Student(BaseModel):
    student_id: int
    telegram_id: int
    name: str
    username: str | None = None


class Question(BaseModel):
    question_id: int
    part: Literal["A", "B"]
    question_number: int
    image_data: bytes
    telegram_file_id: str | None = None
    num_options: int = 5


class Session(BaseModel):
    session_id: int
    student_id: int
    option_id: int
    started_at: datetime
    completed_at: datetime | None = None


class TestResult(BaseModel):
    session_id: int
    exam_type: ExamType = "CT"
    part_a_score: int
    part_b_score: int
    total_score: int
    max_score: int
    time_spent: float = Field(description="Total time in seconds")
    completed_at: datetime
