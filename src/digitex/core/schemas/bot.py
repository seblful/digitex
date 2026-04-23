"""Bot schemas — typed return values from repositories."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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


class Session(BaseModel):
    session_id: int
    student_id: int
    book_id: int
    option_number: int
    started_at: datetime
    completed_at: datetime | None = None


class TestResult(BaseModel):
    session_id: int
    part_a_score: int
    part_b_score: int
    total_score: int
    max_score: int
    time_spent: float = Field(description="Total time in seconds")
    completed_at: datetime
