"""Schemas for centralized testing entities."""

from pathlib import Path

from pydantic import BaseModel, Field


class QuestionA(BaseModel):
    """Question for Part A with multiple choice answer (A-E, mapped to 1-5)."""

    number: int
    image: Path
    answer: int = Field(ge=1, le=5)


class QuestionB(BaseModel):
    """Question for Part B with single text/numeric answer."""

    number: int
    image: Path
    answer: str


class PartA(BaseModel):
    """Part A containing multiple choice questions."""

    questions: list[QuestionA]


class PartB(BaseModel):
    """Part B containing questions with single answers."""

    questions: list[QuestionB]


class Option(BaseModel):
    """Test option (варыант) containing Part A and Part B."""

    option_number: int
    part_a: PartA
    part_b: PartB


class Book(BaseModel):
    """Collection of test options for a subject and year (зборнік)."""

    id: int
    subject: str
    year: int
    options: list[Option]
