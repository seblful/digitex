"""Core functionality."""

from .ocr import TextExtractor
from .schemas import (
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

__all__ = [
    "AnswerRecord",
    "TextExtractor",
    "Book",
    "Option",
    "PartA",
    "PartB",
    "QuestionA",
    "QuestionB",
    "QuestionPart",
    "QuestionRef",
    "Student",
    "StudentProgress",
    "SubjectProgress",
    "TestResult",
]
