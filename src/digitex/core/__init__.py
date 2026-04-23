"""Core functionality."""

from .ocr import TextExtractor
from .schemas import Question, Session, Student, TestResult

__all__ = ["TextExtractor", "Question", "Session", "Student", "TestResult"]
