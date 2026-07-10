"""Tests for the pure answer-checking logic in ``digitex.core.answer``."""

import pytest

from digitex.core.answer import check_answer
from digitex.core.domain import Part


@pytest.mark.parametrize(
    ("part", "student", "correct", "expected"),
    [
        ("A", "3", 3, True),
        ("A", "2", 3, False),
        ("A", " 3 ", 3, True),
        ("B", "ANS", "ANS", True),
        ("B", "ANS1", "ANS1/ANS2", True),
        ("B", "ANS2", "ANS1/ANS2", True),
        ("B", "WRONG", "ANS1/ANS2", False),
        ("B", " ANS1 ", "ANS1 / ANS2", True),
    ],
    ids=[
        "part-a-correct",
        "part-a-wrong",
        "part-a-strips-whitespace",
        "part-b-single-correct",
        "part-b-multi-first-alternative",
        "part-b-multi-second-alternative",
        "part-b-wrong",
        "part-b-strips-whitespace-around-alternatives",
    ],
)
def test_check_answer(
    part: Part, student: str, correct: int | str, expected: bool
) -> None:
    assert check_answer(part, student, correct) is expected
