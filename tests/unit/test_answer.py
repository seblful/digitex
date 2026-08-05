"""Tests for the pure answer-checking logic in ``digitex.core.answer``."""

import pytest

from digitex.bot.keyboards import part_a_kb
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


class TestPlaceholderAnswerIsUnmatchable:
    """``scripts/populate_db.py`` stores a placeholder when no answer key exists.

    It is only safe if nothing a Student can send matches it. The old Part A
    placeholder was ``1``, which told anyone who tapped option 1 they were right
    and scored it into their Session.
    """

    @pytest.mark.parametrize("option", range(1, 11))
    def test_no_selectable_part_a_option_matches_zero(self, option: int) -> None:
        assert not check_answer("A", str(option), 0)

    @pytest.mark.parametrize("num_options", range(1, 9))
    def test_the_option_keyboard_never_offers_zero(self, num_options: int) -> None:
        keyboard = part_a_kb(num_options)

        labels = [button.text for row in keyboard.inline_keyboard for button in row]

        assert labels == [str(n) for n in range(1, num_options + 1)]

    @pytest.mark.parametrize(
        "reply",
        ["ВЕРНАДСКИЙ", "3", " ", ""],
        ids=["text", "digit", "whitespace-only", "empty"],
    )
    def test_no_part_b_reply_matches_the_empty_placeholder(self, reply: str) -> None:
        assert not check_answer("B", reply, "")
