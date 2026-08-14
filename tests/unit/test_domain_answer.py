"""Tests for the answer key's matching rules in ``digitex.domain.answer``."""

import pytest

from digitex.bot.keyboards import part_a_kb
from digitex.domain.answer import AnswerKey
from digitex.domain.entities import Part


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
def test_matches(part: Part, student: str, correct: int | str, expected: bool) -> None:
    assert AnswerKey(part=part, value=correct).matches(student) is expected


class TestStoredForm:
    """``stored`` is the one string form the record and the screen share."""

    def test_a_part_a_key_stores_its_option_index_as_text(self) -> None:
        assert AnswerKey(part="A", value=3).stored == "3"

    def test_a_part_b_key_stores_its_text_unchanged(self) -> None:
        assert AnswerKey(part="B", value="ANS1/ANS2").stored == "ANS1/ANS2"

    def test_a_missing_key_stores_nothing(self) -> None:
        assert AnswerKey(part="B", value=None).stored is None


class TestAMissingAnswerKeyMatchesNothing:
    """A question whose year shipped no answer key stores None for it.

    ``digitex-db populate`` loads such a question anyway, so its image is
    servable — which is only safe if nothing a Student can send scores right
    against it, in either part.
    """

    @pytest.mark.parametrize("option", range(1, 11))
    def test_no_part_a_option_matches_a_missing_key(self, option: int) -> None:
        assert not AnswerKey(part="A", value=None).matches(str(option))

    @pytest.mark.parametrize(
        "reply",
        ["ВЕРНАДСКИЙ", "3", " ", ""],
        ids=["text", "digit", "whitespace-only", "empty"],
    )
    def test_no_part_b_reply_matches_a_missing_key(self, reply: str) -> None:
        assert not AnswerKey(part="B", value=None).matches(reply)

    @pytest.mark.parametrize("reply", ["", "x", " "], ids=["empty", "text", "space"])
    def test_a_blank_part_b_key_matches_nothing_either(self, reply: str) -> None:
        """The same holds for a key that is stored but carries no value."""
        assert not AnswerKey(part="B", value="  ").matches(reply)

    @pytest.mark.parametrize("num_options", range(1, 9))
    def test_the_option_keyboard_offers_only_real_options(
        self, num_options: int
    ) -> None:
        keyboard = part_a_kb(num_options)

        labels = [button.text for row in keyboard.inline_keyboard for button in row]

        assert labels == [str(n) for n in range(1, num_options + 1)]
