"""Tests for bot message format strings."""

import pytest

from digitex.bot.messages import (
    MSG_CORRECT_ANSWER,
    MSG_GREETING,
    MSG_NO_OPTIONS,
    MSG_RESULTS_ERROR_ITEM,
    MSG_RESULTS_OPTION,
    MSG_RESULTS_PART_A,
    MSG_RESULTS_PART_B,
    MSG_RESULTS_SCORE,
    MSG_RESULTS_SUBJECT,
    MSG_RESULTS_TIME,
    MSG_RESULTS_TYPE,
    MSG_RESULTS_YEAR,
    MSG_WRONG_ANSWER,
)


class TestMessageFormatting:
    def test_greeting_formats(self) -> None:
        assert MSG_GREETING.format(name="User") == "Здравствуйте, User! Выберите предмет:"

    def test_wrong_answer_formats(self) -> None:
        result = MSG_WRONG_ANSWER.format(correct_answer="42")
        assert "42" in result

    def test_no_options_formats(self) -> None:
        result = MSG_NO_OPTIONS.format(exam_type="ЦЭ")
        assert "ЦЭ" in result

    def test_results_formats(self) -> None:
        assert "Биология" in MSG_RESULTS_SUBJECT.format(subject_name="Биология")
        assert "2024" in MSG_RESULTS_YEAR.format(year=2024)
        assert "ЦЭ" in MSG_RESULTS_TYPE.format(exam_type="ЦЭ")
        assert "1" in MSG_RESULTS_OPTION.format(option_number=1)
        assert "10" in MSG_RESULTS_SCORE.format(total_score=10, max_score=15)
        assert "6" in MSG_RESULTS_PART_A.format(part_a_score=6)
        assert "4" in MSG_RESULTS_PART_B.format(part_b_score=4)
        assert "120" in MSG_RESULTS_TIME.format(time_spent=120.0)

    def test_error_item_formats(self) -> None:
        result = MSG_RESULTS_ERROR_ITEM.format(qnum=5, user_ans="3", correct_ans="4")
        assert "5" in result
        assert "3" in result
        assert "4" in result

    def test_correct_answer_keywords(self) -> None:
        msg = MSG_CORRECT_ANSWER
        assert len(msg) > 0


def test_all_messages_are_strings() -> None:
    """Verify all messages are non-empty strings."""
    import digitex.bot.messages as mod

    for name in dir(mod):
        if name.startswith("MSG_"):
            value = getattr(mod, name)
            assert isinstance(value, str), f"{name} is not a string"
            assert len(value) > 0, f"{name} is empty"


def test_format_strings_have_consistent_keys() -> None:
    """Verify format strings use valid keyword-only fields."""
    import re

    import digitex.bot.messages as mod

    for name in dir(mod):
        if name.startswith("MSG_"):
            value = getattr(mod, name)
            fields = re.findall(r"\{(\w+)\}", value)
            for field in fields:
                assert field.isidentifier(), f"{name}: invalid field {field!r}"
