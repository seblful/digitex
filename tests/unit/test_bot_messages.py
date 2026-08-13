"""Tests for the user-facing message strings.

Every one of these is a format string rendered in front of a student, so a
typo in a placeholder is a crash in a handler rather than a wrong word. The
sweep over all of them is the point: it covers constants added after this file
was written, which a hand-listed set never would.

The individual cases below cover the ones whose placeholders carry a value the
student reads off the screen — a score, an answer, a name.
"""

from __future__ import annotations

import string

import pytest

from digitex.bot import messages
from digitex.bot.messages import (
    EXAM_LABELS,
    MSG_GREETING,
    MSG_NO_OPTIONS,
    MSG_RESULTS_ERROR_ITEM,
    MSG_RESULTS_SCORE,
    MSG_RESULTS_TIME,
    MSG_WRONG_ANSWER,
)

MESSAGE_NAMES = sorted(name for name in vars(messages) if name.startswith("MSG_"))


def _fields(template: str) -> list[str]:
    """The placeholder names in *template*, in order."""
    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


class TestEveryMessage:
    def test_the_sweep_found_the_messages(self) -> None:
        """Guard the guard: a rename would otherwise empty this whole class."""
        assert len(MESSAGE_NAMES) > 40
        assert "MSG_GREETING" in MESSAGE_NAMES

    @pytest.mark.parametrize("name", MESSAGE_NAMES)
    def test_is_a_non_empty_string(self, name: str) -> None:
        value = getattr(messages, name)

        assert isinstance(value, str)
        assert value.strip()

    @pytest.mark.parametrize("name", MESSAGE_NAMES)
    def test_its_placeholders_are_named_and_balanced(self, name: str) -> None:
        """Callers all use keyword ``.format``, so a positional slot never fills.

        ``Formatter().parse`` is also what raises on an unclosed brace, which is
        the typo that would otherwise reach a student as a ValueError.
        """
        for field in _fields(getattr(messages, name)):
            # "{time_spent:.0f}" parses the format spec off separately, but a
            # nested lookup like "{a.b}" would arrive whole.
            assert field.isidentifier(), f"{name}: unusable placeholder {field!r}"


class TestResultLines:
    @pytest.mark.parametrize(
        ("template", "values", "expected"),
        [
            (MSG_GREETING, {"name": "Аня"}, "Аня"),
            (MSG_WRONG_ANSWER, {"correct_answer": "42"}, "42"),
            (MSG_NO_OPTIONS, {"exam_type": "ЦЭ"}, "ЦЭ"),
            (MSG_RESULTS_SCORE, {"total_score": 10, "max_score": 15}, "10 из 15"),
        ],
        ids=["greeting", "wrong-answer", "no-options", "score"],
    )
    def test_the_value_reaches_the_rendered_line(
        self, template: str, values: dict[str, object], expected: str
    ) -> None:
        assert expected in template.format(**values)

    def test_a_duration_is_rendered_without_decimals(self) -> None:
        """``{time_spent:.0f}`` — a student reads seconds, not float noise."""
        rendered = MSG_RESULTS_TIME.format(time_spent=120.4)

        assert "120" in rendered
        assert "120.4" not in rendered

    def test_an_error_line_shows_both_answers(self) -> None:
        rendered = MSG_RESULTS_ERROR_ITEM.format(qnum=5, user_ans="3", correct_ans="4")

        assert "5" in rendered
        assert "3" in rendered
        assert "4" in rendered


class TestExamLabels:
    def test_both_exam_types_have_a_label_to_show(self) -> None:
        assert set(EXAM_LABELS) == {"CE", "CT"}
        assert all(label.strip() for label in EXAM_LABELS.values())
