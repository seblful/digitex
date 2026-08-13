"""Tests for the inline keyboard builders.

A button's ``callback_data`` is what the handler decodes when it is tapped, so
what each of these checks is the round-trip: the payload the builder packs
unpacks back through the same typed factory the handler filters on. Asserting
the packed string literally would pin the wire format in two places and fail
for a prefix rename that broke nothing.

The empty cases matter because a subject with no years, or a year with no
options, is ordinary — those keyboards must come back empty rather than raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.bot.callbacks import (
    AnswerCB,
    ExamTypeCB,
    ModeCB,
    OptionCB,
    RandomFeedbackCB,
    RandomPartCB,
    RegistrationCB,
    SubjectCB,
    TopicCB,
    YearCB,
)
from digitex.bot.keyboards import (
    COLUMNS_YEARS,
    admin_registration_kb,
    exam_type_kb,
    mode_kb,
    options_kb,
    part_a_kb,
    random_feedback_kb,
    random_part_kb,
    subjects_kb,
    topics_kb,
    years_kb,
)
from digitex.domain.entities import PART_A_OPTION_COUNT

if TYPE_CHECKING:
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def _buttons(markup: InlineKeyboardMarkup) -> list[InlineKeyboardButton]:
    """Every button, flattened out of its row layout."""
    return [button for row in markup.inline_keyboard for button in row]


def _payloads(markup: InlineKeyboardMarkup) -> list[str]:
    payloads: list[str] = []
    for button in _buttons(markup):
        assert button.callback_data is not None, f"{button.text} carries no payload"
        payloads.append(button.callback_data)
    return payloads


class TestSubjects:
    def test_each_button_shows_a_subject_and_carries_its_id(self) -> None:
        markup = subjects_kb([(1, "Биология"), (7, "Математика")])

        assert [b.text for b in _buttons(markup)] == ["Биология", "Математика"]
        assert [SubjectCB.unpack(p).subject_id for p in _payloads(markup)] == [1, 7]

    def test_no_subjects_is_an_empty_keyboard(self) -> None:
        assert subjects_kb([]).inline_keyboard == []


class TestYears:
    def test_each_button_carries_its_year(self) -> None:
        markup = years_kb([2023, 2024, 2025])

        assert [YearCB.unpack(p).year for p in _payloads(markup)] == [2023, 2024, 2025]

    def test_years_are_laid_out_in_a_grid(self) -> None:
        """The one thing ``_grid`` does beyond building buttons."""
        markup = years_kb(list(range(2018, 2025)))

        assert all(len(row) <= COLUMNS_YEARS for row in markup.inline_keyboard)
        assert len(markup.inline_keyboard[0]) == COLUMNS_YEARS

    def test_no_years_is_an_empty_keyboard(self) -> None:
        assert years_kb([]).inline_keyboard == []


class TestOptions:
    def test_each_button_carries_its_option_number(self) -> None:
        markup = options_kb([1, 2, 3])

        assert [OptionCB.unpack(p).option for p in _payloads(markup)] == [1, 2, 3]

    def test_no_options_is_an_empty_keyboard(self) -> None:
        assert options_kb([]).inline_keyboard == []


class TestTopics:
    def test_a_topic_button_carries_its_position_not_its_name(self) -> None:
        """Names are free text and would not survive the payload size limit."""
        topics = ["Клетка", "Генетика", "Эволюция"]

        markup = topics_kb(topics)

        assert [b.text for b in _buttons(markup)] == topics
        assert [TopicCB.unpack(p).index for p in _payloads(markup)] == [0, 1, 2]

    def test_no_topics_is_an_empty_keyboard(self) -> None:
        assert topics_kb([]).inline_keyboard == []


class TestPartAAnswers:
    def test_the_default_offers_every_answer_a_question_has(self) -> None:
        markup = part_a_kb()

        assert [AnswerCB.unpack(p).value for p in _payloads(markup)] == list(
            range(1, PART_A_OPTION_COUNT + 1)
        )

    def test_a_shorter_question_offers_fewer(self) -> None:
        markup = part_a_kb(3)

        assert [AnswerCB.unpack(p).value for p in _payloads(markup)] == [1, 2, 3]


class TestFixedChoiceKeyboards:
    def test_mode_offers_the_three_testing_modes(self) -> None:
        modes = [ModeCB.unpack(p).mode for p in _payloads(mode_kb())]

        assert modes == ["standard", "random", "topics"]

    def test_exam_type_offers_both_variants(self) -> None:
        types = [ExamTypeCB.unpack(p).exam_type for p in _payloads(exam_type_kb())]

        assert types == ["CE", "CT"]

    def test_random_part_offers_both_parts(self) -> None:
        parts = [RandomPartCB.unpack(p).part for p in _payloads(random_part_kb())]

        assert parts == ["A", "B"]

    def test_random_feedback_offers_carrying_on_or_stopping(self) -> None:
        actions = [
            RandomFeedbackCB.unpack(p).action for p in _payloads(random_feedback_kb())
        ]

        assert actions == ["next", "finish"]


class TestAdminRegistration:
    def test_both_decisions_name_the_student_they_are_about(self) -> None:
        """The admin's chat shows many requests, so the id cannot come from state."""
        decisions = [
            RegistrationCB.unpack(p) for p in _payloads(admin_registration_kb(4242))
        ]

        assert [d.action for d in decisions] == ["approve", "reject"]
        assert {d.telegram_id for d in decisions} == {4242}
