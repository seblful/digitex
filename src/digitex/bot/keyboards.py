"""Inline keyboard builders.

Callback payloads are produced via the typed factories in
:mod:`digitex.bot.callbacks` — never as bare formatted strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

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
from digitex.bot.messages import (
    MSG_KB_CE,
    MSG_KB_CT,
    MSG_KB_FINISH,
    MSG_KB_NEXT,
    MSG_KB_PART_A,
    MSG_KB_PART_B,
    MSG_KB_RANDOM,
    MSG_KB_STANDARD,
    MSG_KB_TOPICS,
    MSG_OPTION_PREFIX,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Layout constants — one source of truth instead of magic adjust() numbers.
COLUMNS_SUBJECTS = 1
COLUMNS_YEARS = 3
COLUMNS_OPTIONS = 2
COLUMNS_MODE = 1
COLUMNS_RANDOM_FEEDBACK = 1
COLUMNS_EXAM_TYPE = 2
COLUMNS_RANDOM_PART = 1
COLUMNS_TOPICS = 1
COLUMNS_REGISTRATION = 2


def _grid(items: Iterable[tuple[str, str]], columns: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for text, callback_data in items:
        builder.add(InlineKeyboardButton(text=text, callback_data=callback_data))
    builder.adjust(columns)
    return builder.as_markup()


def subjects_kb(subjects: list[tuple[int, str]]) -> InlineKeyboardMarkup:
    return _grid(
        ((name, SubjectCB(subject_id=sid).pack()) for sid, name in subjects),
        COLUMNS_SUBJECTS,
    )


def years_kb(years: list[int]) -> InlineKeyboardMarkup:
    return _grid(
        ((str(year), YearCB(year=year).pack()) for year in years),
        COLUMNS_YEARS,
    )


def options_kb(options: list[int]) -> InlineKeyboardMarkup:
    return _grid(
        (
            (f"{MSG_OPTION_PREFIX} {opt}", OptionCB(option=opt).pack())
            for opt in options
        ),
        COLUMNS_OPTIONS,
    )


def part_a_kb(num_options: int = 5) -> InlineKeyboardMarkup:
    return _grid(
        ((str(i), AnswerCB(value=i).pack()) for i in range(1, num_options + 1)),
        num_options,
    )


def mode_kb() -> InlineKeyboardMarkup:
    return _grid(
        (
            (MSG_KB_STANDARD, ModeCB(mode="standard").pack()),
            (MSG_KB_RANDOM, ModeCB(mode="random").pack()),
            (MSG_KB_TOPICS, ModeCB(mode="topics").pack()),
        ),
        COLUMNS_MODE,
    )


def random_feedback_kb() -> InlineKeyboardMarkup:
    return _grid(
        (
            (MSG_KB_NEXT, RandomFeedbackCB(action="next").pack()),
            (MSG_KB_FINISH, RandomFeedbackCB(action="finish").pack()),
        ),
        COLUMNS_RANDOM_FEEDBACK,
    )


def exam_type_kb() -> InlineKeyboardMarkup:
    return _grid(
        (
            (MSG_KB_CE, ExamTypeCB(exam_type="CE").pack()),
            (MSG_KB_CT, ExamTypeCB(exam_type="CT").pack()),
        ),
        COLUMNS_EXAM_TYPE,
    )


def random_part_kb() -> InlineKeyboardMarkup:
    return _grid(
        (
            (MSG_KB_PART_A, RandomPartCB(part="A").pack()),
            (MSG_KB_PART_B, RandomPartCB(part="B").pack()),
        ),
        COLUMNS_RANDOM_PART,
    )


def topics_kb(topics: list[str]) -> InlineKeyboardMarkup:
    return _grid(
        ((name, TopicCB(index=i).pack()) for i, name in enumerate(topics)),
        COLUMNS_TOPICS,
    )


def admin_registration_kb(telegram_id: int) -> InlineKeyboardMarkup:
    return _grid(
        (
            (
                "✅ Подтвердить",
                RegistrationCB(action="approve", telegram_id=telegram_id).pack(),
            ),
            (
                "❌ Отклонить",
                RegistrationCB(action="reject", telegram_id=telegram_id).pack(),
            ),
        ),
        COLUMNS_REGISTRATION,
    )
