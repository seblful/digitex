"""Inline keyboard builders."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

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


def subjects_kb(subjects: list[tuple[int, str]]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for subject_id, name in subjects:
        builder.add(InlineKeyboardButton(text=name, callback_data=f"subj:{subject_id}"))
    builder.adjust(1)
    return builder.as_markup()


def years_kb(years: list[int]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for year in years:
        builder.add(InlineKeyboardButton(text=str(year), callback_data=f"year:{year}"))
    builder.adjust(3)
    return builder.as_markup()


def options_kb(options: list[int]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for opt in options:
        builder.add(
            InlineKeyboardButton(
                text=f"{MSG_OPTION_PREFIX} {opt}", callback_data=f"opt:{opt}"
            )
        )
    builder.adjust(2)
    return builder.as_markup()


def part_a_kb(num_options: int = 5) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for i in range(1, num_options + 1):
        builder.add(InlineKeyboardButton(text=str(i), callback_data=f"ans:{i}"))
    builder.adjust(num_options)
    return builder.as_markup()


def mode_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(
        InlineKeyboardButton(text=MSG_KB_STANDARD, callback_data="mode:standard")
    )
    builder.add(InlineKeyboardButton(text=MSG_KB_RANDOM, callback_data="mode:random"))
    builder.add(InlineKeyboardButton(text=MSG_KB_TOPICS, callback_data="mode:topics"))
    builder.adjust(1)
    return builder.as_markup()


def random_feedback_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text=MSG_KB_NEXT, callback_data="random:next"))
    builder.add(InlineKeyboardButton(text=MSG_KB_FINISH, callback_data="random:finish"))
    builder.adjust(1)
    return builder.as_markup()


def exam_type_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text=MSG_KB_CE, callback_data="exam_type:CE"))
    builder.add(InlineKeyboardButton(text=MSG_KB_CT, callback_data="exam_type:CT"))
    builder.adjust(2)
    return builder.as_markup()


def random_part_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text=MSG_KB_PART_A, callback_data="random_part:A"))
    builder.add(InlineKeyboardButton(text=MSG_KB_PART_B, callback_data="random_part:B"))
    builder.adjust(1)
    return builder.as_markup()


def topics_kb(topics: list[str]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for i, name in enumerate(topics):
        builder.add(InlineKeyboardButton(text=name, callback_data=f"topic:{i}"))
    builder.adjust(1)
    return builder.as_markup()


def admin_registration_kb(telegram_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(
        InlineKeyboardButton(
            text="✅ Подтвердить",
            callback_data=f"reg:approve:{telegram_id}",
        )
    )
    builder.add(
        InlineKeyboardButton(
            text="❌ Отклонить",
            callback_data=f"reg:reject:{telegram_id}",
        )
    )
    builder.adjust(2)
    return builder.as_markup()
