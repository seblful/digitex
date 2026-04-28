"""Inline keyboard builders."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


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
        builder.add(InlineKeyboardButton(text=f"Вариант {opt}", callback_data=f"opt:{opt}"))
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
    builder.add(InlineKeyboardButton(text="Стандартный режим", callback_data="mode:standard"))
    builder.add(InlineKeyboardButton(text="Случайные вопросы", callback_data="mode:random"))
    builder.add(InlineKeyboardButton(text="Темы", callback_data="mode:topics"))
    builder.adjust(1)
    return builder.as_markup()


def random_feedback_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Следующий вопрос", callback_data="random:next"))
    builder.add(InlineKeyboardButton(text="Завершить", callback_data="random:finish"))
    builder.adjust(1)
    return builder.as_markup()


def exam_type_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="ЦЭ", callback_data="exam_type:CE"))
    builder.add(InlineKeyboardButton(text="ЦТ", callback_data="exam_type:CT"))
    builder.adjust(2)
    return builder.as_markup()


def random_part_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Часть A", callback_data="random_part:A"))
    builder.add(InlineKeyboardButton(text="Часть B", callback_data="random_part:B"))
    builder.adjust(1)
    return builder.as_markup()


def topics_kb(topics: list[str]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for i, name in enumerate(topics):
        builder.add(InlineKeyboardButton(text=name, callback_data=f"topic:{i}"))
    builder.adjust(1)
    return builder.as_markup()
