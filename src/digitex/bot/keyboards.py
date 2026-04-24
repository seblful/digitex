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


def part_a_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for i in range(1, 6):
        builder.add(InlineKeyboardButton(text=str(i), callback_data=f"ans:{i}"))
    builder.adjust(5)
    return builder.as_markup()
