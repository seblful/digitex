"""The wire format of every inline-keyboard button, declared once per button.

Telegram gives a button 64 bytes of ``callback_data`` and hands the string back
verbatim when it is tapped. aiogram's ``CallbackData`` turns that string into a
schema: the same class packs the payload in :mod:`digitex.bot.keyboards` and
unpacks it — validated, typed — into the handler that filters on it. Nothing in
the bot splits a callback string by hand.

The prefix is the routing key, so it has to be unique across this module; the
field names are the payload and cost bytes, which is why a topic travels as its
position in the keyboard rather than as its name.
"""

from __future__ import annotations

from typing import Literal

from aiogram.filters.callback_data import CallbackData

from digitex.domain.entities import (  # noqa: TC001 — Pydantic needs runtime types
    ExamType,
    Part,
)


class SubjectCB(CallbackData, prefix="subj"):
    subject_id: int


class YearCB(CallbackData, prefix="year"):
    year: int


class OptionCB(CallbackData, prefix="opt"):
    option: int


class AnswerCB(CallbackData, prefix="ans"):
    value: int


class ModeCB(CallbackData, prefix="mode"):
    mode: Literal["standard", "random", "topics"]


class ExamTypeCB(CallbackData, prefix="exam_type"):
    exam_type: ExamType


class RandomPartCB(CallbackData, prefix="random_part"):
    part: Part


class TopicCB(CallbackData, prefix="topic"):
    index: int


class RandomFeedbackCB(CallbackData, prefix="random"):
    action: Literal["next", "finish"]


class RegistrationCB(CallbackData, prefix="reg"):
    action: Literal["approve", "reject"]
    telegram_id: int


__all__ = [
    "AnswerCB",
    "ExamTypeCB",
    "ModeCB",
    "OptionCB",
    "RandomFeedbackCB",
    "RandomPartCB",
    "RegistrationCB",
    "SubjectCB",
    "TopicCB",
    "YearCB",
]
