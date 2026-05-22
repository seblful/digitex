"""Typed CallbackData factories for inline keyboard callbacks.

Every inline keyboard button stores a tiny payload string in ``callback_data``.
Handler functions need to decode that string back into typed values. Using
aiogram's ``CallbackData`` factory lets us declare the schema once and have
both serialization (``.pack()``) and deserialization (``.unpack()``) handled
safely — no more bare ``callback.data.split(":")`` calls scattered across
handlers.
"""

from __future__ import annotations

from typing import Literal

from aiogram.filters.callback_data import CallbackData

from digitex.core.domain import (  # noqa: TC001 — Pydantic needs runtime types
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
