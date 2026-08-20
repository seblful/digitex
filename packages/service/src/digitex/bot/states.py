"""The conversation states an update is routed by.

aiogram matches an update against the *name* a state resolves to, and a
``StatesGroup`` prefixes its own onto every member. That prefix is what keeps
``Testing.answering`` and ``RandomTesting.answering`` from colliding — the two
modes score a reply against different questions, so one shared name would send
a random-mode answer to the standard-mode handler.
"""

from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class Registration(StatesGroup):
    waiting_for_name = State()


class Navigation(StatesGroup):
    select_subject = State()
    select_mode = State()
    select_year = State()
    select_exam_type = State()
    select_option = State()
    select_random_part = State()
    select_random_exam_type = State()
    select_topic = State()


class Testing(StatesGroup):
    answering = State()


class RandomTesting(StatesGroup):
    answering = State()
    feedback = State()
