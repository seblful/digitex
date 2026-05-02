"""FSM states for the bot."""

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
