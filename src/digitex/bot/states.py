"""FSM states for the bot."""

from aiogram.fsm.state import State, StatesGroup


class Navigation(StatesGroup):
    select_subject = State()
    select_year = State()
    select_option = State()


class Testing(StatesGroup):
    answering = State()
