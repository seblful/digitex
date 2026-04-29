"""Tests for bot FSM states."""

from aiogram.fsm.state import State

from digitex.bot.states import Navigation, RandomTesting, Testing


class TestNavigationStates:
    def test_all_states_are_states(self) -> None:
        assert isinstance(Navigation.select_subject, State)
        assert isinstance(Navigation.select_mode, State)
        assert isinstance(Navigation.select_year, State)
        assert isinstance(Navigation.select_exam_type, State)
        assert isinstance(Navigation.select_option, State)
        assert isinstance(Navigation.select_random_part, State)
        assert isinstance(Navigation.select_random_exam_type, State)
        assert isinstance(Navigation.select_topic, State)

    def test_states_are_distinct(self) -> None:
        states = {
            Navigation.select_subject,
            Navigation.select_mode,
            Navigation.select_year,
            Navigation.select_exam_type,
            Navigation.select_option,
            Navigation.select_random_part,
            Navigation.select_random_exam_type,
            Navigation.select_topic,
        }
        assert len(states) == 8


class TestTestingStates:
    def test_answering_is_state(self) -> None:
        assert isinstance(Testing.answering, State)


class TestRandomTestingStates:
    def test_states_are_distinct(self) -> None:
        states = {RandomTesting.answering, RandomTesting.feedback}
        assert len(states) == 2
