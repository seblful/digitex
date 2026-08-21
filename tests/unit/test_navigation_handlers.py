"""Tests for the navigation screens that hand off into a random round.

The module's contract is that every screen edits the message the tap came
from, so one keyboard replaces the next. The part and topic screens are the
two that start a round instead of showing another keyboard — they must still
dismiss themselves, or the chat keeps offering both parts under the question.
The one exception is an empty draw: the round never started, so the keyboard
stays live and the student can pick a part or topic that has questions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from digitex.bot.answer_flow import Round
from digitex.bot.callbacks import RandomPartCB, TopicCB
from digitex.bot.handlers.navigation import on_random_part_selected, on_topic_selected
from digitex.bot.messages import MSG_START_TESTING

ROUND_START_PATH = "digitex.bot.handlers.navigation.start_random_question"


@dataclass
class FakeState:
    """Stands in for aiogram's FSMContext over the keys navigation reads."""

    data: dict[str, Any] = field(default_factory=dict)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def set_data(self, data: dict[str, Any]) -> None:
        self.data = dict(data)


def _message() -> Any:
    message = MagicMock()
    message.edit_text = AsyncMock()
    return message


def _callback() -> Any:
    callback = MagicMock()
    callback.answer = AsyncMock()
    return callback


def _round(state: FakeState) -> Round:
    """A Round over the fake state; seams the patched start never reaches stay None."""
    return Round(
        cast("Any", None), cast("Any", state), Path("unused"), cast("Any", None)
    )


async def _tap_part(msg: Any, *, question_shown: bool) -> None:
    state = FakeState(data={"subject_id": 1, "exam_type": "CE"})
    with patch(ROUND_START_PATH, new_callable=AsyncMock, return_value=question_shown):
        await on_random_part_selected(
            _callback(),
            RandomPartCB(part="A"),
            cast("Any", state),
            msg,
            _round(state),
        )


async def _tap_topic(msg: Any, *, question_shown: bool) -> None:
    state = FakeState(data={"subject_id": 1, "topic_names": ["Клетка"]})
    with patch(ROUND_START_PATH, new_callable=AsyncMock, return_value=question_shown):
        await on_topic_selected(
            _callback(),
            TopicCB(index=0),
            cast("Any", state),
            msg,
            _round(state),
        )


class TestPartScreenDismissesItself:
    async def test_the_part_keyboard_is_edited_away_when_the_round_starts(
        self,
    ) -> None:
        msg = _message()

        await _tap_part(msg, question_shown=True)

        msg.edit_text.assert_awaited_once_with(MSG_START_TESTING)

    async def test_an_empty_draw_keeps_the_keyboard_for_another_pick(self) -> None:
        msg = _message()

        await _tap_part(msg, question_shown=False)

        msg.edit_text.assert_not_awaited()


class TestTopicScreenDismissesItself:
    async def test_the_topic_list_is_edited_away_when_the_round_starts(self) -> None:
        msg = _message()

        await _tap_topic(msg, question_shown=True)

        msg.edit_text.assert_awaited_once_with(MSG_START_TESTING)

    async def test_an_empty_draw_keeps_the_list_for_another_pick(self) -> None:
        msg = _message()

        await _tap_topic(msg, question_shown=False)

        msg.edit_text.assert_not_awaited()
