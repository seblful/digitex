"""Tests for the random / topic mode handlers' stale-keyboard guards.

Topic mode draws from both part tables, so a Part A keyboard left in the chat
can still be tapped while a Part B question is on screen. Both handlers must
refuse a tap that does not match the Part currently showing — otherwise the
answer is scored against the wrong Question and disclosed to the Student.

``process_random_answer`` is the scoring path, so "was it awaited" is the whole
contract under test. The aiogram objects are real: ``on_random_part_a_answer``
runs an ``isinstance`` check on ``callback.message``, which a cast cannot pass.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from aiogram import types

from digitex.bot.callbacks import AnswerCB
from digitex.bot.handlers.random import (
    on_random_part_a_answer,
    on_random_part_b_answer,
)

SCORING_PATH = "digitex.bot.handlers.random.process_random_answer"


@dataclass
class FakeState:
    """Stands in for aiogram's FSMContext over the keys RandomState declares."""

    data: dict[str, Any] = field(default_factory=dict)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def update_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)


def _state(current_part: str | None, question_id: int | None = 7) -> FakeState:
    return FakeState(
        data={
            "subject_id": 1,
            "topic_name": "Клетка",
            "current_question_id": question_id,
            "current_part": current_part,
        }
    )


def _message(text: str | None = None) -> types.Message:
    """A real Message — the Part A handler's isinstance check demands one."""
    return types.Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=types.Chat(id=1, type="private"),
        text=text,
    )


def _callback() -> Any:
    """A callback query carrying a real Message, with the API calls stubbed."""
    callback = MagicMock()
    callback.message = _message()
    callback.answer = AsyncMock()
    return callback


async def _tap_part_a(callback: Any, state: FakeState) -> AsyncMock:
    with patch(SCORING_PATH, new_callable=AsyncMock) as scoring:
        await on_random_part_a_answer(
            cast("types.CallbackQuery", callback),
            AnswerCB(value=3),
            cast("Any", state),
            cast("Any", None),
        )
    return scoring


class TestPartAKeyboardWhilePartBIsShowing:
    async def test_stale_tap_is_not_scored_against_the_part_b_question(self) -> None:
        callback = _callback()

        scoring = await _tap_part_a(callback, _state(current_part="B"))

        scoring.assert_not_awaited()

    async def test_stale_tap_is_still_acknowledged(self) -> None:
        callback = _callback()

        await _tap_part_a(callback, _state(current_part="B"))

        callback.answer.assert_awaited_once()

    async def test_tap_is_not_scored_when_no_question_is_active(self) -> None:
        callback = _callback()

        scoring = await _tap_part_a(
            callback, _state(current_part=None, question_id=None)
        )

        scoring.assert_not_awaited()

    async def test_matching_tap_is_scored(self) -> None:
        callback = _callback()

        scoring = await _tap_part_a(callback, _state(current_part="A"))

        scoring.assert_awaited_once()


class TestPartBTextWhilePartAIsShowing:
    """The sibling guard this one mirrors, pinned so the pair stays symmetric."""

    async def _send_text(self, state: FakeState) -> AsyncMock:
        with patch(SCORING_PATH, new_callable=AsyncMock) as scoring:
            await on_random_part_b_answer(
                _message(text="ВЕРНАДСКИЙ"), cast("Any", state), cast("Any", None)
            )
        return scoring

    async def test_text_is_not_scored_while_a_part_a_question_is_showing(self) -> None:
        scoring = await self._send_text(_state(current_part="A"))

        scoring.assert_not_awaited()

    async def test_matching_text_is_scored(self) -> None:
        scoring = await self._send_text(_state(current_part="B"))

        scoring.assert_awaited_once()
