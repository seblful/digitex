"""Tests for the random / topic mode handlers' stale-keyboard guards.

Topic mode draws from both part tables, so a Part A keyboard left in the chat
can still be tapped while a Part B question is on screen. Both handlers must
refuse a tap that does not match the Part currently showing — otherwise the
answer is scored against the wrong Question and disclosed to the Student.

``process_random_answer`` is the scoring path, so "was it awaited" is the whole
contract under test. The message arrives as a handler argument, narrowed by
``AccessibleMessageMiddleware``, so these tests need no real aiogram objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from digitex.bot.answer_flow import Round
from digitex.bot.callbacks import AnswerCB
from digitex.bot.handlers.random import (
    on_random_part_a_answer,
    on_random_part_b_answer,
    process_random_answer,
)
from digitex.domain.answer import AnswerKey

SCORING_PATH = "digitex.bot.handlers.random.process_random_answer"


@dataclass
class FakeState:
    """Stands in for aiogram's FSMContext over the keys RandomState declares."""

    data: dict[str, Any] = field(default_factory=dict)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def update_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)


@dataclass
class FakeMessage:
    """The narrowed message a callback handler is handed."""

    text: str | None = None


def _state(current_part: str | None, question_id: int | None = 7) -> FakeState:
    return FakeState(
        data={
            "subject_id": 1,
            "topic_name": "Клетка",
            "current_question_id": question_id,
            "current_part": current_part,
            "waiting_for_answer": True,
        }
    )


def _callback() -> Any:
    """A callback query with its API calls stubbed."""
    callback = MagicMock()
    callback.answer = AsyncMock()
    return callback


async def _tap_part_a(callback: Any, state: FakeState) -> AsyncMock:
    with patch(SCORING_PATH, new_callable=AsyncMock) as scoring:
        await on_random_part_a_answer(
            callback,
            AnswerCB(value=3),
            cast("Any", state),
            cast("Any", FakeMessage()),
            cast("Any", None),
            cast("Any", None),
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
                cast("Any", FakeMessage(text="ВЕРНАДСКИЙ")),
                cast("Any", state),
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
            )
        return scoring

    async def test_text_is_not_scored_while_a_part_a_question_is_showing(self) -> None:
        scoring = await self._send_text(_state(current_part="A"))

        scoring.assert_not_awaited()

    async def test_matching_text_is_scored(self) -> None:
        scoring = await self._send_text(_state(current_part="B"))

        scoring.assert_awaited_once()


class TestAnAnswerIsScoredOnlyOnce:
    """A second reply must not reach scoring while the first is in flight.

    ``show_random_question`` arms ``waiting_for_answer`` and the guard disarms
    it before scoring — the same protocol the standard loop uses, shared by
    both modes now that RandomState declares the field.
    """

    async def test_a_tap_after_the_guard_disarmed_is_not_scored(self) -> None:
        state = _state(current_part="A")
        state.data["waiting_for_answer"] = False

        scoring = await _tap_part_a(_callback(), state)

        scoring.assert_not_awaited()

    async def test_the_first_tap_disarms_the_guard(self) -> None:
        state = _state(current_part="A")

        await _tap_part_a(_callback(), state)

        assert state.data["waiting_for_answer"] is False


class TestWrongAnswerRendering:
    """The Part B answer key is free text and the reply goes out as HTML.

    An unescaped "<" makes Telegram reject the message; the raised error would
    skip the transition to ``feedback``, leaving the Student unable to continue.
    """

    async def _reply(self, correct_answer: str) -> tuple[str, Any]:
        message = MagicMock()
        message.answer = AsyncMock()
        state = MagicMock()
        state.get_data = AsyncMock(return_value=_state(current_part="B").data)
        state.update_data = AsyncMock()
        state.set_state = AsyncMock()

        class _Uow:
            async def __aenter__(self) -> Any:
                return MagicMock()

            async def __aexit__(self, *args: object) -> None:
                return None

        round = Round(
            cast("Any", None),
            cast("Any", state),
            cast("Any", None),
            Path("unused"),
            open_uow=lambda: cast("Any", _Uow()),
        )

        with patch(
            "digitex.bot.handlers.random.evaluate_random_answer",
            new_callable=AsyncMock,
            return_value=(False, AnswerKey(part="B", value=correct_answer)),
        ):
            await process_random_answer(cast("Any", message), round, "нет")

        return message.answer.call_args.args[0], state

    async def test_html_special_characters_are_escaped(self) -> None:
        text, _ = await self._reply("x < 5 & y > 2")

        assert "x &lt; 5 &amp; y &gt; 2" in text

    async def test_the_round_still_advances_to_feedback(self) -> None:
        _, state = await self._reply("x < 5")

        state.set_state.assert_awaited_once()
