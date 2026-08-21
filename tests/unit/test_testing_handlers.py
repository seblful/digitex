"""Tests for the standard testing loop's stale-keyboard guards.

Old Part A keyboards stay live in the chat after their question is answered,
so a tap can arrive for a question that is already scored — and unlike random
mode, this loop records answers to a Session, so a reply that slips past the
guard is written against the next, unseen question. Both handlers claim the
reply through the round before anything is scored.

``_record_and_advance`` is the scoring path, so "was it awaited" is the whole
contract under test. The message arrives as a handler argument, narrowed by
``AccessibleMessageMiddleware``; the round arrives the same way, built by
``RoundMiddleware`` — so these tests need no real aiogram objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from digitex.bot.answer_flow import Round
from digitex.bot.callbacks import AnswerCB
from digitex.bot.handlers.testing import on_part_a_answer, on_part_b_answer

SCORING_PATH = "digitex.bot.handlers.testing._record_and_advance"


@dataclass
class FakeState:
    """Stands in for aiogram's FSMContext over the keys TestingState declares."""

    data: dict[str, Any] = field(default_factory=dict)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def update_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)


@dataclass
class FakeMessage:
    """The narrowed message a callback handler is handed."""

    text: str | None = None


def _state(current_part: str | None) -> FakeState:
    return FakeState(
        data={
            "session_id": 7,
            "question_ids": [[10, "A"], [20, "B"]],
            "current_index": 0,
            "current_part": current_part,
            "waiting_for_answer": True,
        }
    )


def _round(state: FakeState) -> Round:
    """A Round over the fake state; seams a refused reply never reaches stay None."""
    return Round(
        cast("Any", None), cast("Any", state), Path("unused"), cast("Any", None)
    )


def _callback() -> Any:
    """A callback query with its API calls stubbed."""
    callback = MagicMock()
    callback.answer = AsyncMock()
    return callback


async def _tap_part_a(callback: Any, state: FakeState) -> AsyncMock:
    with patch(SCORING_PATH, new_callable=AsyncMock) as scoring:
        await on_part_a_answer(
            callback,
            AnswerCB(value=3),
            cast("Any", FakeMessage()),
            _round(state),
        )
    return scoring


class TestPartAKeyboardWhilePartBIsShowing:
    async def test_stale_tap_is_not_recorded_against_the_part_b_question(self) -> None:
        callback = _callback()

        scoring = await _tap_part_a(callback, _state(current_part="B"))

        scoring.assert_not_awaited()

    async def test_stale_tap_is_still_acknowledged(self) -> None:
        callback = _callback()

        await _tap_part_a(callback, _state(current_part="B"))

        callback.answer.assert_awaited_once()

    async def test_matching_tap_is_scored(self) -> None:
        callback = _callback()

        scoring = await _tap_part_a(callback, _state(current_part="A"))

        scoring.assert_awaited_once()


class TestPartBTextWhilePartAIsShowing:
    """The sibling guard this one mirrors, pinned so the pair stays symmetric."""

    async def _send_text(
        self, state: FakeState, text: str | None = "ФОТОСИНТЕЗ"
    ) -> AsyncMock:
        with patch(SCORING_PATH, new_callable=AsyncMock) as scoring:
            await on_part_b_answer(cast("Any", FakeMessage(text=text)), _round(state))
        return scoring

    async def test_text_is_not_scored_while_a_part_a_question_is_showing(self) -> None:
        scoring = await self._send_text(_state(current_part="A"))

        scoring.assert_not_awaited()

    async def test_matching_text_is_scored(self) -> None:
        scoring = await self._send_text(_state(current_part="B"))

        scoring.assert_awaited_once()

    async def test_an_empty_message_is_not_scored(self) -> None:
        scoring = await self._send_text(_state(current_part="B"), text=None)

        scoring.assert_not_awaited()


class TestAnAnswerIsScoredOnlyOnce:
    """A second reply must not reach scoring while the first is in flight.

    ``show_testing_question`` arms ``waiting_for_answer`` and ``claim_reply``
    disarms it before scoring — the same protocol random mode uses, shared by
    both modes through the ``ReplyGuard`` slice.
    """

    async def test_a_tap_after_the_guard_disarmed_is_not_scored(self) -> None:
        state = _state(current_part="A")
        state.data["waiting_for_answer"] = False

        scoring = await _tap_part_a(_callback(), state)

        scoring.assert_not_awaited()

    async def test_the_guard_is_disarmed_before_scoring_runs(self) -> None:
        """The reply is spent first, so a crash mid-scoring cannot replay it."""
        state = _state(current_part="A")
        armed_at_scoring: list[bool] = []

        with patch(SCORING_PATH, new_callable=AsyncMock) as scoring:
            scoring.side_effect = lambda *args, **kwargs: armed_at_scoring.append(
                state.data["waiting_for_answer"]
            )
            await on_part_a_answer(
                _callback(),
                AnswerCB(value=3),
                cast("Any", FakeMessage()),
                _round(state),
            )

        assert armed_at_scoring == [False]
