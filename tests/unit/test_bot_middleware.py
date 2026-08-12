"""Tests for the callback-query middleware.

``AccessibleMessageMiddleware`` is the seam that lets every callback handler
declare ``msg: Message`` instead of re-checking an optional one, so what it
passes through and what it refuses is the whole contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from aiogram import types

from digitex.bot.middleware import AccessibleMessageMiddleware


@dataclass
class _Recorder:
    """Stands in for the downstream handler."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, event: Any, data: dict[str, Any]) -> str:
        self.calls.append(data)
        return "handled"


def _message() -> types.Message:
    return types.Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=types.Chat(id=1, type="private"),
    )


def _inaccessible() -> types.InaccessibleMessage:
    """What Telegram sends once the original message is gone or too old."""
    return types.InaccessibleMessage(
        message_id=1,
        chat=types.Chat(id=1, type="private"),
    )


@dataclass
class _FakeCallback:
    """The two attributes the middleware touches.

    A real ``CallbackQuery.answer`` is a request object bound to a Bot, so a
    fake is the only way to observe the acknowledgement without a network call.
    """

    message: Any
    acks: list[int] = field(default_factory=list)

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        self.acks.append(1)


def _callback(message: Any) -> _FakeCallback:
    return _FakeCallback(message=message)


class TestAccessibleMessageMiddleware:
    async def test_an_accessible_message_is_injected_as_msg(self) -> None:
        message = _message()
        callback = _callback(message)
        handler = _Recorder()

        result = await AccessibleMessageMiddleware()(
            handler, cast("Any", callback), {"state": "x"}
        )

        assert result == "handled"
        assert handler.calls[0]["msg"] is message
        assert callback.acks == []

    async def test_an_inaccessible_message_is_acknowledged_not_handled(self) -> None:
        callback = _callback(_inaccessible())
        handler = _Recorder()

        result = await AccessibleMessageMiddleware()(handler, cast("Any", callback), {})

        assert result is None
        assert handler.calls == []
        assert callback.acks == [1]

    async def test_a_missing_message_is_acknowledged_not_handled(self) -> None:
        callback = _callback(None)
        handler = _Recorder()

        await AccessibleMessageMiddleware()(handler, cast("Any", callback), {})

        assert handler.calls == []
        assert callback.acks == [1]
