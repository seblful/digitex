"""Tests for the callback-query middleware.

``AccessibleMessageMiddleware`` is the seam that lets every callback handler
declare ``msg: Message`` instead of re-checking an optional one, so what it
passes through and what it refuses is the whole contract.

``AuthMiddleware`` sits on both observers, so what it returns matters as much
as what it calls: the dispatcher reads the result to know whether an update
was handled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from aiogram import types
from aiogram.dispatcher.event.bases import UNHANDLED

from digitex.bot.middleware import AccessibleMessageMiddleware, AuthMiddleware


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


@dataclass
class _FakeStudents:
    authorized: bool

    async def is_authorized(self, telegram_id: int) -> bool:
        return self.authorized


def _open_uow(*, authorized: bool) -> Any:
    """A transaction factory whose students answer ``is_authorized``.

    Handed to the middleware rather than patched over a module global: the
    middleware takes the factory now, so there is nothing to reach in and
    replace.
    """

    class _FakeUow:
        def __init__(self) -> None:
            self.students = _FakeStudents(authorized)

        async def __aenter__(self) -> _FakeUow:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    return _FakeUow


def _tap(user_id: int) -> tuple[types.CallbackQuery, dict[str, Any]]:
    """A real CallbackQuery — the middleware narrows with isinstance."""
    user = types.User(id=user_id, is_bot=False, first_name="u")
    query = types.CallbackQuery(id="1", from_user=user, chat_instance="ci", data="x")
    return query, {"event_from_user": user}


class TestAuthMiddleware:
    async def test_a_plain_message_passes_through_with_its_result(self) -> None:
        handler = _Recorder()
        middleware = AuthMiddleware(admin_user_id=1, open_uow=cast("Any", None))

        result = await middleware(handler, _message(), {})

        assert result == "handled"

    async def test_an_authorized_tap_passes_through_with_its_result(self) -> None:
        handler = _Recorder()
        middleware = AuthMiddleware(
            admin_user_id=1, open_uow=_open_uow(authorized=True)
        )
        query, data = _tap(user_id=2)

        result = await middleware(handler, query, data)

        assert result == "handled"

    async def test_an_unauthorized_tap_reports_unhandled(self) -> None:
        """None would look handled to the dispatcher; UNHANDLED is what it reads."""
        handler = _Recorder()
        middleware = AuthMiddleware(
            admin_user_id=1, open_uow=_open_uow(authorized=False)
        )
        query, data = _tap(user_id=2)

        result = await middleware(handler, query, data)

        assert result is UNHANDLED
        assert handler.calls == []


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
