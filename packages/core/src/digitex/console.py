"""Terminal plumbing shared by every entry point.

Two helpers, both about the boundary between a command and its process: how an
async command gets an event loop, and how a failure reaches the operator. They
live in core because both members' CLIs need them — `digitex-bot` and
`digitex-db` from the service, `digitex-extract` and friends from the studio.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from collections.abc import Coroutine


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run``, with the event loop psycopg needs on Windows.

    Windows defaults to the ProactorEventLoop, which psycopg rejects, so every
    command that opens a database connection has to run on a selector loop
    instead. Keeping that paired with the call to run it is what stops a new
    command from remembering one half.

    A `Runner` taking the loop directly, rather than
    ``set_event_loop_policy`` — event-loop policies are deprecated and go away
    in Python 3.16.
    """
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(coro)
    return asyncio.run(coro)


def abort(message: str) -> typer.Exit:
    """Render *message* on stderr and return the exit to raise."""
    typer.echo(typer.style(message, fg="red", bold=True), err=True)
    return typer.Exit(code=1)
