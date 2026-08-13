"""Helpers shared by the CLI command modules."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from collections.abc import Coroutine


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` with the event-loop policy psycopg needs on Windows.

    Windows' default ProactorEventLoop is rejected by psycopg, so every
    command that opens a database connection installs the selector policy
    first. Keeping the pair together is what stops a new command from
    forgetting the first half.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def abort(message: str) -> typer.Exit:
    """Render *message* on stderr and return the exit to raise."""
    typer.echo(typer.style(message, fg="red", bold=True), err=True)
    return typer.Exit(code=1)
