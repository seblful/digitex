"""Async DB helpers wrapping the sync UnitOfWork."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from digitex.core.db import UnitOfWork

T = TypeVar("T")


async def with_uow(db_path: str, callback: Callable[[UnitOfWork], T]) -> T:
    """Run a sync callback with a UnitOfWork in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_in_uow, db_path, callback)


def _run_in_uow(db_path: str, callback: Callable[[UnitOfWork], Any]) -> Any:
    with UnitOfWork(db_path) as uow:
        return callback(uow)
