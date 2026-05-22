"""Handler dependency helpers.

aiogram injects ``pool: AsyncConnectionPool`` into handlers from
``workflow_data``. Most handlers immediately turn around and open a
``UnitOfWork`` against that pool. ``with_uow`` lifts the boilerplate so the
handler signature can take ``uow`` directly and the open/close happens once
around the body.

Handlers that need multiple separate transactions can still open UoWs
manually — this helper is for the common case of one transaction per handler.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any


def with_uow[**P, R](
    fn: Callable[P, Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """Wrap a handler so ``pool`` is exchanged for ``uow``.

    The wrapped handler must accept a keyword ``uow: UnitOfWork``. The wrapper
    pulls ``pool: AsyncConnectionPool`` from aiogram's injected kwargs, opens
    a UoW, and forwards everything else through.
    """

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        pool = kwargs.pop("pool")
        async with UnitOfWork(pool) as uow:
            return await fn(*args, uow=uow, **kwargs)

    return wrapper


__all__ = ["with_uow"]
