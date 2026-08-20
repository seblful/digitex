"""The row shape this layer speaks, and the one way it becomes a typed object.

The pool sets ``dict_row`` as its row factory, so a fetch hands back a plain
``dict[str, Any]``. Two aliases name that shape once and one function validates
it, which is all the repositories share: no column name is spelled here, and no
repository has to repeat the same ``model_validate`` call.

Nothing here builds SQL or opens anything. A row arrives already fetched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from psycopg import AsyncConnection


type DictRow = dict[str, Any]
"""One row, as ``dict_row`` produces it."""

type DictConn = "AsyncConnection[DictRow]"
"""The connection type every repository is annotated with.

psycopg's stubs default a connection's row type to ``tuple``, so annotating with
the plain class would make every ``row["column"]`` in the layer a type error.
Naming the dict-row connection here means the cast that establishes it happens
once, where the pool hands a connection over, instead of at each fetch.
"""


def row_to_model[T: BaseModel](row: Mapping[str, Any], model: type[T]) -> T:
    """Validate a dict-shaped row against a Pydantic model.

    Extra keys are dropped — Pydantic's default for a model that has not asked
    for anything else — so a select may carry columns the model does not name.

    Raises:
        ValidationError: If the row is missing a field the model requires. That
            is a mismatch between the model and the select, so it surfaces on
            the first read rather than as a missing attribute later.
    """
    return model.model_validate(dict(row))


__all__ = ["DictConn", "DictRow", "row_to_model"]
