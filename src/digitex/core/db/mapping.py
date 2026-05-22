"""Generic helpers for mapping database rows to typed objects.

Repositories receive ``dict[str, Any]`` rows from psycopg (``dict_row`` is the
pool's default ``row_factory``). Pydantic models and dataclasses both know how
to validate a dict; the helpers below remove the per-repo boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def row_to_model[T: BaseModel](row: Mapping[str, Any], model: type[T]) -> T:
    """Validate a dict-shaped row against a Pydantic model.

    Extra keys are dropped (the model's ``model_config`` decides this);
    missing keys raise a ``ValidationError`` from Pydantic.
    """
    return model.model_validate(dict(row))


__all__ = ["row_to_model"]
