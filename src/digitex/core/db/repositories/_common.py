"""Shared SQL fragments and row types used by the repositories.

Why this module exists. The five repositories all interpolate a ``Part``
literal (``"A"`` / ``"B"``) into the SQL table name. Keeping the whitelist of
safe table names plus the few query-building helpers in one place lets each
repository file stay focused on its aggregate's reads and writes.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any

    from psycopg import AsyncConnection

    from digitex.core.domain import Part

# Whitelist of safe table names for f-string interpolation. Any code that
# substitutes a Part into a SQL string MUST go through ``_part_table()``.
_PART_TABLES = MappingProxyType({"A": "part_a_questions", "B": "part_b_questions"})


def _part_table(part: str) -> str:
    """Return the SQL table name for the given part, or raise."""
    try:
        return _PART_TABLES[part]
    except KeyError as e:
        raise ValueError(f"Unknown part {part!r}; expected 'A' or 'B'") from e


def _validate_part(part: str) -> Part:
    if part not in _PART_TABLES:
        raise ValueError(f"Unknown part {part!r}; expected 'A' or 'B'")
    return part  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Row types — lightweight containers for query results
# ---------------------------------------------------------------------------


class SubjectRow(NamedTuple):
    id: int
    name: str


class SessionInfo(NamedTuple):
    subject_name: str
    year: int
    option_number: int


class WrongAnswer(NamedTuple):
    question_number: int
    part: str
    student_answer: str
    correct_answer: str


class QuestionOrigin(NamedTuple):
    year: int
    option_number: int
    exam_type: str


# ---------------------------------------------------------------------------
# Question query fragments
# ---------------------------------------------------------------------------


def _question_base(part: Part) -> str:
    table = _part_table(part)
    return (
        f"SELECT q.question_id, '{part}' AS part, q.question_number,"
        " b.a_num_options, i.image_data, i.telegram_file_id"
        f"  FROM {table} q"
        "  JOIN options o ON q.option_id = o.option_id"
        "  JOIN books b ON o.book_id = b.book_id"
        f"  LEFT JOIN images i ON i.question_id = q.question_id AND i.part = '{part}'"
    )


def _question_full(part: Part) -> str:
    table = _part_table(part)
    return (
        f"SELECT q.question_id, '{part}' AS part, q.question_number,"
        " b.a_num_options, i.image_data, i.telegram_file_id,"
        " b.year_value, o.option_number, o.exam_type"
        f"  FROM {table} q"
        "  JOIN options o ON q.option_id = o.option_id"
        "  JOIN books b ON o.book_id = b.book_id"
        f"  LEFT JOIN images i ON i.question_id = q.question_id AND i.part = '{part}'"
    )


async def _get_or_create(
    conn: AsyncConnection,
    table: str,
    id_col: str,
    where: dict[str, Any],
) -> int:
    """Insert or fetch a row, returning its id, in one round-trip.

    Uses ``ON CONFLICT … DO UPDATE`` so a row is always returned by the
    ``RETURNING`` clause (``DO NOTHING`` would suppress the row on conflict).
    The update is a no-op (re-assigning the conflict columns to themselves).
    """
    cols = list(where.keys())
    values = list(where.values())
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    conflict_cols = ", ".join(cols)
    # Re-assign the conflict columns to themselves so RETURNING always fires.
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
    cur = await conn.execute(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        f" ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}"
        f" RETURNING {id_col}",
        values,
    )
    row = await cur.fetchone()
    assert row is not None
    return row[id_col]


async def _union_both_parts(
    conn: AsyncConnection,
    select_a: str,
    joins: str = "",
    where: str = "",
    order_by: str = "",
    limit: str = "",
    params: tuple = (),
    select_b: str | None = None,
    joins_b: str | None = None,
) -> list[dict[str, Any]]:
    """Run a UNION-ALL query across both ``part_*_questions`` tables.

    The standard ``JOIN options o`` and ``JOIN books b`` are always added so
    ``o.*`` / ``b.*`` are available. *select_b* / *joins_b* override the first
    half when the two halves differ (e.g. a hard-coded ``'A'`` vs ``'B'`` part
    literal).

    Both halves share the same parameter list — *params* is duplicated when
    bound, so each ``%s`` placeholder in *where* should appear in both halves.
    """
    if select_b is None:
        select_b = select_a
    if joins_b is None:
        joins_b = joins

    base = (
        "SELECT {select}"
        " FROM {table} q"
        " JOIN options o ON q.option_id = o.option_id"
        " JOIN books b ON o.book_id = b.book_id"
        " {joins}"
        " {where}"
    )
    union = (
        base.format(select=select_a, table=_part_table("A"), joins=joins, where=where)
        + " UNION ALL "
        + base.format(
            select=select_b, table=_part_table("B"), joins=joins_b, where=where
        )
    )
    sql = union
    if order_by or limit:
        sql = f"SELECT * FROM ({union}) u"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"
    cur = await conn.execute(sql, params + params)
    return await cur.fetchall()


__all__ = [
    "QuestionOrigin",
    "SessionInfo",
    "SubjectRow",
    "WrongAnswer",
    "_get_or_create",
    "_part_table",
    "_question_base",
    "_question_full",
    "_union_both_parts",
    "_validate_part",
]
