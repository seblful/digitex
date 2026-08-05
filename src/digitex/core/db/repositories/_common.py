"""Shared SQL builders and row types used by the repositories.

Why this module exists. Questions live in two tables — ``part_a_questions`` and
``part_b_questions`` — so nearly every question query either picks one table by
part (:func:`question_select`) or runs over both and UNION ALLs the halves
(:class:`BothParts`). The whitelist that makes interpolating a part into SQL
safe lives here too, so each repository file stays focused on its aggregate's
reads and writes.

These names are public despite the module's underscore: three of the five
repositories import them. The SELECT / JOIN / WHERE fragments they accept are
repository-supplied literals, never user input — a part is the only value ever
interpolated, and it must go through :func:`part_table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, cast

if TYPE_CHECKING:
    from typing import Any, LiteralString

    from digitex.core.db.mapping import DictConn, DictRow
    from digitex.core.domain import Part

# Whitelist of safe table names for interpolation. Any code that substitutes a
# Part into a SQL string MUST go through ``part_table()``.
_PART_TABLES = MappingProxyType({"A": "part_a_questions", "B": "part_b_questions"})

# The two halves of every BothParts query, in the order results come back.
PARTS: tuple[Part, Part] = ("A", "B")


def part_table(part: str) -> LiteralString:
    """Return the SQL table name for the given part, or raise.

    The returned value is one of two whitelisted literals, so callers can
    safely interpolate it into a query and the result stays a ``LiteralString``
    that ``psycopg.execute`` accepts.
    """
    try:
        return cast("LiteralString", _PART_TABLES[part])
    except KeyError as e:
        raise ValueError(f"Unknown part {part!r}; expected 'A' or 'B'") from e


def validate_part(part: str) -> Part:
    """Narrow a string to a ``Part``, or raise."""
    if part not in _PART_TABLES:
        raise ValueError(f"Unknown part {part!r}; expected 'A' or 'B'")
    return cast("Part", part)


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
# Question queries
# ---------------------------------------------------------------------------

_QUESTION_COLUMNS = (
    "q.question_id, '{part}' AS part, q.question_number,"
    " b.a_num_options, i.telegram_file_id"
)
_ORIGIN_COLUMNS = "b.year_value, o.option_number, o.exam_type"

_QUESTION_FROM = (
    "SELECT {columns}"
    "  FROM {table} q"
    "  JOIN options o ON q.option_id = o.option_id"
    "  JOIN books b ON o.book_id = b.book_id"
    "  LEFT JOIN images i ON i.question_id = q.question_id AND i.part = '{part}'"
)


def question_select(part: Part, *, with_origin: bool = False) -> LiteralString:
    """One part's question metadata, optionally with its origin columns.

    The BYTEA payload is deliberately not selected. Callers that need to upload
    a fresh image (a cache miss) fetch the bytes via
    :meth:`~digitex.core.db.repositories.question.QuestionRepository.get_image`.

    With *with_origin*, the year / option / exam-type columns are added, which
    is the only way the two question shapes differ.
    """
    columns = _QUESTION_COLUMNS
    if with_origin:
        columns = f"{columns}, {_ORIGIN_COLUMNS}"
    # Every substituted value is a literal, so the result stays a LiteralString.
    return (
        _QUESTION_FROM.replace("{columns}", columns)
        .replace("{table}", part_table(part))
        .replace("{part}", part)
    )


_BOTH_PARTS_HALF = (
    "SELECT {select}"
    " FROM {table} q"
    " JOIN options o ON q.option_id = o.option_id"
    " JOIN books b ON o.book_id = b.book_id"
    " {joins}"
    " {where}"
)


@dataclass(frozen=True)
class BothParts:
    """A question query run over both part tables and UNION ALLed together.

    *select* and *joins* are templates in which ``{part}`` expands to that
    half's part literal, so one spelling covers both halves — there are no
    separate overrides for the B side. ``JOIN options o`` and ``JOIN books b``
    are always present, so ``o.*`` and ``b.*`` are available without asking.

    :meth:`fetch` binds its parameters once per half, so callers pass each
    value exactly once no matter that the SQL mentions it twice.
    """

    select: str
    joins: str = ""
    where: str = ""
    order_by: str = ""

    def _half(self, part: Part) -> str:
        # Substitution is by ``replace``, not ``format``, so stray braces in a
        # caller's fragment are harmless. Table and part go last so they also
        # expand inside the injected select / joins / where.
        return (
            _BOTH_PARTS_HALF.replace("{select}", self.select)
            .replace("{joins}", self.joins)
            .replace("{where}", self.where)
            .replace("{table}", part_table(part))
            .replace("{part}", part)
        )

    def _sql(self) -> LiteralString:
        union = " UNION ALL ".join(self._half(part) for part in PARTS)
        if not self.order_by:
            return cast("LiteralString", union)
        return cast(
            "LiteralString", f"SELECT * FROM ({union}) u ORDER BY {self.order_by}"
        )

    async def fetch(self, conn: DictConn, *params: Any) -> list[DictRow]:
        """Run the query, binding *params* into each half of the union."""
        cur = await conn.execute(self._sql(), params * len(PARTS))
        return await cur.fetchall()


async def get_or_create(
    conn: DictConn,
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
    # Re-assign the conflict columns to themselves so RETURNING always fires.
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
    # Column names come from the caller-controlled ``where`` dict literals,
    # never from user input — safe to interpolate.
    sql = cast(
        "LiteralString",
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        f" ON CONFLICT ({col_list}) DO UPDATE SET {update_clause}"
        f" RETURNING {id_col}",
    )
    cur = await conn.execute(sql, values)
    row = await cur.fetchone()
    assert row is not None
    return row[id_col]


__all__ = [
    "PARTS",
    "BothParts",
    "QuestionOrigin",
    "SessionInfo",
    "SubjectRow",
    "WrongAnswer",
    "get_or_create",
    "part_table",
    "question_select",
    "validate_part",
]
