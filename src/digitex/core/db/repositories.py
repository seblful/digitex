"""Repository classes — the only layer that touches raw SQL.

All methods are ``async`` and run against an ``AsyncConnection`` borrowed from
the application's :class:`psycopg_pool.AsyncConnectionPool`. The pool's default
``row_factory`` is ``dict_row``, so every fetched row is a ``dict[str, Any]``.

Why two question tables. ``part_a_questions`` has an integer answer
(1..5); ``part_b_questions`` has a free-form text answer. Keeping them split
preserves the type-safety of ``answer``. The price is a small amount of
union/dispatch logic — see :data:`_PART_TABLES` and :func:`_union_both_parts`.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from digitex.bot.schemas import AuthorizedUser
from digitex.core.schemas import Question, Session, Student, TestResult

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from digitex.core.value_objects import QuestionKey


Part = Literal["A", "B"]

# Whitelist of safe table names for f-string interpolation. Any code that
# substitutes a Part into a SQL string MUST go through ``_part_table()``.
_PART_TABLES = MappingProxyType({"A": "part_a_questions", "B": "part_b_questions"})


def _part_table(part: str) -> str:
    """Return the SQL table name for the given part, or raise."""
    try:
        return _PART_TABLES[part]
    except KeyError as e:
        raise ValueError(f"Unknown part {part!r}; expected 'A' or 'B'") from e


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


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class BookRepository:
    """Reads/writes for subjects, books, and options."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get_or_create_subject(self, name: str) -> int:
        return await _get_or_create(
            self._conn, "subjects", "subject_id", {"name": name}
        )

    async def get_or_create_book(
        self,
        subject_id: int,
        year: int,
        a_num_options: int = 5,  # noqa: ARG002 — kept for callsite compatibility
    ) -> int:
        return await _get_or_create(
            self._conn,
            "books",
            "book_id",
            {"subject_id": subject_id, "year_value": year},
        )

    async def get_or_create_option(
        self,
        book_id: int,
        option_number: int,
        exam_type: str = "CT",  # noqa: ARG002 — see get_or_create_book
    ) -> int:
        return await _get_or_create(
            self._conn,
            "options",
            "option_id",
            {"book_id": book_id, "option_number": option_number},
        )

    async def list_subjects(self) -> list[SubjectRow]:
        cur = await self._conn.execute(
            "SELECT subject_id, name FROM subjects ORDER BY name"
        )
        rows = await cur.fetchall()
        return [SubjectRow(r["subject_id"], r["name"]) for r in rows]

    async def list_years(self, subject_id: int) -> list[int]:
        cur = await self._conn.execute(
            "SELECT year_value FROM books"
            " WHERE subject_id = %s ORDER BY year_value DESC",
            (subject_id,),
        )
        rows = await cur.fetchall()
        return [r["year_value"] for r in rows]

    async def list_options(self, book_id: int, exam_type: str) -> list[int]:
        cur = await self._conn.execute(
            "SELECT option_number FROM options"
            " WHERE book_id = %s AND exam_type = %s ORDER BY option_number",
            (book_id, exam_type),
        )
        rows = await cur.fetchall()
        return [r["option_number"] for r in rows]

    async def get_option_id(self, book_id: int, option_number: int) -> int:
        cur = await self._conn.execute(
            "SELECT option_id FROM options WHERE book_id = %s AND option_number = %s",
            (book_id, option_number),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Option {option_number} not found for book {book_id}")
        return row["option_id"]


class QuestionRepository:
    """Repository for questions, images, answers, and topic mappings."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    # -- CRUD ----------------------------------------------------------------

    async def get_or_create(self, option_id: int, key: QuestionKey, answer: str) -> int:
        if key.part == "A":
            if not answer.isdigit():
                raise ValueError(f"Part A answer must be a digit, got {answer!r}")
            answer_val: int | str = int(answer)
        else:
            answer_val = answer
        table = _part_table(key.part)

        cur = await self._conn.execute(
            f"INSERT INTO {table} (option_id, question_number, answer)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (option_id, question_number)"
            " DO UPDATE SET answer = EXCLUDED.answer"
            " RETURNING question_id",
            (option_id, key.number, answer_val),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["question_id"]

    async def insert_image(
        self, question_id: int, part: str, image_data: bytes
    ) -> None:
        # Skip the write if the BYTEA payload hasn't changed; this avoids
        # rewriting multi-MB rows during idempotent re-runs.
        await self._conn.execute(
            "INSERT INTO images (question_id, part, image_data)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (question_id, part)"
            " DO UPDATE SET image_data = EXCLUDED.image_data"
            " WHERE images.image_data IS DISTINCT FROM EXCLUDED.image_data",
            (question_id, part, image_data),
        )

    async def cache_file_id(
        self, question_id: int, part: str, telegram_file_id: str
    ) -> None:
        await self._conn.execute(
            "UPDATE images SET telegram_file_id = %s"
            " WHERE question_id = %s AND part = %s",
            (telegram_file_id, question_id, part),
        )

    # -- topic mappings (used by populate_db.py) -----------------------------

    async def delete_topic(
        self,
        option_id: int,
        question_number: int,
        part: str,
        topic_name: str,
    ) -> None:
        table = _part_table(part)
        await self._conn.execute(
            "DELETE FROM question_topics"
            " WHERE part = %s AND topic_name = %s AND question_id IN"
            f" (SELECT q.question_id FROM {table} q"
            "  WHERE q.option_id = %s AND q.question_number = %s)",
            (part, topic_name, option_id, question_number),
        )

    async def upsert_topic(
        self,
        option_id: int,
        question_number: int,
        part: str,
        topic_name: str,
    ) -> None:
        table = _part_table(part)
        await self._conn.execute(
            "INSERT INTO question_topics (question_id, part, topic_name)"
            f" SELECT q.question_id, %s, %s FROM {table} q"
            "  WHERE q.option_id = %s AND q.question_number = %s"
            " ON CONFLICT (question_id, part, topic_name) DO NOTHING",
            (part, topic_name, option_id, question_number),
        )

    async def count_topics(self) -> int:
        cur = await self._conn.execute("SELECT COUNT(*) AS n FROM question_topics")
        row = await cur.fetchone()
        assert row is not None
        return row["n"]

    # -- queries -------------------------------------------------------------

    async def get(self, question_id: int, part: str) -> Question:
        base = _question_base(_validate_part(part))
        cur = await self._conn.execute(
            base + " WHERE q.question_id = %s", (question_id,)
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        return _row_to_question(row)

    async def list_for_option(self, option_id: int, part: str) -> list[Question]:
        base = _question_base(_validate_part(part))
        cur = await self._conn.execute(
            base + " WHERE q.option_id = %s ORDER BY q.question_number",
            (option_id,),
        )
        rows = await cur.fetchall()
        return [_row_to_question(r) for r in rows]

    async def get_correct_answer(self, question_id: int, part: str) -> int | str:
        """Return the correct answer for a question.

        Part A answers are integers (option index); Part B are free-form text.
        """
        table = _part_table(part)
        cur = await self._conn.execute(
            f"SELECT answer FROM {table} WHERE question_id = %s",
            (question_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No answer for question {question_id}")
        return int(row["answer"]) if part == "A" else str(row["answer"])

    async def get_random_question_id(
        self,
        subject_id: int,
        part: str,
        exam_type: str | None = None,
    ) -> int:
        table = _part_table(part)
        params: list[Any] = [subject_id]
        where = "b.subject_id = %s"
        if exam_type:
            where += " AND o.exam_type = %s"
            params.append(exam_type)
        cur = await self._conn.execute(
            f"SELECT q.question_id FROM {table} q"
            " JOIN options o ON q.option_id = o.option_id"
            " JOIN books b ON o.book_id = b.book_id"
            f" WHERE {where}"
            " ORDER BY RANDOM() LIMIT 1",
            params,
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No {part} questions found for subject {subject_id}")
        return row["question_id"]

    async def get_topics_for_subject(self, subject_id: int) -> list[str]:
        rows = await _union_both_parts(
            self._conn,
            select_a="DISTINCT qt.topic_name",
            joins=(
                "JOIN question_topics qt"
                " ON qt.question_id = q.question_id AND qt.part = 'A'"
            ),
            joins_b=(
                "JOIN question_topics qt"
                " ON qt.question_id = q.question_id AND qt.part = 'B'"
            ),
            where="WHERE b.subject_id = %s",
            order_by="topic_name",
            params=(subject_id,),
        )
        return list(dict.fromkeys(r["topic_name"] for r in rows))

    async def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> tuple[int, str]:
        rows = await _union_both_parts(
            self._conn,
            select_a="qt.question_id, qt.part",
            joins=(
                "JOIN question_topics qt"
                " ON qt.question_id = q.question_id AND qt.part = 'A'"
            ),
            joins_b=(
                "JOIN question_topics qt"
                " ON qt.question_id = q.question_id AND qt.part = 'B'"
            ),
            where="WHERE b.subject_id = %s AND qt.topic_name = %s",
            order_by="RANDOM()",
            limit="1",
            params=(subject_id, topic_name),
        )
        if not rows:
            raise KeyError(
                f"No questions found for topic {topic_name!r} in subject {subject_id}"
            )
        return rows[0]["question_id"], rows[0]["part"]

    async def get_question_origin(self, question_id: int) -> QuestionOrigin:
        rows = await _union_both_parts(
            self._conn,
            select_a="b.year_value, o.option_number, o.exam_type",
            where="WHERE q.question_id = %s",
            params=(question_id,),
        )
        if not rows:
            raise KeyError(f"Origin not found for question {question_id}")
        r = rows[0]
        return QuestionOrigin(r["year_value"], r["option_number"], r["exam_type"])

    async def get_full(
        self, question_id: int, part: str
    ) -> tuple[Question, QuestionOrigin]:
        base = _question_full(_validate_part(part))
        cur = await self._conn.execute(
            base + " WHERE q.question_id = %s", (question_id,)
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        question = _row_to_question(row)
        origin = QuestionOrigin(
            year=row["year_value"],
            option_number=row["option_number"],
            exam_type=row["exam_type"],
        )
        return question, origin


class StudentRepository:
    """Repository for Telegram users (students)."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get_or_create(
        self, telegram_id: int, name: str, username: str | None = None
    ) -> Student:
        # One round-trip get-or-create: if the row exists, RETURNING fires via
        # the no-op DO UPDATE; if it doesn't, the INSERT fires. The row's
        # original ``created_at`` is preserved because we don't touch it.
        cur = await self._conn.execute(
            "INSERT INTO students (telegram_id, name, username)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (telegram_id)"
            " DO UPDATE SET name = EXCLUDED.name, username = EXCLUDED.username"
            " RETURNING student_id, telegram_id, name, username",
            (telegram_id, name, username),
        )
        row = await cur.fetchone()
        assert row is not None
        return Student(
            student_id=row["student_id"],
            telegram_id=row["telegram_id"],
            name=row["name"],
            username=row["username"],
        )


class SessionRepository:
    """Repository for test sessions and per-question answers."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def create(self, student_id: int, option_id: int) -> Session:
        cur = await self._conn.execute(
            "INSERT INTO test_sessions (student_id, option_id)"
            " VALUES (%s, %s)"
            " RETURNING session_id, student_id, option_id, started_at, completed_at",
            (student_id, option_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return Session(
            session_id=row["session_id"],
            student_id=row["student_id"],
            option_id=row["option_id"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    async def record_answer(
        self,
        session_id: int,
        question_id: int,
        student_answer: str,
        is_correct: bool,
        time_spent: float,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO session_answers"
            "  (session_id, question_id, student_answer, is_correct, time_spent)"
            " VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (session_id, question_id) DO NOTHING",
            (session_id, question_id, student_answer, is_correct, time_spent),
        )

    async def complete(self, session_id: int) -> TestResult:
        await self._conn.execute(
            "UPDATE test_sessions SET completed_at = NOW() WHERE session_id = %s",
            (session_id,),
        )
        return await self.get_result(session_id)

    async def get_session_info(self, session_id: int) -> SessionInfo:
        cur = await self._conn.execute(
            "SELECT s.name AS subject_name, b.year_value, o.option_number"
            "  FROM test_sessions ts"
            "  JOIN options o ON o.option_id = ts.option_id"
            "  JOIN books b ON b.book_id = o.book_id"
            "  JOIN subjects s ON s.subject_id = b.subject_id"
            " WHERE ts.session_id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Session {session_id} not found")
        return SessionInfo(row["subject_name"], row["year_value"], row["option_number"])

    async def get_wrong_answers(self, session_id: int) -> list[WrongAnswer]:
        rows = await _union_both_parts(
            self._conn,
            select_a=(
                "q.question_number AS question_number, 'A' AS part,"
                " sa.student_answer AS student_answer, q.answer::text AS correct_answer"
            ),
            select_b=(
                "q.question_number AS question_number, 'B' AS part,"
                " sa.student_answer AS student_answer, q.answer AS correct_answer"
            ),
            joins="JOIN session_answers sa ON sa.question_id = q.question_id",
            where="WHERE sa.session_id = %s AND sa.is_correct = FALSE",
            order_by="part, question_number",
            params=(session_id,),
        )
        return [
            WrongAnswer(
                question_number=r["question_number"],
                part=r["part"],
                student_answer=r["student_answer"],
                correct_answer=r["correct_answer"],
            )
            for r in rows
        ]

    async def get_result(self, session_id: int) -> TestResult:
        cur = await self._conn.execute(
            "SELECT o.exam_type, o.option_number, ts.started_at, ts.completed_at"
            "  FROM test_sessions ts"
            "  JOIN options o ON o.option_id = ts.option_id"
            " WHERE ts.session_id = %s",
            (session_id,),
        )
        session_row = await cur.fetchone()
        if session_row is None:
            raise KeyError(f"Session {session_id} not found")

        cur_a = await self._conn.execute(
            "SELECT COUNT(*) FILTER (WHERE sa.is_correct) AS correct,"
            "       COUNT(*) AS total"
            "  FROM session_answers sa"
            "  JOIN part_a_questions q ON sa.question_id = q.question_id"
            " WHERE sa.session_id = %s",
            (session_id,),
        )
        part_a = await cur_a.fetchone()

        cur_b = await self._conn.execute(
            "SELECT COUNT(*) FILTER (WHERE sa.is_correct) AS correct,"
            "       COUNT(*) AS total"
            "  FROM session_answers sa"
            "  JOIN part_b_questions q ON sa.question_id = q.question_id"
            " WHERE sa.session_id = %s",
            (session_id,),
        )
        part_b = await cur_b.fetchone()
        assert part_a is not None
        assert part_b is not None

        started = session_row["started_at"]
        completed = session_row["completed_at"]
        return TestResult(
            session_id=session_id,
            exam_type=session_row["exam_type"],
            part_a_score=part_a["correct"],
            part_b_score=part_b["correct"],
            total_score=part_a["correct"] + part_b["correct"],
            max_score=part_a["total"] + part_b["total"],
            time_spent=(completed - started).total_seconds(),
            completed_at=completed,
        )


class AuthorizedUserRepository:
    """Repository for the registration / approval workflow."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get_status(self, telegram_id: int) -> str | None:
        cur = await self._conn.execute(
            "SELECT status FROM authorized_users WHERE telegram_id = %s",
            (telegram_id,),
        )
        row = await cur.fetchone()
        return row["status"] if row else None

    async def create_request(
        self,
        telegram_id: int,
        full_name: str,
        telegram_username: str | None = None,
    ) -> AuthorizedUser:
        # Re-applying preserves the original ``created_at`` and clears the
        # handled_at / handled_by fields from any previous decision.
        cur = await self._conn.execute(
            "INSERT INTO authorized_users"
            " (telegram_id, full_name, telegram_username, status)"
            " VALUES (%s, %s, %s, 'pending')"
            " ON CONFLICT (telegram_id) DO UPDATE SET"
            "   full_name = EXCLUDED.full_name,"
            "   telegram_username = EXCLUDED.telegram_username,"
            "   status = 'pending',"
            "   handled_at = NULL,"
            "   handled_by = NULL"
            " RETURNING telegram_id, full_name, telegram_username,"
            "          status, created_at, handled_at, handled_by",
            (telegram_id, full_name, telegram_username),
        )
        row = await cur.fetchone()
        assert row is not None
        return _row_to_authorized_user(row)

    async def approve(self, telegram_id: int, admin_id: int) -> AuthorizedUser:
        return await self._set_status(telegram_id, admin_id, "approved")

    async def reject(self, telegram_id: int, admin_id: int) -> AuthorizedUser:
        return await self._set_status(telegram_id, admin_id, "rejected")

    async def _set_status(
        self, telegram_id: int, admin_id: int, status: str
    ) -> AuthorizedUser:
        cur = await self._conn.execute(
            "UPDATE authorized_users"
            " SET status = %s, handled_at = NOW(), handled_by = %s"
            " WHERE telegram_id = %s"
            " RETURNING telegram_id, full_name, telegram_username,"
            "          status, created_at, handled_at, handled_by",
            (status, admin_id, telegram_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No registration request found for {telegram_id}")
        return _row_to_authorized_user(row)

    async def delete_request(self, telegram_id: int) -> None:
        await self._conn.execute(
            "DELETE FROM authorized_users WHERE telegram_id = %s",
            (telegram_id,),
        )

    async def get_request(self, telegram_id: int) -> AuthorizedUser | None:
        cur = await self._conn.execute(
            "SELECT telegram_id, full_name, telegram_username,"
            "       status, created_at, handled_at, handled_by"
            "  FROM authorized_users WHERE telegram_id = %s",
            (telegram_id,),
        )
        row = await cur.fetchone()
        return _row_to_authorized_user(row) if row else None

    async def is_authorized(self, telegram_id: int) -> bool:
        cur = await self._conn.execute(
            "SELECT 1 FROM authorized_users"
            " WHERE telegram_id = %s AND status = 'approved'",
            (telegram_id,),
        )
        return await cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_part(part: str) -> Part:
    if part not in _PART_TABLES:
        raise ValueError(f"Unknown part {part!r}; expected 'A' or 'B'")
    return part  # type: ignore[return-value]


def _row_to_question(row: dict[str, Any]) -> Question:
    return Question(
        question_id=row["question_id"],
        part=row["part"],
        question_number=row["question_number"],
        num_options=row["a_num_options"],
        image_data=bytes(row["image_data"]) if row["image_data"] is not None else b"",
        telegram_file_id=row["telegram_file_id"],
    )


def _row_to_authorized_user(row: dict[str, Any]) -> AuthorizedUser:
    return AuthorizedUser(
        telegram_id=row["telegram_id"],
        full_name=row["full_name"],
        telegram_username=row["telegram_username"],
        status=row["status"],
        created_at=row["created_at"],
        handled_at=row["handled_at"],
        handled_by=row["handled_by"],
    )


__all__ = [
    "AuthorizedUserRepository",
    "BookRepository",
    "QuestionOrigin",
    "QuestionRepository",
    "SessionInfo",
    "SessionRepository",
    "StudentRepository",
    "SubjectRow",
    "WrongAnswer",
]
