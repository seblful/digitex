"""Repository classes — the only layer that touches raw SQL."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import NamedTuple

from digitex.bot.schemas import Question, Session, Student, TestResult
from digitex.core.value_objects import QuestionKey


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
# Helpers
# ---------------------------------------------------------------------------


def _get_or_create(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    where: dict,
) -> int:
    """Insert or get an existing row, returning its id.

    Uses ``ON CONFLICT … DO UPDATE … RETURNING`` so only one round-trip is
    needed (SQLite ≥ 3.35).
    """
    cols = list(where.keys())
    values = list(where.values())
    placeholders = ", ".join("?" * len(cols))
    col_list = ", ".join(cols)
    conflict_cols = ", ".join(cols)
    set_clause = ", ".join(f"{c} = excluded.{c}" for c in cols)
    row = conn.execute(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        f" ON CONFLICT({conflict_cols}) DO UPDATE SET {set_clause}"
        f" RETURNING {id_col}",
        values,
    ).fetchone()
    return row[0]


def _union_both_parts(
    conn: sqlite3.Connection,
    select_a: str,
    joins: str = "",
    where: str = "",
    order_by: str = "",
    limit: str = "",
    params: tuple = (),
    select_b: str | None = None,
    joins_b: str | None = None,
) -> list:
    """Execute a UNION-ALL query across ``part_a_*`` / ``part_b_*``.

    The template always includes the standard ``JOIN options o …`` and
    ``JOIN books b`` so that ``o.*`` and ``b.*`` columns are available.

    *select_b* / *joins_b* override the first-half values when the two
    halves differ (e.g. a hard-coded part literal ``'A'`` vs ``'B'``).
    """
    if select_b is None:
        select_b = select_a
    if joins_b is None:
        joins_b = joins

    base = (
        f"SELECT {{select}}"
        f" FROM {{table}} q"
        f" JOIN options o ON q.option_id = o.option_id"
        f" JOIN books b ON o.book_id = b.book_id"
        f" {{joins}}"
        f" {{where}}"
    )
    union = (
        base.format(select=select_a, table="part_a_questions", joins=joins, where=where)
        + " UNION ALL "
        + base.format(select=select_b, table="part_b_questions", joins=joins_b, where=where)
    )
    # SQLite requires a subquery wrapper for ORDER BY / LIMIT on compound
    # SELECTs when the ordering expression is not a result-set column.
    if order_by or limit:
        sql = f"SELECT * FROM ({union})"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"
    else:
        sql = union
    return conn.execute(sql, params + params).fetchall()


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class BookRepository:
    """Write-side repository for loading extraction data into the DB."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create_subject(self, name: str) -> int:
        return _get_or_create(self._conn, "subjects", "subject_id", {"name": name})

    def get_or_create_book(
        self, subject_id: int, year: int, a_num_options: int = 5
    ) -> int:
        book_id = _get_or_create(
            self._conn,
            "books",
            "book_id",
            {"subject_id": subject_id, "year_value": year},
        )
        self._conn.execute(
            "UPDATE books SET a_num_options = ? WHERE book_id = ? AND a_num_options != ?",
            (a_num_options, book_id, a_num_options),
        )
        return book_id

    def get_or_create_option(
        self, book_id: int, option_number: int, exam_type: str = "CT"
    ) -> int:
        option_id = _get_or_create(
            self._conn,
            "options",
            "option_id",
            {"book_id": book_id, "option_number": option_number},
        )
        self._conn.execute(
            "UPDATE options SET exam_type = ? WHERE option_id = ? AND exam_type != ?",
            (exam_type, option_id, exam_type),
        )
        return option_id

    def list_subjects(self) -> list[SubjectRow]:
        rows = self._conn.execute(
            "SELECT subject_id, name FROM subjects ORDER BY name"
        ).fetchall()
        return [SubjectRow(r[0], r[1]) for r in rows]

    def list_years(self, subject_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT year_value FROM books WHERE subject_id = ? ORDER BY year_value DESC",
            (subject_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def list_options(self, book_id: int, exam_type: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT option_number FROM options"
            " WHERE book_id = ? AND exam_type = ? ORDER BY option_number",
            (book_id, exam_type),
        ).fetchall()
        return [r[0] for r in rows]

    def get_option_id(self, book_id: int, option_number: int) -> int:
        row = self._conn.execute(
            "SELECT option_id FROM options WHERE book_id = ? AND option_number = ?",
            (book_id, option_number),
        ).fetchone()
        if row is None:
            raise KeyError(f"Option {option_number} not found for book {book_id}")
        return row[0]


class QuestionRepository:
    """Repository for questions, images, and answers."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- helpers ---------------------------------------------------------------

    def _question_sql(self, table: str) -> str:
        part_literal = "'A'" if table == "part_a_questions" else "'B'"
        return (
            f"SELECT q.question_id, {part_literal} AS part,"
            f"       q.question_number, b.a_num_options,"
            f"       i.image_data, i.telegram_file_id"
            f"  FROM {table} q"
            f"  JOIN options o ON q.option_id = o.option_id"
            f"  JOIN books b ON o.book_id = b.book_id"
            f"  JOIN images i ON i.question_id = q.question_id AND i.part = {part_literal}"
        )

    # -- CRUD ------------------------------------------------------------------

    def get_or_create(self, option_id: int, key: QuestionKey, answer: str) -> int:
        if key.part == "A":
            if not answer.isdigit():
                raise ValueError(f"Part A answer must be a digit, got {answer!r}")
            answer_val = int(answer)
            table = "part_a_questions"
        else:
            answer_val = answer
            table = "part_b_questions"

        self._conn.execute(
            f"INSERT OR IGNORE INTO {table} (option_id, question_number, answer)"
            " VALUES (?, ?, ?)",
            (option_id, key.number, answer_val),
        )

        qid = self._conn.execute(
            f"SELECT question_id FROM {table} WHERE option_id = ? AND question_number = ?",
            (option_id, key.number),
        ).fetchone()[0]

        self._conn.execute(
            f"UPDATE {table} SET answer = ? WHERE question_id = ? AND answer != ?",
            (answer_val, qid, answer_val),
        )

        return qid

    def insert_image(
        self, question_id: int, part: str, image_data: bytes
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO images (question_id, part, image_data)"
            " VALUES (?, ?, ?)",
            (question_id, part, image_data),
        )

    def cache_file_id(
        self, question_id: int, part: str, telegram_file_id: str
    ) -> None:
        self._conn.execute(
            "UPDATE images SET telegram_file_id = ? WHERE question_id = ? AND part = ?",
            (telegram_file_id, question_id, part),
        )

    # -- queries ---------------------------------------------------------------

    def get(self, question_id: int, part: str) -> Question:
        table = "part_a_questions" if part == "A" else "part_b_questions"
        row = self._conn.execute(
            self._question_sql(table) + " WHERE q.question_id = ?",
            (question_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        return Question(
            question_id=row[0],
            part=row[1],
            question_number=row[2],
            num_options=row[3],
            image_data=row[4],
            telegram_file_id=row[5],
        )

    def list_for_option(self, option_id: int, part: str) -> list[Question]:
        table = "part_a_questions" if part == "A" else "part_b_questions"
        rows = self._conn.execute(
            self._question_sql(table)
            + " WHERE q.option_id = ? ORDER BY q.question_number",
            (option_id,),
        ).fetchall()
        return [
            Question(
                question_id=r[0],
                part=r[1],
                question_number=r[2],
                num_options=r[3],
                image_data=r[4],
                telegram_file_id=r[5],
            )
            for r in rows
        ]

    def get_correct_answer(self, question_id: int, part: str) -> int | str:
        """Return the correct answer for a question.

        Part A answers are *integers* (option index); Part B answers are
        free-form *strings*.
        """
        table = "part_a_questions" if part == "A" else "part_b_questions"
        row = self._conn.execute(
            f"SELECT answer FROM {table} WHERE question_id = ?",
            (question_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No answer for question {question_id}")
        return int(row[0]) if part == "A" else str(row[0])

    def get_random_question_id(
        self, subject_id: int, part: str, exam_type: str | None = None
    ) -> int:
        """Pick a random question for the given subject / part / exam type."""
        table = "part_a_questions" if part == "A" else "part_b_questions"
        where = "b.subject_id = ?"
        params: list = [subject_id]
        if exam_type:
            where += " AND o.exam_type = ?"
            params.append(exam_type)
        row = self._conn.execute(
            f"SELECT q.question_id FROM {table} q"
            " JOIN options o ON q.option_id = o.option_id"
            f" JOIN books b ON o.book_id = b.book_id"
            f" WHERE {where}"
            " ORDER BY RANDOM() LIMIT 1",
            params,
        ).fetchone()
        if row is None:
            raise KeyError(
                f"No {part} questions found for subject {subject_id}"
            )
        return row[0]

    def get_topics_for_subject(self, subject_id: int) -> list[str]:
        rows = _union_both_parts(self._conn,
            select_a="DISTINCT qt.topic_name",
            joins="JOIN question_topics qt ON qt.question_id = q.question_id AND qt.part = 'A'",
            joins_b="JOIN question_topics qt ON qt.question_id = q.question_id AND qt.part = 'B'",
            where="WHERE b.subject_id = ?",
            order_by="topic_name",
            params=(subject_id,),
        )
        return [r[0] for r in rows]

    def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> tuple[int, str]:
        rows = _union_both_parts(self._conn,
            select_a="qt.question_id, qt.part",
            joins=(
                "JOIN question_topics qt"
                " ON qt.question_id = q.question_id AND qt.part = 'A'"
            ),
            joins_b=(
                "JOIN question_topics qt"
                " ON qt.question_id = q.question_id AND qt.part = 'B'"
            ),
            where="WHERE b.subject_id = ? AND qt.topic_name = ?",
            order_by="RANDOM()",
            limit="1",
            params=(subject_id, topic_name),
        )
        if not rows:
            raise KeyError(
                f"No questions found for topic {topic_name!r} in subject {subject_id}"
            )
        return rows[0][0], rows[0][1]

    def get_question_origin(self, question_id: int) -> QuestionOrigin:
        """Return (year, option_number, exam_type) for a question."""
        rows = _union_both_parts(self._conn,
            select_a="b.year_value, o.option_number, o.exam_type",
            where="WHERE q.question_id = ?",
            params=(question_id,),
        )
        if not rows:
            raise KeyError(f"Origin not found for question {question_id}")
        return QuestionOrigin(*rows[0])


class StudentRepository:
    """Repository for Telegram users."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create(
        self, telegram_id: int, name: str, username: str | None = None
    ) -> Student:
        self._conn.execute(
            "INSERT OR IGNORE INTO students (telegram_id, name, username)"
            " VALUES (?, ?, ?)",
            (telegram_id, name, username),
        )
        row = self._conn.execute(
            "SELECT student_id, telegram_id, name, username"
            "  FROM students WHERE telegram_id = ?",
            (telegram_id,),
        ).fetchone()
        return Student(
            student_id=row[0],
            telegram_id=row[1],
            name=row[2],
            username=row[3],
        )


class SessionRepository:
    """Repository for test sessions and per-question answers."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, student_id: int, option_id: int) -> Session:
        cur = self._conn.execute(
            "INSERT INTO test_sessions (student_id, option_id)"
            " VALUES (?, ?)",
            (student_id, option_id),
        )
        row = self._conn.execute(
            "SELECT session_id, student_id, option_id, started_at, completed_at"
            "  FROM test_sessions WHERE session_id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return Session(
            session_id=row[0],
            student_id=row[1],
            option_id=row[2],
            started_at=row[3],
            completed_at=row[4],
        )

    def record_answer(
        self,
        session_id: int,
        question_id: int,
        student_answer: str,
        is_correct: bool,
        time_spent: float,
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO session_answers"
            "  (session_id, question_id, student_answer, is_correct, time_spent)"
            "  VALUES (?, ?, ?, ?, ?)",
            (session_id, question_id, student_answer, int(is_correct), time_spent),
        )

    def complete(self, session_id: int) -> TestResult:
        self._conn.execute(
            "UPDATE test_sessions SET completed_at = CURRENT_TIMESTAMP"
            " WHERE session_id = ?",
            (session_id,),
        )
        return self.get_result(session_id)

    def get_session_info(self, session_id: int) -> SessionInfo:
        row = self._conn.execute(
            "SELECT s.name, b.year_value, o.option_number"
            "  FROM test_sessions ts"
            "  JOIN options o ON o.option_id = ts.option_id"
            "  JOIN books b ON b.book_id = o.book_id"
            "  JOIN subjects s ON s.subject_id = b.subject_id"
            " WHERE ts.session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Session {session_id} not found")
        return SessionInfo(row[0], row[1], row[2])

    def get_wrong_answers(self, session_id: int) -> list[WrongAnswer]:
        rows = _union_both_parts(self._conn,
            select_a="q.question_number, 'A', sa.student_answer, q.answer",
            joins="JOIN session_answers sa ON sa.question_id = q.question_id",
            where="WHERE sa.session_id = ? AND sa.is_correct = 0",
            order_by="2, 1",
            params=(session_id,),
            select_b="q.question_number, 'B', sa.student_answer, q.answer",
        )
        return [WrongAnswer(*r) for r in rows]

    def get_result(self, session_id: int) -> TestResult:
        session_row = self._conn.execute(
            "SELECT o.exam_type, o.option_number, ts.started_at, ts.completed_at"
            "  FROM test_sessions ts"
            "  JOIN options o ON o.option_id = ts.option_id"
            " WHERE ts.session_id = ?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            raise KeyError(f"Session {session_id} not found")

        answer_rows = _union_both_parts(self._conn,
            select_a="sa.is_correct, 'A'",
            joins="JOIN session_answers sa ON sa.question_id = q.question_id",
            where="WHERE sa.session_id = ?",
            params=(session_id,),
            select_b="sa.is_correct, 'B'",
        )

        part_a_score = sum(1 for c, _ in answer_rows if c)
        part_b_score = sum(1 for c, p in answer_rows if c and p == "B")
        max_a = sum(1 for _, p in answer_rows if p == "A")
        max_b = sum(1 for _, p in answer_rows if p == "B")

        exam_type = session_row[0]
        started = datetime.fromisoformat(session_row[2])
        completed = datetime.fromisoformat(session_row[3])

        return TestResult(
            session_id=session_id,
            exam_type=exam_type,
            part_a_score=part_a_score,
            part_b_score=part_b_score,
            total_score=part_a_score + part_b_score,
            max_score=max_a + max_b,
            time_spent=(completed - started).total_seconds(),
            completed_at=completed,
        )
