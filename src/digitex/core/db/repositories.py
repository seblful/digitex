"""Repository classes — the only layer that touches raw SQL."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from digitex.bot.schemas import Question, Session, Student, TestResult
from digitex.core.value_objects import QuestionKey


def _get_or_create(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    where: dict,
) -> int:
    cols = ", ".join(where.keys())
    placeholders = ", ".join("?" * len(where))
    conn.execute(
        f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})",
        tuple(where.values()),
    )
    conditions = " AND ".join(f"{k} = ?" for k in where)
    return conn.execute(
        f"SELECT {id_col} FROM {table} WHERE {conditions}",
        tuple(where.values()),
    ).fetchone()[0]


class BookRepository:
    """Write-side repository for loading extraction data into the DB."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create_subject(self, name: str) -> int:
        return _get_or_create(self._conn, "subjects", "subject_id", {"name": name})

    def get_or_create_book(self, subject_id: int, year: int) -> int:
        return _get_or_create(
            self._conn, "books", "book_id",
            {"subject_id": subject_id, "year_value": year},
        )

    def get_or_create_option(self, book_id: int, option_number: int) -> int:
        return _get_or_create(
            self._conn, "options", "option_id",
            {"book_id": book_id, "option_number": option_number},
        )


class QuestionRepository:
    """Repository for questions, images, and answers."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create(self, option_id: int, key: QuestionKey) -> int:
        return _get_or_create(
            self._conn, "questions", "question_id",
            {"option_id": option_id, "part": key.part, "question_number": key.number},
        )

    def insert_image(self, question_id: int, image_data: bytes) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO images (question_id, image_data, image_order)"
            " VALUES (?, ?, 1)",
            (question_id, image_data),
        )

    def insert_answer(self, question_id: int, key: QuestionKey, answer: str) -> None:
        if key.part == "A":
            if not answer.isdigit():
                raise ValueError(f"Part A answer must be a digit, got {answer!r}")
            self._conn.execute(
                "INSERT OR IGNORE INTO part_a_answers (question_id, answer) VALUES (?, ?)",
                (question_id, int(answer)),
            )
        else:
            self._conn.execute(
                "INSERT OR IGNORE INTO part_b_answers (question_id, answer) VALUES (?, ?)",
                (question_id, answer),
            )

    def cache_file_id(self, question_id: int, telegram_file_id: str) -> None:
        self._conn.execute(
            "UPDATE images SET telegram_file_id = ? WHERE question_id = ? AND image_order = 1",
            (telegram_file_id, question_id),
        )

    def get(self, question_id: int) -> Question:
        row = self._conn.execute(
            "SELECT q.question_id, q.part, q.question_number,"
            "       i.image_data, i.telegram_file_id"
            "  FROM questions q"
            "  JOIN images i ON i.question_id = q.question_id AND i.image_order = 1"
            " WHERE q.question_id = ?",
            (question_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        return Question(
            question_id=row[0],
            part=row[1],
            question_number=row[2],
            image_data=row[3],
            telegram_file_id=row[4],
        )

    def list_for_option(self, option_id: int, part: str) -> list[Question]:
        rows = self._conn.execute(
            "SELECT q.question_id, q.part, q.question_number,"
            "       i.image_data, i.telegram_file_id"
            "  FROM questions q"
            "  JOIN images i ON i.question_id = q.question_id AND i.image_order = 1"
            " WHERE q.option_id = ? AND q.part = ?"
            " ORDER BY q.question_number",
            (option_id, part),
        ).fetchall()
        return [
            Question(
                question_id=r[0], part=r[1], question_number=r[2],
                image_data=r[3], telegram_file_id=r[4],
            )
            for r in rows
        ]

    def get_correct_answer(self, question_id: int, part: str) -> str:
        row = self._conn.execute(
            "SELECT answer FROM part_a_answers WHERE question_id = ?"
            " UNION ALL "
            "SELECT answer FROM part_b_answers WHERE question_id = ?",
            (question_id, question_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"No answer for question {question_id}")
        return str(row[0])


class StudentRepository:
    """Repository for Telegram users."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create(
        self, telegram_id: int, name: str, username: str | None = None
    ) -> Student:
        self._conn.execute(
            "INSERT OR IGNORE INTO students (telegram_id, name, username) VALUES (?, ?, ?)",
            (telegram_id, name, username),
        )
        row = self._conn.execute(
            "SELECT student_id, telegram_id, name, username FROM students WHERE telegram_id = ?",
            (telegram_id,),
        ).fetchone()
        return Student(
            student_id=row[0], telegram_id=row[1], name=row[2], username=row[3]
        )


class SessionRepository:
    """Repository for test sessions and per-question answers."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, student_id: int, book_id: int, option_number: int) -> Session:
        cur = self._conn.execute(
            "INSERT INTO test_sessions (student_id, book_id, option_number) VALUES (?, ?, ?)",
            (student_id, book_id, option_number),
        )
        row = self._conn.execute(
            "SELECT session_id, student_id, book_id, option_number, started_at, completed_at"
            "  FROM test_sessions WHERE session_id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return Session(
            session_id=row[0], student_id=row[1], book_id=row[2],
            option_number=row[3], started_at=row[4], completed_at=row[5],
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
            "UPDATE test_sessions SET completed_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )
        return self.get_result(session_id)

    def get_result(self, session_id: int) -> TestResult:
        session_row = self._conn.execute(
            "SELECT book_id, option_number, started_at, completed_at"
            "  FROM test_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            raise KeyError(f"Session {session_id} not found")

        answer_rows = self._conn.execute(
            "SELECT sa.is_correct, q.part"
            "  FROM session_answers sa"
            "  JOIN questions q ON q.question_id = sa.question_id"
            " WHERE sa.session_id = ?",
            (session_id,),
        ).fetchall()

        part_a_score = sum(1 for is_correct, part in answer_rows if is_correct and part == "A")
        part_b_score = sum(1 for is_correct, part in answer_rows if is_correct and part == "B")
        max_a = sum(1 for _, part in answer_rows if part == "A")
        max_b = sum(1 for _, part in answer_rows if part == "B")

        started = datetime.fromisoformat(session_row[2])
        completed = datetime.fromisoformat(session_row[3])

        return TestResult(
            session_id=session_id,
            part_a_score=part_a_score,
            part_b_score=part_b_score,
            total_score=part_a_score + part_b_score,
            max_score=max_a + max_b,
            time_spent=(completed - started).total_seconds(),
            completed_at=completed,
        )
