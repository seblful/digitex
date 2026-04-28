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

    def get_or_create_book(self, subject_id: int, year: int, a_num_options: int = 5) -> int:
        book_id = _get_or_create(
            self._conn, "books", "book_id",
            {"subject_id": subject_id, "year_value": year},
        )
        self._conn.execute(
            "UPDATE books SET a_num_options = ? WHERE book_id = ? AND a_num_options != ?",
            (a_num_options, book_id, a_num_options),
        )
        return book_id

    def get_or_create_option(self, book_id: int, option_number: int, exam_type: str = "CT") -> int:
        option_id = _get_or_create(
            self._conn, "options", "option_id",
            {"book_id": book_id, "option_number": option_number},
        )
        self._conn.execute(
            "UPDATE options SET exam_type = ? WHERE option_id = ? AND exam_type != ?",
            (exam_type, option_id, exam_type),
        )
        return option_id


class QuestionRepository:
    """Repository for questions, images, and answers."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

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

    def insert_image(self, question_id: int, part: str, image_data: bytes) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO images (question_id, part, image_data)"
            " VALUES (?, ?, ?)",
            (question_id, part, image_data),
        )

    def cache_file_id(self, question_id: int, part: str, telegram_file_id: str) -> None:
        self._conn.execute(
            "UPDATE images SET telegram_file_id = ? WHERE question_id = ? AND part = ?",
            (telegram_file_id, question_id, part),
        )

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

    def get(self, question_id: int, part: str) -> Question:
        table = "part_a_questions" if part == "A" else "part_b_questions"
        row = self._conn.execute(
            self._question_sql(table) + " WHERE q.question_id = ?",
            (question_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        return Question(
            question_id=row[0], part=row[1], question_number=row[2],
            num_options=row[3],
            image_data=row[4], telegram_file_id=row[5],
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
                question_id=r[0], part=r[1], question_number=r[2],
                num_options=r[3],
                image_data=r[4], telegram_file_id=r[5],
            )
            for r in rows
        ]

    def get_correct_answer(self, question_id: int, part: str) -> str:
        table = "part_a_questions" if part == "A" else "part_b_questions"
        row = self._conn.execute(
            f"SELECT answer FROM {table} WHERE question_id = ?",
            (question_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No answer for question {question_id}")
        return str(row[0])

    def get_random_question_id(self, subject_id: int, part: str, exam_type: str | None = None) -> int:
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
            raise KeyError(f"No {part} questions found for subject {subject_id}")
        return row[0]

    def get_topics_for_subject(self, subject_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT qt.topic_name"
            "  FROM question_topics qt"
            "  JOIN part_a_questions q ON q.question_id = qt.question_id AND qt.part = 'A'"
            "  JOIN options o ON q.option_id = o.option_id"
            "  JOIN books b ON o.book_id = b.book_id"
            " WHERE b.subject_id = ?"
            " UNION"
            " SELECT DISTINCT qt.topic_name"
            "  FROM question_topics qt"
            "  JOIN part_b_questions q ON q.question_id = qt.question_id AND qt.part = 'B'"
            "  JOIN options o ON q.option_id = o.option_id"
            "  JOIN books b ON o.book_id = b.book_id"
            " WHERE b.subject_id = ?"
            " ORDER BY topic_name",
            (subject_id, subject_id),
        ).fetchall()
        return [r[0] for r in rows]

    def get_random_question_id_by_topic(self, subject_id: int, topic_name: str) -> tuple[int, str]:
        row = self._conn.execute(
            "SELECT * FROM ("
            "  SELECT qt.question_id, qt.part"
            "    FROM question_topics qt"
            "    JOIN part_a_questions q ON q.question_id = qt.question_id AND qt.part = 'A'"
            "    JOIN options o ON q.option_id = o.option_id"
            "    JOIN books b ON o.book_id = b.book_id"
            "   WHERE b.subject_id = ? AND qt.topic_name = ?"
            "   UNION ALL"
            "  SELECT qt.question_id, qt.part"
            "    FROM question_topics qt"
            "    JOIN part_b_questions q ON q.question_id = qt.question_id AND qt.part = 'B'"
            "    JOIN options o ON q.option_id = o.option_id"
            "    JOIN books b ON o.book_id = b.book_id"
            "   WHERE b.subject_id = ? AND qt.topic_name = ?"
            ") ORDER BY RANDOM() LIMIT 1",
            (subject_id, topic_name, subject_id, topic_name),
        ).fetchone()
        if row is None:
            raise KeyError(f"No questions found for topic {topic_name!r} in subject {subject_id}")
        return row[0], row[1]

    def get_question_origin(self, question_id: int) -> tuple[int, int, str]:
        """Return (year, option_number, exam_type) for a question."""
        row = self._conn.execute(
            "SELECT b.year_value, o.option_number, o.exam_type"
            "  FROM part_a_questions q"
            "  JOIN options o ON q.option_id = o.option_id"
            "  JOIN books b ON o.book_id = b.book_id"
            " WHERE q.question_id = ?"
            " UNION ALL"
            " SELECT b.year_value, o.option_number, o.exam_type"
            "  FROM part_b_questions q"
            "  JOIN options o ON q.option_id = o.option_id"
            "  JOIN books b ON o.book_id = b.book_id"
            " WHERE q.question_id = ?",
            (question_id, question_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Origin not found for question {question_id}")
        return row[0], row[1], row[2]


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

    def create(self, student_id: int, book_id: int, option_number: int, exam_type: str = "CT") -> Session:
        cur = self._conn.execute(
            "INSERT INTO test_sessions (student_id, book_id, option_number, exam_type)"
            " VALUES (?, ?, ?, ?)",
            (student_id, book_id, option_number, exam_type),
        )
        row = self._conn.execute(
            "SELECT session_id, student_id, book_id, option_number,"
            "       exam_type, started_at, completed_at"
            "  FROM test_sessions WHERE session_id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return Session(
            session_id=row[0], student_id=row[1], book_id=row[2],
            option_number=row[3], exam_type=row[4], started_at=row[5],
            completed_at=row[6],
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
            "SELECT book_id, option_number, exam_type, started_at, completed_at"
            "  FROM test_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            raise KeyError(f"Session {session_id} not found")

        answer_rows = self._conn.execute(
            "SELECT sa.is_correct, 'A' FROM session_answers sa"
            "  JOIN part_a_questions q ON q.question_id = sa.question_id"
            " WHERE sa.session_id = ?"
            " UNION ALL"
            " SELECT sa.is_correct, 'B' FROM session_answers sa"
            "  JOIN part_b_questions q ON q.question_id = sa.question_id"
            " WHERE sa.session_id = ?",
            (session_id, session_id),
        ).fetchall()

        part_a_score = sum(1 for is_correct, part in answer_rows if is_correct and part == "A")
        part_b_score = sum(1 for is_correct, part in answer_rows if is_correct and part == "B")
        max_a = sum(1 for _, part in answer_rows if part == "A")
        max_b = sum(1 for _, part in answer_rows if part == "B")

        started = datetime.fromisoformat(session_row[3])
        completed = datetime.fromisoformat(session_row[4])

        return TestResult(
            session_id=session_id,
            exam_type=session_row[2],
            part_a_score=part_a_score,
            part_b_score=part_b_score,
            total_score=part_a_score + part_b_score,
            max_score=max_a + max_b,
            time_spent=(completed - started).total_seconds(),
            completed_at=completed,
        )
