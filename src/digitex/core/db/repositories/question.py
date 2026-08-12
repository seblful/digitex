"""Repository for questions, images, answers, and topic mappings."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from digitex.core.db.mapping import row_to_model
from digitex.core.domain import Part, Question, QuestionOrigin

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn
    from digitex.core.domain import ExamType, QuestionKey

# Questions live in one table with a ``part`` column, so the part is always a
# bound parameter and never interpolated into SQL.
_QUESTION_SELECT = (
    "SELECT q.question_id, q.part, q.question_number, i.telegram_file_id"
    "  FROM questions q"
    "  LEFT JOIN images i"
    "    ON i.question_id = q.question_id AND i.part = q.part"
)

# Only the origin needs the book a question came from.
_QUESTION_WITH_ORIGIN_SELECT = (
    "SELECT q.question_id, q.part, q.question_number, i.telegram_file_id,"
    "       b.year_value, o.option_number, o.exam_type"
    "  FROM questions q"
    "  JOIN options o ON q.option_id = o.option_id"
    "  JOIN books b ON o.book_id = b.book_id"
    "  LEFT JOIN images i"
    "    ON i.question_id = q.question_id AND i.part = q.part"
)


def _row_to_question(row: dict[str, Any]) -> Question:
    """Build a metadata-only ``Question`` (no ``image_data``).

    Question selects do not carry the BYTEA payload — fetch it explicitly with
    :meth:`QuestionRepository.get_image` when a cache miss requires uploading a
    fresh image.
    """
    return row_to_model(
        {
            "question_id": row["question_id"],
            "part": row["part"],
            "question_number": row["question_number"],
            "telegram_file_id": row["telegram_file_id"],
        },
        Question,
    )


class QuestionRepository:
    """Repository for questions, images, answers, and topic mappings."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    # -- CRUD ----------------------------------------------------------------

    async def get_or_create(self, option_id: int, key: QuestionKey, answer: str) -> int:
        """Insert or update one question's answer key, returning its id.

        Part A answers are option indices, so a non-numeric one is a bad answer
        key rather than a storable value. Both parts are stored as text; the
        numeric reading happens in :meth:`get_correct_answer`.
        """
        if key.part == "A" and not answer.isdigit():
            raise ValueError(f"Part A answer must be a digit, got {answer!r}")

        cur = await self._conn.execute(
            "INSERT INTO questions (option_id, part, question_number, answer)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (option_id, part, question_number)"
            " DO UPDATE SET answer = EXCLUDED.answer"
            " RETURNING question_id",
            (option_id, key.part, key.number, answer),
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
        """Unmap one topic from one question.

        No production caller today — ``upsert_topic`` is idempotent, so the
        populate script needs no delete-before-insert. Kept as the topic
        mapping's other half.
        """
        await self._conn.execute(
            "DELETE FROM question_topics"
            " WHERE part = %s AND topic_name = %s AND question_id IN"
            " (SELECT q.question_id FROM questions q"
            "  WHERE q.option_id = %s AND q.part = %s AND q.question_number = %s)",
            (part, topic_name, option_id, part, question_number),
        )

    async def upsert_topic(
        self,
        option_id: int,
        question_number: int,
        part: str,
        topic_name: str,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO question_topics (question_id, part, topic_name)"
            " SELECT q.question_id, q.part, %s FROM questions q"
            "  WHERE q.option_id = %s AND q.part = %s AND q.question_number = %s"
            " ON CONFLICT (question_id, part, topic_name) DO NOTHING",
            (topic_name, option_id, part, question_number),
        )

    async def count_topics(self) -> int:
        cur = await self._conn.execute("SELECT COUNT(*) AS n FROM question_topics")
        row = await cur.fetchone()
        assert row is not None
        return row["n"]

    # -- queries -------------------------------------------------------------

    async def get(self, question_id: int, part: str) -> Question:
        cur = await self._conn.execute(
            _QUESTION_SELECT + " WHERE q.question_id = %s AND q.part = %s",
            (question_id, part),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        return _row_to_question(row)

    async def list_ids_for_option(self, option_id: int) -> list[tuple[int, Part]]:
        """Return ``(question_id, part)`` pairs for an option, A then B.

        Used to build the testing-loop playlist from the option screen — only
        ids are needed up front; metadata and images are fetched per-question
        as the student advances.
        """
        cur = await self._conn.execute(
            "SELECT question_id, part FROM questions"
            " WHERE option_id = %s"
            " ORDER BY part, question_number",
            (option_id,),
        )
        rows = await cur.fetchall()
        return [(r["question_id"], r["part"]) for r in rows]

    async def get_image(self, question_id: int, part: str) -> bytes:
        """Fetch the raw image bytes for a question.

        Separate from :meth:`get` so callers that only need to render a cached
        Telegram ``file_id`` do not pull megabytes from the DB.
        """
        cur = await self._conn.execute(
            "SELECT image_data FROM images WHERE question_id = %s AND part = %s",
            (question_id, part),
        )
        row = await cur.fetchone()
        if row is None or row["image_data"] is None:
            raise KeyError(f"No image stored for question {question_id} part {part}")
        return bytes(row["image_data"])

    async def get_correct_answer(self, question_id: int, part: str) -> int | str:
        """Return the correct answer for a question.

        Part A answers are integers (option index); Part B are free-form text.
        """
        cur = await self._conn.execute(
            "SELECT answer FROM questions WHERE question_id = %s AND part = %s",
            (question_id, part),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No answer for question {question_id}")
        return int(row["answer"]) if part == "A" else str(row["answer"])

    async def get_random_question_id(
        self,
        subject_id: int,
        part: str,
        exam_type: ExamType | None = None,
    ) -> int:
        # ORDER BY RANDOM() forces a full scan + per-row random() evaluation.
        # COUNT + OFFSET scans only OFFSET+1 rows on the second query and keeps
        # the first query indexable.
        params: list[Any] = [subject_id, part]
        where = "b.subject_id = %s AND q.part = %s"
        if exam_type:
            where += " AND o.exam_type = %s"
            params.append(exam_type)
        base = (
            " FROM questions q"
            " JOIN options o ON q.option_id = o.option_id"
            " JOIN books b ON o.book_id = b.book_id"
            f" WHERE {where}"
        )
        cur = await self._conn.execute("SELECT COUNT(*) AS n" + base, params)
        row = await cur.fetchone()
        n = row["n"] if row else 0
        if n == 0:
            raise KeyError(f"No {part} questions found for subject {subject_id}")
        offset = secrets.randbelow(n)
        cur = await self._conn.execute(
            "SELECT q.question_id" + base + " ORDER BY q.question_id OFFSET %s LIMIT 1",
            [*params, offset],
        )
        row = await cur.fetchone()
        assert row is not None
        return row["question_id"]

    async def get_topics_for_subject(self, subject_id: int) -> list[str]:
        cur = await self._conn.execute(
            "SELECT DISTINCT qt.topic_name"
            " FROM questions q"
            " JOIN options o ON q.option_id = o.option_id"
            " JOIN books b ON o.book_id = b.book_id"
            " JOIN question_topics qt"
            "   ON qt.question_id = q.question_id AND qt.part = q.part"
            " WHERE b.subject_id = %s"
            " ORDER BY qt.topic_name",
            (subject_id,),
        )
        rows = await cur.fetchall()
        return [r["topic_name"] for r in rows]

    async def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> tuple[int, str]:
        # Topic-filtered sets are small (rarely more than a few dozen rows).
        # Pull the candidate ids and pick one client-side — cheaper than
        # ORDER BY RANDOM() over the topic join.
        cur = await self._conn.execute(
            "SELECT qt.question_id, qt.part"
            " FROM questions q"
            " JOIN options o ON q.option_id = o.option_id"
            " JOIN books b ON o.book_id = b.book_id"
            " JOIN question_topics qt"
            "   ON qt.question_id = q.question_id AND qt.part = q.part"
            " WHERE b.subject_id = %s AND qt.topic_name = %s",
            (subject_id, topic_name),
        )
        rows = await cur.fetchall()
        if not rows:
            raise KeyError(
                f"No questions found for topic {topic_name!r} in subject {subject_id}"
            )
        pick = rows[secrets.randbelow(len(rows))]
        return pick["question_id"], pick["part"]

    async def get_full(
        self, question_id: int, part: str
    ) -> tuple[Question, QuestionOrigin]:
        cur = await self._conn.execute(
            _QUESTION_WITH_ORIGIN_SELECT + " WHERE q.question_id = %s AND q.part = %s",
            (question_id, part),
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


__all__ = ["QuestionRepository"]
