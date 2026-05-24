"""Repository for questions, images, answers, and topic mappings."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from digitex.core.db.mapping import row_to_model
from digitex.core.db.repositories._common import (
    QuestionOrigin,
    _part_table,
    _question_base,
    _question_full,
    _union_both_parts,
    _validate_part,
)
from digitex.core.domain import Part, Question

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn
    from digitex.core.domain import QuestionKey


def _row_to_question(row: dict[str, Any]) -> Question:
    """Build a metadata-only ``Question`` (no ``image_data``).

    Rows produced by ``_question_base`` / ``_question_full`` no longer carry
    the BYTEA payload — fetch it explicitly with :meth:`QuestionRepository.get_image`
    when a cache miss requires uploading a fresh image.
    """
    return row_to_model(
        {
            "question_id": row["question_id"],
            "part": row["part"],
            "question_number": row["question_number"],
            "num_options": row["a_num_options"],
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

    async def list_ids_for_option(self, option_id: int) -> list[tuple[int, Part]]:
        """Return ``(question_id, part)`` pairs for an option, A then B.

        Used to build the testing-loop playlist from the option screen — only
        ids are needed up front; metadata and images are fetched per-question
        as the student advances.
        """
        rows = await _union_both_parts(
            self._conn,
            select_a="q.question_id, 'A' AS part, q.question_number",
            select_b="q.question_id, 'B' AS part, q.question_number",
            where="WHERE q.option_id = %s",
            order_by="part, question_number",
            params=(option_id,),
        )
        return [(r["question_id"], r["part"]) for r in rows]

    async def get_image(self, question_id: int, part: str) -> bytes:
        """Fetch the raw image bytes for a question.

        Separate from :meth:`get` so callers that only need to render a cached
        Telegram ``file_id`` do not pull megabytes from the DB.
        """
        _validate_part(part)
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
        # ORDER BY RANDOM() forces a full scan + per-row random() evaluation.
        # COUNT + OFFSET scans only OFFSET+1 rows on the second query and keeps
        # the first query indexable.
        table = _part_table(part)
        params: list[Any] = [subject_id]
        where = "b.subject_id = %s"
        if exam_type:
            where += " AND o.exam_type = %s"
            params.append(exam_type)
        base = (
            f" FROM {table} q"
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
        # Topic-filtered sets are small (rarely more than a few dozen rows).
        # Pull the candidate ids and pick one client-side — cheaper than
        # ORDER BY RANDOM() on the UNION-ALL of both part tables.
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
            params=(subject_id, topic_name),
        )
        if not rows:
            raise KeyError(
                f"No questions found for topic {topic_name!r} in subject {subject_id}"
            )
        pick = rows[secrets.randbelow(len(rows))]
        return pick["question_id"], pick["part"]

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


__all__ = ["QuestionRepository"]
