"""Repository for questions, images, answers, and topic mappings."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from digitex.core.db.mapping import row_to_model
from digitex.core.domain import Question, QuestionOrigin

if TYPE_CHECKING:
    from digitex.core.db.mapping import DictConn
    from digitex.core.domain import ExamType, Part, QuestionKey

# ``question_id`` identifies a question on its own — the part is a column of the
# row it names, so nothing that references a question restates it.
_QUESTION_SELECT = (
    "SELECT q.question_id, q.part, q.question_number, i.telegram_file_id"
    "  FROM questions q"
    "  LEFT JOIN images i ON i.question_id = q.question_id"
)

# Only the origin needs the book a question came from.
_QUESTION_WITH_ORIGIN_SELECT = (
    "SELECT q.question_id, q.part, q.question_number, i.telegram_file_id,"
    "       b.year_value, o.option_number, o.exam_type"
    "  FROM questions q"
    "  JOIN options o ON q.option_id = o.option_id"
    "  JOIN books b ON o.book_id = b.book_id"
    "  LEFT JOIN images i ON i.question_id = q.question_id"
)

# Topic mappings are addressed by the question's natural key: the populate
# script walks options and question numbers off the filesystem and never holds
# an id.
_QUESTION_BY_KEY = (
    " (SELECT question_id FROM questions"
    "   WHERE option_id = %s AND part = %s AND question_number = %s)"
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

    async def get_or_create(
        self, option_id: int, key: QuestionKey, answer: str | None
    ) -> int:
        """Insert or update one question's answer key, returning its id.

        *answer* is None for a question whose key is missing or unusable: the
        question is still stored, so its image is servable, and scoring can
        never match it. Part A answers are option indices, so a non-numeric one
        is a bad answer key rather than a storable value.
        """
        if key.part == "A" and answer is not None and not answer.isdigit():
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

    async def insert_image(self, question_id: int, image_data: bytes) -> None:
        # Skip the write if the BYTEA payload hasn't changed; this avoids
        # rewriting multi-MB rows during idempotent re-runs.
        #
        # New bytes drop the cached file_id: it names an image already uploaded
        # to Telegram, and send_question prefers it over the payload — so a
        # re-seed that kept it would serve the old image forever. The DISTINCT
        # guard is what makes this safe to pair with the update: an idempotent
        # re-run never reaches the SET, so a valid cache survives it.
        await self._conn.execute(
            "INSERT INTO images (question_id, image_data)"
            " VALUES (%s, %s)"
            " ON CONFLICT (question_id)"
            " DO UPDATE SET image_data = EXCLUDED.image_data,"
            " telegram_file_id = NULL"
            " WHERE images.image_data IS DISTINCT FROM EXCLUDED.image_data",
            (question_id, image_data),
        )

    async def cache_file_id(self, question_id: int, telegram_file_id: str) -> None:
        await self._conn.execute(
            "UPDATE images SET telegram_file_id = %s WHERE question_id = %s",
            (telegram_file_id, question_id),
        )

    # -- topic mappings (used by populate_db.py) -----------------------------

    async def get_or_create_topic(self, subject_id: int, topic_name: str) -> int:
        """Return the id of a subject's topic, naming it if it is new.

        Topics are referenced by id, so the name is stored once: a rename is one
        UPDATE, and a misspelling cannot become a second topic behind a mapping.
        """
        # DO UPDATE, not DO NOTHING: RETURNING is suppressed on a conflict that
        # does nothing, and this needs the id either way. The update is a no-op.
        cur = await self._conn.execute(
            "INSERT INTO topics (subject_id, name) VALUES (%s, %s)"
            " ON CONFLICT (subject_id, name) DO UPDATE SET name = EXCLUDED.name"
            " RETURNING topic_id",
            (subject_id, topic_name),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["topic_id"]

    async def delete_topic(
        self,
        option_id: int,
        question_number: int,
        part: Part,
        topic_id: int,
    ) -> None:
        """Unmap one topic from one question.

        No production caller today — ``upsert_topic`` is idempotent, so the
        populate script needs no delete-before-insert. Kept as the topic
        mapping's other half.
        """
        await self._conn.execute(
            "DELETE FROM question_topics"
            " WHERE topic_id = %s AND question_id IN" + _QUESTION_BY_KEY,
            (topic_id, option_id, part, question_number),
        )

    async def upsert_topic(
        self,
        option_id: int,
        question_number: int,
        part: Part,
        topic_id: int,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO question_topics (question_id, topic_id)"
            " SELECT question_id, %s FROM questions"
            "  WHERE option_id = %s AND part = %s AND question_number = %s"
            " ON CONFLICT (question_id, topic_id) DO NOTHING",
            (topic_id, option_id, part, question_number),
        )

    async def count_topics(self) -> int:
        cur = await self._conn.execute("SELECT COUNT(*) AS n FROM question_topics")
        row = await cur.fetchone()
        assert row is not None
        return row["n"]

    # -- queries -------------------------------------------------------------

    async def get(self, question_id: int) -> Question:
        cur = await self._conn.execute(
            _QUESTION_SELECT + " WHERE q.question_id = %s",
            (question_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        return _row_to_question(row)

    async def list_ids_for_option(self, option_id: int) -> list[tuple[int, Part]]:
        """Return ``(question_id, part)`` pairs for an option, A then B.

        Used to build the testing-loop playlist from the option screen — only
        ids are needed up front; metadata and images are fetched per-question
        as the student advances. The part rides along because the answer guards
        need it before the question itself is loaded.
        """
        cur = await self._conn.execute(
            "SELECT question_id, part FROM questions"
            " WHERE option_id = %s"
            " ORDER BY part, question_number",
            (option_id,),
        )
        rows = await cur.fetchall()
        return [(r["question_id"], r["part"]) for r in rows]

    async def get_image(self, question_id: int) -> bytes:
        """Fetch the raw image bytes for a question.

        Separate from :meth:`get` so callers that only need to render a cached
        Telegram ``file_id`` do not pull megabytes from the DB.
        """
        cur = await self._conn.execute(
            "SELECT image_data FROM images WHERE question_id = %s",
            (question_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"No image stored for question {question_id}")
        return bytes(row["image_data"])

    async def get_correct_answer(self, question_id: int) -> int | str | None:
        """Return the correct answer for a question, or None if it has no key.

        Part A answers are integers (option index); Part B are free-form text.
        The part is read from the row rather than passed in, so a caller cannot
        ask for the answer under the wrong one.
        """
        cur = await self._conn.execute(
            "SELECT part, answer FROM questions WHERE question_id = %s",
            (question_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        if row["answer"] is None:
            return None
        return int(row["answer"]) if row["part"] == "A" else str(row["answer"])

    async def get_random_question_id(
        self,
        subject_id: int,
        part: Part,
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
        """Every topic of *subject_id* that has at least one question mapped.

        A topic with no questions would open a round with nothing to ask, so it
        is not offered.
        """
        cur = await self._conn.execute(
            "SELECT t.name FROM topics t"
            " WHERE t.subject_id = %s"
            "   AND EXISTS (SELECT 1 FROM question_topics qt"
            "                WHERE qt.topic_id = t.topic_id)"
            " ORDER BY t.name",
            (subject_id,),
        )
        rows = await cur.fetchall()
        return [r["name"] for r in rows]

    async def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> int:
        # Topic-filtered sets are small (rarely more than a few dozen rows).
        # Pull the candidate ids and pick one client-side — cheaper than
        # ORDER BY RANDOM() over the topic join. The topic carries its subject
        # and the mapping carries the ids, so neither the questions table nor
        # the books above it has to be visited to make the draw.
        cur = await self._conn.execute(
            "SELECT qt.question_id"
            " FROM topics t"
            " JOIN question_topics qt ON qt.topic_id = t.topic_id"
            " WHERE t.subject_id = %s AND t.name = %s",
            (subject_id, topic_name),
        )
        rows = await cur.fetchall()
        if not rows:
            raise KeyError(
                f"No questions found for topic {topic_name!r} in subject {subject_id}"
            )
        return rows[secrets.randbelow(len(rows))]["question_id"]

    async def get_full(self, question_id: int) -> tuple[Question, QuestionOrigin]:
        cur = await self._conn.execute(
            _QUESTION_WITH_ORIGIN_SELECT + " WHERE q.question_id = %s",
            (question_id,),
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
