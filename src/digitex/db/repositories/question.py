"""Questions, images, answers and topics — five roles over one connection.

This was one ``QuestionRepository`` with fourteen methods spanning five
unrelated jobs: reading a question to serve, drawing one at random, the topic
map, Telegram's file_id cache, and loading the corpus in. No caller wanted more
than three of them — the random-question handler wanted two — and every fake
written against it had to carry the whole surface.

They are five classes now, sharing the connection the unit of work opened. The
SQL did not change; what changed is that a caller can be handed the role it
uses, which is what lets a use case declare the two methods it needs instead of
depending on all fourteen.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from digitex.db.mapping import row_to_model
from digitex.domain.answer import AnswerKey
from digitex.domain.entities import Question, QuestionOrigin

if TYPE_CHECKING:
    from digitex.db.mapping import DictConn
    from digitex.domain.entities import ExamType, Part, QuestionKey

# ``question_id`` identifies a question on its own — the part is a column of the
# row it names, so nothing that references a question restates it.
_QUESTION_SELECT = (
    "SELECT q.question_id, q.part, q.question_number,"
    "       i.object_key, i.telegram_file_id"
    "  FROM questions q"
    "  LEFT JOIN images i ON i.question_id = q.question_id"
)

# Only the origin needs the book a question came from.
_QUESTION_WITH_ORIGIN_SELECT = (
    "SELECT q.question_id, q.part, q.question_number,"
    "       i.object_key, i.telegram_file_id,"
    "       b.year_value, o.option_number, o.exam_type"
    "  FROM questions q"
    "  JOIN options o ON q.option_id = o.option_id"
    "  JOIN books b ON o.book_id = b.book_id"
    "  LEFT JOIN images i ON i.question_id = q.question_id"
)


def _row_to_question(row: dict[str, Any]) -> Question:
    """Build a ``Question``, image key and all.

    The key is a short string, so there is no reason to leave it behind and
    fetch it in a second round-trip the way an image payload would deserve:
    every question select carries everything a render needs.
    """
    return row_to_model(
        {
            "question_id": row["question_id"],
            "part": row["part"],
            "question_number": row["question_number"],
            "image_key": row["object_key"],
            "telegram_file_id": row["telegram_file_id"],
        },
        Question,
    )


class QuestionCatalog:
    """Reading a question the bot is about to serve.

    Everything a render needs comes back in one row — the image key is a short
    string, so leaving it behind would cost a second round-trip for nothing.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

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

    async def get_correct_answer(self, question_id: int) -> AnswerKey:
        """Return the question's answer key; its value is None without one.

        Part A values are integers (option index); Part B are free-form text.
        The part is read from the row rather than passed in, so a caller cannot
        ask for the answer under the wrong one — the key carries its own
        matching rules from here on.
        """
        cur = await self._conn.execute(
            "SELECT part, answer FROM questions WHERE question_id = %s",
            (question_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        if row["answer"] is None:
            return AnswerKey(part=row["part"], value=None)
        return AnswerKey(
            part=row["part"],
            value=int(row["answer"]) if row["part"] == "A" else str(row["answer"]),
        )

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


class QuestionDraw:
    """Picking a question at random, for the modes with no playlist.

    Random testing draws one question at a time rather than walking an option,
    so this is a different question of the corpus than the catalog asks.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

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


class TopicIndex:
    """The subject-level topic map.

    Two subjects may use the same topic name without sharing questions, which
    is why a topic is created against a subject rather than looked up by name.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

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


class FileIdCache:
    """Telegram's own id for an image already uploaded once.

    The bot parks a new ``file_id`` in FSM state and settles it inside the next
    round's transaction, so this is written far from where it is earned. One
    method, because that debt is the whole of what this is for.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def cache_file_id(self, question_id: int, telegram_file_id: str) -> None:
        await self._conn.execute(
            "UPDATE images SET telegram_file_id = %s WHERE question_id = %s",
            (telegram_file_id, question_id),
        )


class QuestionCorpus:
    """Writing the extraction output into the database.

    The seeding path, not a serving one: nothing the bot does while a student
    is answering reaches any of this.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

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

    async def set_image(
        self, question_id: int, object_key: str, content_hash: str
    ) -> None:
        """Point a question at its image file on disk.

        A changed image drops the cached file_id: that id names an image already
        uploaded to Telegram, and send_question prefers it over the file — so a
        re-seed that kept it would serve the superseded image forever. The key
        alone cannot detect the change, because re-extracting a question rewrites
        the same path; the hash is what makes it visible. The DISTINCT guard is
        what makes clearing the cache safe to pair with the update: an idempotent
        re-run never reaches the SET, so a valid cache survives it.
        """
        await self._conn.execute(
            "INSERT INTO images (question_id, object_key, content_hash)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (question_id)"
            " DO UPDATE SET object_key = EXCLUDED.object_key,"
            " content_hash = EXCLUDED.content_hash,"
            " telegram_file_id = NULL"
            " WHERE images.content_hash IS DISTINCT FROM EXCLUDED.content_hash"
            "    OR images.object_key IS DISTINCT FROM EXCLUDED.object_key",
            (question_id, object_key, content_hash),
        )

    async def list_images(self) -> list[tuple[str, str]]:
        """Every ``(object_key, content_hash)`` the corpus claims to have.

        The whole table, because the reconcile check has to answer "which files
        is nothing pointing at?" as well as "which rows point at nothing" — and
        the second question cannot be asked one row at a time.
        """
        cur = await self._conn.execute(
            "SELECT object_key, content_hash FROM images ORDER BY object_key"
        )
        rows = await cur.fetchall()
        return [(r["object_key"], r["content_hash"]) for r in rows]
