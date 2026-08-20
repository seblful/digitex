"""The five roles a question is addressed through, over one connection.

Reading a question to serve, drawing one at random, the topic map, Telegram's
``file_id`` cache, and loading the extraction output in are five unrelated jobs
that happen to touch the same tables. As one class they were fourteen methods no
caller wanted more than three of, and every fake written against it had to carry
the whole surface. As five, a use case can be handed the role it uses.

They are not merged and they are not layered: each takes the connection the unit
of work opened and issues its own SQL. The shapes they return live in
:mod:`digitex.domain.entities`, because callers outside this layer read them.

``question_id`` names a question on its own — the part is a column of the row it
names, not part of its address — so nothing here passes a part alongside an id,
and the part is a bound parameter wherever it does appear.
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

# Everything a render needs, in one row. The image key is a short string, so
# leaving it for a second round-trip — the way an image payload would deserve —
# would buy nothing. LEFT JOIN because a question is loadable before its image
# row is seeded.
_QUESTION_SELECT = """
    SELECT q.question_id, q.part, q.question_number,
           i.object_key, i.telegram_file_id
      FROM questions q
      LEFT JOIN images i ON i.question_id = q.question_id
"""

# The same row plus the book and option it came from. Only a randomly drawn
# question needs that, which is why the two joins are not in the select above.
_QUESTION_WITH_ORIGIN_SELECT = """
    SELECT q.question_id, q.part, q.question_number,
           i.object_key, i.telegram_file_id,
           b.year_value, o.option_number, o.exam_type
      FROM questions q
      JOIN options o ON q.option_id = o.option_id
      JOIN books b ON o.book_id = b.book_id
      LEFT JOIN images i ON i.question_id = q.question_id
"""

# Which rows a random draw may pick from. Shared by the count and the pick so
# the two queries cannot come to disagree about the size of the set they are
# indexing into.
_DRAW_SCOPE = """
      FROM questions q
      JOIN options o ON q.option_id = o.option_id
      JOIN books b ON o.book_id = b.book_id
     WHERE b.subject_id = %s AND q.part = %s
"""


def _row_to_question(row: dict[str, Any]) -> Question:
    """Build a ``Question`` from either question select.

    The one rename between the tables and the model: ``images.object_key`` is
    the question's ``image_key``. Columns the model does not name — the origin
    ones — are dropped on validation.
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
    """Reading a question the bot is about to serve."""

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get(self, question_id: int) -> Question:
        """One question, image key and cached ``file_id`` included.

        Raises:
            KeyError: If no question has that id.
        """
        cur = await self._conn.execute(
            _QUESTION_SELECT + " WHERE q.question_id = %s",
            (question_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        return _row_to_question(row)

    async def list_ids_for_option(self, option_id: int) -> list[tuple[int, Part]]:
        """The playlist for an option: ``(question_id, part)``, A then B.

        This order is the order a student answers in. Only ids are read up
        front — metadata and images are fetched per question as they advance —
        but the part rides along, because the answer guards need it before the
        question itself is loaded.
        """
        cur = await self._conn.execute(
            """
            SELECT question_id, part
              FROM questions
             WHERE option_id = %s
             ORDER BY part, question_number
            """,
            (option_id,),
        )
        rows = await cur.fetchall()
        return [(row["question_id"], row["part"]) for row in rows]

    async def get_correct_answer(self, question_id: int) -> AnswerKey:
        """The question's answer key — value None when its year shipped none.

        The part is read off the row rather than passed in, so no caller can ask
        for an answer under the wrong one. From here the key carries its own
        matching rules, which is why Part A comes back as an ``int`` (an option
        index) and Part B as ``str`` (free text) rather than as the ``TEXT``
        both are stored in.

        Raises:
            KeyError: If no question has that id.
        """
        cur = await self._conn.execute(
            """
            SELECT part, answer
              FROM questions
             WHERE question_id = %s
            """,
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
        """A question and the book it came from.

        Random mode shows the origin — the student did not pick a year, so the
        message has to say which one they were given.

        Raises:
            KeyError: If no question has that id.
        """
        cur = await self._conn.execute(
            _QUESTION_WITH_ORIGIN_SELECT + " WHERE q.question_id = %s",
            (question_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Question {question_id} not found")
        origin = QuestionOrigin(
            year=row["year_value"],
            option_number=row["option_number"],
            exam_type=row["exam_type"],
        )
        return _row_to_question(row), origin


class QuestionDraw:
    """Picking a question at random, for the modes with no playlist.

    Random testing asks the corpus a different question than the catalog does:
    not "give me this one" but "give me any one matching these filters".
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_random_question_id(
        self,
        subject_id: int,
        part: Part,
        exam_type: ExamType | None = None,
    ) -> int:
        """Draw one question of *part* from *subject_id*, optionally by exam type.

        Two queries rather than ``ORDER BY RANDOM()``, which would force a full
        scan and evaluate ``random()`` per row. A COUNT stays indexable, and the
        OFFSET pick then reads only ``offset + 1`` rows.

        Raises:
            KeyError: If nothing matches — an empty subject, or a part it has no
                questions in. The bot turns that into "no questions here" rather
                than an error, so it is a normal outcome for a filter that
                narrowed to nothing.
        """
        if exam_type is None:
            scope = _DRAW_SCOPE
            params: list[Any] = [subject_id, part]
        else:
            scope = _DRAW_SCOPE + " AND o.exam_type = %s"
            params = [subject_id, part, exam_type]

        cur = await self._conn.execute("SELECT COUNT(*) AS n" + scope, params)
        count_row = await cur.fetchone()
        total = count_row["n"] if count_row else 0
        if total == 0:
            raise KeyError(f"No {part} questions found for subject {subject_id}")

        cur = await self._conn.execute(
            "SELECT q.question_id"
            + scope
            + " ORDER BY q.question_id OFFSET %s LIMIT 1",
            [*params, secrets.randbelow(total)],
        )
        row = await cur.fetchone()
        # The count above just proved the offset is in range.
        assert row is not None
        return row["question_id"]

    async def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> int:
        """Draw one question mapped to a subject's topic.

        All candidate ids come back and the pick happens here, rather than as a
        count-then-offset: a topic holds a few dozen questions at most, so one
        query over the index is cheaper than two. The topic carries its subject
        and the mapping carries the ids, so neither ``questions`` nor the books
        above it has to be visited to make the draw.

        Raises:
            KeyError: If the subject has no such topic, or has it with nothing
                mapped to it.
        """
        cur = await self._conn.execute(
            """
            SELECT qt.question_id
              FROM topics t
              JOIN question_topics qt ON qt.topic_id = t.topic_id
             WHERE t.subject_id = %s AND t.name = %s
            """,
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

    A topic belongs to a subject, so it is created against one rather than
    looked up by name: two subjects may both offer "Оптика" without sharing a
    single question between them.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_or_create_topic(self, subject_id: int, topic_name: str) -> int:
        """Return the id of a subject's topic, naming it if it is new.

        The name is stored once, here, and mappings carry the id — so a rename
        is one UPDATE, and a misspelling cannot become a second topic hidden
        behind its own mappings.
        """
        # DO UPDATE rather than DO NOTHING, and the update is deliberately a
        # no-op: RETURNING yields no row for a conflict that changed nothing,
        # and the id is needed either way.
        cur = await self._conn.execute(
            """
            INSERT INTO topics (subject_id, name) VALUES (%s, %s)
                 ON CONFLICT (subject_id, name) DO UPDATE SET name = EXCLUDED.name
              RETURNING topic_id
            """,
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
        """Map a topic onto one question, addressed the way the map names it.

        The hand-written topic map names questions by option and key, not by id,
        so the INSERT selects the id instead of taking one: a key naming a
        question that was never extracted maps nothing rather than failing.
        """
        await self._conn.execute(
            """
            INSERT INTO question_topics (question_id, topic_id)
                 SELECT question_id, %s
                   FROM questions
                  WHERE option_id = %s AND part = %s AND question_number = %s
                 ON CONFLICT (question_id, topic_id) DO NOTHING
            """,
            (topic_id, option_id, part, question_number),
        )

    async def count_topics(self) -> int:
        """How many question-to-topic mappings exist, across every subject.

        Mappings, not topics: what the seeder reports having loaded is the
        number of questions it managed to tag, and a named topic that tagged
        nothing is not progress.
        """
        cur = await self._conn.execute("SELECT COUNT(*) AS n FROM question_topics")
        row = await cur.fetchone()
        assert row is not None
        return row["n"]

    async def get_topics_for_subject(self, subject_id: int) -> list[str]:
        """The topics of *subject_id* that have at least one question mapped.

        A topic with nothing mapped to it would open a round with nothing to
        ask, so it is not offered. EXISTS rather than a join, because the
        question is whether any mapping exists and not how many.
        """
        cur = await self._conn.execute(
            """
            SELECT t.name
              FROM topics t
             WHERE t.subject_id = %s
               AND EXISTS (SELECT 1
                             FROM question_topics qt
                            WHERE qt.topic_id = t.topic_id)
             ORDER BY t.name
            """,
            (subject_id,),
        )
        rows = await cur.fetchall()
        return [row["name"] for row in rows]


class FileIdCache:
    """Telegram's own id for an image it has already been sent once.

    One method, because that is the whole of the role. The bot earns a
    ``file_id`` when it uploads an image, parks it in FSM state, and settles it
    inside the *next* round's transaction — so this is always written far from
    where it was earned.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def cache_file_id(self, question_id: int, telegram_file_id: str) -> None:
        """Record the id Telegram gave this question's image."""
        await self._conn.execute(
            """
            UPDATE images
               SET telegram_file_id = %s
             WHERE question_id = %s
            """,
            (telegram_file_id, question_id),
        )


class QuestionCorpus:
    """Writing the extraction output into the database.

    The seeding path only. Nothing the bot does while a student is answering
    reaches any of this, and every write is an upsert, so re-running a seed over
    output that has not changed is a no-op rather than a duplicate.
    """

    def __init__(self, conn: DictConn) -> None:
        self._conn = conn

    async def get_or_create(
        self, option_id: int, key: QuestionKey, answer: str | None
    ) -> int:
        """Store one question's answer key, returning its id.

        *answer* is None for a question whose key is missing or unusable. The
        question is stored anyway, so its image stays servable, and a NULL key
        matches nothing a student can send.

        Raises:
            ValueError: If a Part A answer is not a digit. Part A answers are
                option indices, so a word there is a bad answer key rather than
                a value to store — the seeder catches this and loads the
                question without a key.
        """
        if key.part == "A" and answer is not None and not answer.isdigit():
            raise ValueError(f"Part A answer must be a digit, got {answer!r}")

        cur = await self._conn.execute(
            """
            INSERT INTO questions (option_id, part, question_number, answer)
                 VALUES (%s, %s, %s, %s)
                 ON CONFLICT (option_id, part, question_number)
                 DO UPDATE SET answer = EXCLUDED.answer
              RETURNING question_id
            """,
            (option_id, key.part, key.number, answer),
        )
        row = await cur.fetchone()
        assert row is not None
        return row["question_id"]

    async def set_image(
        self, question_id: int, object_key: str, content_hash: str
    ) -> None:
        """Point a question at its image file, dropping a superseded upload.

        A changed image must lose the cached ``file_id``: that id names an
        upload Telegram already holds and the renderer prefers it over the file,
        so a re-seed that kept it would serve the old image forever. The key
        alone cannot detect the change — re-extracting a question rewrites the
        same path — which is what the hash is for.

        The guard on the UPDATE is what makes clearing the cache safe to pair
        with the write: an idempotent re-run never reaches the SET, so a valid
        ``file_id`` survives it.
        """
        await self._conn.execute(
            """
            INSERT INTO images (question_id, object_key, content_hash)
                 VALUES (%s, %s, %s)
                 ON CONFLICT (question_id)
                 DO UPDATE SET object_key = EXCLUDED.object_key,
                               content_hash = EXCLUDED.content_hash,
                               telegram_file_id = NULL
                           WHERE images.content_hash
                                 IS DISTINCT FROM EXCLUDED.content_hash
                              OR images.object_key
                                 IS DISTINCT FROM EXCLUDED.object_key
            """,
            (question_id, object_key, content_hash),
        )

    async def list_images(self) -> list[tuple[str, str]]:
        """Every ``(object_key, content_hash)`` the corpus claims to hold.

        The whole table, because reconciling asks two questions and only one of
        them can be asked a row at a time: which rows point at nothing, and
        which files nothing points at.
        """
        cur = await self._conn.execute(
            """
            SELECT object_key, content_hash
              FROM images
             ORDER BY object_key
            """
        )
        rows = await cur.fetchall()
        return [(row["object_key"], row["content_hash"]) for row in rows]
