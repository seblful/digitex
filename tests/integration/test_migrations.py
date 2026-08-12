"""Migration 0003's data migration, against real Postgres.

The repository tests run ``upgrade`` on an empty database, which never exercises
the part of 0003 that carries existing rows across: two tables whose identity
sequences overlap become one, and every ``images`` / ``question_topics`` /
``session_answers`` row has to follow the question it belonged to onto a new id.

This module gets its own container so it can walk the revisions without
disturbing the session-scoped schema the other tests share.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from psycopg.rows import dict_row

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import LiteralString

pytestmark = pytest.mark.integration

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def migration_dsn() -> Iterator[str]:
    """A Postgres of its own, left at revision 0002."""
    testcontainers = pytest.importorskip("testcontainers.postgres")
    try:
        # Constructing the container already reaches for the Docker daemon.
        container = testcontainers.PostgresContainer("postgres:17-alpine")
        container.start()
    except Exception as e:  # pragma: no cover - environment guard
        pytest.skip(f"Cannot start Postgres container (is Docker running?): {e}")
    try:
        yield container.get_connection_url().replace(
            "postgresql+psycopg2://", "postgresql://"
        )
    finally:
        container.stop()


def _alembic(dsn: str) -> Any:
    """An Alembic config pointed at *dsn*."""
    import os

    from alembic.config import Config

    os.environ["DATABASE_URL"] = dsn
    from digitex import config as config_module

    config_module._settings = None

    cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    return cfg


def _seed_split_schema(dsn: str) -> None:
    """Fill the two-table schema, with Part A and Part B ids that collide.

    Both tables' identity sequences start at 1, so question 1 exists twice —
    which is what made the child tables ambiguous.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO subjects (subject_id, name)"
            " OVERRIDING SYSTEM VALUE VALUES (1, 'Physics')"
        )
        conn.execute(
            "INSERT INTO books (book_id, subject_id, year_value)"
            " OVERRIDING SYSTEM VALUE VALUES (1, 1, 2024)"
        )
        conn.execute(
            "INSERT INTO options (option_id, book_id, option_number, exam_type)"
            " OVERRIDING SYSTEM VALUE VALUES (1, 1, 1, 'CT')"
        )
        conn.execute(
            "INSERT INTO students (student_id, telegram_id, name)"
            " OVERRIDING SYSTEM VALUE VALUES (1, 99, 'Ada')"
        )
        conn.execute(
            "INSERT INTO test_sessions (session_id, student_id, option_id)"
            " OVERRIDING SYSTEM VALUE VALUES (1, 1, 1)"
        )
        conn.execute(
            "INSERT INTO part_a_questions"
            " (question_id, option_id, question_number, answer)"
            " OVERRIDING SYSTEM VALUE VALUES (1, 1, 1, 3), (2, 1, 2, 5)"
        )
        conn.execute(
            "INSERT INTO part_b_questions"
            " (question_id, option_id, question_number, answer)"
            " OVERRIDING SYSTEM VALUE"
            " VALUES (1, 1, 1, 'neutron'), (2, 1, 2, 'photon')"
        )
        conn.execute(
            "INSERT INTO images (question_id, part, image_data, telegram_file_id)"
            " VALUES (1, 'A', %s, 'tg-a1'), (2, 'A', %s, 'tg-a2'),"
            "        (1, 'B', %s, 'tg-b1'), (2, 'B', %s, 'tg-b2')",
            (b"a1", b"a2", b"b1", b"b2"),
        )
        conn.execute(
            "INSERT INTO question_topics (question_id, part, topic_name)"
            " VALUES (1, 'A', 'kinematics'), (1, 'B', 'optics')"
        )
        conn.execute(
            "INSERT INTO session_answers"
            " (session_id, question_id, part, student_answer, is_correct, time_spent)"
            " VALUES (1, 1, 'A', '3', TRUE, 1.0), (1, 1, 'B', 'proton', FALSE, 2.0)"
        )


@pytest.fixture(scope="module")
def merged(migration_dsn: str) -> str:
    """Seed the split schema, then upgrade past the merge."""
    from alembic import command

    cfg = _alembic(migration_dsn)
    command.upgrade(cfg, "0002")
    _seed_split_schema(migration_dsn)
    command.upgrade(cfg, "head")
    return migration_dsn


def _rows(dsn: str, sql: LiteralString) -> list[dict[str, Any]]:
    with (
        psycopg.connect(dsn, autocommit=True) as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        return cur.execute(sql).fetchall()


class TestQuestionsTableMerge:
    def test_every_question_carries_over_with_its_answer(self, merged: str) -> None:
        rows = _rows(
            merged,
            "SELECT part, question_number, answer FROM questions"
            " ORDER BY part, question_number",
        )
        assert [(r["part"], r["question_number"], r["answer"]) for r in rows] == [
            ("A", 1, "3"),
            ("A", 2, "5"),
            ("B", 1, "neutron"),
            ("B", 2, "photon"),
        ]

    def test_ids_are_unique_across_both_parts(self, merged: str) -> None:
        rows = _rows(merged, "SELECT question_id FROM questions")
        assert len({r["question_id"] for r in rows}) == len(rows) == 4

    def test_every_image_still_points_at_its_own_question(self, merged: str) -> None:
        """The remapping's whole job: tg-b1 must not end up on the Part A row."""
        rows = _rows(
            merged,
            "SELECT i.telegram_file_id, q.part, q.question_number"
            "  FROM images i JOIN questions q"
            "    ON q.question_id = i.question_id AND q.part = i.part"
            " ORDER BY i.telegram_file_id",
        )
        assert [
            (r["telegram_file_id"], r["part"], r["question_number"]) for r in rows
        ] == [
            ("tg-a1", "A", 1),
            ("tg-a2", "A", 2),
            ("tg-b1", "B", 1),
            ("tg-b2", "B", 2),
        ]

    def test_topics_follow_their_question(self, merged: str) -> None:
        rows = _rows(
            merged,
            "SELECT qt.topic_name, q.part, q.question_number"
            "  FROM question_topics qt JOIN questions q"
            "    ON q.question_id = qt.question_id AND q.part = qt.part"
            " ORDER BY qt.topic_name",
        )
        assert [(r["topic_name"], r["part"], r["question_number"]) for r in rows] == [
            ("kinematics", "A", 1),
            ("optics", "B", 1),
        ]

    def test_recorded_answers_survive_on_the_right_questions(self, merged: str) -> None:
        rows = _rows(
            merged,
            "SELECT sa.student_answer, q.part, q.question_number"
            "  FROM session_answers sa JOIN questions q"
            "    ON q.question_id = sa.question_id AND q.part = sa.part"
            " ORDER BY q.part",
        )
        assert [
            (r["student_answer"], r["part"], r["question_number"]) for r in rows
        ] == [("3", "A", 1), ("proton", "B", 1)]

    def test_no_child_row_was_left_orphaned(self, merged: str) -> None:
        for table in ("images", "question_topics", "session_answers"):
            rows = _rows(
                merged,
                f"SELECT count(*) AS n FROM {table} c"
                " WHERE NOT EXISTS (SELECT 1 FROM questions q"
                "   WHERE q.question_id = c.question_id AND q.part = c.part)",
            )
            assert rows[0]["n"] == 0, table


class TestMergedSchemaConstraints:
    def test_an_image_for_an_unknown_question_is_rejected(self, merged: str) -> None:
        """The foreign key a dual-parent reference could not express."""
        with (
            psycopg.connect(merged, autocommit=True) as conn,
            pytest.raises(psycopg.errors.ForeignKeyViolation),
        ):
            conn.execute(
                "INSERT INTO images (question_id, part, image_data)"
                " VALUES (99999, 'A', %s)",
                (b"x",),
            )

    def test_a_child_cannot_claim_a_part_its_question_lacks(self, merged: str) -> None:
        part_a_id = _rows(
            merged,
            "SELECT question_id FROM questions"
            " WHERE part = 'A' AND question_number = 1",
        )[0]["question_id"]

        with (
            psycopg.connect(merged, autocommit=True) as conn,
            pytest.raises(psycopg.errors.ForeignKeyViolation),
        ):
            conn.execute(
                "INSERT INTO question_topics (question_id, part, topic_name)"
                " VALUES (%s, 'B', 'mismatch')",
                (part_a_id,),
            )

    def test_the_unmatchable_part_a_placeholder_is_storable(self, merged: str) -> None:
        """``populate_db`` writes '0' for a year with no answer key.

        The old ``CHECK (answer BETWEEN 1 AND 5)`` rejected it and rolled back
        the whole year.
        """
        with psycopg.connect(merged, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO questions (option_id, part, question_number, answer)"
                " VALUES (1, 'A', 90, '0')"
            )
            conn.execute("DELETE FROM questions WHERE question_number = 90")

    def test_a_non_numeric_part_a_answer_is_still_rejected(self, merged: str) -> None:
        with (
            psycopg.connect(merged, autocommit=True) as conn,
            pytest.raises(psycopg.errors.CheckViolation),
        ):
            conn.execute(
                "INSERT INTO questions (option_id, part, question_number, answer)"
                " VALUES (1, 'A', 91, 'abc')"
            )


class TestDowngrade:
    def test_downgrade_splits_the_table_back_apart(self, merged: str) -> None:
        """Run last in the module — it leaves the schema back at head."""
        from alembic import command

        cfg = _alembic(merged)
        command.downgrade(cfg, "0002")
        try:
            part_a = _rows(
                merged,
                "SELECT question_number, answer FROM part_a_questions"
                " ORDER BY question_number",
            )
            part_b = _rows(
                merged,
                "SELECT question_number, answer FROM part_b_questions"
                " ORDER BY question_number",
            )
            # Part A narrows back to INTEGER, so the '0' placeholder cannot
            # come with it — the migration drops those rows deliberately.
            assert [(r["question_number"], r["answer"]) for r in part_a] == [
                (1, 3),
                (2, 5),
            ]
            assert [(r["question_number"], r["answer"]) for r in part_b] == [
                (1, "neutron"),
                (2, "photon"),
            ]
        finally:
            command.upgrade(cfg, "head")
