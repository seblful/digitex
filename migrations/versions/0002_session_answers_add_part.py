"""Add part column to session_answers.

session_answers.question_id references either part_a_questions or
part_b_questions. Both tables use independent identity sequences, so their
question_ids overlap. Without a part column there is no reliable way to
determine which table an answer belongs to, causing UNION ALL queries to
return duplicate rows.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE session_answers ADD COLUMN part TEXT NOT NULL DEFAULT 'A'"
        " CHECK (part IN ('A', 'B'))"
    )
    op.execute("ALTER TABLE session_answers ALTER COLUMN part DROP DEFAULT")


def downgrade() -> None:
    op.execute("ALTER TABLE session_answers DROP COLUMN part")
