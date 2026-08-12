"""Drop books.a_num_options — read on every question query, written by nobody.

The column implied a per-book override for how many numbered answers a Part A
question offers, but no repository method or migration ever set it, so every
row carried the default and ``Question.num_options`` was always 5. The count is
now a domain constant, ``PART_A_OPTION_COUNT``.

Dropping it also lets ``QuestionRepository.get`` stop joining ``options`` and
``books`` — that join existed only to fetch this column.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-12
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE books DROP COLUMN a_num_options")


def downgrade() -> None:
    op.execute("ALTER TABLE books ADD COLUMN a_num_options INTEGER NOT NULL DEFAULT 5")
