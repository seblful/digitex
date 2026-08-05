"""Key session_answers by (session_id, question_id, part).

``session_answers.question_id`` has two possible parents — ``part_a_questions``
and ``part_b_questions`` — and each table has its own identity sequence, so both
hold a ``question_id`` 1. An id alone therefore does not identify a question,
which is why every dual-parent row also carries ``part`` (see the 0001 notes).

The original UNIQUE omitted ``part``. Because a session's playlist covers Part A
then Part B for one option, a Part B answer whose id matched an already-recorded
Part A answer in the same session collided, and ``record_answer``'s
``ON CONFLICT DO NOTHING`` discarded it: the answer went unscored and its
mistake never reached the results screen. ``images`` and ``question_topics``
already key on ``(question_id, part)``; this brings ``session_answers`` in line.

Widening a unique key only ever admits more rows, so no existing row can
conflict. Answers already dropped are unrecoverable — this stops the loss, it
does not repair history.

The dropped constraint carries the name PostgreSQL generated for 0001's inline
``UNIQUE (session_id, question_id)``.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-05
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_OLD_CONSTRAINT = "session_answers_session_id_question_id_key"
_NEW_CONSTRAINT = "session_answers_session_id_question_id_part_key"


def upgrade() -> None:
    op.execute(f"ALTER TABLE session_answers DROP CONSTRAINT {_OLD_CONSTRAINT}")
    op.execute(
        f"ALTER TABLE session_answers ADD CONSTRAINT {_NEW_CONSTRAINT}"
        " UNIQUE (session_id, question_id, part)"
    )


def downgrade() -> None:
    """Narrow the key back.

    Fails if any session holds the same ``question_id`` for both parts — the
    rows this revision exists to allow.
    """
    op.execute(f"ALTER TABLE session_answers DROP CONSTRAINT {_NEW_CONSTRAINT}")
    op.execute(
        f"ALTER TABLE session_answers ADD CONSTRAINT {_OLD_CONSTRAINT}"
        " UNIQUE (session_id, question_id)"
    )
