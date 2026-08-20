"""Typed conversation state — one model per shape a conversation can be in.

aiogram stores state as a free-form dict, so a key is whatever the last writer
spelled it: a typo reads back as a default and takes a wrong turn, with no error
anywhere. The models below name every key exactly once, and the three functions
here are the only way the bot reaches the dict.

Three models rather than one because the two testing modes genuinely differ —
standard mode walks a playlist and records to a Session, random mode holds one
question at a time and records nothing. :class:`RoundDebt` is the slice both
share, so a round can be settled without asking which mode it was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from digitex.domain.entities import (  # noqa: TC001 — Pydantic needs runtime types
    ExamType,
    Part,
)

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext


class NavigationState(BaseModel):
    """State carried while the user is picking subject / year / option.

    Every field is optional: navigation fills them in one screen at a time, and
    a handler reached with a step missing answers the tap and stops.
    """

    subject_id: int | None = None
    year: int | None = None
    exam_type: ExamType | None = None
    book_id: int | None = None
    topic_names: list[str] | None = None
    topic_name: str | None = None
    random_part: Part | None = None


class TestingState(BaseModel):
    """State for the standard testing loop (records answers to a Session)."""

    session_id: int
    question_ids: list[tuple[int, Part]]
    current_index: int = 0
    current_part: Part | None = None
    question_start_time: float | None = None
    waiting_for_answer: bool = False
    # ``(question_id, telegram_file_id)`` owed by the last render. Settled
    # inside the next round's transaction — see `answer_flow` for why.
    pending_file_id_cache: tuple[int, str] | None = None


class RoundDebt(BaseModel):
    """Just the ``file_id`` debt, read out of either mode's state.

    Both round states declare the key; this model exists so the debt can be
    settled without knowing which mode the conversation was in.
    """

    pending_file_id_cache: tuple[int, str] | None = None


class RandomState(BaseModel):
    """State for random / topic question mode (no Session recording).

    Constructed whole by navigation when the round starts — the required
    ``subject_id`` is the constructor making that explicit.
    """

    subject_id: int
    topic_name: str | None = None
    exam_type: ExamType | None = None
    random_part: Part | None = None
    current_question_id: int | None = None
    current_part: Part | None = None
    question_start_time: float | None = None
    waiting_for_answer: bool = False
    pending_file_id_cache: tuple[int, str] | None = None


async def load[T: BaseModel](state: FSMContext, model: type[T]) -> T:
    """Load FSM state into the given Pydantic model.

    Missing keys take their model defaults; extra keys are dropped.
    """
    return model.model_validate(await state.get_data())


async def save(state: FSMContext, payload: BaseModel) -> None:
    """Persist a Pydantic model back to FSM state, overwriting all keys."""
    await state.set_data(payload.model_dump())


async def merge(state: FSMContext, model: type[BaseModel], **fields: Any) -> None:
    """Update a subset of *model*'s keys without round-tripping a whole model.

    The write is validated against the one model the conversation is in: a
    key that model does not declare would be stored, then silently dropped by
    ``load`` — which is how a field written for the wrong mode loses its
    value with no error anywhere.

    Raises:
        ValueError: If a key is not a field of *model*.
    """
    unknown = sorted(set(fields) - set(model.model_fields))
    if unknown:
        raise ValueError(
            f"Unknown field(s) {unknown} for {model.__name__} — declare them"
            " on the state model in fsm_data"
        )
    await state.update_data(**fields)


__all__ = [
    "NavigationState",
    "RandomState",
    "RoundDebt",
    "TestingState",
    "load",
    "merge",
    "save",
]
