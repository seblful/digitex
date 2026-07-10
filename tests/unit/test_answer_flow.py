"""Tests for the question-round module — its interface is the test surface.

No aiogram objects and no Postgres: the round functions take the typed FSM
state and a UnitOfWork-shaped object, and return outcomes as values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from digitex.bot.answer_flow import (
    NextQuestion,
    RoundFinished,
    evaluate_random_answer,
    file_id_debt,
    pick_random_question,
    run_testing_round,
)
from digitex.bot.fsm_data import RandomState, TestingState
from digitex.core.db.repositories._common import QuestionOrigin
from digitex.core.domain import Question

if TYPE_CHECKING:
    from digitex.core.db import UnitOfWork
    from digitex.core.domain import Part


@dataclass
class FakeQuestions:
    by_key: dict[tuple[int, str], Question] = field(default_factory=dict)
    correct: dict[tuple[int, str], int | str] = field(default_factory=dict)
    images: dict[tuple[int, str], bytes] = field(default_factory=dict)
    full: dict[tuple[int, str], tuple[Question, QuestionOrigin]] = field(
        default_factory=dict
    )
    random_result: tuple[int, str] | None = None
    topic_result: tuple[int, str] | None = None
    cached: list[tuple[int, str, str]] = field(default_factory=list)
    image_fetches: list[tuple[int, str]] = field(default_factory=list)

    async def cache_file_id(self, question_id: int, part: str, file_id: str) -> None:
        self.cached.append((question_id, part, file_id))

    async def get_correct_answer(self, question_id: int, part: str) -> int | str:
        return self.correct[(question_id, part)]

    async def get(self, question_id: int, part: str) -> Question:
        return self.by_key[(question_id, part)]

    async def get_image(self, question_id: int, part: str) -> bytes:
        self.image_fetches.append((question_id, part))
        return self.images[(question_id, part)]

    async def get_full(
        self, question_id: int, part: str
    ) -> tuple[Question, QuestionOrigin]:
        return self.full[(question_id, part)]

    async def get_random_question_id(
        self, subject_id: int, part: str, exam_type: str | None
    ) -> int:
        if self.random_result is None:
            raise KeyError("no questions")
        return self.random_result[0]

    async def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> tuple[int, str]:
        if self.topic_result is None:
            raise KeyError("no questions")
        return self.topic_result


@dataclass
class FakeSessions:
    recorded: list[dict[str, Any]] = field(default_factory=list)

    async def record_answer(self, **kwargs: Any) -> None:
        self.recorded.append(kwargs)


@dataclass
class FakeUow:
    questions: FakeQuestions = field(default_factory=FakeQuestions)
    sessions: FakeSessions = field(default_factory=FakeSessions)


def as_uow(fake: FakeUow) -> UnitOfWork:
    """The fakes satisfy UnitOfWork's contract structurally; cast for the checker."""
    return cast("UnitOfWork", fake)


def _question(question_id: int, part: Part, file_id: str | None = None) -> Question:
    return Question(
        question_id=question_id,
        part=part,
        question_number=1,
        telegram_file_id=file_id,
    )


class TestRunTestingRound:
    async def test_correct_answer_recorded_and_next_question_returned(self) -> None:
        uow = FakeUow()
        uow.questions.correct[(10, "A")] = 3
        next_q = _question(20, "B", file_id="cached")
        uow.questions.by_key[(20, "B")] = next_q
        testing = TestingState(
            session_id=7,
            question_ids=[(10, "A"), (20, "B")],
            current_index=0,
            question_start_time=100.0,
        )

        outcome = await run_testing_round(as_uow(uow), testing, " 3 ", now=112.5)

        assert uow.sessions.recorded == [
            {
                "session_id": 7,
                "question_id": 10,
                "part": "A",
                "student_answer": "3",
                "is_correct": True,
                "time_spent": 12.5,
            }
        ]
        assert outcome == NextQuestion(question=next_q, next_index=1)

    async def test_wrong_answer_recorded_as_incorrect(self) -> None:
        uow = FakeUow()
        uow.questions.correct[(10, "A")] = 3
        testing = TestingState(session_id=7, question_ids=[(10, "A")])

        outcome = await run_testing_round(as_uow(uow), testing, "2", now=1.0)

        assert uow.sessions.recorded[0]["is_correct"] is False
        assert outcome == RoundFinished(next_index=1)

    async def test_settles_pending_file_id_debt_first(self) -> None:
        uow = FakeUow()
        uow.questions.correct[(10, "A")] = 1
        testing = TestingState(
            session_id=7,
            question_ids=[(10, "A")],
            pending_file_id_cache=(5, "A", "file123"),
        )

        await run_testing_round(as_uow(uow), testing, "1", now=1.0)

        assert uow.questions.cached == [(5, "A", "file123")]

    async def test_next_question_image_fetched_only_on_cache_miss(self) -> None:
        uow = FakeUow()
        uow.questions.correct[(10, "A")] = 1
        uow.questions.by_key[(20, "B")] = _question(20, "B", file_id=None)
        uow.questions.images[(20, "B")] = b"image-bytes"
        testing = TestingState(session_id=7, question_ids=[(10, "A"), (20, "B")])

        outcome = await run_testing_round(as_uow(uow), testing, "1", now=1.0)

        assert isinstance(outcome, NextQuestion)
        assert outcome.question.image_data == b"image-bytes"
        assert uow.questions.image_fetches == [(20, "B")]

    async def test_cached_next_question_skips_image_fetch(self) -> None:
        uow = FakeUow()
        uow.questions.correct[(10, "A")] = 1
        uow.questions.by_key[(20, "B")] = _question(20, "B", file_id="cached")
        testing = TestingState(session_id=7, question_ids=[(10, "A"), (20, "B")])

        await run_testing_round(as_uow(uow), testing, "1", now=1.0)

        assert uow.questions.image_fetches == []


class TestFileIdDebt:
    def test_no_debt_when_cached_file_id_was_reused(self) -> None:
        assert file_id_debt(_question(10, "A"), None) is None

    def test_debt_carries_question_identity(self) -> None:
        assert file_id_debt(_question(10, "A"), "new-id") == (10, "A", "new-id")


class TestPickRandomQuestion:
    def _random_state(self, **overrides: Any) -> RandomState:
        defaults: dict[str, Any] = {"subject_id": 1, "random_part": "A"}
        defaults.update(overrides)
        return RandomState(**defaults)

    async def test_returns_none_when_filters_incomplete(self) -> None:
        picked = await pick_random_question(
            as_uow(FakeUow()), self._random_state(random_part=None)
        )
        assert picked is None

    async def test_returns_none_when_no_question_matches(self) -> None:
        picked = await pick_random_question(as_uow(FakeUow()), self._random_state())
        assert picked is None

    async def test_picks_by_part_and_settles_debt(self) -> None:
        uow = FakeUow()
        question = _question(10, "A", file_id="cached")
        origin = QuestionOrigin(2023, 1, "CE")
        uow.questions.random_result = (10, "A")
        uow.questions.full[(10, "A")] = (question, origin)

        picked = await pick_random_question(
            as_uow(uow),
            self._random_state(pending_file_id_cache=(5, "B", "file9")),
        )

        assert picked == (question, origin)
        assert uow.questions.cached == [(5, "B", "file9")]

    async def test_topic_mode_uses_topic_lookup(self) -> None:
        uow = FakeUow()
        question = _question(11, "B", file_id=None)
        uow.questions.topic_result = (11, "B")
        uow.questions.full[(11, "B")] = (question, QuestionOrigin(2020, 2, "CT"))
        uow.questions.images[(11, "B")] = b"img"

        picked = await pick_random_question(
            as_uow(uow), self._random_state(topic_name="Cells", random_part=None)
        )

        assert picked is not None
        assert picked[0].image_data == b"img"


class TestEvaluateRandomAnswer:
    async def test_none_without_active_question(self) -> None:
        rnd = RandomState(subject_id=1)
        assert await evaluate_random_answer(as_uow(FakeUow()), rnd, "x") is None

    async def test_scores_part_b_alternatives(self) -> None:
        uow = FakeUow()
        uow.questions.correct[(11, "B")] = "ANS1/ANS2"
        rnd = RandomState(subject_id=1, current_question_id=11, current_part="B")

        verdict = await evaluate_random_answer(as_uow(uow), rnd, "ANS2")

        assert verdict == (True, "ANS1/ANS2")
