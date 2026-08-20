"""Tests for the question-round module — its interface is the test surface.

No aiogram objects and no Postgres: the round functions take the typed FSM
state and a UnitOfWork-shaped object, and return outcomes as values. The
Round's own methods are driven through fakes standing at its real seams —
the bot, the FSM context, and the ``open_uow`` transaction factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from digitex.bot import fsm_data
from digitex.bot.answer_flow import (
    NextQuestion,
    Round,
    RoundFinished,
    evaluate_random_answer,
    pick_random_question,
    run_testing_round,
)
from digitex.bot.fsm_data import RandomState, TestingState
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.domain.answer import AnswerKey
from digitex.domain.entities import Question, QuestionOrigin

if TYPE_CHECKING:
    from aiogram import Bot, types
    from aiogram.fsm.context import FSMContext

    from digitex.db import UnitOfWork
    from digitex.domain.entities import Part


@dataclass
class FakeCatalog:
    """Reading a question. Keyed by id alone — the part is a column of the row."""

    by_id: dict[int, Question] = field(default_factory=dict)
    correct: dict[int, AnswerKey] = field(default_factory=dict)
    full: dict[int, tuple[Question, QuestionOrigin]] = field(default_factory=dict)
    lookups: list[int] = field(default_factory=list)

    async def get_correct_answer(self, question_id: int) -> AnswerKey:
        return self.correct[question_id]

    async def get(self, question_id: int) -> Question:
        self.lookups.append(question_id)
        return self.by_id[question_id]

    async def get_full(self, question_id: int) -> tuple[Question, QuestionOrigin]:
        return self.full[question_id]


@dataclass
class FakeDraw:
    """Drawing one at random. A missing result raises, as an empty corpus does."""

    random_result: int | None = None
    topic_result: int | None = None

    async def get_random_question_id(
        self, subject_id: int, part: str, exam_type: str | None
    ) -> int:
        if self.random_result is None:
            raise KeyError("no questions")
        return self.random_result

    async def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> int:
        if self.topic_result is None:
            raise KeyError("no questions")
        return self.topic_result


@dataclass
class FakeFileIds:
    """The parked file_id debt, recorded in the order it was settled."""

    cached: list[tuple[int, str]] = field(default_factory=list)

    async def cache_file_id(self, question_id: int, file_id: str) -> None:
        self.cached.append((question_id, file_id))


@dataclass
class FakeSessions:
    recorded: list[dict[str, Any]] = field(default_factory=list)

    async def record_answer(self, **kwargs: Any) -> None:
        self.recorded.append(kwargs)


@dataclass
class FakeUow:
    """One fake per role the bot actually reaches for.

    Four small objects instead of one carrying fourteen methods, which is the
    split under test as much as the repository is: a fake that has to implement
    a method nobody calls is a fake that will drift from the real thing.
    """

    questions: FakeCatalog = field(default_factory=FakeCatalog)
    draw: FakeDraw = field(default_factory=FakeDraw)
    file_ids: FakeFileIds = field(default_factory=FakeFileIds)
    sessions: FakeSessions = field(default_factory=FakeSessions)


def as_uow(fake: FakeUow) -> UnitOfWork:
    """The fakes satisfy UnitOfWork's contract structurally; cast for the checker."""
    return cast("UnitOfWork", fake)


# ---------------------------------------------------------------------------
# Telegram-side fakes — enough of aiogram's shape for the Round, no mocks
# ---------------------------------------------------------------------------


@dataclass
class FakePhoto:
    file_id: str


@dataclass
class FakeSentMessage:
    photo: list[FakePhoto]


@dataclass
class FakeBot:
    """Records send_photo calls; yields *fresh_file_id* when an upload happens."""

    fresh_file_id: str | None = None
    sent: list[dict[str, Any]] = field(default_factory=list)

    async def send_photo(self, **kwargs: Any) -> FakeSentMessage:
        self.sent.append(kwargs)
        if self.fresh_file_id is None:
            return FakeSentMessage(photo=[])
        return FakeSentMessage(photo=[FakePhoto(self.fresh_file_id)])


@dataclass
class FakeChat:
    id: int = 99


@dataclass
class FakeMessage:
    chat: FakeChat = field(default_factory=FakeChat)
    answers: list[str] = field(default_factory=list)

    async def answer(self, text: str, **kwargs: Any) -> None:
        self.answers.append(text)


@dataclass
class FakeState:
    """Stands in for aiogram's FSMContext — the conversation data dict."""

    data: dict[str, Any] = field(default_factory=dict)
    cleared: bool = False

    async def update_data(self, **fields: Any) -> None:
        self.data.update(fields)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def clear(self) -> None:
        self.data.clear()
        self.cleared = True


def as_bot(fake: FakeBot) -> Bot:
    return cast("Bot", fake)


def as_message(fake: FakeMessage) -> types.Message:
    return cast("types.Message", fake)


def as_state(fake: FakeState) -> FSMContext:
    return cast("FSMContext", fake)


class TestMerge:
    async def test_a_key_the_model_does_not_declare_is_refused(self) -> None:
        """Stored-then-dropped is how a renamed field loses data silently."""
        state = FakeState()

        with pytest.raises(ValueError, match="Unknown field"):
            await fsm_data.merge(as_state(state), TestingState, current_question_idd=7)

        assert state.data == {}

    async def test_a_key_only_another_mode_declares_is_refused(self) -> None:
        """A field written for the wrong mode is exactly the silent-drop bug."""
        state = FakeState()

        with pytest.raises(ValueError, match="Unknown field"):
            await fsm_data.merge(as_state(state), TestingState, current_question_id=7)

        assert state.data == {}

    async def test_declared_keys_pass_through(self) -> None:
        state = FakeState()

        await fsm_data.merge(as_state(state), RandomState, current_question_id=7)

        assert state.data == {"current_question_id": 7}


def _question(question_id: int, part: Part, file_id: str | None = None) -> Question:
    return Question(
        question_id=question_id,
        part=part,
        question_number=1,
        image_key=f"biology/2016/1/{part}/{question_id}.jpg",
        telegram_file_id=file_id,
    )


# The renders below never reach the filesystem: a cached file_id short-circuits
# the upload, and FSInputFile does not open its path until aiogram sends it —
# which the FakeBot here never does.
CORPUS = Path("corpus")


def _uow_factory(uow: FakeUow, opened: list[object]) -> Any:
    """A transaction factory for the Round's ``open_uow`` seam."""

    class _Ctx:
        async def __aenter__(self) -> FakeUow:
            opened.append(uow)
            return uow

        async def __aexit__(self, *args: object) -> None:
            return None

    return lambda: cast("Any", _Ctx())


def _round(
    state: FakeState,
    *,
    bot: FakeBot | None = None,
    uow: FakeUow | None = None,
    opened: list[object] | None = None,
) -> Round:
    """A Round over fakes; ``pool`` is never touched, the factory replaces it."""
    return Round(
        as_bot(bot or FakeBot()),
        as_state(state),
        cast("Any", None),
        CORPUS,
        open_uow=_uow_factory(uow or FakeUow(), opened if opened is not None else []),
    )


class TestRunTestingRound:
    async def test_correct_answer_recorded_and_next_question_returned(self) -> None:
        uow = FakeUow()
        uow.questions.correct[10] = AnswerKey(part="A", value=3)
        next_q = _question(20, "B", file_id="cached")
        uow.questions.by_id[20] = next_q
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
                "student_answer": "3",
                "correct_answer": AnswerKey(part="A", value=3),
                "is_correct": True,
                "time_spent_seconds": 12.5,
            }
        ]
        assert outcome == NextQuestion(question=next_q, next_index=1)

    async def test_wrong_answer_recorded_as_incorrect(self) -> None:
        uow = FakeUow()
        uow.questions.correct[10] = AnswerKey(part="A", value=3)
        testing = TestingState(session_id=7, question_ids=[(10, "A")])

        outcome = await run_testing_round(as_uow(uow), testing, "2", now=1.0)

        assert uow.sessions.recorded[0]["is_correct"] is False
        assert outcome == RoundFinished(next_index=1)

    async def test_an_answer_is_recorded_with_the_key_it_was_judged_against(
        self,
    ) -> None:
        """The verdict and the key it came from are written together."""
        uow = FakeUow()
        uow.questions.correct[10] = AnswerKey(part="B", value="neutron")
        testing = TestingState(session_id=7, question_ids=[(10, "B")])

        await run_testing_round(as_uow(uow), testing, "proton", now=1.0)

        recorded = uow.sessions.recorded[0]
        assert recorded["correct_answer"] == AnswerKey(part="B", value="neutron")
        assert recorded["is_correct"] is False

    async def test_a_question_with_no_key_is_recorded_wrong_with_no_key(self) -> None:
        uow = FakeUow()
        uow.questions.correct[10] = AnswerKey(part="B", value=None)
        testing = TestingState(session_id=7, question_ids=[(10, "B")])

        await run_testing_round(as_uow(uow), testing, "anything", now=1.0)

        recorded = uow.sessions.recorded[0]
        assert recorded["correct_answer"] == AnswerKey(part="B", value=None)
        assert recorded["is_correct"] is False

    async def test_settles_pending_file_id_debt_first(self) -> None:
        uow = FakeUow()
        uow.questions.correct[10] = AnswerKey(part="A", value=1)
        testing = TestingState(
            session_id=7,
            question_ids=[(10, "A")],
            pending_file_id_cache=(5, "file123"),
        )

        await run_testing_round(as_uow(uow), testing, "1", now=1.0)

        assert uow.file_ids.cached == [(5, "file123")]

    async def test_next_question_carries_its_image_key_in_one_lookup(self) -> None:
        """An uncached question costs no extra round-trip: the key rides along."""
        uow = FakeUow()
        uow.questions.correct[10] = AnswerKey(part="A", value=1)
        uow.questions.by_id[20] = _question(20, "B", file_id=None)
        testing = TestingState(session_id=7, question_ids=[(10, "A"), (20, "B")])

        outcome = await run_testing_round(as_uow(uow), testing, "1", now=1.0)

        assert isinstance(outcome, NextQuestion)
        assert outcome.question.image_key == "biology/2016/1/B/20.jpg"
        assert uow.questions.lookups == [20]


class TestShowQuestion:
    """The render + debt protocol, driven through the round's show methods."""

    async def _show_testing(
        self,
        question: Question,
        *,
        fresh_file_id: str | None = None,
        state: FakeState | None = None,
        index: int = 0,
        started_at: float = 100.0,
    ) -> tuple[FakeState, FakeBot, FakeMessage]:
        fake_state = state or FakeState()
        bot = FakeBot(fresh_file_id=fresh_file_id)
        message = FakeMessage()
        await _round(fake_state, bot=bot).show_testing_question(
            as_message(message), question, index=index, started_at=started_at
        )
        return fake_state, bot, message

    async def _show_random(
        self,
        question: Question,
        *,
        fresh_file_id: str | None = None,
        started_at: float = 100.0,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> tuple[FakeState, FakeBot, FakeMessage]:
        fake_state = FakeState()
        bot = FakeBot(fresh_file_id=fresh_file_id)
        message = FakeMessage()
        await _round(fake_state, bot=bot).show_random_question(
            as_message(message),
            question,
            started_at=started_at,
            caption=caption,
            parse_mode=parse_mode,
        )
        return fake_state, bot, message

    async def test_a_testing_render_records_its_playlist_position(self) -> None:
        state, _, _ = await self._show_testing(
            _question(10, "A", file_id="cached"), index=4, started_at=123.5
        )

        assert state.data["current_index"] == 4
        assert state.data["current_part"] == "A"
        assert state.data["question_start_time"] == 123.5
        assert state.data["waiting_for_answer"] is True

    async def test_a_random_render_records_the_question_itself(self) -> None:
        """No playlist in random mode — scoring looks the question up by id."""
        state, _, _ = await self._show_random(_question(10, "A", file_id="cached"))

        assert state.data["current_question_id"] == 10
        assert state.data["current_part"] == "A"
        assert state.data["waiting_for_answer"] is True
        assert "current_index" not in state.data

    async def test_cached_file_id_incurs_no_debt(self) -> None:
        state, _, _ = await self._show_testing(_question(10, "A", file_id="cached"))

        assert state.data["pending_file_id_cache"] is None

    async def test_fresh_upload_parks_a_debt_carrying_question_identity(self) -> None:
        state, _, _ = await self._show_testing(
            _question(10, "A"), fresh_file_id="new-id"
        )

        assert state.data["pending_file_id_cache"] == (10, "new-id")

    async def test_upload_without_a_photo_in_the_response_incurs_no_debt(self) -> None:
        state, _, _ = await self._show_testing(_question(10, "A"), fresh_file_id=None)

        assert state.data["pending_file_id_cache"] is None

    async def test_each_render_clears_the_debt_the_round_settled(self) -> None:
        state = FakeState(data={"pending_file_id_cache": (5, "stale")})

        await self._show_testing(_question(10, "A", file_id="cached"), state=state)

        assert state.data["pending_file_id_cache"] is None

    async def test_part_a_goes_out_with_the_option_keyboard(self) -> None:
        _, bot, message = await self._show_testing(_question(10, "A", file_id="cached"))

        assert bot.sent[0]["reply_markup"] is not None
        assert message.answers == []

    async def test_part_b_gets_a_follow_up_prompt_and_no_keyboard(self) -> None:
        _, bot, message = await self._show_testing(_question(11, "B", file_id="cached"))

        assert bot.sent[0]["reply_markup"] is None
        assert message.answers == [MSG_ENTER_ANSWER]

    async def test_caption_and_parse_mode_reach_telegram(self) -> None:
        _, bot, _ = await self._show_random(
            _question(10, "A", file_id="cached"),
            caption="Тема: Cells",
            parse_mode="HTML",
        )

        assert bot.sent[0]["caption"] == "Тема: Cells"
        assert bot.sent[0]["parse_mode"] == "HTML"


class TestEndRound:
    """Leaving a round settles the debt and clears the state, together."""

    async def test_parked_file_id_is_written_before_the_state_goes_away(self) -> None:
        uow = FakeUow()
        state = FakeState(data={"pending_file_id_cache": (5, "file9")})

        await _round(state, uow=uow).end()

        assert uow.file_ids.cached == [(5, "file9")]
        assert state.cleared is True
        assert state.data == {}

    async def test_a_round_that_owes_nothing_opens_no_transaction(self) -> None:
        """Rendering from cache costs no round-trip on the way out either."""
        opened: list[object] = []
        state = FakeState(data={"current_question_id": 10, "current_part": "A"})

        await _round(state, opened=opened).end()

        assert opened == []
        assert state.cleared is True

    async def test_the_debt_is_read_out_of_either_mode(self) -> None:
        """Standard mode parks extra keys; the debt is found all the same."""
        uow = FakeUow()
        testing = TestingState(
            session_id=7,
            question_ids=[(10, "A")],
            pending_file_id_cache=(5, "file9"),
        )
        state = FakeState(data=testing.model_dump())

        await _round(state, uow=uow).end()

        assert uow.file_ids.cached == [(5, "file9")]


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
        uow.draw.random_result = 10
        uow.questions.full[10] = (question, origin)

        picked = await pick_random_question(
            as_uow(uow),
            self._random_state(pending_file_id_cache=(5, "file9")),
        )

        assert picked == (question, origin)
        assert uow.file_ids.cached == [(5, "file9")]

    async def test_topic_mode_uses_topic_lookup(self) -> None:
        uow = FakeUow()
        question = _question(11, "B", file_id=None)
        uow.draw.topic_result = 11
        uow.questions.full[11] = (question, QuestionOrigin(2020, 2, "CT"))

        picked = await pick_random_question(
            as_uow(uow), self._random_state(topic_name="Cells", random_part=None)
        )

        assert picked is not None
        assert picked[0].image_key == "biology/2016/1/B/11.jpg"


class TestEvaluateRandomAnswer:
    async def test_none_without_active_question(self) -> None:
        rnd = RandomState(subject_id=1)
        assert await evaluate_random_answer(as_uow(FakeUow()), rnd, "x") is None

    async def test_scores_part_b_alternatives(self) -> None:
        uow = FakeUow()
        uow.questions.correct[11] = AnswerKey(part="B", value="ANS1/ANS2")
        rnd = RandomState(subject_id=1, current_question_id=11, current_part="B")

        verdict = await evaluate_random_answer(as_uow(uow), rnd, "ANS2")

        assert verdict == (True, AnswerKey(part="B", value="ANS1/ANS2"))
