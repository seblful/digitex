"""What the bot needs from a database, stated without naming one.

Handlers used to import ``digitex.db.UnitOfWork`` and construct it from a
``psycopg`` pool. That is ordinary layering — dependencies pointing down into
infrastructure — but it is not inversion, and the difference was being paid for
in `pyproject.toml`: four `import-linter` contracts, two of them near-duplicate
``forbidden`` lists with a comment warning they must be kept in sync by hand,
plus a third list in ``tests/contracts``. Three hand-maintained lists defending
one boundary, because the boundary was policy rather than structure.

These protocols invert it. The bot says what it needs; ``digitex.db`` provides
classes that answer to it; only a composition root names both. The boundary
becomes a fact about the direction of imports, which a contract can state once.

Deliberately narrow. Each protocol lists what the bot calls and nothing more —
``QuestionCatalog`` here has four methods where the Postgres class has the same
four, but ``TopicIndex`` has one where the class has four, because the bot only
ever reads a subject's topics. A protocol that mirrored the implementation
would be a second spelling of it rather than a statement of what is required.

Runtime-checkable so a test can assert the concrete classes still answer; that
check is method presence only, and whether the signatures line up is ``ty``'s
job at every call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from digitex.domain.answer import AnswerKey
    from digitex.domain.entities import (
        ExamType,
        Part,
        Question,
        QuestionOrigin,
        Session,
        SessionInfo,
        Student,
        SubjectRow,
        TestResult,
        WrongAnswer,
    )


@runtime_checkable
class QuestionCatalog(Protocol):
    """Reading a question the bot is about to serve."""

    async def get(self, question_id: int) -> Question: ...

    async def get_full(self, question_id: int) -> tuple[Question, QuestionOrigin]: ...

    async def get_correct_answer(self, question_id: int) -> AnswerKey: ...

    async def list_ids_for_option(self, option_id: int) -> list[tuple[int, Part]]: ...


@runtime_checkable
class QuestionDraw(Protocol):
    """Picking a question at random, for the modes with no playlist."""

    async def get_random_question_id(
        self, subject_id: int, part: Part, exam_type: ExamType | None
    ) -> int: ...

    async def get_random_question_id_by_topic(
        self, subject_id: int, topic_name: str
    ) -> int: ...


@runtime_checkable
class TopicIndex(Protocol):
    """The topic names a subject offers.

    One method: creating and upserting topics belongs to seeding, which is not
    something the bot does.
    """

    async def get_topics_for_subject(self, subject_id: int) -> list[str]: ...


@runtime_checkable
class FileIdCache(Protocol):
    """Telegram's own id for an image already uploaded once."""

    async def cache_file_id(self, question_id: int, telegram_file_id: str) -> None: ...


@runtime_checkable
class CatalogIndex(Protocol):
    """Walking the corpus the way the navigation keyboards do."""

    async def list_subjects(self) -> list[SubjectRow]: ...

    async def list_years(self, subject_id: int) -> list[int]: ...

    async def list_options(self, book_id: int, exam_type: ExamType) -> list[int]: ...

    async def get_book(self, subject_id: int, year: int) -> int | None: ...

    async def get_option_id(self, book_id: int, option_number: int) -> int: ...


@runtime_checkable
class SessionLog(Protocol):
    """A student's attempt at a set of questions, and what it scored."""

    async def create(self, student_telegram_id: int, option_id: int) -> Session: ...

    async def record_answer(
        self,
        session_id: int,
        question_id: int,
        student_answer: str,
        correct_answer: AnswerKey,
        is_correct: bool,
        time_spent_seconds: float,
    ) -> None: ...

    async def complete(self, session_id: int) -> TestResult: ...

    async def get_session_info(self, session_id: int) -> SessionInfo: ...

    async def get_wrong_answers(self, session_id: int) -> list[WrongAnswer]: ...

    async def get_result(self, session_id: int) -> TestResult: ...


@runtime_checkable
class StudentDirectory(Protocol):
    """The person behind a Telegram id, and whether they may take a test."""

    async def get_or_create(
        self,
        telegram_id: int,
        telegram_name: str,
        telegram_username: str | None = None,
    ) -> Student: ...

    async def create_request(
        self,
        telegram_id: int,
        full_name: str,
        telegram_name: str,
        telegram_username: str | None = None,
    ) -> Student: ...

    async def approve(self, telegram_id: int, admin_id: int) -> Student: ...

    async def reject(self, telegram_id: int, admin_id: int) -> Student: ...

    async def get(self, telegram_id: int) -> Student | None: ...

    async def is_authorized(self, telegram_id: int) -> bool: ...


@runtime_checkable
class Repositories(Protocol):
    """The roles reachable inside one transaction.

    Named for the attributes a unit of work exposes, so the concrete one
    satisfies this without being told about it — and so a fake is five small
    objects rather than one carrying every method any of them has.
    """

    @property
    def books(self) -> CatalogIndex: ...

    @property
    def questions(self) -> QuestionCatalog: ...

    @property
    def draw(self) -> QuestionDraw: ...

    @property
    def topics(self) -> TopicIndex: ...

    @property
    def file_ids(self) -> FileIdCache: ...

    @property
    def sessions(self) -> SessionLog: ...

    @property
    def students(self) -> StudentDirectory: ...


type OpenUow = Callable[[], AbstractAsyncContextManager[Repositories]]
"""Start a transaction and hand back the roles inside it.

A factory rather than an open transaction: a handler decides when its
transaction begins and ends, and the two things it does with a round — render a
question, settle the parked ``file_id`` — happen in different ones.

A ``type`` alias, so the names it is built from stay behind ``TYPE_CHECKING``:
this module needs nothing at runtime and is imported by the layer that must not
grow a runtime dependency.
"""
