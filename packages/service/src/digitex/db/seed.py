"""Loading the extraction output into the database, and checking it stayed true.

Two directions over the same corpus. :func:`populate` turns the output tree
(``{subject}/{year}/{option}/{part}/{n}.png`` plus a per-year ``answers.json``,
plus the book archive's hand-written ``topics.json``) into rows. Every write is
an upsert, so re-running after a new extraction adds what is new and leaves the
rest alone — which is what makes re-seeding a routine step rather than a
migration.

Images are the exception to "becomes rows": the file stays on disk and the row
records its key and content hash. That keeps the database a few megabytes and
the seed cheap, and costs the guarantee that a row and its bytes still agree —
which is what :func:`check_images` exists to re-establish, and why the runbook
syncs files *before* seeding rather than after.

Migrating the schema first is :mod:`digitex.service.cli.db`'s job; the caller
hands in an open pool and the two corpus roots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm

from digitex.db import UnitOfWork
from digitex.domain.corpus import (
    book_topics_file,
    file_digest,
    question_image_number,
    question_object_key,
    walk_question_images,
)
from digitex.domain.entities import QuestionKey, exam_type_for, parse_exam_type

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from psycopg_pool import AsyncConnectionPool

    from digitex.domain.entities import ExamType

# The output tree names subjects in English because that is what the pipeline
# writes; the bot shows a student the Russian name. Anything unmapped is shown
# capitalised, so a new subject seeds before it is translated.
SUBJECT_NAMES = {
    "biology": "Биология",
    "chemistry": "Химия",
    "physics": "Физика",
    "math": "Математика",
    "russian": "Русский язык",
    "history": "История",
    "social": "Обществознание",
}


def get_subject_name(subject: str) -> str:
    """The name a subject directory is stored and displayed under."""
    return SUBJECT_NAMES.get(subject.lower(), subject.capitalize())


# ---------------------------------------------------------------------------
# Walking the output tree
# ---------------------------------------------------------------------------


def _numbered_subdirs(parent: Path) -> list[Path]:
    """The numerically named subdirectories of *parent*, in numeric order.

    Both numbered levels of the output tree — years under a subject, options
    under a year — are read through this, so ``10`` lands after ``9`` at each
    of them rather than after ``1``. Anything else in the directory belongs to
    something other than the corpus and is skipped.
    """
    numbered = (
        path for path in parent.iterdir() if path.is_dir() and path.name.isdigit()
    )
    return sorted(numbered, key=lambda path: int(path.name))


def _question_images(option_dir: Path) -> Iterator[tuple[QuestionKey, Path]]:
    """Every question image under one option, Part A first and then by number.

    Sorting the part directories puts A before B; the numbers are sorted as
    numbers, because ``10.png`` is not between ``1.png`` and ``2.png``. That
    ordering is the order rows are written in, so a re-seed replays the same
    sequence it did the first time.
    """
    for part_dir in sorted(option_dir.iterdir()):
        if not part_dir.is_dir() or part_dir.name not in ("A", "B"):
            continue
        numbered = sorted(
            (number, path)
            for path in part_dir.iterdir()
            if (number := question_image_number(path)) is not None
        )
        for number, path in numbered:
            yield QuestionKey.parse(f"{part_dir.name}{number}"), path


def _read_answers(year_dir: Path) -> dict[str, dict[str, str]]:
    """One year's answer key, as ``{option number: {question key: answer}}``.

    Empty when the year shipped without one. Worth saying out loud, because
    every question in that year will load with a NULL key — but not worth
    refusing the year over: the images are still servable.
    """
    answers_file = year_dir / "answers.json"
    if not answers_file.exists():
        tqdm.write(f"  Warning: no answers.json in {year_dir}")
        return {}
    return json.loads(answers_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Populate
# ---------------------------------------------------------------------------


async def _load_question(
    uow: UnitOfWork, option_id: int, key: QuestionKey, raw_answer: str | None
) -> tuple[int, bool]:
    """Store one question; return its id and whether a key came with it.

    A question whose ``answers.json`` entry is absent, blank or unusable is
    loaded anyway with a NULL key, so its image stays servable and nothing a
    student sends can match it. An answer that was present but rejected earns a
    warning — that is a typo in a hand-checked file rather than an absence.
    """
    if raw_answer:
        try:
            return await uow.corpus.get_or_create(option_id, key, raw_answer), True
        except ValueError as e:
            tqdm.write(f"  Warning: {e} — storing it without a key")
    return await uow.corpus.get_or_create(option_id, key, None), False


async def _populate_year(
    uow: UnitOfWork, subject_id: int, output_dir: Path, year_dir: Path
) -> tuple[int, int]:
    """Load one year: its options, questions and images.

    Returns ``(questions loaded, answer keys among them)`` — the two numbers the
    caller reports, and the gap between them is how much of the year is
    unscoreable.
    """
    year = int(year_dir.name)
    answers = _read_answers(year_dir)

    book_id = await uow.books.get_book(subject_id, year)
    if book_id is None:
        book_id = await uow.books.create_book(subject_id, year)

    questions_loaded = 0
    answers_loaded = 0

    for option_dir in _numbered_subdirs(year_dir):
        option_number = int(option_dir.name)
        # Which exam variant an option is depends on its year and number, so it
        # is derived here rather than read off the tree, which does not say.
        option_id = await uow.books.get_or_create_option(
            book_id, option_number, exam_type_for(year, option_number)
        )
        option_answers = answers.get(str(option_number), {})

        for key, image in _question_images(option_dir):
            question_id, keyed = await _load_question(
                uow, option_id, key, option_answers.get(str(key))
            )
            await uow.corpus.set_image(
                question_id,
                question_object_key(output_dir, image),
                file_digest(image),
            )
            questions_loaded += 1
            if keyed:
                answers_loaded += 1

    return questions_loaded, answers_loaded


async def _option_ids(uow: UnitOfWork, book_id: int, exam_type: ExamType) -> list[int]:
    """The ids of one book's options in one exam variant.

    An option id depends only on its book and number, so a topic's whole key
    list resolves against one of these rather than a lookup per key.
    """
    return [
        await uow.books.get_option_id(book_id, option_number)
        for option_number in await uow.books.list_options(book_id, exam_type)
    ]


async def _populate_topics(
    pool: AsyncConnectionPool, subject_id: int, topics_file: Path
) -> int:
    """Map a subject's hand-written topics onto the questions already loaded.

    Returns how many mappings the database holds afterwards. Optional: a subject
    with no ``topics.json`` keeps every question and simply offers no topic
    rounds.

    A topic names questions by year, exam type and key rather than by id, and it
    names them for every option of that year at once — the same question number
    is the same topic whichever option a student draws it from.
    """
    if not topics_file.exists():
        tqdm.write(f"  No {topics_file}, skipping topics")
        return 0

    topics_data = json.loads(topics_file.read_text(encoding="utf-8"))

    async with UnitOfWork(pool) as uow:
        for topic_name, years in tqdm(topics_data.items(), desc="topics"):
            topic_id = await uow.topics.get_or_create_topic(subject_id, topic_name)
            for year_str, exam_types in years.items():
                book_id = await uow.books.get_book(subject_id, int(year_str))
                # A topic may name a year that was never extracted.
                if book_id is None:
                    continue
                for exam_type_name, keys in exam_types.items():
                    option_ids = await _option_ids(
                        uow, book_id, parse_exam_type(exam_type_name)
                    )
                    for raw_key in keys:
                        key = QuestionKey.parse(raw_key)
                        for option_id in option_ids:
                            await uow.topics.upsert_topic(
                                option_id, key.number, key.part, topic_id
                            )
        return await uow.topics.count_topics()

    # Reached only when ``__aexit__`` suppressed a ``Rollback``: psycopg's
    # transaction manager does that, so the ``async with`` can finish without its
    # body having. Nothing committed on that path means nothing was mapped, and
    # the count the caller prints has to say so.
    return 0


async def populate_subject(
    pool: AsyncConnectionPool, output_dir: Path, books_dir: Path, subject: str
) -> None:
    """Load one subject's years from *output_dir*, then its topic mappings.

    Both roots are needed because the topic map is hand-written and lives in the
    book archive rather than in the extraction output.

    A transaction per year, not per subject: a year is the unit worth keeping
    when the next one fails, and a whole subject in one transaction would hold a
    connection open for the length of the hashing.
    """
    subject_dir = output_dir / subject
    if not subject_dir.exists():
        tqdm.write(f"Not found: {subject_dir}")
        return

    year_dirs = _numbered_subdirs(subject_dir)
    if not year_dirs:
        tqdm.write(f"No year directories found under {subject_dir}")
        return

    tqdm.write(f"\n{subject} — {len(year_dirs)} year(s)")

    async with UnitOfWork(pool) as uow:
        subject_id = await uow.books.get_or_create_subject(get_subject_name(subject))

    for year_dir in tqdm(year_dirs, desc=subject):
        async with UnitOfWork(pool) as uow:
            questions, answers = await _populate_year(
                uow, subject_id, output_dir, year_dir
            )
        tqdm.write(f"  {year_dir.name}: {questions} questions, {answers} answers")

    topic_count = await _populate_topics(
        pool, subject_id, book_topics_file(books_dir, subject)
    )
    if topic_count:
        tqdm.write(f"  {topic_count} topic mappings loaded")


async def populate(
    pool: AsyncConnectionPool,
    output_dir: Path,
    books_dir: Path,
    subject: str | None = None,
) -> None:
    """Load *subject* from *output_dir*, or every subject the tree holds."""
    if subject is not None:
        await populate_subject(pool, output_dir, books_dir, subject)
        return

    subjects = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
    if not subjects:
        tqdm.write("No subjects found in extraction output.")
        return

    for name in subjects:
        await populate_subject(pool, output_dir, books_dir, name)


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------


def _all_question_images(output_dir: Path) -> Iterator[Path]:
    """Every question image in the corpus, across all subjects and years."""
    for subject_dir in sorted(output_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        for year_dir in _numbered_subdirs(subject_dir):
            for image in walk_question_images(year_dir):
                yield image.path


@dataclass(frozen=True)
class ImageCheck:
    """Where the ``images`` rows and the image files disagree.

    Three lists of object keys, kept apart because each names a different fix:
    *missing* means the files never reached this machine (sync the corpus),
    *stale* means they did but the rows were not re-seeded afterwards (populate),
    and *orphaned* means files nothing points at — a subject that was never
    seeded, or output that outlived its questions.
    """

    missing: list[str]
    stale: list[str]
    orphaned: list[str]

    @property
    def ok(self) -> bool:
        """True when every row has its file and every file has its row."""
        return not (self.missing or self.stale or self.orphaned)


async def check_images(pool: AsyncConnectionPool, output_dir: Path) -> ImageCheck:
    """Compare every stored key and hash against the files on this machine.

    Hashing is the expensive half and only the keys present on both sides need
    it: a key with no file is already missing, and a file with no key is already
    orphaned, whatever their bytes say.
    """
    async with UnitOfWork(pool) as uow:
        stored = dict(await uow.corpus.list_images())

    on_disk = {
        question_object_key(output_dir, path): path
        for path in _all_question_images(output_dir)
    }

    both = stored.keys() & on_disk.keys()
    return ImageCheck(
        missing=sorted(stored.keys() - on_disk.keys()),
        stale=sorted(
            key
            for key in tqdm(sorted(both), desc="hashing")
            if file_digest(on_disk[key]) != stored[key]
        ),
        orphaned=sorted(on_disk.keys() - stored.keys()),
    )
