"""Load extraction output into the database, and check that it still matches.

The on-disk corpus (``{subject}/{year}/{option}/{part}/{n}.png`` plus its
``answers.json`` and ``topic_to_year.json``) becomes rows. Every write goes
through ``get_or_create``, so a re-run of the same output is a no-op rather
than a duplicate — which is what makes re-seeding after new extractions safe.

Images are the exception to "becomes rows": the file stays on disk and the row
records its key and content hash. That buys a small database and a cheap seed,
and costs the guarantee that a row and its bytes agree — which is what
:func:`check_images` is for, and why the runbook syncs the files before seeding
rather than after.

The caller supplies the pool and the output directory; migrating the schema
first is :mod:`digitex.cli.db`'s job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm

from digitex.db import UnitOfWork
from digitex.domain.corpus import (
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
    """Translate subject name to Russian if mapping exists."""
    return SUBJECT_NAMES.get(subject.lower(), subject.capitalize())


async def _populate_year(
    uow: UnitOfWork, subject_id: int, output_dir: Path, year_dir: Path
) -> tuple[int, int]:
    year = int(year_dir.name)

    answers: dict[str, dict[str, str]] = {}
    answers_file = year_dir / "answers.json"
    if answers_file.exists():
        answers = json.loads(answers_file.read_text(encoding="utf-8"))
    else:
        tqdm.write(f"  Warning: no answers.json in {year_dir}")

    book_id = await uow.books.get_book(subject_id, year)
    if book_id is None:
        book_id = await uow.books.create_book(subject_id, year)

    questions_loaded = 0
    answers_loaded = 0

    option_dirs = sorted(
        (d for d in year_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for option_dir in option_dirs:
        option_number = int(option_dir.name)
        exam_type = exam_type_for(year, option_number)
        option_id = await uow.books.get_or_create_option(
            book_id, option_number, exam_type
        )
        option_answers = answers.get(str(option_number), {})

        for part_dir in sorted(option_dir.iterdir()):
            if not part_dir.is_dir() or part_dir.name not in ("A", "B"):
                continue

            numbered = sorted(
                (number, f)
                for f in part_dir.iterdir()
                if (number := question_image_number(f)) is not None
            )

            for number, img_file in numbered:
                key = QuestionKey.parse(f"{part_dir.name}{number}")
                raw_answer = option_answers.get(str(key))

                question_id: int | None = None
                if raw_answer:
                    try:
                        question_id = await uow.questions.get_or_create(
                            option_id, key, raw_answer
                        )
                        answers_loaded += 1
                    except ValueError as e:
                        tqdm.write(f"  Warning: {e} — storing it without a key")

                if question_id is None:
                    # A Question whose answers.json entry is missing or unusable
                    # is still loaded, so its image is servable — with a NULL
                    # key, which no reply can match.
                    question_id = await uow.questions.get_or_create(
                        option_id, key, None
                    )

                await uow.questions.set_image(
                    question_id,
                    question_object_key(output_dir, img_file),
                    file_digest(img_file),
                )
                questions_loaded += 1

    return questions_loaded, answers_loaded


async def _populate_topics(
    pool: AsyncConnectionPool, subject_id: int, subject_dir: Path
) -> int:
    """Populate question_topics from topic_to_year.json. Returns mapping count."""
    topics_file = subject_dir / "topic_to_year.json"
    if not topics_file.exists():
        print(f"  No topic_to_year.json in {subject_dir}, skipping topics")
        return 0

    topics_data = json.loads(topics_file.read_text(encoding="utf-8"))

    async with UnitOfWork(pool) as uow:
        for topic_name, years in tqdm(topics_data.items(), desc="topics"):
            # The name is stored once, on the topic; the mappings below carry
            # its id.
            topic_id = await uow.questions.get_or_create_topic(subject_id, topic_name)
            for year_str, exam_types in years.items():
                year = int(year_str)
                book_id = await uow.books.get_book(subject_id, year)
                if book_id is None:
                    continue
                for exam_type_name, keys in exam_types.items():
                    exam_type = parse_exam_type(exam_type_name)
                    option_numbers = await uow.books.list_options(book_id, exam_type)
                    # Option ids depend only on (book_id, option_number), so
                    # resolve them once rather than per topic key.
                    option_ids = [
                        await uow.books.get_option_id(book_id, option_number)
                        for option_number in option_numbers
                    ]
                    for key in keys:
                        question_key = QuestionKey.parse(key)
                        for option_id in option_ids:
                            await uow.questions.upsert_topic(
                                option_id,
                                question_key.number,
                                question_key.part,
                                topic_id,
                            )
        return await uow.questions.count_topics()


async def populate_subject(
    pool: AsyncConnectionPool, output_dir: Path, subject: str
) -> None:
    """Load one subject's years, then its topic mappings."""
    subject_dir = output_dir / subject
    if not subject_dir.exists():
        print(f"Not found: {subject_dir}")
        return

    year_dirs = sorted(
        (d for d in subject_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not year_dirs:
        print(f"No year directories found under {subject_dir}")
        return

    print(f"\n{subject} — {len(year_dirs)} year(s)")

    async with UnitOfWork(pool) as uow:
        subject_id = await uow.books.get_or_create_subject(get_subject_name(subject))

    for year_dir in tqdm(year_dirs, desc=subject):
        async with UnitOfWork(pool) as uow:
            questions, answers = await _populate_year(
                uow, subject_id, output_dir, year_dir
            )
        tqdm.write(f"  {year_dir.name}: {questions} questions, {answers} answers")

    topic_count = await _populate_topics(pool, subject_id, subject_dir)
    if topic_count:
        print(f"  {topic_count} topic mappings loaded")


async def populate(
    pool: AsyncConnectionPool, output_dir: Path, subject: str | None = None
) -> None:
    """Load *subject* from *output_dir*, or every subject found there."""
    if subject is not None:
        await populate_subject(pool, output_dir, subject)
        return

    subjects = sorted(d.name for d in output_dir.iterdir() if d.is_dir())
    if not subjects:
        print("No subjects found in extraction output.")
        return

    for name in subjects:
        await populate_subject(pool, output_dir, name)


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------


def _all_question_images(output_dir: Path) -> Iterator[Path]:
    """Every question image in the corpus, across all subjects and years."""
    for subject_dir in sorted(output_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        for year_dir in sorted(subject_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for image in walk_question_images(year_dir):
                yield image.path


@dataclass(frozen=True)
class ImageCheck:
    """Where the ``images`` table and the image corpus disagree.

    Each list holds object keys. They are separate because the fixes differ:
    *missing* means the files never reached this machine (sync them), *stale*
    means they did but the rows were not re-seeded afterwards (populate), and
    *orphaned* means files nothing points at (a subject that was never seeded,
    or extraction output that outlived its questions).
    """

    missing: list[str]
    stale: list[str]
    orphaned: list[str]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.stale or self.orphaned)


async def check_images(pool: AsyncConnectionPool, output_dir: Path) -> ImageCheck:
    """Compare every stored image key and hash against the files on disk."""
    async with UnitOfWork(pool) as uow:
        stored = dict(await uow.questions.list_images())

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
