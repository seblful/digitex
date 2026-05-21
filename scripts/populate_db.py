"""Populate the database from extraction output.

Runs ``alembic upgrade head`` before populating so the schema is always at the
latest revision. Idempotent — safe to re-run.

Usage::

    uv run python scripts/populate_db.py              # all subjects
    uv run python scripts/populate_db.py biology      # single subject
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from tqdm import tqdm

from digitex.config import get_settings
from digitex.core.db import UnitOfWork, pool_lifespan
from digitex.core.value_objects import QuestionKey

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def _alembic_upgrade() -> None:
    cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    command.upgrade(cfg, "head")


async def _populate_year(  # noqa: PLR0912 — linear ETL pipeline; branches reflect the directory layout
    uow: UnitOfWork, subject_id: int, year_dir: Path
) -> tuple[int, int]:
    year = int(year_dir.name)

    answers: dict[str, dict[str, str]] = {}
    answers_file = year_dir / "answers.json"
    if answers_file.exists():
        answers = json.loads(answers_file.read_text(encoding="utf-8"))
    else:
        tqdm.write(f"  Warning: no answers.json in {year_dir}")

    a_num_options = 0
    for q_answers in answers.values():
        for label, answer in q_answers.items():
            if label.startswith("A") and answer.isdigit():
                a_num_options = max(a_num_options, int(answer))
    if a_num_options == 0:
        a_num_options = 5

    book_id = await uow.books.get_or_create_book(subject_id, year, a_num_options)

    questions_loaded = 0
    answers_loaded = 0

    option_dirs = sorted(
        (d for d in year_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for option_dir in option_dirs:
        option_number = int(option_dir.name)
        exam_type = "CE" if year >= 2023 and option_number <= 5 else "CT"
        option_id = await uow.books.get_or_create_option(
            book_id, option_number, exam_type
        )
        option_answers = answers.get(str(option_number), {})

        for part_dir in sorted(option_dir.iterdir()):
            if not part_dir.is_dir() or part_dir.name not in ("A", "B"):
                continue

            img_files = sorted(
                (
                    f
                    for f in part_dir.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in IMAGE_EXTENSIONS
                    and f.stem.isdigit()
                ),
                key=lambda f: int(f.stem),
            )

            for img_file in img_files:
                key = QuestionKey(part=part_dir.name, number=int(img_file.stem))  # type: ignore[arg-type]
                raw_answer = option_answers.get(str(key))
                if raw_answer:
                    try:
                        question_id = await uow.questions.get_or_create(
                            option_id, key, raw_answer
                        )
                        answers_loaded += 1
                    except ValueError as e:
                        tqdm.write(f"  Warning: {e} (skipped)")
                        fallback = "1" if key.part == "A" else ""
                        question_id = await uow.questions.get_or_create(
                            option_id, key, fallback
                        )
                else:
                    fallback = "1" if key.part == "A" else ""
                    question_id = await uow.questions.get_or_create(
                        option_id, key, fallback
                    )
                await uow.questions.insert_image(
                    question_id, key.part, img_file.read_bytes()
                )
                questions_loaded += 1

    return questions_loaded, answers_loaded


async def _populate_topics(pool, subject_id: int, subject_dir: Path) -> int:
    """Populate question_topics from topic_to_year.json. Returns mapping count."""
    topics_file = subject_dir / "topic_to_year.json"
    if not topics_file.exists():
        print(f"  No topic_to_year.json in {subject_dir}, skipping topics")
        return 0

    topics_data = json.loads(topics_file.read_text(encoding="utf-8"))

    async with UnitOfWork(pool) as uow:
        for topic_name, years in tqdm(topics_data.items(), desc="topics"):
            for year_str, exam_types in years.items():
                year = int(year_str)
                book_id = await uow.books.get_or_create_book(subject_id, year)
                for exam_type, keys in exam_types.items():
                    option_numbers = await uow.books.list_options(book_id, exam_type)
                    for key in keys:
                        part = key[0]
                        qnum = int(key[1:])
                        for option_number in option_numbers:
                            option_id = await uow.books.get_option_id(
                                book_id, option_number
                            )
                            await uow.questions.delete_topic(
                                option_id, qnum, part, topic_name
                            )
                            await uow.questions.upsert_topic(
                                option_id, qnum, part, topic_name
                            )
        return await uow.questions.count_topics()


async def populate_subject(pool, output_dir: Path, subject: str) -> None:
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
            questions, answers = await _populate_year(uow, subject_id, year_dir)
        tqdm.write(f"  {year_dir.name}: {questions} questions, {answers} answers")

    topic_count = await _populate_topics(pool, subject_id, subject_dir)
    if topic_count:
        print(f"  {topic_count} topic mappings loaded")


async def _amain() -> None:
    _alembic_upgrade()

    settings = get_settings()
    output_dir = settings.paths.extraction_output_dir

    if not output_dir.exists():
        print(f"Extraction output not found: {output_dir}")
        sys.exit(1)

    async with pool_lifespan(settings.database) as pool:
        if len(sys.argv) > 1:
            await populate_subject(pool, output_dir, sys.argv[1])
        else:
            subjects = sorted(d.name for d in output_dir.iterdir() if d.is_dir())
            if not subjects:
                print("No subjects found in extraction output.")
                sys.exit(0)
            for subject in subjects:
                await populate_subject(pool, output_dir, subject)

    print("\nDone.")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
