"""Populate the database from extraction output.

Runs ``alembic upgrade head`` before populating so the schema is always at the
latest revision. Idempotent — safe to re-run.

Usage::

    uv run python scripts/populate_db.py              # all subjects
    uv run python scripts/populate_db.py biology      # single subject
    uv run python scripts/populate_db.py --help
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import TYPE_CHECKING, Annotated

import typer
from alembic import command
from alembic.config import Config
from tqdm import tqdm

from digitex.config import BASE_DIR, get_settings
from digitex.core.corpus import question_image_number
from digitex.core.db import UnitOfWork, null_pool_lifespan
from digitex.core.domain import QuestionKey, exam_type_for, parse_exam_type

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(help="Load extraction output into the database.")


def _abort(message: str) -> typer.Exit:
    """Render *message* on stderr and return the exit to raise."""
    typer.echo(typer.style(message, fg="red", bold=True), err=True)
    return typer.Exit(code=1)


# A Question whose answers.json entry is missing or unusable is still loaded, so
# its image is servable — but with an answer no reply can match. Part A answers
# are integers and the option buttons start at 1, so 0 is unreachable; Part B
# compares against text, and "" matches nothing.
PLACEHOLDER_ANSWER = {"A": "0", "B": ""}

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
    # BASE_DIR, not this file's parent: alembic.ini and migrations/ ship with
    # the package, wherever the process happens to be running from.
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    command.upgrade(cfg, "head")


async def _populate_year(
    uow: UnitOfWork, subject_id: int, year_dir: Path
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
                        tqdm.write(f"  Warning: {e} — storing placeholder answer")

                if question_id is None:
                    question_id = await uow.questions.get_or_create(
                        option_id, key, PLACEHOLDER_ANSWER[key.part]
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
                                topic_name,
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


async def _amain(subject: str | None) -> None:
    _alembic_upgrade()

    settings = get_settings()
    output_dir = settings.paths.extraction_output_dir

    if not output_dir.exists():
        raise _abort(f"Extraction output not found: {output_dir}")

    async with null_pool_lifespan(settings.database) as pool:
        if subject is not None:
            await populate_subject(pool, output_dir, subject)
        else:
            subjects = sorted(d.name for d in output_dir.iterdir() if d.is_dir())
            if not subjects:
                typer.echo("No subjects found in extraction output.")
                return
            for name in subjects:
                await populate_subject(pool, output_dir, name)

    typer.echo("\nDone.")


@app.command()
def populate(
    subject: Annotated[
        str | None,
        typer.Argument(help="Subject to load; omit to load every subject"),
    ] = None,
) -> None:
    """Load extraction output into the database, migrating the schema first."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_amain(subject))


if __name__ == "__main__":
    app()
