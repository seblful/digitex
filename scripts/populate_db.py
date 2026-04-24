"""Populate database from extraction output. Initializes DB automatically if needed.

Idempotent — safe to re-run.

Usage:
    python scripts/populate_db.py              # all subjects
    python scripts/populate_db.py biology      # single subject
"""

import json
import sqlite3
import sys
from pathlib import Path

from tqdm import tqdm

from digitex.config import get_settings
from digitex.core.db import UnitOfWork
from digitex.core.value_objects import QuestionKey

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SQL_SCRIPT_PATH = Path("scripts/script.sql")

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


def init_db(db_path: str) -> None:
    db_file = Path(db_path)
    if db_file.exists():
        return
    db_file.parent.mkdir(parents=True, exist_ok=True)
    sql = SQL_SCRIPT_PATH.read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql)
        conn.commit()
        print(f"Initialized database at {db_path}")
    finally:
        conn.close()


def _populate_year(uow: UnitOfWork, subject_id: int, year_dir: Path) -> tuple[int, int]:
    """Returns (questions_loaded, answers_loaded)."""
    year = int(year_dir.name)
    book_id = uow.books.get_or_create_book(subject_id, year)

    answers: dict[str, dict[str, str]] = {}
    answers_file = year_dir / "answers.json"
    if answers_file.exists():
        answers = json.loads(answers_file.read_text(encoding="utf-8"))
    else:
        tqdm.write(f"  Warning: no answers.json in {year_dir}")

    questions_loaded = 0
    answers_loaded = 0

    option_dirs = sorted(
        (d for d in year_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for option_dir in option_dirs:
        option_number = int(option_dir.name)
        option_id = uow.books.get_or_create_option(book_id, option_number)
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
                question_id = uow.questions.get_or_create(option_id, key)
                uow.questions.insert_image(question_id, img_file.read_bytes())
                questions_loaded += 1

                raw_answer = option_answers.get(str(key))
                if raw_answer:
                    try:
                        uow.questions.insert_answer(question_id, key, raw_answer)
                        answers_loaded += 1
                    except ValueError as e:
                        tqdm.write(f"  Warning: {e} (skipped)")

    return questions_loaded, answers_loaded


def populate_subject(db_path: str, output_dir: Path, subject: str) -> None:
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

    for year_dir in tqdm(year_dirs, desc=subject):
        with UnitOfWork(db_path) as uow:
            subject_id = uow.books.get_or_create_subject(get_subject_name(subject))
            questions, answers = _populate_year(uow, subject_id, year_dir)
        tqdm.write(f"  {year_dir.name}: {questions} questions, {answers} answers")


def main() -> None:
    settings = get_settings()
    output_dir = settings.paths.extraction_output_dir
    db_path = settings.database.path

    if not output_dir.exists():
        print(f"Extraction output not found: {output_dir}")
        sys.exit(1)

    init_db(db_path)

    if len(sys.argv) > 1:
        populate_subject(db_path, output_dir, sys.argv[1])
    else:
        subjects = sorted(d.name for d in output_dir.iterdir() if d.is_dir())
        if not subjects:
            print("No subjects found in extraction output.")
            sys.exit(0)
        for subject in subjects:
            populate_subject(db_path, output_dir, subject)

    print("\nDone.")


if __name__ == "__main__":
    main()
