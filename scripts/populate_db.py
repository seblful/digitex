"""Populate database from extraction output.

Run after init_db.py. Idempotent — safe to re-run.

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# --- helpers -----------------------------------------------------------------

def _get_or_create(cur: sqlite3.Cursor, table: str, id_col: str, where: dict) -> int:
    cols = ", ".join(where.keys())
    placeholders = ", ".join("?" * len(where))
    cur.execute(
        f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})",
        tuple(where.values()),
    )
    conditions = " AND ".join(f"{k} = ?" for k in where)
    return cur.execute(
        f"SELECT {id_col} FROM {table} WHERE {conditions}",
        tuple(where.values()),
    ).fetchone()[0]


def _subject_id(cur: sqlite3.Cursor, name: str) -> int:
    return _get_or_create(cur, "subjects", "subject_id", {"name": name})


def _book_id(cur: sqlite3.Cursor, subject_id: int, year: int) -> int:
    return _get_or_create(cur, "books", "book_id", {"subject_id": subject_id, "year_value": year})


def _option_id(cur: sqlite3.Cursor, book_id: int, option_number: int) -> int:
    return _get_or_create(cur, "options", "option_id", {"book_id": book_id, "option_number": option_number})


def _question_id(cur: sqlite3.Cursor, option_id: int, part: str, question_number: int) -> int:
    return _get_or_create(cur, "questions", "question_id", {
        "option_id": option_id,
        "part": part,
        "question_number": question_number,
    })


def _insert_image(cur: sqlite3.Cursor, question_id: int, image_data: bytes) -> None:
    cur.execute(
        "INSERT OR IGNORE INTO images (question_id, image_data, is_table, image_order) VALUES (?, ?, 0, 1)",
        (question_id, image_data),
    )


def _insert_answer(cur: sqlite3.Cursor, question_id: int, part: str, answer: str) -> None:
    if part == "A":
        if not answer.isdigit():
            tqdm.write(f"  Warning: unexpected Part A answer '{answer}' for question_id {question_id}, skipping")
            return
        cur.execute(
            "INSERT OR IGNORE INTO part_a_answers (question_id, correct_order) VALUES (?, ?)",
            (question_id, int(answer)),
        )
    else:
        cur.execute(
            "INSERT OR IGNORE INTO part_b_answers (question_id, answer_text) VALUES (?, ?)",
            (question_id, answer),
        )


# --- core --------------------------------------------------------------------

def _populate_year(cur: sqlite3.Cursor, subject_id: int, year_dir: Path) -> tuple[int, int]:
    """Returns (questions_loaded, answers_loaded)."""
    year = int(year_dir.name)
    book_id = _book_id(cur, subject_id, year)

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
        option_id = _option_id(cur, book_id, option_number)
        option_answers = answers.get(str(option_number), {})

        for part_dir in sorted(option_dir.iterdir()):
            if not part_dir.is_dir() or part_dir.name not in ("A", "B"):
                continue

            part = part_dir.name
            img_files = sorted(
                (f for f in part_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f.stem.isdigit()),
                key=lambda f: int(f.stem),
            )

            for img_file in img_files:
                question_number = int(img_file.stem)
                question_id = _question_id(cur, option_id, part, question_number)
                _insert_image(cur, question_id, img_file.read_bytes())
                questions_loaded += 1

                answer_key = f"{part}{question_number}"
                if answer_key in option_answers:
                    _insert_answer(cur, question_id, part, option_answers[answer_key])
                    answers_loaded += 1

    return questions_loaded, answers_loaded


def populate_subject(conn: sqlite3.Connection, output_dir: Path, subject: str) -> None:
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
    cur = conn.cursor()
    subject_id = _subject_id(cur, subject)

    for year_dir in tqdm(year_dirs, desc=subject):
        questions, answers = _populate_year(cur, subject_id, year_dir)
        tqdm.write(f"  {year_dir.name}: {questions} questions, {answers} answers")

    conn.commit()


def main() -> None:
    settings = get_settings()
    output_dir = settings.paths.extraction_output_dir
    db_path = settings.database.path

    if not output_dir.exists():
        print(f"Extraction output not found: {output_dir}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    try:
        if len(sys.argv) > 1:
            populate_subject(conn, output_dir, sys.argv[1])
        else:
            subjects = sorted(d.name for d in output_dir.iterdir() if d.is_dir())
            if not subjects:
                print("No subjects found in extraction output.")
                sys.exit(0)
            for subject in subjects:
                populate_subject(conn, output_dir, subject)
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
