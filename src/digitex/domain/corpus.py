"""On-disk corpus layout — the book archive and the extraction output tree.

The corpus lives in two trees:

- book archive:      ``books/{subject}/{variant}/pages/{year}/{page}.{ext}``
  and ``books/{subject}/{variant}/answers/{year}_{n}.{ext}``, where *variant*
  is ``raw`` (the scans as they arrived) or ``processed`` (the same files,
  corrected, file for file), plus a per-subject ``topics.json`` above both
  variants
- extraction output: ``output/{subject}/{year}/{option}/{part}/{number}.{ext}``
  plus a per-year ``answers.json``

Every module that walks these trees or parses/formats their filenames goes
through this one, so a layout change is a one-file edit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

IMAGE_EXTENSIONS: Final = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
)

# The sheet number is optional: a year that fits on one sheet is often exported
# as just ``2016``, and that sheet is sheet 1.
_ANSWER_SHEET_STEM = re.compile(r"(\d{4})(?:_(\d+))?")

PAGE_NUMBER_WIDTH: Final = 3

# The two variants a subject's scans exist in. ``raw`` is what came off the
# scanner and is never written to again; ``processed`` is derived from it and
# can be rebuilt at any time.
RAW: Final = "raw"
PROCESSED: Final = "processed"


def is_image(path: Path) -> bool:
    """Return True for files with a known image extension."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def book_variant_dir(books_dir: Path, subject: str, variant: str) -> Path:
    """Root of one subject's scans in one variant."""
    return books_dir / subject / variant


def book_pages_dir(books_dir: Path, subject: str, variant: str) -> Path:
    """Where a subject's scanned pages live, a directory per year below."""
    return book_variant_dir(books_dir, subject, variant) / "pages"


def book_answers_dir(books_dir: Path, subject: str, variant: str) -> Path:
    """Where a subject's answer sheets live."""
    return book_variant_dir(books_dir, subject, variant) / "answers"


def book_topics_file(books_dir: Path, subject: str) -> Path:
    """Where a subject's topic map lives — hand-written, so above the variants.

    Topics name questions by year and key, not by scan, so the file belongs to
    the subject rather than to ``raw`` or ``processed``.
    """
    return books_dir / subject / "topics.json"


def book_subjects(books_dir: Path) -> list[str]:
    """Every subject the archive holds, named by its directory."""
    if not books_dir.is_dir():
        return []
    return sorted(path.name for path in books_dir.iterdir() if path.is_dir())


def walk_book_pages(
    books_dir: Path, variant: str, subject: str | None = None
) -> Iterator[Path]:
    """Every scanned page in one variant, of *subject* or of every subject.

    Answer sheets are not pages and are not yielded — they sit beside
    ``pages`` rather than under it.
    """
    subjects = [subject] if subject is not None else book_subjects(books_dir)
    for name in subjects:
        pages_dir = book_pages_dir(books_dir, name, variant)
        if not pages_dir.is_dir():
            continue
        for path in sorted(pages_dir.rglob("*")):
            if is_image(path):
                yield path


def book_page_name(number: int, image_format: str) -> str:
    """Name a scanned page by its position in the year, zero-padded.

    The subject, variant and year are all in the path already, so the filename
    carries only the page number. Padded, because reading order and
    lexicographic order then agree everywhere the corpus is looked at — a file
    browser, a Label Studio task list — and not just where the code remembers
    to sort numerically.
    """
    return f"{number:0{PAGE_NUMBER_WIDTH}d}.{image_format}"


def question_image_number(path: Path) -> int | None:
    """The question number *path* carries, or None if it is not a question image.

    Question images are named ``{number}.{ext}`` with a positive integer, so a
    stem that is not all digits belongs to something else — a stray export, a
    thumbnail. Every walker of the output tree asks this one question, and each
    decides for itself whether to warn about a None.
    """
    if not is_image(path) or not path.stem.isdigit():
        return None
    return int(path.stem)


def question_image_path(
    year_dir: Path, option: int | str, part: str, number: int, image_format: str
) -> Path:
    """Where one question's image is written under a year's output tree."""
    return year_dir / str(option) / part / f"{number}.{image_format}"


def question_slot_taken(
    year_dir: Path, option: int | str, part: str, number: int
) -> bool:
    """True when a question image already occupies this option/part/number.

    Format-agnostic on purpose: one run can have written ``1.png`` and the next
    be configured for jpg, and either file means the slot is taken.
    """
    folder = year_dir / str(option) / part
    return folder.is_dir() and any(
        question_image_number(path) == number for path in folder.glob(f"{number}.*")
    )


def highest_question_number(year_dir: Path, option: int | str, part: str) -> int:
    """The highest question number already written to one option/part, 0 if none.

    What the next question written there must be numbered, minus one — so a
    caller can tell a continuation from a collision or a gap before writing.
    """
    folder = year_dir / str(option) / part
    if not folder.is_dir():
        return 0
    numbers = (question_image_number(path) for path in folder.iterdir())
    return max((number for number in numbers if number is not None), default=0)


def question_object_key(output_dir: Path, image_path: Path) -> str:
    """The stored key for a question image: its path relative to the corpus root.

    POSIX-separated whichever platform writes it, because the key is seeded from
    a Windows laptop and resolved on a Linux server — a backslash in the column
    would name nothing there.
    """
    return image_path.relative_to(output_dir).as_posix()


def file_digest(path: Path) -> str:
    """Hex SHA-256 of a file's contents.

    Seeded alongside the key so a question re-extracted to the same path is
    still recognisable as changed — see ``QuestionCorpus.set_image``.
    """
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort key that orders embedded numbers numerically.

    Keeps ``page_2`` ahead of ``page_10``, which plain lexicographic sorting
    gets backwards — and page order decides question numbering.
    """
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.stem):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return parts


@dataclass(frozen=True)
class QuestionImage:
    """One numbered question image inside a year's extraction output."""

    option: str
    part: str
    number: int
    path: Path


def walk_question_images(year_dir: Path) -> Iterator[QuestionImage]:
    """Yield every numbered question image under ``{option}/{part}/``.

    Image files whose stem is not an integer are skipped.
    """
    for option_dir in year_dir.iterdir():
        if not option_dir.is_dir():
            continue
        for part_dir in option_dir.iterdir():
            if not part_dir.is_dir():
                continue
            for img in part_dir.iterdir():
                number = question_image_number(img)
                if number is None:
                    continue
                yield QuestionImage(option_dir.name, part_dir.name, number, img)


def parse_answer_sheet_stem(stem: str) -> tuple[int, int] | None:
    """Parse a ``{year}`` or ``{year}_{sheet}`` stem into (year, sheet_number).

    A year whose key fits on one sheet is often exported under the bare year,
    and reading that as sheet 1 is what the name means. Only the year is ever
    used downstream; the sheet number orders a year's sheets and keeps their
    names apart.
    """
    match = _ANSWER_SHEET_STEM.match(stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 1)


def parse_book_page_path(page_path: Path) -> tuple[str, str]:
    """Extract (subject, year) from ``{subject}/{variant}/pages/{year}/{page}``.

    Anchored on the ``pages`` segment — the subject sits two above it, the
    year directly below — so a page parses the same in either variant, and a
    raw page and its processed twin give the same
    :func:`training_page_name`.

    Raises:
        ValueError: If the path has no ``pages`` segment, or the segment sits
            too near an end for a subject and a year to be around it.
    """
    parts = page_path.parts
    marker = parts.index("pages") if "pages" in parts else 0
    # Two above and one below is the whole rule; anything shorter is some other
    # tree that happens to have a ``pages`` directory in it.
    if marker < 2 or marker + 1 >= len(parts):
        raise ValueError(f"No subject/year segment in {page_path}")
    return parts[marker - 2], parts[marker + 1]


def training_page_name(subject: str, year: str, stem: str) -> str:
    """Name a book page copied into the training images pool."""
    return f"{subject}_{year}_{stem}.jpg"
