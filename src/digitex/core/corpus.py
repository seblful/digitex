"""On-disk corpus layout — the book archive and the extraction output tree.

The corpus lives in two trees:

- book archive:      ``books/{subject}/images/{year}/{page}.{ext}`` and
  ``books/{subject}/answers/{year}_{n}.{ext}``
- extraction output: ``output/{subject}/{year}/{option}/{part}/{number}.{ext}``
  plus a per-year ``answers.json``

Every module that walks these trees or parses/formats their filenames goes
through this one, so a layout change is a one-file edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

IMAGE_EXTENSIONS: Final = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
)

_MANUAL_NAME = re.compile(r"^(\d{4})_(\d+)_([AB])_(\d+)\.png$")
_ANSWER_SHEET_STEM = re.compile(r"(\d{4})_(\d+)")


def is_image(path: Path) -> bool:
    """Return True for files with a known image extension."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


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


@dataclass(frozen=True)
class ManualImageName:
    """Parsed ``{year}_{option}_{part}_{question}.png`` manual-image filename."""

    year: int
    option: int
    part: str
    question: int

    @classmethod
    def parse(cls, filename: str) -> ManualImageName | None:
        match = _MANUAL_NAME.match(filename)
        if not match:
            return None
        return cls(
            year=int(match.group(1)),
            option=int(match.group(2)),
            part=match.group(3),
            question=int(match.group(4)),
        )


def parse_answer_sheet_stem(stem: str) -> tuple[int, int] | None:
    """Parse a ``{year}_{sheet}`` answer-sheet stem into (year, sheet_number)."""
    match = _ANSWER_SHEET_STEM.match(stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_book_page_path(page_path: Path) -> tuple[str, str]:
    """Extract (subject, year) from ``books/{subject}/images/{year}/{page}``.

    Raises:
        ValueError: If the path has no ``books`` or ``images`` segment, or the
            segment is last so nothing names the subject or year.
    """
    parts = page_path.parts
    try:
        return parts[parts.index("books") + 1], parts[parts.index("images") + 1]
    except (ValueError, IndexError) as e:
        # ValueError when a marker segment is absent, IndexError when it is last.
        # Both mean the same thing to a caller, so both say so.
        raise ValueError(f"No subject/year segment in {page_path}") from e


def training_page_name(subject: str, year: str, stem: str) -> str:
    """Name a book page copied into the training images pool."""
    return f"{subject}_{year}_{stem}.jpg"
