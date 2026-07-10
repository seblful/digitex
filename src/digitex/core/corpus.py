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


def count_images(folder: Path) -> int:
    """Count image files directly inside *folder*."""
    return sum(1 for p in folder.iterdir() if is_image(p))


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
                if not is_image(img):
                    continue
                try:
                    number = int(img.stem)
                except ValueError:
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
        ValueError: If the path has no ``books`` or ``images`` segment.
    """
    parts = page_path.parts
    return parts[parts.index("books") + 1], parts[parts.index("images") + 1]


def training_page_name(subject: str, year: str, stem: str) -> str:
    """Name a book page copied into the training images pool."""
    return f"{subject}_{year}_{stem}.jpg"
