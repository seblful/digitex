"""Fixtures for the differential suite — replaying a recorded book.

Every test here needs a recording made by ``digitex-extract record-golden``,
which needs the real checkpoint and the book it came off. Neither is in the
checkout, so the suite skips itself wherever they are missing — the same way
the integration suite skips without Docker and the UI suite without a display.

What it does *not* need is a GPU, a checkpoint or tesseract: the recording
holds every answer those would have given, so a replay is pure pixel
arithmetic. Only the pipeline extra has to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from digitex.config import get_settings
from digitex.domain.corpus import PROCESSED, book_pages_dir
from digitex.pipeline.recording import Recording, golden_dir

if TYPE_CHECKING:
    from pathlib import Path


def _recording_files() -> list[Path]:
    """Every recording in the data root, or an empty list if there are none."""
    directory = golden_dir(get_settings().paths.data_root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run every test that wants a recording once per recording found.

    Parametrized rather than fixed to one book so that recording a second one
    — a subject with a different page layout, a year that exercises page-break
    joins — extends the suite by dropping a file into the data root.
    """
    if "recording_file" not in metafunc.fixturenames:
        return

    files = _recording_files()
    metafunc.parametrize(
        "recording_file",
        files or [None],
        ids=[path.stem for path in files] or ["no-recording"],
    )


@pytest.fixture
def recording(recording_file: Path | None) -> Recording:
    """Load one recording, or skip when the data root holds none."""
    if recording_file is None:
        pytest.skip(
            "No recording in the data root — make one with"
            " `digitex-extract record-golden <subject> <year>`"
        )
    return Recording.load(recording_file)


@pytest.fixture
def recorded_pages(recording: Recording) -> Path:
    """The page directory the recording came off.

    Skips rather than fails when the book has moved: a recording outliving the
    scans it was taken from is a stale fixture, not a regression.
    """
    subject, _, year = recording.source.partition("/")
    books_dir = get_settings().paths.books_dir
    pages_dir = book_pages_dir(books_dir, subject, PROCESSED) / year
    if not pages_dir.is_dir():
        pytest.skip(f"Recorded book {recording.source} is no longer at {pages_dir}")
    return pages_dir
