"""Get the scan archive into the shape everything downstream reads.

Two passes over ``books/{subject}/raw/``, both safe to re-run, and
:func:`preprocess_scans` runs them in order:

- :func:`rename_pages` gives every page its canonical ``{number}.{ext}``,
  numbered in reading order within its year. Scanner exports arrive named for
  the batch that made them (``Химия.001.png``) or for nothing in particular
  (``10.jpg``, which sorts ahead of ``2.jpg``). Answer sheets keep their names:
  ``{year}_{n}`` is what says which year and sheet they are.
- :func:`preprocess_scans` corrects every scan — pages and answer sheets alike
  — into ``books/{subject}/processed/``.

Correcting ahead of time rather than on the way into each run is what keeps the
corpus honest: annotation, training and inference all see the same bytes, so a
model cannot be trained on one rendering of a page and asked to predict on
another.

The scanner's white canvas is cut off on the way out, which is worth about 6%
of a page and is why nothing downstream ever wants the raw variant. It also
means a processed scan is *not* pixel-aligned with its original — the crop box
is measured per scan — so geometry only ever refers to the processed file. Draw
annotations on the processed page, and reprocess with ``force`` only when you
are willing to redraw them: a correction that moves an edge moves every
percentage coordinate with it.

A scan costs a few seconds and the archive rebuilds across a process pool, so
there is no cache to invalidate: an existing output is skipped, and ``force``
redoes the lot.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from PIL import Image
from tqdm import tqdm

from digitex.domain.corpus import (
    PROCESSED,
    RAW,
    book_page_name,
    book_pages_dir,
    book_subjects,
    book_variant_dir,
    is_image,
    natural_sort_key,
)
from digitex.imaging import correct_document
from digitex.pipeline.exceptions import DirectoryNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()

PROCESSED_FORMAT = "png"


@dataclass(frozen=True)
class PreprocessResult:
    """What one pass over the archive did.

    ``skipped`` is scans already processed, which is the steady state — a run
    that processes nothing and skips everything is the archive being up to
    date, not a failure. ``renamed`` is the pages the run gave a canonical name
    before correcting anything.
    """

    processed: int = 0
    skipped: int = 0
    renamed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.errors)


@dataclass(frozen=True)
class RenameResult:
    """What one renaming pass did, ``unchanged`` being pages already correct."""

    renamed: int = 0
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.errors)


def preprocess_scan(source: Path, target: Path) -> None:
    """Correct one scan onto *target* — the unit of work a worker takes.

    Answer sheets keep the plain correction, without the shadow flatten: their
    printed row shading is content, and the flatten would bleach it to white
    wherever a fold's shadow crossed it. Pages carry no light print worth
    keeping, so they get the full treatment.

    Module-level, and taking only paths, because a process pool has to pickle
    it by name and hand it arguments that survive the trip.
    """
    with Image.open(source) as scan:
        dpi = scan.info.get("dpi")
        corrected = correct_document(scan, flatten="answers" not in source.parts)
    corrected.save(target, **({"dpi": dpi} if dpi else {}))


def _twin(source: Path, raw_root: Path, processed_root: Path) -> Path:
    """Where *source*'s processed counterpart sits, same place, same name."""
    return (processed_root / source.relative_to(raw_root)).with_suffix(
        f".{PROCESSED_FORMAT}"
    )


def _scans(raw_root: Path) -> list[Path]:
    """Every image under one subject's raw tree — pages and answer sheets."""
    return sorted(path for path in raw_root.rglob("*") if is_image(path))


def _plan(books_dir: Path, *, force: bool) -> tuple[list[tuple[Path, Path]], int]:
    """Pair each raw scan with where it goes, dropping the ones already done.

    Returns the work and the number skipped, so a caller can report both
    without walking the archive twice.
    """
    work: list[tuple[Path, Path]] = []
    skipped = 0
    for subject in book_subjects(books_dir):
        raw_root = book_variant_dir(books_dir, subject, RAW)
        processed_root = book_variant_dir(books_dir, subject, PROCESSED)
        for source in _scans(raw_root):
            target = _twin(source, raw_root, processed_root)
            if target.exists() and not force:
                skipped += 1
                continue
            work.append((source, target))
    return work, skipped


def preprocess_scans(books_dir: Path, *, force: bool = False) -> PreprocessResult:
    """Name every page canonically, then correct the raw scans into *processed*.

    :func:`rename_pages` runs first, because a processed scan is written under
    its raw page's name: correcting a page called ``bio.01.png`` and renaming
    afterwards would move a file this run had just written, and a run that
    stopped in between would leave the archive half-named.

    Scans go out to worker processes: the filter is seconds of arithmetic each
    and no scan depends on another. One that cannot be read is counted and
    named rather than raised, so a single bad file does not cost the run.

    Args:
        books_dir: Root of the archive, ``{subject}/{variant}/`` below.
        force: Reprocess scans that already have an output.

    Raises:
        DirectoryNotFoundError: If *books_dir* does not exist.
    """
    if not books_dir.exists():
        raise DirectoryNotFoundError(books_dir)

    rename = rename_pages(books_dir)

    work, skipped = _plan(books_dir, force=force)
    if not work:
        logger.info("Nothing to preprocess", books_dir=str(books_dir), skipped=skipped)
        return PreprocessResult(
            skipped=skipped, renamed=rename.renamed, errors=rename.errors
        )

    # Once, here, rather than racing to create the same year directory from
    # twenty workers.
    for _, target in work:
        target.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    with ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(preprocess_scan, source, target): source
            for source, target in work
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Preprocessing scans"
        ):
            source = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(
                    "Failed to preprocess scan",
                    path=str(source),
                    error=str(e),
                    exc_info=True,
                )
                errors.append(f"{source.name}: {e}")

    logger.info(
        "Preprocessed archive",
        books_dir=str(books_dir),
        processed=len(work) - len(errors),
        skipped=skipped,
        renamed=rename.renamed,
        failed=len(errors) + rename.failed,
    )
    return PreprocessResult(
        processed=len(work) - len(errors),
        skipped=skipped,
        renamed=rename.renamed,
        errors=rename.errors + errors,
    )


def _years(pages: list[Path]) -> dict[Path, list[Path]]:
    """Pages grouped by their year directory, each group in reading order.

    Numbering restarts per year, because a page number means "the nth page of
    this book", and a book is one subject's one year.
    """
    groups: defaultdict[Path, list[Path]] = defaultdict(list)
    for page in pages:
        groups[page.parent].append(page)
    return {year: sorted(found, key=natural_sort_key) for year, found in groups.items()}


def rename_pages(books_dir: Path) -> RenameResult:
    """Renumber every page to its canonical name, in both variants at once.

    A page's processed twin follows it, so the two trees never disagree about
    what a page is called. Pages already correctly named are left alone, which
    is what makes this safe to run over a half-renamed archive — and what makes
    the steady state cheap.

    Args:
        books_dir: Root of the archive, ``{subject}/{variant}/`` below.

    Raises:
        DirectoryNotFoundError: If *books_dir* does not exist.
    """
    if not books_dir.exists():
        raise DirectoryNotFoundError(books_dir)

    renamed = 0
    unchanged = 0
    errors: list[str] = []

    for subject in book_subjects(books_dir):
        raw_root = book_variant_dir(books_dir, subject, RAW)
        processed_root = book_variant_dir(books_dir, subject, PROCESSED)
        pages_dir = book_pages_dir(books_dir, subject, RAW)
        if not pages_dir.is_dir():
            continue

        pages = [path for path in pages_dir.rglob("*") if is_image(path)]
        for year in _years(pages).values():
            for number, source in enumerate(year, start=1):
                target = source.with_name(
                    book_page_name(number, source.suffix.lstrip("."))
                )
                if target == source:
                    unchanged += 1
                    continue

                twin = _twin(source, raw_root, processed_root)
                twin_target = _twin(target, raw_root, processed_root)
                # Checked before anything moves, and the twin moves first, so
                # a name already taken leaves both variants as they were rather
                # than half-renamed.
                if target.exists() or (twin.exists() and twin_target.exists()):
                    errors.append(f"{source.name}: {target.name} is already taken")
                    continue

                if twin.exists():
                    twin.rename(twin_target)
                source.rename(target)
                renamed += 1

    logger.info(
        "Renamed pages",
        books_dir=str(books_dir),
        renamed=renamed,
        unchanged=unchanged,
        failed=len(errors),
    )
    return RenameResult(renamed=renamed, unchanged=unchanged, errors=errors)
