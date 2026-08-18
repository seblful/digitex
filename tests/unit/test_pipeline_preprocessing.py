"""Tests for the archive preparation — canonical names, then corrected pixels.

Three things decide whether this stage is safe to build a corpus on: that a
processed scan lands at the mirrored path with the mirrored name, that it keeps
the raw scan's geometry, and that renaming a page carries its processed twin
along. The first two are what let every other module go on parsing paths the
way it always did and let annotations drawn on one variant apply to the other;
the third is what stops the variants drifting apart.

The plan is tested apart from the run, because deciding what to do is cheap and
doing it costs a process pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from digitex.imaging import correct_document
from digitex.pipeline.exceptions import DirectoryNotFoundError
from digitex.pipeline.preprocessing import (
    _plan,
    preprocess_scan,
    preprocess_scans,
    rename_pages,
)

if TYPE_CHECKING:
    from pathlib import Path


def _scan(path: Path, *, paper: int = 210, ink: int = 30, margin: int = 0) -> Path:
    """Write a small gray-paper scan with a band of ink, and grain on both.

    The grain is not decoration: the correction reads the *shape* of the
    histogram's peaks, so a page of two flat tones is not the page the
    algorithm was written against. *margin* surrounds the page with the
    scanner's pure-white canvas, which is what the crop looks for.
    """
    rng = np.random.default_rng(0)
    pixels = np.full((160, 120), 255.0)
    page = rng.normal(paper, 4, (160 - 2 * margin, 120 - 2 * margin)).clip(0, 255)
    pixels[margin : 160 - margin, margin : 120 - margin] = page
    pixels[60:100, 20:100] = rng.normal(ink, 6, (40, 80)).clip(0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)
    return path


@pytest.fixture
def books_dir(tmp_path: Path) -> Path:
    return tmp_path / "books"


class TestPlan:
    def test_a_scan_keeps_its_place_and_changes_its_format(
        self, books_dir: Path
    ) -> None:
        """``{subject}/{variant}/images/{year}/`` is what every module parses."""
        _scan(books_dir / "biology" / "raw" / "images" / "2016" / "001.jpg")

        work, _ = _plan(books_dir, force=False)

        assert [target for _, target in work] == [
            books_dir / "biology" / "processed" / "images" / "2016" / "001.png"
        ]

    def test_answer_sheets_are_processed_too(self, books_dir: Path) -> None:
        """A vision model reads them, and it reads them better cleaned."""
        _scan(books_dir / "biology" / "raw" / "images" / "2016" / "001.jpg")
        _scan(books_dir / "biology" / "raw" / "answers" / "2016_1.jpg")

        work, _ = _plan(books_dir, force=False)

        assert sorted(source.name for source, _ in work) == ["001.jpg", "2016_1.jpg"]

    def test_every_subject_is_walked(self, books_dir: Path) -> None:
        _scan(books_dir / "biology" / "raw" / "images" / "2016" / "001.jpg")
        _scan(books_dir / "chemistry" / "raw" / "images" / "2016" / "001.png")

        work, _ = _plan(books_dir, force=False)

        assert len(work) == 2

    def test_an_existing_output_is_skipped_until_forced(self, books_dir: Path) -> None:
        """The steady state is a full archive and an empty plan."""
        _scan(books_dir / "biology" / "raw" / "images" / "2016" / "001.jpg")
        _scan(books_dir / "biology" / "processed" / "images" / "2016" / "001.png")

        skipping, skipped = _plan(books_dir, force=False)
        forced, none_skipped = _plan(books_dir, force=True)

        assert (skipping, skipped) == ([], 1)
        assert len(forced) == 1
        assert none_skipped == 0


class TestPreprocessScan:
    def test_the_paper_goes_white(self, tmp_path: Path) -> None:
        source = _scan(tmp_path / "001.jpg")
        target = tmp_path / "001.png"

        preprocess_scan(source, target)

        with Image.open(target) as after:
            pixels = np.array(after)
        # The paper was 210 and carries most of the page.
        assert np.median(pixels) == 255
        # The ink is still ink — a page burnt out to blank would also pass the
        # assertion above.
        assert pixels.min() < 60

    def test_an_answer_sheet_keeps_the_plain_correction(self, tmp_path: Path) -> None:
        """A sheet's printed shading is content, so the flatten stays off."""
        source = _scan(tmp_path / "answers" / "2016_1.png")
        target = tmp_path / "out.png"

        preprocess_scan(source, target)

        with Image.open(source) as scan, Image.open(target) as got:
            plain = correct_document(scan, flatten=False)
            flattened = correct_document(scan, flatten=True)
            assert np.array_equal(np.array(got), np.array(plain))
            assert not np.array_equal(np.array(got), np.array(flattened))

    def test_the_scanners_canvas_is_cut_off(self, tmp_path: Path) -> None:
        """Blank border is model input spent on nothing — about 6% of a real page."""
        source = _scan(tmp_path / "001.jpg", margin=20)
        target = tmp_path / "001.png"

        preprocess_scan(source, target)

        with Image.open(source) as before, Image.open(target) as after:
            assert after.size == (before.width - 40, before.height - 40)


class TestPreprocessScans:
    def test_a_missing_archive_is_the_callers_problem(self, books_dir: Path) -> None:
        with pytest.raises(DirectoryNotFoundError):
            preprocess_scans(books_dir)

    def test_nothing_to_do_reports_the_scans_it_left_alone(
        self, books_dir: Path
    ) -> None:
        """Processed nothing, skipped everything is up to date, not a failure."""
        _scan(books_dir / "biology" / "raw" / "images" / "2016" / "001.jpg")
        _scan(books_dir / "biology" / "processed" / "images" / "2016" / "001.png")

        result = preprocess_scans(books_dir)

        assert (result.processed, result.skipped, result.failed) == (0, 1, 0)

    def test_one_unreadable_scan_does_not_cost_the_run(self, books_dir: Path) -> None:
        """A thousand pages is too many to restart because one is truncated."""
        _scan(books_dir / "biology" / "raw" / "images" / "2016" / "001.jpg")
        _scan(books_dir / "biology" / "raw" / "images" / "2017" / "001.jpg")
        broken = books_dir / "biology" / "raw" / "images" / "2017" / "002.jpg"
        broken.write_bytes(b"not an image")

        result = preprocess_scans(books_dir)

        assert (result.processed, result.failed) == (2, 1)
        assert "002.jpg" in result.errors[0]
        processed = books_dir / "biology" / "processed" / "images"
        assert (processed / "2016" / "001.png").exists()
        assert (processed / "2017" / "001.png").exists()


class TestRenamePages:
    def test_pages_are_renumbered_in_reading_order_and_padded(
        self, books_dir: Path
    ) -> None:
        """``10.jpg`` sorts ahead of ``2.jpg`` anywhere that reads names flat."""
        year = books_dir / "biology" / "raw" / "images" / "2016"
        for name in ("1.jpg", "2.jpg", "10.jpg"):
            _scan(year / name)

        result = rename_pages(books_dir)

        assert sorted(p.name for p in year.iterdir()) == [
            "001.jpg",
            "002.jpg",
            "003.jpg",
        ]
        assert (result.renamed, result.unchanged) == (3, 0)

    def test_a_scanners_own_name_keeps_its_format(self, books_dir: Path) -> None:
        """Renaming is not converting — a PNG stays a PNG."""
        year = books_dir / "chemistry" / "raw" / "images" / "2016"
        _scan(year / "Химия.001.png")
        _scan(year / "Химия.002.png")

        rename_pages(books_dir)

        assert sorted(p.name for p in year.iterdir()) == ["001.png", "002.png"]

    def test_numbering_restarts_each_year(self, books_dir: Path) -> None:
        """A page number means the nth page of this book, and a book is a year."""
        images = books_dir / "biology" / "raw" / "images"
        _scan(images / "2016" / "7.jpg")
        _scan(images / "2017" / "9.jpg")

        rename_pages(books_dir)

        assert (images / "2016" / "001.jpg").exists()
        assert (images / "2017" / "001.jpg").exists()

    def test_the_processed_twin_follows_its_page(self, books_dir: Path) -> None:
        """Or the two variants stop agreeing about what a page is called."""
        _scan(books_dir / "biology" / "raw" / "images" / "2016" / "5.jpg")
        _scan(books_dir / "biology" / "processed" / "images" / "2016" / "5.png")

        rename_pages(books_dir)

        processed = books_dir / "biology" / "processed" / "images" / "2016"
        assert [p.name for p in processed.iterdir()] == ["001.png"]

    def test_answer_sheets_are_left_alone(self, books_dir: Path) -> None:
        """``{year}_{n}`` is what says which year and sheet a sheet is."""
        sheets = books_dir / "biology" / "raw" / "answers"
        _scan(sheets / "2016_1.jpg")

        rename_pages(books_dir)

        assert [p.name for p in sheets.iterdir()] == ["2016_1.jpg"]

    def test_an_already_named_archive_is_left_alone(self, books_dir: Path) -> None:
        """What makes the steady state cheap and re-running safe."""
        year = books_dir / "biology" / "raw" / "images" / "2016"
        _scan(year / "001.jpg")
        _scan(year / "002.jpg")

        result = rename_pages(books_dir)

        assert (result.renamed, result.unchanged, result.failed) == (0, 2, 0)

    def test_a_taken_name_leaves_both_variants_as_they_were(
        self, books_dir: Path
    ) -> None:
        """A stale processed file must not be renamed over, or it is lost.

        Both variants are checked before either moves, so the page is reported
        and skipped rather than half-renamed into a tree that disagrees with
        itself.
        """
        year = books_dir / "biology" / "raw" / "images" / "2016"
        processed = books_dir / "biology" / "processed" / "images" / "2016"
        _scan(year / "5.jpg")
        _scan(processed / "5.png")
        _scan(processed / "001.png")

        result = rename_pages(books_dir)

        assert result.failed == 1
        assert (year / "5.jpg").exists()
        assert sorted(p.name for p in processed.iterdir()) == ["001.png", "5.png"]
