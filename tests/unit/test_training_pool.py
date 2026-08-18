"""Tests for the training pool — the book pages copied out for annotation.

Both entry points (``create``, which samples a books tree, and
``add_from_file``, which takes a hand-written list) funnel into the same
per-page save, so what each page is worth is the same question either way:
saved, already there, or a path the corpus layout does not recognize.

``create`` samples at random, so tests that assert on a particular output file
offer exactly one page to choose from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

from digitex.pipeline.training_pool import PageDataCreator, PageOutcome

if TYPE_CHECKING:
    from pathlib import Path


def _page(
    root: Path,
    rel: str,
    *,
    size: tuple[int, int] = (400, 400),
    mode: str = "RGB",
) -> Path:
    """Write one page image at *rel* under *root*, creating its parents."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, "red").save(path)
    return path


@pytest.fixture
def books_dir(tmp_path: Path) -> Path:
    return tmp_path / "books"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir(parents=True)
    return out


@pytest.fixture
def creator() -> PageDataCreator:
    return PageDataCreator(image_size=100)


class TestSaveOutcome:
    """Why the two not-saved outcomes are counted apart.

    "Unrecognized path" and "already present" used to be counted together, so
    an operator read the total as "already added".
    """

    def test_an_unrecognized_path_is_counted_apart_from_an_existing_one(
        self, tmp_path: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        recognized = _page(tmp_path, "books/biology/processed/pages/2016/001.jpg")
        unrecognized = _page(tmp_path, "scans/2016/1.jpg")

        first = creator._save_images([recognized], output_dir, "t")
        again = creator._save_images([recognized, unrecognized], output_dir, "t")

        assert first[PageOutcome.SAVED] == 1
        assert again[PageOutcome.ALREADY_PRESENT] == 1
        assert again[PageOutcome.UNRECOGNIZED_PATH] == 1
        assert again[PageOutcome.SAVED] == 0

    def test_a_directory_path_is_unrecognized_not_a_crash(
        self, tmp_path: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        """A year-less ``pages`` directory passes exists() and used to raise."""
        directory = tmp_path / "books" / "biology" / "processed" / "pages"
        directory.mkdir(parents=True)

        counts = creator._save_images([directory], output_dir, "t")

        assert counts[PageOutcome.UNRECOGNIZED_PATH] == 1


class TestCreate:
    def test_a_page_is_named_for_the_book_it_came_from(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        """The pool is flat, so subject and year have to survive in the name."""
        _page(books_dir, "math/processed/pages/2024/page1.jpg")

        creator.create(books_dir, output_dir, num_images=1)

        assert (output_dir / "math_2024_page1.jpg").exists()

    def test_pages_are_resized_to_the_training_size(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        _page(books_dir, "math/processed/pages/2024/page1.jpg", size=(400, 400))

        creator.create(books_dir, output_dir, num_images=1)

        with Image.open(output_dir / "math_2024_page1.jpg") as saved:
            assert saved.size == (100, 100)

    def test_resizing_preserves_the_aspect_ratio(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        """A squashed page would teach the model the wrong proportions."""
        _page(books_dir, "math/processed/pages/2024/page1.jpg", size=(400, 200))

        creator.create(books_dir, output_dir, num_images=1)

        with Image.open(output_dir / "math_2024_page1.jpg") as saved:
            assert saved.size == (100, 50)

    def test_a_transparent_page_is_flattened_to_rgb(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        """The pool is written as JPEG, which has no alpha channel to write."""
        _page(
            books_dir,
            "math/processed/pages/2024/page1.png",
            mode="RGBA",
            size=(100, 100),
        )

        creator.create(books_dir, output_dir, num_images=1)

        with Image.open(output_dir / "math_2024_page1.jpg") as saved:
            assert saved.mode == "RGB"

    def test_exactly_the_requested_number_is_taken(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        for i in range(5):
            _page(books_dir, f"math/processed/pages/2024/page{i}.jpg", size=(100, 100))

        creator.create(books_dir, output_dir, num_images=3)

        assert len(list(output_dir.glob("*.jpg"))) == 3

    def test_asking_for_more_pages_than_exist_takes_them_all(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        for i in range(2):
            _page(books_dir, f"math/processed/pages/2024/page{i}.jpg", size=(100, 100))

        creator.create(books_dir, output_dir, num_images=99)

        assert len(list(output_dir.glob("*.jpg"))) == 2

    def test_the_output_directory_is_created_when_missing(
        self, books_dir: Path, tmp_path: Path, creator: PageDataCreator
    ) -> None:
        _page(books_dir, "math/processed/pages/2024/page1.jpg", size=(100, 100))
        nested = tmp_path / "output" / "nested"

        creator.create(books_dir, nested, num_images=1)

        assert (nested / "math_2024_page1.jpg").exists()

    def test_an_empty_books_tree_is_a_failure_not_an_empty_run(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        """Silently producing nothing reads as "the pool is up to date"."""
        books_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="No images found"):
            creator.create(books_dir, output_dir, num_images=1)


class TestCreateOneSubject:
    """Sampling one subject, so a new book can be annotated on its own."""

    def test_only_the_named_subject_is_sampled(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        _page(books_dir, "biology/processed/pages/2024/page1.jpg", size=(100, 100))
        _page(books_dir, "chemistry/processed/pages/2024/page1.jpg", size=(100, 100))

        creator.create(books_dir, output_dir, num_images=99, subject="biology")

        assert [p.name for p in output_dir.glob("*.jpg")] == ["biology_2024_page1.jpg"]

    def test_a_subject_with_no_pages_is_a_failure_naming_it(
        self, books_dir: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        """Otherwise the message blames the whole archive for one empty book."""
        _page(books_dir, "biology/processed/pages/2024/page1.jpg", size=(100, 100))

        with pytest.raises(FileNotFoundError, match="chemistry"):
            creator.create(books_dir, output_dir, num_images=1, subject="chemistry")


class TestAddFromFile:
    """The hand-written list, whose paths nothing has validated."""

    def test_the_listed_pages_are_added(
        self,
        tmp_path: Path,
        books_dir: Path,
        output_dir: Path,
        creator: PageDataCreator,
    ) -> None:
        page = _page(books_dir, "math/processed/pages/2024/page1.jpg", size=(100, 100))
        listing = tmp_path / "pages.txt"
        listing.write_text(f"{page}\n", encoding="utf-8")

        creator.add_from_file(listing, output_dir)

        assert (output_dir / "math_2024_page1.jpg").exists()

    def test_a_path_that_is_not_there_is_skipped_not_fatal(
        self,
        tmp_path: Path,
        books_dir: Path,
        output_dir: Path,
        creator: PageDataCreator,
    ) -> None:
        """The list is typed by hand, so one bad line must not lose the rest."""
        page = _page(books_dir, "math/processed/pages/2024/page1.jpg", size=(100, 100))
        listing = tmp_path / "pages.txt"
        listing.write_text(
            f"{books_dir / 'math/processed/pages/2024/gone.jpg'}\n{page}\n",
            encoding="utf-8",
        )

        creator.add_from_file(listing, output_dir)

        assert (output_dir / "math_2024_page1.jpg").exists()
        assert len(list(output_dir.glob("*.jpg"))) == 1

    def test_a_blank_line_between_entries_is_ignored(
        self,
        tmp_path: Path,
        books_dir: Path,
        output_dir: Path,
        creator: PageDataCreator,
    ) -> None:
        """It has to sit between two entries — the file's own ends are stripped."""
        first = _page(books_dir, "math/processed/pages/2024/page1.jpg", size=(100, 100))
        second = _page(
            books_dir, "math/processed/pages/2024/page2.jpg", size=(100, 100)
        )
        listing = tmp_path / "pages.txt"
        listing.write_text(f"{first}\n\n   \n{second}\n", encoding="utf-8")

        creator.add_from_file(listing, output_dir)

        assert len(list(output_dir.glob("*.jpg"))) == 2

    def test_an_empty_listing_writes_nothing(
        self, tmp_path: Path, output_dir: Path, creator: PageDataCreator
    ) -> None:
        listing = tmp_path / "pages.txt"
        listing.write_text("", encoding="utf-8")

        creator.add_from_file(listing, output_dir)

        assert list(output_dir.iterdir()) == []

    def test_the_destination_is_created_when_missing(
        self, tmp_path: Path, books_dir: Path, creator: PageDataCreator
    ) -> None:
        page = _page(books_dir, "math/processed/pages/2024/page1.jpg", size=(100, 100))
        listing = tmp_path / "pages.txt"
        listing.write_text(f"{page}\n", encoding="utf-8")
        destination = tmp_path / "pool" / "nested"

        creator.add_from_file(listing, destination)

        assert (destination / "math_2024_page1.jpg").exists()
