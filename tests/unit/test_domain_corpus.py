"""Tests for the corpus layout module."""

from pathlib import Path

import pytest

from digitex.domain.corpus import (
    PROCESSED,
    QuestionImage,
    book_page_name,
    file_digest,
    highest_question_number,
    is_image,
    natural_sort_key,
    parse_answer_sheet_stem,
    parse_book_page_path,
    question_image_path,
    question_object_key,
    question_slot_taken,
    training_page_name,
    walk_book_pages,
    walk_question_images,
)


class TestQuestionObjectKey:
    def test_key_is_the_path_below_the_corpus_root(self, tmp_path: Path) -> None:
        image = tmp_path / "biology" / "2016" / "1" / "A" / "3.jpg"
        assert question_object_key(tmp_path, image) == "biology/2016/1/A/3.jpg"

    def test_key_uses_forward_slashes_on_every_platform(self, tmp_path: Path) -> None:
        """The key is written on Windows and resolved on Linux."""
        image = tmp_path.joinpath("biology", "2016", "1", "A", "3.jpg")
        assert "\\" not in question_object_key(tmp_path, image)

    def test_an_image_outside_the_corpus_has_no_key(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"elsewhere"):
            question_object_key(tmp_path / "output", tmp_path / "elsewhere" / "3.jpg")


class TestFileDigest:
    def test_same_bytes_hash_the_same(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(b"payload")
        (tmp_path / "b.jpg").write_bytes(b"payload")
        assert file_digest(tmp_path / "a.jpg") == file_digest(tmp_path / "b.jpg")

    def test_changed_bytes_hash_differently(self, tmp_path: Path) -> None:
        image = tmp_path / "a.jpg"
        image.write_bytes(b"payload")
        before = file_digest(image)
        image.write_bytes(b"corrected payload")
        assert file_digest(image) != before


class TestIsImage:
    def test_recognizes_image_files(self, tmp_path: Path) -> None:
        img = tmp_path / "1.JPG"
        img.touch()
        assert is_image(img) is True

    def test_rejects_non_images_and_directories(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").touch()
        (tmp_path / "folder").mkdir()
        assert is_image(tmp_path / "notes.txt") is False
        assert is_image(tmp_path / "folder") is False


class TestQuestionSlots:
    """Where a question is written, and what is already sitting there.

    These are what keeps the output tree in order: the review window refuses
    numbering that does not continue from `highest_question_number`, and the
    extractor refuses to overwrite a slot `question_slot_taken` reports.
    """

    def test_path_follows_the_output_tree_layout(self, tmp_path: Path) -> None:
        assert question_image_path(tmp_path, 3, "A", 7, "jpg") == (
            tmp_path / "3" / "A" / "7.jpg"
        )

    def test_slot_is_taken_whatever_format_holds_it(self, tmp_path: Path) -> None:
        (tmp_path / "3" / "A").mkdir(parents=True)
        (tmp_path / "3" / "A" / "7.png").touch()

        assert question_slot_taken(tmp_path, 3, "A", 7) is True

    def test_free_slot_and_missing_folder_are_both_free(self, tmp_path: Path) -> None:
        (tmp_path / "3" / "A").mkdir(parents=True)
        (tmp_path / "3" / "A" / "7.jpg").touch()

        assert question_slot_taken(tmp_path, 3, "A", 8) is False
        assert question_slot_taken(tmp_path, 4, "A", 1) is False

    def test_highest_number_ignores_gaps_and_strays(self, tmp_path: Path) -> None:
        (tmp_path / "3" / "A").mkdir(parents=True)
        for name in ("1.jpg", "2.jpg", "9.jpg", "draft.jpg", "10.txt"):
            (tmp_path / "3" / "A" / name).touch()

        assert highest_question_number(tmp_path, 3, "A") == 9

    def test_highest_number_is_zero_for_an_untouched_folder(
        self, tmp_path: Path
    ) -> None:
        assert highest_question_number(tmp_path, 1, "A") == 0


class TestWalkQuestionImages:
    def test_yields_numbered_images_with_option_and_part(self, tmp_path: Path) -> None:
        (tmp_path / "1" / "A").mkdir(parents=True)
        (tmp_path / "1" / "B").mkdir()
        (tmp_path / "1" / "A" / "1.jpg").touch()
        (tmp_path / "1" / "A" / "2.jpg").touch()
        (tmp_path / "1" / "B" / "1.png").touch()

        found = set(walk_question_images(tmp_path))

        assert found == {
            QuestionImage("1", "A", 1, tmp_path / "1" / "A" / "1.jpg"),
            QuestionImage("1", "A", 2, tmp_path / "1" / "A" / "2.jpg"),
            QuestionImage("1", "B", 1, tmp_path / "1" / "B" / "1.png"),
        }

    def test_skips_non_numeric_stems_and_loose_files(self, tmp_path: Path) -> None:
        (tmp_path / "1" / "A").mkdir(parents=True)
        (tmp_path / "1" / "A" / "cover.jpg").touch()
        (tmp_path / "answers.json").touch()

        assert list(walk_question_images(tmp_path)) == []


class TestAnswerSheetStem:
    def test_parses_year_and_sheet_number(self) -> None:
        assert parse_answer_sheet_stem("2016_1") == (2016, 1)
        assert parse_answer_sheet_stem("2024_12") == (2024, 12)

    def test_rejects_invalid_stem(self) -> None:
        assert parse_answer_sheet_stem("invalid") is None
        assert parse_answer_sheet_stem("16_1") is None


class TestBookPagePath:
    def test_round_trip_with_training_page_name(self) -> None:
        page = Path("books") / "biology" / "raw" / "images" / "2008" / "012.jpg"
        subject, year = parse_book_page_path(page)
        assert (subject, year) == ("biology", "2008")
        assert training_page_name(subject, year, page.stem) == "biology_2008_012.jpg"

    def test_a_page_and_its_processed_twin_name_the_same_thing(self) -> None:
        """The property the training pool rests on: one page, one pool name.

        The variant segment sits between the subject and ``images``, so reading
        the subject as ``images``' parent would name every processed page
        ``processed_2008_…`` and every raw one ``raw_2008_…``.
        """
        raw = Path("books/biology/raw/images/2008/012.jpg")
        processed = Path("books/biology/processed/images/2008/012.png")

        assert parse_book_page_path(raw) == parse_book_page_path(processed)
        assert training_page_name(*parse_book_page_path(processed), processed.stem) == (
            "biology_2008_012.jpg"
        )

    @pytest.mark.parametrize(
        "raw",
        [
            "scans/biology/2008/12.jpg",
            "books/biology/raw/images",
            "books",
            "raw/images/2008/12.jpg",
        ],
        ids=[
            "no-marker-segment",
            "nothing-after-images",
            "no-images-at-all",
            "no-subject-above-images",
        ],
    )
    def test_unusable_paths_all_raise_value_error(self, raw: str) -> None:
        """A marker too near either end has no subject or no year around it.

        Reading back two segments from an ``images`` at index 1 wraps around to
        the end of the path, which would return a filename as the subject.
        """
        with pytest.raises(ValueError, match="No subject/year segment"):
            parse_book_page_path(Path(raw))


class TestBookPageName:
    def test_pads_so_reading_order_survives_a_flat_sort(self) -> None:
        """The reason for padding: ``10`` must not sort ahead of ``2``."""
        names = [book_page_name(n, "png") for n in (1, 2, 10)]

        assert names == ["001.png", "002.png", "010.png"]
        assert sorted(names) == names

    def test_keeps_the_format_it_is_given(self) -> None:
        assert book_page_name(7, "jpg") == "007.jpg"


class TestWalkBookPages:
    def test_yields_one_variant_and_no_answer_sheets(self, tmp_path: Path) -> None:
        """Annotating a raw page teaches the model a rendering it never sees."""
        books = tmp_path / "books"
        for rel in (
            "biology/processed/images/2016/001.png",
            "biology/processed/answers/2016_1.png",
            "biology/raw/images/2016/001.jpg",
            "chemistry/processed/images/2016/001.png",
        ):
            path = books / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

        found = list(walk_book_pages(books, PROCESSED))

        assert [p.relative_to(books).as_posix() for p in found] == [
            "biology/processed/images/2016/001.png",
            "chemistry/processed/images/2016/001.png",
        ]


class TestNaturalSortKey:
    def test_orders_embedded_numbers_numerically(self) -> None:
        paths = [Path("page_10.jpg"), Path("page_2.jpg"), Path("page_1.jpg")]

        assert [p.name for p in sorted(paths, key=natural_sort_key)] == [
            "page_1.jpg",
            "page_2.jpg",
            "page_10.jpg",
        ]

    def test_orders_numbers_inside_longer_names(self) -> None:
        paths = [
            Path("page_20_image.png"),
            Path("page_3_image.png"),
            Path("page_10_image.png"),
        ]

        assert [p.name for p in sorted(paths, key=natural_sort_key)] == [
            "page_3_image.png",
            "page_10_image.png",
            "page_20_image.png",
        ]

    def test_is_case_insensitive(self) -> None:
        paths = [Path("Image_B.jpg"), Path("Image_A.jpg"), Path("Image_b.jpg")]

        assert [p.name for p in sorted(paths, key=natural_sort_key)] == [
            "Image_A.jpg",
            "Image_B.jpg",
            "Image_b.jpg",
        ]
