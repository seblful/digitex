"""Tests for the corpus layout module."""

from pathlib import Path

from digitex.core.corpus import (
    ManualImageName,
    QuestionImage,
    count_images,
    is_image,
    natural_sort_key,
    parse_answer_sheet_stem,
    parse_book_page_path,
    training_page_name,
    walk_question_images,
)


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

    def test_count_images(self, tmp_path: Path) -> None:
        (tmp_path / "1.jpg").touch()
        (tmp_path / "2.png").touch()
        (tmp_path / "answers.json").touch()
        assert count_images(tmp_path) == 2


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


class TestManualImageName:
    def test_parses_valid_name(self) -> None:
        parsed = ManualImageName.parse("2016_3_A_20.png")
        assert parsed == ManualImageName(year=2016, option=3, part="A", question=20)

    def test_rejects_bad_part_missing_fields_and_wrong_extension(self) -> None:
        assert ManualImageName.parse("2016_3_C_20.png") is None
        assert ManualImageName.parse("2016_3_20.png") is None
        assert ManualImageName.parse("2016_3_A_.png") is None
        assert ManualImageName.parse("2016_3_A_20.jpg") is None


class TestAnswerSheetStem:
    def test_parses_year_and_sheet_number(self) -> None:
        assert parse_answer_sheet_stem("2016_1") == (2016, 1)
        assert parse_answer_sheet_stem("2024_12") == (2024, 12)

    def test_rejects_invalid_stem(self) -> None:
        assert parse_answer_sheet_stem("invalid") is None
        assert parse_answer_sheet_stem("16_1") is None


class TestBookPagePath:
    def test_round_trip_with_training_page_name(self) -> None:
        page = Path("books") / "biology" / "images" / "2008" / "12_old.jpg"
        subject, year = parse_book_page_path(page)
        assert (subject, year) == ("biology", "2008")
        assert training_page_name(subject, year, page.stem) == (
            "biology_2008_12_old.jpg"
        )


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
