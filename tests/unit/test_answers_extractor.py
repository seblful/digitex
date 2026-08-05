"""Tests for AnswersExtractor's pure parsing and normalization logic."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from digitex.extractors import AnswersExtractor


@pytest.fixture
def extractor(tmp_path: Path) -> AnswersExtractor:
    """A real extractor with the OpenRouter client stubbed out."""
    return AnswersExtractor(
        api_key="test-key",
        books_dir=tmp_path / "books",
        output_dir=tmp_path / "output",
        model="test-model",
        base_url="https://example.invalid/v1",
        client=MagicMock(),
    )


class TestAnswersExtractor:
    def test_extract_year_and_sheet(self, extractor: AnswersExtractor) -> None:
        assert extractor._extract_year_and_sheet(Path("2016_1.jpg")) == (2016, 1)
        assert extractor._extract_year_and_sheet(Path("2024_2.png")) == (2024, 2)

    def test_extract_year_and_sheet_invalid(self, extractor: AnswersExtractor) -> None:
        with pytest.raises(ValueError, match="Invalid filename format"):
            extractor._extract_year_and_sheet(Path("invalid.jpg"))

    def test_normalize_label_cyrillic_to_latin(
        self, extractor: AnswersExtractor
    ) -> None:
        assert extractor._normalize_label("А1") == "A1"
        assert extractor._normalize_label("В2") == "B2"
        assert extractor._normalize_label("A1") == "A1"
        assert extractor._normalize_label("B2") == "B2"

    @pytest.mark.parametrize(
        "label",
        ["A", "1", "A1B", "B-1", "", "AB", "A 1"],
        ids=[
            "letter-only",
            "digit-only",
            "trailing-letter",
            "separator",
            "empty",
            "no-digits",
            "inner-space",
        ],
    )
    def test_normalize_label_rejects_anything_but_part_plus_number(
        self, extractor: AnswersExtractor, label: str
    ) -> None:
        """The sort key downstream assumes this shape; nothing else is usable."""
        with pytest.raises(ValueError, match="Invalid question label"):
            extractor._normalize_label(label)

    def test_normalize_label_tolerates_surrounding_whitespace(
        self, extractor: AnswersExtractor
    ) -> None:
        assert extractor._normalize_label(" a1 ") == "A1"

    def test_normalize_answer_latin_to_cyrillic(
        self, extractor: AnswersExtractor
    ) -> None:
        assert extractor._normalize_answer("A1B2C3") == "А1В2С3"
        assert extractor._normalize_answer("134") == "134"
        assert extractor._normalize_answer("А1Б1В5") == "А1Б1В5"

    @pytest.mark.parametrize(
        ("raw", "normalized"),
        [
            ("11", "1"),
            ("12", "2"),
            ("20", "10"),
            ("31", "1"),
            ("32", "2"),
            ("40", "10"),
            ("1", "1"),
            ("10", "10"),
        ],
        ids=[
            "11-to-1",
            "12-to-2",
            "20-to-10",
            "31-to-1",
            "32-to-2",
            "40-to-10",
            "1-unchanged",
            "10-unchanged",
        ],
    )
    def test_normalize_option_maps_book_ranges_to_one_to_ten(
        self, extractor: AnswersExtractor, raw: str, normalized: str
    ) -> None:
        assert extractor._normalize_option(raw) == normalized

    def test_sort_answers_by_option_number(self, extractor: AnswersExtractor) -> None:
        answers = {
            "5": {"A1": "test"},
            "2": {"A1": "test"},
            "10": {"A1": "test"},
            "1": {"A1": "test"},
        }
        sorted_answers = extractor._sort_answers(answers)
        assert list(sorted_answers.keys()) == ["1", "2", "5", "10"]


class TestAnswersExtractorExtract:
    """``extract`` over a directory of answer sheets, with ``ocr`` stubbed."""

    @staticmethod
    def _seed_sheets(tmp_path: Path, *names: str) -> None:
        answers_dir = tmp_path / "books" / "bio" / "answers"
        answers_dir.mkdir(parents=True)
        for name in names:
            (answers_dir / name).touch()

    def test_writes_answers_when_every_sheet_succeeds(
        self,
        extractor: AnswersExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._seed_sheets(tmp_path, "2016_1.jpg", "2016_2.jpg")
        monkeypatch.setattr(extractor, "ocr", lambda _: {"1": {"A1": "3"}})

        result = extractor.extract("bio")

        assert result.success
        assert (tmp_path / "output" / "bio" / "2016" / "answers.json").exists()

    def test_partial_failure_leaves_the_year_unwritten(
        self,
        extractor: AnswersExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A year spans several sheets, and a written year is never retried.

        Committing the sheets that did succeed would strand the rest: the next
        run sees answers.json and skips the year.
        """
        self._seed_sheets(tmp_path, "2016_1.jpg", "2016_2.jpg")

        def ocr(image_path: Path) -> dict[str, dict[str, str]]:
            if image_path.name == "2016_2.jpg":
                raise RuntimeError("api timeout")
            return {"1": {"A1": "3"}}

        monkeypatch.setattr(extractor, "ocr", ocr)

        result = extractor.extract("bio")

        assert not result.success
        assert not (tmp_path / "output" / "bio" / "2016" / "answers.json").exists()

    def test_a_malformed_label_fails_only_its_own_year(
        self,
        extractor: AnswersExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bad label is a sheet failure, not a run failure.

        Sorting the output assumes ``{A|B}{digits}``, so a label the model got
        wrong used to abort the whole command from the write loop — after every
        API call had been paid for, and before any later year was written.
        """
        self._seed_sheets(tmp_path, "2016_1.jpg", "2017_1.jpg")

        def ocr(image_path: Path) -> dict[str, dict[str, str]]:
            if image_path.name == "2016_1.jpg":
                return {"1": {"A1)": "3"}}
            return {"1": {"A1": "3"}}

        monkeypatch.setattr(extractor, "ocr", ocr)

        result = extractor.extract("bio")

        assert not result.success
        assert not (tmp_path / "output" / "bio" / "2016" / "answers.json").exists()
        assert (tmp_path / "output" / "bio" / "2017" / "answers.json").exists()

    def test_an_unaffected_year_is_still_written(
        self,
        extractor: AnswersExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._seed_sheets(tmp_path, "2016_1.jpg", "2017_1.jpg")

        def ocr(image_path: Path) -> dict[str, dict[str, str]]:
            if image_path.name == "2016_1.jpg":
                raise RuntimeError("api timeout")
            return {"1": {"A1": "3"}}

        monkeypatch.setattr(extractor, "ocr", ocr)

        extractor.extract("bio")

        assert not (tmp_path / "output" / "bio" / "2016" / "answers.json").exists()
        assert (tmp_path / "output" / "bio" / "2017" / "answers.json").exists()
