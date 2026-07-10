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
        client=MagicMock(),
    )


class TestAnswersExtractor:
    def test_extract_year_and_part(self, extractor: AnswersExtractor) -> None:
        assert extractor._extract_year_and_part(Path("2016_1.jpg")) == (2016, 1)
        assert extractor._extract_year_and_part(Path("2024_2.png")) == (2024, 2)

    def test_extract_year_and_part_invalid(self, extractor: AnswersExtractor) -> None:
        with pytest.raises(ValueError, match="Invalid filename format"):
            extractor._extract_year_and_part(Path("invalid.jpg"))

    def test_normalize_label_cyrillic_to_latin(
        self, extractor: AnswersExtractor
    ) -> None:
        assert extractor._normalize_label("А1") == "A1"
        assert extractor._normalize_label("В2") == "B2"
        assert extractor._normalize_label("A1") == "A1"
        assert extractor._normalize_label("B2") == "B2"

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
