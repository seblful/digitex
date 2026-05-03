"""Tests for AnswersExtractor."""

from pathlib import Path

import pytest

from digitex.extractors import AnswersExtractor


class TestAnswersExtractor:
    """Test cases for AnswersExtractor."""

    @pytest.fixture
    def extractor(self) -> AnswersExtractor:
        """Create an AnswersExtractor instance."""
        return AnswersExtractor.__new__(AnswersExtractor)

    def test_extract_year_and_part(self, extractor: AnswersExtractor) -> None:
        """Test year and part extraction from filename."""
        year, part = extractor._extract_year_and_part(Path("2016_1.jpg"))
        assert year == 2016
        assert part == 1

        year, part = extractor._extract_year_and_part(Path("2024_2.png"))
        assert year == 2024
        assert part == 2

    def test_extract_year_and_part_invalid(self, extractor: AnswersExtractor) -> None:
        """Test invalid filename raises ValueError."""
        with pytest.raises(ValueError, match="Invalid filename format"):
            extractor._extract_year_and_part(Path("invalid.jpg"))

    def test_normalize_label(self, extractor: AnswersExtractor) -> None:
        """Test label normalization (Cyrillic to Latin)."""
        assert extractor._normalize_label("А1") == "A1"
        assert extractor._normalize_label("В2") == "B2"
        assert extractor._normalize_label("A1") == "A1"
        assert extractor._normalize_label("B2") == "B2"

    def test_normalize_answer(self, extractor: AnswersExtractor) -> None:
        """Test answer normalization (Latin to Cyrillic)."""
        assert extractor._normalize_answer("A1B2C3") == "А1В2С3"
        assert extractor._normalize_answer("134") == "134"
        assert extractor._normalize_answer("А1Б1В5") == "А1Б1В5"

    def test_sort_answers(self, extractor: AnswersExtractor) -> None:
        """Test answer sorting by option number."""
        answers = {
            "5": {"A1": "test"},
            "2": {"A1": "test"},
            "10": {"A1": "test"},
            "1": {"A1": "test"},
        }
        sorted_answers = extractor._sort_answers(answers)
        keys = list(sorted_answers.keys())
        assert keys == ["1", "2", "5", "10"]
