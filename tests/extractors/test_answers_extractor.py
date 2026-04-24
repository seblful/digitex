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

    def test_parse_markdown_table(self, extractor: AnswersExtractor) -> None:
        """Test markdown table parsing."""
        markdown = """
| Задание | 11 | 12 | 13 |
| --- | --- | --- | --- |
| A1 | 2 | 3 | 1 |
| A2 | 4 | 1 | 2 |
"""
        rows = extractor._parse_markdown_table(markdown)
        assert len(rows) == 4
        assert rows[0] == ["Задание", "11", "12", "13"]
        assert rows[2] == ["A1", "2", "3", "1"]
        assert rows[3] == ["A2", "4", "1", "2"]

    def test_parse_answers_simple(self, extractor: AnswersExtractor) -> None:
        """Test parsing simple answer table."""
        markdown = """
| Задание | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| A1 | 2 | 2 | 1 | 4 | 4 |
| A2 | 3 | 4 | 1 | 3 | 3 |
"""
        result = extractor._parse_answers_from_markdown(markdown, part=1)
        assert "1" in result
        assert result["1"]["A1"] == "2"
        assert result["1"]["A2"] == "3"
        assert result["2"]["A1"] == "2"
        assert result["5"]["A1"] == "4"

    def test_parse_answers_complex(self, extractor: AnswersExtractor) -> None:
        """Test parsing answer table with complex answers."""
        markdown = """
| Задание | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| B1 | 134 | 123 | 125 | 134 | 124 |
| B2 | А1Б1В5 | А2Б2В5 | А5Б5В1 | А5Б5В2 | А2Б1В1 |
"""
        result = extractor._parse_answers_from_markdown(markdown, part=1)
        assert "1" in result
        assert result["1"]["B1"] == "134"
        assert result["1"]["B2"] == "А1Б1В5"
        assert result["2"]["B1"] == "123"
        assert result["2"]["B2"] == "А2Б2В5"

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

    def test_parse_answers_part2(self, extractor: AnswersExtractor) -> None:
        """Test parsing answers for part 2 (options 6-10)."""
        markdown = """
| Задание | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- |
| A1 | 1 | 2 | 3 | 4 | 1 |
"""
        result = extractor._parse_answers_from_markdown(markdown, part=2)
        assert "6" in result
        assert result["6"]["A1"] == "1"
        assert result["7"]["A1"] == "2"
        assert result["10"]["A1"] == "1"

    def test_parse_answers_empty(self, extractor: AnswersExtractor) -> None:
        """Test parsing empty markdown returns empty dict."""
        result = extractor._parse_answers_from_markdown("", part=1)
        assert result == {}

    def test_parse_answers_no_table(self, extractor: AnswersExtractor) -> None:
        """Test parsing markdown without table returns empty dict."""
        markdown = "Just some text without tables"
        result = extractor._parse_answers_from_markdown(markdown, part=1)
        assert result == {}

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
