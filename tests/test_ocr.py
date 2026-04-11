"""Tests for the Core OCR module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from digitex.core.ocr import (
    _TESSERACT_CONFIG_DEFAULT,
    _TESSERACT_CONFIG_DIGITS,
    TextExtractor,
)


class TestTextExtractor:
    """Test suite for TextExtractor class."""

    def test_init_default_language(self) -> None:
        """Test initialization with default language."""
        extractor = TextExtractor()
        assert extractor.language == "rus"

    def test_init_custom_language(self) -> None:
        """Test initialization with custom language."""
        extractor = TextExtractor(language="eng")
        assert extractor.language == "eng"

    def test_language_property(self) -> None:
        """Test language property returns correct value."""
        extractor = TextExtractor(language="eng")
        assert extractor.language == "eng"

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text(self, mock_pytesseract: MagicMock) -> None:
        """Test extract_text calls pytesseract correctly."""
        mock_pytesseract.return_value = "Hello World"

        extractor = TextExtractor(language="eng")
        image = Image.new("RGB", (100, 100), color="white")

        result = extractor.extract_text(image)

        assert result == "Hello World"
        mock_pytesseract.assert_called_once_with(
            image, lang="eng", config=_TESSERACT_CONFIG_DEFAULT
        )

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text_strips_whitespace(self, mock_pytesseract: MagicMock) -> None:
        """Test extract_text strips leading/trailing whitespace."""
        mock_pytesseract.return_value = "  Hello World  \n"

        extractor = TextExtractor()
        image = Image.new("RGB", (100, 100), color="white")

        result = extractor.extract_text(image)

        assert result == "Hello World"

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text_with_custom_config(self, mock_pytesseract: MagicMock) -> None:
        """Test extract_text uses custom config when provided."""
        mock_pytesseract.return_value = "test"

        extractor = TextExtractor()
        image = Image.new("RGB", (100, 100), color="white")
        custom_config = "--psm 6"

        extractor.extract_text(image, config=custom_config)

        mock_pytesseract.assert_called_once_with(
            image, lang="rus", config=custom_config
        )

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text_overrides_language(self, mock_pytesseract: MagicMock) -> None:
        """Test extract_text uses lang parameter instead of instance language."""
        mock_pytesseract.return_value = "test"

        extractor = TextExtractor(language="rus")
        image = Image.new("RGB", (100, 100), color="white")

        extractor.extract_text(image, lang="eng")

        mock_pytesseract.assert_called_once_with(
            image, lang="eng", config=_TESSERACT_CONFIG_DEFAULT
        )

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_digits(self, mock_pytesseract: MagicMock) -> None:
        """Test extract_digits extracts numbers correctly."""
        mock_pytesseract.return_value = "Question 5 and 10"

        extractor = TextExtractor()
        image = Image.new("RGB", (100, 100), color="white")

        result = extractor.extract_digits(image)

        assert result == [5, 10]
        mock_pytesseract.assert_called_once_with(
            image, lang="rus", config=_TESSERACT_CONFIG_DIGITS
        )

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_digits_no_numbers(self, mock_pytesseract: MagicMock) -> None:
        """Test extract_digits returns empty list when no numbers found."""
        mock_pytesseract.return_value = "No numbers here"

        extractor = TextExtractor()
        image = Image.new("RGB", (100, 100), color="white")

        result = extractor.extract_digits(image)

        assert result == []

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_digits_multiple_digits(self, mock_pytesseract: MagicMock) -> None:
        """Test extract_digits handles multi-digit numbers."""
        mock_pytesseract.return_value = "Number 123 and 456"

        extractor = TextExtractor()
        image = Image.new("RGB", (100, 100), color="white")

        result = extractor.extract_digits(image)

        assert result == [123, 456]

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_digits_overrides_language(
        self, mock_pytesseract: MagicMock
    ) -> None:
        """Test extract_digits uses lang parameter when provided."""
        mock_pytesseract.return_value = "123"

        extractor = TextExtractor(language="rus")
        image = Image.new("RGB", (100, 100), color="white")

        extractor.extract_digits(image, lang="eng")

        mock_pytesseract.assert_called_once_with(
            image, lang="eng", config=_TESSERACT_CONFIG_DIGITS
        )


class TestTesseractConfigConstants:
    """Test suite for Tesseract configuration constants."""

    def test_default_config_format(self) -> None:
        """Test default config contains expected parameters."""
        assert "--psm 7" in _TESSERACT_CONFIG_DEFAULT
        assert "--oem 1" in _TESSERACT_CONFIG_DEFAULT

    def test_digits_config_includes_whitelist(self) -> None:
        """Test digits config includes character whitelist."""
        assert "-c tessedit_char_whitelist=0123456789" in _TESSERACT_CONFIG_DIGITS

    def test_digits_config_includes_default_params(self) -> None:
        """Test digits config includes default parameters."""
        assert "--psm 7" in _TESSERACT_CONFIG_DIGITS
        assert "--oem 1" in _TESSERACT_CONFIG_DIGITS
