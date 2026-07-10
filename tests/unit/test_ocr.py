"""Tests for the OCR adapter around pytesseract.

``TextExtractor`` is the project's thin adapter over pytesseract, so
pytesseract itself is patched and the adapter's contract (language, config,
post-processing) is what's asserted.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from digitex.core.ocr import (
    _TESSERACT_CONFIG_DEFAULT,
    _TESSERACT_CONFIG_DIGITS,
    TextExtractor,
)


def _image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="white")


class TestTextExtractor:
    def test_default_language_is_russian(self) -> None:
        assert TextExtractor().language == "rus"

    def test_custom_language(self) -> None:
        assert TextExtractor(language="eng").language == "eng"

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text_uses_instance_language_and_default_config(
        self, mock_pytesseract: MagicMock
    ) -> None:
        mock_pytesseract.return_value = "Hello World"
        image = _image()

        result = TextExtractor(language="eng").extract_text(image)

        assert result == "Hello World"
        mock_pytesseract.assert_called_once_with(
            image, lang="eng", config=_TESSERACT_CONFIG_DEFAULT
        )

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text_strips_whitespace(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.return_value = "  Hello World  \n"
        assert TextExtractor().extract_text(_image()) == "Hello World"

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text_custom_config_overrides_default(
        self, mock_pytesseract: MagicMock
    ) -> None:
        mock_pytesseract.return_value = "test"
        image = _image()

        TextExtractor().extract_text(image, config="--psm 6")

        mock_pytesseract.assert_called_once_with(image, lang="rus", config="--psm 6")

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_text_lang_parameter_overrides_instance(
        self, mock_pytesseract: MagicMock
    ) -> None:
        mock_pytesseract.return_value = "test"
        image = _image()

        TextExtractor(language="rus").extract_text(image, lang="eng")

        mock_pytesseract.assert_called_once_with(
            image, lang="eng", config=_TESSERACT_CONFIG_DEFAULT
        )

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    @pytest.mark.parametrize(
        ("ocr_text", "expected"),
        [
            ("Question 5 and 10", [5, 10]),
            ("No numbers here", []),
            ("Number 123 and 456", [123, 456]),
        ],
        ids=["two-numbers", "no-numbers", "multi-digit"],
    )
    def test_extract_digits_parses_numbers(
        self, mock_pytesseract: MagicMock, ocr_text: str, expected: list[int]
    ) -> None:
        mock_pytesseract.return_value = ocr_text
        image = _image()

        result = TextExtractor().extract_digits(image)

        assert result == expected
        mock_pytesseract.assert_called_once_with(
            image, lang="rus", config=_TESSERACT_CONFIG_DIGITS
        )

    @patch("digitex.core.ocr.pytesseract.image_to_string")
    def test_extract_digits_lang_parameter_overrides_instance(
        self, mock_pytesseract: MagicMock
    ) -> None:
        mock_pytesseract.return_value = "123"
        image = _image()

        TextExtractor(language="rus").extract_digits(image, lang="eng")

        mock_pytesseract.assert_called_once_with(
            image, lang="eng", config=_TESSERACT_CONFIG_DIGITS
        )


class TestTesseractConfigConstants:
    def test_default_config_format(self) -> None:
        assert "--psm 7" in _TESSERACT_CONFIG_DEFAULT
        assert "--oem 1" in _TESSERACT_CONFIG_DEFAULT

    def test_digits_config_includes_whitelist(self) -> None:
        assert "-c tessedit_char_whitelist=0123456789" in _TESSERACT_CONFIG_DIGITS

    def test_digits_config_includes_default_params(self) -> None:
        assert "--psm 7" in _TESSERACT_CONFIG_DIGITS
        assert "--oem 1" in _TESSERACT_CONFIG_DIGITS
