"""Tests for the OCR adapter around pytesseract.

``TextExtractor`` is the project's thin adapter over pytesseract — the one
third-party call that needs a binary on PATH. pytesseract is patched here and
the adapter's own contract is what gets asserted: which language and config
reach the call, and what the adapter does to the text on the way back.

The digits path matters most: it reads option and question numbers off page
markers, and a stray non-digit there re-files a whole book.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from PIL import Image

from digitex.imaging import ocr
from digitex.imaging.ocr import (
    _TESSERACT_CONFIG_DEFAULT,
    _TESSERACT_CONFIG_DIGITS,
    _TESSERACT_CONFIG_HOCR,
    TextExtractor,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def tesseract(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Stand in for the pytesseract module, recording how it was called."""
    fake = MagicMock()
    fake.image_to_string.return_value = ""
    monkeypatch.setattr(ocr, "pytesseract", fake)
    return fake


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="white")


class TestLanguage:
    def test_the_corpus_language_is_the_default(self) -> None:
        """The books are Russian throughout, so nothing has to ask for it."""
        assert TextExtractor().language == "rus"

    def test_the_instance_language_is_used_for_a_read(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        TextExtractor(language="eng").extract_text(image)

        assert tesseract.image_to_string.call_args.kwargs["lang"] == "eng"

    def test_a_per_call_language_wins_over_the_instance(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        TextExtractor(language="rus").extract_text(image, lang="eng")

        assert tesseract.image_to_string.call_args.kwargs["lang"] == "eng"


class TestExtractText:
    def test_the_recognized_text_comes_back(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        tesseract.image_to_string.return_value = "Hello World"

        assert TextExtractor().extract_text(image) == "Hello World"

    def test_surrounding_whitespace_is_stripped(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        """Tesseract ends every read with a newline; callers compare on the text."""
        tesseract.image_to_string.return_value = "  Hello World  \n"

        assert TextExtractor().extract_text(image) == "Hello World"

    def test_a_read_uses_the_single_line_config_by_default(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        TextExtractor().extract_text(image)

        assert (
            tesseract.image_to_string.call_args.kwargs["config"]
            == _TESSERACT_CONFIG_DEFAULT
        )

    def test_a_caller_can_override_the_config(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        TextExtractor().extract_text(image, config="--psm 6")

        assert tesseract.image_to_string.call_args.kwargs["config"] == "--psm 6"

    def test_a_missing_pytesseract_says_how_to_install_it(
        self, monkeypatch: pytest.MonkeyPatch, image: Image.Image
    ) -> None:
        """The one dependency needing a binary on PATH, so this is a real path."""
        monkeypatch.setattr(ocr, "pytesseract", None)

        with pytest.raises(ImportError, match="uv add pytesseract"):
            TextExtractor().extract_text(image)


def _hocr(*slopes: float) -> bytes:
    """An hOCR document with one text line per baseline slope."""
    lines = "".join(
        f"<span class='ocr_line' title='bbox 0 0 100 20; baseline {slope} -3'>"
        for slope in slopes
    )
    return f"<html><body>{lines}</body></html>".encode()


class TestDetectSkew:
    """The skew angle is read off the baselines tesseract fits per text line."""

    def test_the_median_slope_becomes_the_angle(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        """One misread line must not tilt the answer, hence the median."""
        # tan(2°) ≈ 0.0349; the 0.9 outlier is a line tesseract misfit.
        tesseract.image_to_pdf_or_hocr.return_value = _hocr(0.0349, 0.0349, 0.9)

        angle = TextExtractor().detect_skew(image)

        assert angle == pytest.approx(2.0, abs=0.01)

    def test_text_leaning_the_other_way_reads_negative(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        tesseract.image_to_pdf_or_hocr.return_value = _hocr(-0.0349)

        assert TextExtractor().detect_skew(image) == pytest.approx(-2.0, abs=0.01)

    def test_a_crop_without_text_reads_as_level(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        """A blank or pictorial crop must pass through unrotated, not crash."""
        tesseract.image_to_pdf_or_hocr.return_value = _hocr()

        assert TextExtractor().detect_skew(image) == 0.0

    def test_the_read_asks_tesseract_for_hocr(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        tesseract.image_to_pdf_or_hocr.return_value = _hocr(0.0)

        TextExtractor().detect_skew(image)

        kwargs = tesseract.image_to_pdf_or_hocr.call_args.kwargs
        assert kwargs["extension"] == "hocr"
        assert kwargs["lang"] == "rus"
        assert kwargs["config"] == _TESSERACT_CONFIG_HOCR

    def test_a_missing_pytesseract_says_how_to_install_it(
        self, monkeypatch: pytest.MonkeyPatch, image: Image.Image
    ) -> None:
        monkeypatch.setattr(ocr, "pytesseract", None)

        with pytest.raises(ImportError, match="uv add pytesseract"):
            TextExtractor().detect_skew(image)


class TestExtractDigits:
    @pytest.mark.parametrize(
        ("recognized", "expected"),
        [
            ("Question 5 and 10", [5, 10]),
            ("No numbers here", []),
            ("", []),
            ("Number 123 and 456", [123, 456]),
            ("11", [11]),
        ],
        ids=["two-numbers", "no-numbers", "empty", "multi-digit", "option-marker"],
    )
    def test_every_run_of_digits_is_read_as_a_number(
        self,
        tesseract: MagicMock,
        image: Image.Image,
        recognized: str,
        expected: list[int],
    ) -> None:
        tesseract.image_to_string.return_value = recognized

        assert TextExtractor().extract_digits(image) == expected

    def test_a_digit_read_restricts_tesseract_to_digits(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        """Whitelisting is what stops a smudge being read as a letter."""
        TextExtractor().extract_digits(image)

        config = tesseract.image_to_string.call_args.kwargs["config"]
        assert config == _TESSERACT_CONFIG_DIGITS
        assert "tessedit_char_whitelist=0123456789" in config

    def test_a_per_call_language_reaches_a_digit_read_too(
        self, tesseract: MagicMock, image: Image.Image
    ) -> None:
        TextExtractor(language="rus").extract_digits(image, lang="eng")

        assert tesseract.image_to_string.call_args.kwargs["lang"] == "eng"
