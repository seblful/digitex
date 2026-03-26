"""OCR utilities using Tesseract."""

import logging
import re
from typing import Final

import pytesseract
from PIL import Image

from digitex.config import get_settings

logger = logging.getLogger(__name__)

_TESSERACT_CONFIG_DEFAULT: Final = "--psm 7 --oem 1"
_TESSERACT_CONFIG_DIGITS: Final = f"{_TESSERACT_CONFIG_DEFAULT} -c tessedit_char_whitelist=0123456789"


class TextExtractor:
    """Extract text from images using OCR."""

    def __init__(self, language: str | None = None) -> None:
        """Initialize the text extractor.

        Args:
            language: Tesseract language code. If None, uses settings default.
        """
        self._language = language

    @property
    def language(self) -> str:
        """Get the OCR language code."""
        if self._language is None:
            return get_settings().ocr.language
        return self._language

    def extract_text(
        self,
        image: Image.Image,
        config: str = _TESSERACT_CONFIG_DEFAULT,
        lang: str | None = None,
    ) -> str:
        """Extract text from an image.

        Args:
            image: PIL Image to extract text from.
            config: Tesseract configuration string.
            lang: Language code (overrides instance default).

        Returns:
            Extracted text string.
        """
        language = lang if lang is not None else self.language
        text = pytesseract.image_to_string(image, lang=language, config=config)
        logger.debug(f"OCR text: '{text.strip()}'")
        return text.strip()

    def extract_digits(
        self,
        image: Image.Image,
        lang: str | None = None,
    ) -> list[int]:
        """Extract digits from an image.

        Args:
            image: PIL Image to extract digits from.
            lang: Language code (overrides instance default).

        Returns:
            List of extracted integers.
        """
        language = lang if lang is not None else self.language
        text = self.extract_text(image, config=_TESSERACT_CONFIG_DIGITS, lang=language)
        numbers = re.findall(r"\d+", text)
        logger.debug(f"OCR digits: {numbers}")
        return [int(n) for n in numbers]
