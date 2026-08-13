"""OCR utilities using Tesseract."""

import re
from types import ModuleType
from typing import Final, cast

import structlog
from PIL import Image

try:
    import pytesseract as _pytesseract_mod

    pytesseract: ModuleType | None = _pytesseract_mod
except ImportError:
    pytesseract = cast("ModuleType | None", None)

logger = structlog.get_logger()

_TESSERACT_CONFIG_DEFAULT: Final = "--psm 7 --oem 1"
_TESSERACT_CONFIG_DIGITS: Final = (
    f"{_TESSERACT_CONFIG_DEFAULT} -c tessedit_char_whitelist=0123456789"
)


class TextExtractor:
    """Extract text from images using OCR."""

    def __init__(self, language: str = "rus") -> None:
        self._language = language

    @property
    def language(self) -> str:
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
        if pytesseract is None:
            msg = "pytesseract is not installed. Install it with: uv add pytesseract"
            raise ImportError(msg)
        text = pytesseract.image_to_string(image, lang=language, config=config)
        logger.debug("OCR text", text=text.strip())
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
        logger.debug("OCR digits", numbers=numbers)
        return [int(n) for n in numbers]
