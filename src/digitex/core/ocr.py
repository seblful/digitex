"""OCR utilities using Tesseract."""

import logging
import re
from typing import Final

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

_TESSERACT_CONFIG_DEFAULT: Final = "--psm 7 --oem 1"
_TESSERACT_CONFIG_DIGITS: Final = f"{_TESSERACT_CONFIG_DEFAULT} -c tessedit_char_whitelist=0123456789"


class TextExtractor:
    """Extract text from images using OCR."""

    def extract_text(
        self,
        image: Image.Image,
        config: str = _TESSERACT_CONFIG_DEFAULT,
        lang: str = "rus",
    ) -> str:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        logger.debug(f"OCR text: '{text.strip()}'")
        return text.strip()

    def extract_digits(
        self,
        image: Image.Image,
        lang: str = "rus",
    ) -> list[int]:
        text = self.extract_text(image, config=_TESSERACT_CONFIG_DIGITS, lang=lang)
        numbers = re.findall(r"\d+", text)
        logger.debug(f"OCR digits: {numbers}")
        return [int(n) for n in numbers]
