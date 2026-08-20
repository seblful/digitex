"""OCR utilities using Tesseract."""

import math
import re
import statistics
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

# The corpus is Russian throughout.
OCR_LANGUAGE = "rus"

_TESSERACT_CONFIG_DEFAULT: Final = "--psm 7 --oem 1"
_TESSERACT_CONFIG_DIGITS: Final = (
    f"{_TESSERACT_CONFIG_DEFAULT} -c tessedit_char_whitelist=0123456789"
)
# A question crop is one uniform block of text, which is what psm 6 assumes.
_TESSERACT_CONFIG_HOCR: Final = "--psm 6 --oem 1"

# hOCR gives every text line as `baseline <slope> <offset>` inside its title
# attribute. The slope is dy/dx in image coordinates.
_BASELINE_RE: Final = re.compile(rb"baseline (-?[\d.]+) -?[\d.]+")


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

    def detect_skew(
        self,
        image: Image.Image,
        lang: str | None = None,
    ) -> float:
        """The rotation, in degrees counterclockwise, that would level the text.

        Read off the baselines tesseract fits through the image's text lines —
        the median of their slopes, so a single misread line cannot tilt the
        answer. Positive means the content leans clockwise and a
        counterclockwise turn corrects it, matching cv2's rotation convention.

        Args:
            image: PIL Image to measure.
            lang: Language code (overrides instance default).

        Returns:
            Correction angle in degrees, 0.0 when no text line is found.
        """
        language = lang if lang is not None else self.language
        if pytesseract is None:
            msg = "pytesseract is not installed. Install it with: uv add pytesseract"
            raise ImportError(msg)
        hocr = pytesseract.image_to_pdf_or_hocr(
            image, extension="hocr", lang=language, config=_TESSERACT_CONFIG_HOCR
        )
        slopes = [float(m.group(1)) for m in _BASELINE_RE.finditer(hocr)]
        if not slopes:
            return 0.0
        angle = math.degrees(math.atan(statistics.median(slopes)))
        logger.debug("OCR skew", angle=angle, lines=len(slopes))
        return angle

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
        text = self.extract_text(image, config=_TESSERACT_CONFIG_DIGITS, lang=lang)
        numbers = re.findall(r"\d+", text)
        logger.debug("OCR digits", numbers=numbers)
        return [int(n) for n in numbers]
