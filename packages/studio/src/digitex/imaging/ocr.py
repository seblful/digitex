"""Reading a crop's text off tesseract.

The project's one adapter over pytesseract, and deliberately not re-exported
from the package: pytesseract is the single dependency that wants a binary on
PATH, so importing it should be a decision rather than a side effect of
importing :mod:`digitex.imaging`. Missing entirely, it stays missing until
something actually asks for a read.

Nothing here decides what a read means. Turning ``11`` into an option number,
or a baseline slope into a rotation, belongs to the extraction pipeline — this
module hands back what tesseract said and how far the lines lean.
"""

import math
import re
import statistics
from types import ModuleType
from typing import Final

import structlog
from PIL import Image

pytesseract: ModuleType | None
try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = structlog.get_logger()

# The corpus is Russian throughout.
OCR_LANGUAGE = "rus"

# psm 7 is "one line of text", which is what a marker crop holds; oem 1 picks
# the LSTM engine over the legacy one.
_TESSERACT_CONFIG_DEFAULT: Final = "--psm 7 --oem 1"
_TESSERACT_CONFIG_DIGITS: Final = (
    f"{_TESSERACT_CONFIG_DEFAULT} -c tessedit_char_whitelist=0123456789"
)
# A question crop is one uniform block of text, which is what psm 6 assumes.
_TESSERACT_CONFIG_HOCR: Final = "--psm 6 --oem 1"

# hOCR gives every text line as `baseline <slope> <offset>` inside its title
# attribute. The slope is dy/dx in image coordinates.
_BASELINE_RE: Final = re.compile(rb"baseline (-?[\d.]+) -?[\d.]+")


def _require_tesseract() -> ModuleType:
    """The pytesseract module, refusing to carry on without it.

    Read out of the module globals on each call rather than bound once, which
    is what lets a test stand a double in its place.

    Raises:
        ImportError: If pytesseract is not installed.
    """
    if pytesseract is None:
        msg = "pytesseract is not installed. Install it with: uv add pytesseract"
        raise ImportError(msg)
    return pytesseract


class TextExtractor:
    """Tesseract, asked one crop at a time.

    Carries the corpus language so no caller has to name it, and answers to
    :class:`digitex.pipeline.ports.TextReader` — which is narrower, naming
    neither the config nor the per-call language override.
    """

    def __init__(self, language: str = OCR_LANGUAGE) -> None:
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

        Stripped on the way out — tesseract ends every read with a newline,
        and callers compare on the text, not on the whitespace around it.

        Args:
            image: PIL Image to extract text from.
            config: Tesseract configuration string.
            lang: Language code (overrides instance default).

        Returns:
            Extracted text string.

        Raises:
            ImportError: If pytesseract is not installed.
        """
        language = self.language if lang is None else lang
        read = _require_tesseract().image_to_string(image, lang=language, config=config)
        text = read.strip()
        logger.debug("OCR text", text=text)
        return text

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

        Raises:
            ImportError: If pytesseract is not installed.
        """
        language = self.language if lang is None else lang
        hocr = _require_tesseract().image_to_pdf_or_hocr(
            image, extension="hocr", lang=language, config=_TESSERACT_CONFIG_HOCR
        )
        slopes = [float(m.group(1)) for m in _BASELINE_RE.finditer(hocr)]
        # A blank or pictorial crop fits no baseline, and needs no rotation.
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

        The read is whitelisted to digits, which is what makes tesseract pick
        a digit over its look-alike letter: left to itself it reads an option
        marker's ``11`` as ``ll`` and the number is lost outright.

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
