"""The two things page extraction cannot work out for itself, as interfaces.

Everything extraction does is arithmetic on pixels except for two answers it
has to be given: which labelled regions the segmentation model found on a page,
and what OCR reads off a crop. Naming those two by interface rather than by
class is what lets the differential harness replay a whole book with neither
installed, and what stops ``import digitex.pipeline.page`` pulling in ~3 GB of
CUDA wheels to crop a rectangle.

They live here and not in :mod:`digitex.domain` because both speak PIL images,
and ``domain`` may not: the deployed bot imports ``domain`` and the production
image installs no image library. A port belongs to the layer that can name its
vocabulary.

Both are narrower than the classes that satisfy them, on purpose.
``TextExtractor`` also accepts a tesseract config string and a language
override; extraction passes neither, so neither is asked for here — an
interface listing them would oblige every stand-in to accept arguments nothing
ever sends.

Both are ``runtime_checkable`` so a test can assert the concrete classes still
answer to them. That check sees method *names* only; whether the signatures fit
is ``ty``'s job, which it does at every call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL import Image

    from digitex.domain.entities import Detection


@runtime_checkable
class RegionDetector(Protocol):
    """Finds the labelled regions on a page image."""

    def predict(self, image: Image.Image) -> list[Detection]:
        """Every region found on *image*, in no particular order.

        The caller sorts into reading order — *which* regions come back is the
        model's business, what order they are consumed in is the numbering's.
        """
        ...


@runtime_checkable
class TextReader(Protocol):
    """Reads what a crop says, and how far it leans."""

    def extract_text(self, image: Image.Image) -> str:
        """The text on *image*, stripped. Empty when nothing was read."""
        ...

    def extract_digits(self, image: Image.Image) -> list[int]:
        """Every run of digits on *image*, in order. Empty when none was read."""
        ...

    def detect_skew(self, image: Image.Image) -> float:
        """Degrees counterclockwise that would level the text. 0.0 when unknown."""
        ...
