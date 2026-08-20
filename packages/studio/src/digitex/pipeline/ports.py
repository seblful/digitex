"""What page extraction needs from the outside world, as interfaces.

Extraction reaches outside itself for exactly two things: what the segmentation
model found on a page, and what OCR read off a crop. Everything else it does is
arithmetic on pixels. Naming those two by interface rather than by class is
what lets the differential harness replay a run with neither installed, and
what stops importing the page extractor from dragging in ~3 GB of CUDA wheels
to do arithmetic.

They live here rather than in :mod:`digitex.domain` because both speak in PIL
images, and ``domain`` may not import PIL — the deployed bot imports ``domain``
and the production image installs no image library. A port belongs with the
layer that can name its vocabulary.

Both are deliberately narrower than the classes that satisfy them.
``TextExtractor`` also takes a tesseract config string and a language override;
extraction passes neither, so neither is asked for here. An interface that
listed them would oblige every stand-in to accept arguments nothing sends.

Both are ``runtime_checkable`` so a test can assert the concrete classes still
answer to them. That check is method presence only — whether the signatures
line up is ``ty``'s job, and it does it at every call site.
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

        The caller sorts into reading order — which regions come back is the
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
