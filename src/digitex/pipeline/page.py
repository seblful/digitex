"""Page extractor for extracting question images from a single page.

Reading a page is split from writing it: :meth:`PageExtractor.read_page` turns
a page image into labelled regions (YOLO + OCR), and :meth:`PageExtractor.extract`
walks those regions through the numbering in `placement`, saving a crop for
each question. A `PageReviewer` sits between the two, which is where a human
gets to correct a polygon or a misread marker before anything is written.
"""

from collections import Counter
from pathlib import Path

import structlog
from PIL import Image

from digitex.domain.corpus import question_image_path, question_slot_taken
from digitex.domain.entities import Detection, PixelPolygon, normalize_option_number
from digitex.imaging import (
    ImageCropper,
    SegmentProcessor,
    resize_image,
)
from digitex.imaging.ocr import TextExtractor
from digitex.ml.predictors import YOLO_SegmentationPredictor
from digitex.pipeline.base import ExtractionConfig
from digitex.pipeline.placement import (
    PageExtractionState,
    PageRegion,
    QuestionPlacement,
    place_questions,
    reading_order_key,
)
from digitex.pipeline.review import PageProposal, PageReviewer, accept_page

logger = structlog.get_logger()

OCR_LANGUAGE = "rus"


class PageExtractor:
    """Extract question images from a single page using YOLO segmentation."""

    def __init__(
        self,
        config: ExtractionConfig,
        predictor: YOLO_SegmentationPredictor | None = None,
        segment_processor: SegmentProcessor | None = None,
        image_cropper: ImageCropper | None = None,
        text_extractor: TextExtractor | None = None,
        on_review: PageReviewer | None = None,
    ) -> None:
        self.config = config

        self._predictor = predictor
        self._segment_processor = segment_processor or SegmentProcessor()
        self._image_cropper = image_cropper or ImageCropper()
        self._text_extractor = text_extractor or TextExtractor(language=OCR_LANGUAGE)
        self._on_review = on_review or accept_page

    @property
    def predictor(self) -> YOLO_SegmentationPredictor:
        """Get or initialize the YOLO predictor."""
        if self._predictor is None:
            self._predictor = YOLO_SegmentationPredictor(str(self.config.model_path))
        return self._predictor

    def _crop(self, image: Image.Image, polygon: PixelPolygon) -> Image.Image:
        """Cut *polygon* out of the page and process it into a question image."""
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        cropped = resize_image(
            cropped, self.config.question_max_width, self.config.question_max_height
        )
        return self._segment_processor.process(cropped)

    def _extract_option_number(
        self, image: Image.Image, polygon: PixelPolygon
    ) -> int | None:
        """Extract option number from image region."""
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        digits = self._text_extractor.extract_digits(cropped)
        if digits:
            return normalize_option_number(digits[0])
        return None

    def _extract_part_letter(
        self, image: Image.Image, polygon: PixelPolygon
    ) -> str | None:
        """Extract part letter (A/B) from image region."""
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        text = self._text_extractor.extract_text(cropped).upper()
        # Uppercase and drop the part word before transliterating. Its second
        # letter is a Cyrillic A, which maps to a Latin "A" and would win the
        # Part A test below for every marker, Part B included.
        text = text.replace("ЧАСТЬ", "").strip()
        text_normalized = text.translate(str.maketrans("АБВ", "ABB"))

        if "A" in text_normalized:
            return "A"
        if "B" in text_normalized:
            return "B"
        return None

    def _detect(self, image: Image.Image) -> list[Detection]:
        """Run YOLO prediction and return detections in reading order.

        Raises:
            ValueError: If no detections are found on the page.
        """
        detections = self.predictor.predict(image)

        if not detections:
            raise ValueError("No detections found on page")

        class_counts = Counter(det.label for det in detections)
        logger.debug("Predictions", class_counts=dict(class_counts))

        return sorted(detections, key=lambda det: reading_order_key(det.polygon))

    def read_page(self, image: Image.Image) -> list[PageRegion]:
        """Detect the page's regions and read what its markers say.

        Returned in reading order, which is the order the numbering consumes
        them in. A detection carrying a label the model's class map doesn't
        cover is dropped — it has no place in the numbering either way.

        Raises:
            ValueError: If the page has no detections.
        """
        regions: list[PageRegion] = []

        for det in self._detect(image):
            if det.label == "option":
                regions.append(
                    PageRegion(
                        label="option",
                        polygon=det.polygon,
                        reading=self._extract_option_number(image, det.polygon),
                    )
                )
            elif det.label == "part":
                regions.append(
                    PageRegion(
                        label="part",
                        polygon=det.polygon,
                        reading=self._extract_part_letter(image, det.polygon),
                    )
                )
            elif det.label == "question":
                regions.append(PageRegion(label="question", polygon=det.polygon))
            else:
                logger.warning("Ignoring region with unknown label", label=det.label)

        return regions

    def _write_question(
        self,
        image: Image.Image,
        region: PageRegion,
        placement: QuestionPlacement,
        output_dir: Path,
    ) -> bool:
        """Save one placed question's crop, unless its slot is already taken.

        A taken slot means this page's numbering has run into output an earlier
        page already wrote. Overwriting would destroy an extracted question, so
        the existing file wins and False comes back for the caller to report —
        and `--review` marks it on the page before anything is written.
        """
        logger.debug(
            "Extracting question",
            option=placement.option,
            part=placement.part,
            question=placement.number,
        )

        if question_slot_taken(
            output_dir, placement.option, placement.part, placement.number
        ):
            logger.error(
                "Question slot already taken, keeping the existing image",
                option=placement.option,
                part=placement.part,
                question=placement.number,
            )
            return False

        output_path = question_image_path(
            output_dir,
            placement.option,
            placement.part,
            placement.number,
            self.config.image_format,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._crop(image, region.polygon).save(output_path)
        return True

    def extract(
        self,
        image: Image.Image,
        output_dir: Path,
        state: PageExtractionState,
        page_number: int = 0,
        page_count: int = 0,
    ) -> list[QuestionPlacement]:
        """Extract questions from a single page image, advancing *state*.

        *state* is mutated in place — it belongs to the caller, which threads
        one state across a whole book so question numbering continues across
        page boundaries. If a page raises partway through, the state reflects
        the detections handled up to that point; the caller decides whether
        that is recoverable.

        The reviewer sees the page's regions before any of them is cropped, and
        may correct them, move where the page starts numbering, or skip the
        page entirely — in which case nothing is written and *state* is left
        exactly where it was.

        Args:
            image: PIL Image of the page.
            output_dir: Base output directory.
            state: Question-numbering state, advanced by this call.
            page_number: This page's 1-based place in its book, for the
                reviewer to report progress with. 0 outside a book.
            page_count: How many pages the book holds. 0 outside a book.

        Returns:
            The placements whose slot was already taken. Their crops were not
            written — the existing files were kept — and a caller reporting an
            honest result must say so.

        Raises:
            ValueError: If the page has no detections, or a question comes
                before any option/part marker.
            ReviewAborted: If the reviewer stopped the run.
        """
        regions = self.read_page(image)

        reviewed = self._on_review(
            PageProposal(
                image=image,
                regions=regions,
                state=state,
                output_dir=output_dir,
                # BookExtractor opens pages from disk, so PIL knows the
                # filename — though only ImageFile declares it.
                page_name=Path(str(getattr(image, "filename", ""))).name,
                # Bound to this page, so a reviewer previewing a region sees
                # the file that would be written rather than a lookalike.
                crop=lambda polygon: self._crop(image, polygon),
                page_number=page_number,
                page_count=page_count,
            )
        )
        if reviewed is None:
            logger.info("Page skipped by reviewer")
            return []

        state.adopt(reviewed.state)

        collisions: list[QuestionPlacement] = []

        def write(region: PageRegion, placement: QuestionPlacement) -> None:
            if not self._write_question(image, region, placement, output_dir):
                collisions.append(placement)

        place_questions(reviewed.regions, state, write=write)
        return collisions
