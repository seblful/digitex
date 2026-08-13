"""Page extractor for extracting question images from a single page.

Reading a page is split from writing it: :meth:`PageExtractor.read_page` turns
a page image into labelled regions (YOLO + OCR), and :meth:`PageExtractor.extract`
walks those regions through the numbering in `placement`, saving a crop for
each question. A `PageReviewer` sits between the two, which is where a human
gets to correct a polygon or a misread marker before anything is written.
"""

from pathlib import Path

import structlog
from PIL import Image

from digitex.core import TextExtractor
from digitex.core.domain import Detection, PixelPolygon, normalize_option_number
from digitex.core.processors import (
    ImageCropper,
    SegmentProcessor,
    resize_image,
)
from digitex.extractors.base import ExtractionConfig
from digitex.extractors.conflict_resolution import (
    Conflict,
    ConflictResolver,
    keep_current_option,
)
from digitex.extractors.placement import (
    CORRECTED_PART,
    PageExtractionState,
    PageRegion,
    QuestionPlacement,
    place_questions,
    reading_order_key,
)
from digitex.extractors.review import PageProposal, PageReviewer, accept_page
from digitex.ml.predictors import YOLO_SegmentationPredictor

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
        on_conflict: ConflictResolver | None = None,
        on_review: PageReviewer | None = None,
    ) -> None:
        self.config = config

        self._predictor = predictor
        self._segment_processor = segment_processor or SegmentProcessor()
        self._image_cropper = image_cropper or ImageCropper()
        self._text_extractor = text_extractor or TextExtractor(language=OCR_LANGUAGE)
        self._on_conflict = on_conflict or keep_current_option
        self._on_review = on_review or accept_page

    @property
    def predictor(self) -> YOLO_SegmentationPredictor:
        """Get or initialize the YOLO predictor."""
        if self._predictor is None:
            self._predictor = YOLO_SegmentationPredictor(str(self.config.model_path))
        return self._predictor

    def _crop_and_save(
        self,
        image: Image.Image,
        polygon: PixelPolygon,
        output_path: Path,
        current_option: int,
        output_dir: Path,
    ) -> int:
        """Crop, process, and save extracted image. Returns resolved option number."""
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        cropped = resize_image(
            cropped, self.config.question_max_width, self.config.question_max_height
        )
        processed = self._segment_processor.process(cropped)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            return self._handle_existing_file(
                output_path, processed, current_option, output_dir
            )

        processed.save(output_path)
        return current_option

    def _handle_existing_file(
        self,
        output_path: Path,
        new_image: Image.Image,
        current_option: int,
        output_dir: Path,
    ) -> int:
        """Ask the resolver where a colliding question belongs.

        Returns the option the question ended up under — *current_option* when
        the existing file is kept, which is also what happens when the resolver
        names an option whose slot is taken too.
        """
        resolved_option = self._on_conflict(
            Conflict(
                new_image=new_image,
                existing_path=output_path,
                current_option=current_option,
            )
        )

        if resolved_option == current_option:
            return current_option

        correct_path = (
            output_dir / str(resolved_option) / CORRECTED_PART / output_path.name
        )
        if correct_path.exists():
            # Moving the crop here would overwrite another question's image, so
            # the collision stands and the state is left where it was.
            logger.error(
                "Corrected path already taken, keeping existing file",
                from_path=str(output_path),
                to_path=str(correct_path),
            )
            return current_option

        correct_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Saving corrected image",
            from_path=str(output_path),
            to_path=str(correct_path),
        )
        new_image.save(str(correct_path))
        output_path.unlink()
        return resolved_option

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

        class_counts: dict[str, int] = {}
        for det in detections:
            class_counts[det.label] = class_counts.get(det.label, 0) + 1
        logger.debug("Predictions", class_counts=class_counts)

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
    ) -> int:
        """Save one placed question's crop. Returns the option it landed under."""
        logger.debug(
            "Extracting question",
            option=placement.option,
            part=placement.part,
            question=placement.number,
        )
        output_path = (
            output_dir
            / str(placement.option)
            / placement.part
            / f"{placement.number}.{self.config.image_format}"
        )
        return self._crop_and_save(
            image, region.polygon, output_path, placement.option, output_dir
        )

    def extract(
        self,
        image: Image.Image,
        output_dir: Path,
        state: PageExtractionState,
    ) -> None:
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
            )
        )
        if reviewed is None:
            logger.info("Page skipped by reviewer")
            return

        state.adopt(reviewed.state)

        def write(region: PageRegion, placement: QuestionPlacement) -> int:
            return self._write_question(image, region, placement, output_dir)

        place_questions(reviewed.regions, state, write=write)
