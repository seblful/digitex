"""Page extractor for extracting question images from a single page."""

from dataclasses import dataclass
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
from digitex.ml.predictors import YOLO_SegmentationPredictor

logger = structlog.get_logger()

OCR_LANGUAGE = "rus"

# A conflict-resolver correction moves the question to a different option, and
# an option always starts at Part A. Named once so the state machine and the
# path it lands at cannot disagree.
CORRECTED_PART = "A"


@dataclass(frozen=True)
class QuestionPlacement:
    """Where one detected question lands in the extraction output."""

    option: int
    part: str
    number: int


@dataclass
class PageExtractionState:
    """Question-numbering state machine, threaded across a book's pages.

    Owns every decision about which option/part/number a detection belongs
    to. Consumes the page's markers in reading order (``on_option`` /
    ``on_part``), hands out placements as values (``next_question`` +
    ``commit_question``), and takes conflict-resolver corrections back via
    ``correct_option``. Performs no I/O — reading markers off the page and
    saving crops belong to PageExtractor.
    """

    option: int = 0
    part: str = ""
    question: int = 0

    def on_option(self, new_option: int | None) -> bool:
        """Advance when a marker continues the option sequence.

        Anything that is not exactly the next option number is treated as an
        OCR misread and ignored. Returns True on change.
        """
        if new_option is not None and new_option == self.option + 1:
            self.option = new_option
            self.part = "A"
            self.question = 0
            return True
        return False

    def on_part(self, new_part: str | None) -> bool:
        """Switch part when a different part marker is read. Returns True on change."""
        if new_part is not None and new_part != self.part:
            self.part = new_part
            self.question = 0
            return True
        return False

    def next_question(self) -> QuestionPlacement:
        """Return the placement the next question will get, without committing.

        The caller commits via :meth:`commit_question` only after the crop is
        saved, so a failed save doesn't consume a question number.
        """
        return QuestionPlacement(self.option, self.part, self.question + 1)

    def commit_question(self) -> None:
        """Consume the question number handed out by :meth:`next_question`."""
        self.question += 1

    def correct_option(self, resolved_option: int) -> bool:
        """Apply a conflict-resolver decision. Returns True if the option moved.

        The question counter deliberately keeps running — the corrected
        question retains its number under the new option.
        """
        if resolved_option == self.option:
            return False
        self.option = resolved_option
        self.part = CORRECTED_PART
        return True


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
    ) -> None:
        self.config = config

        self._predictor = predictor
        self._segment_processor = segment_processor or SegmentProcessor()
        self._image_cropper = image_cropper or ImageCropper()
        self._text_extractor = text_extractor or TextExtractor(language=OCR_LANGUAGE)
        self._on_conflict = on_conflict or keep_current_option

    @property
    def predictor(self) -> YOLO_SegmentationPredictor:
        """Get or initialize the YOLO predictor."""
        if self._predictor is None:
            self._predictor = YOLO_SegmentationPredictor(str(self.config.model_path))
        return self._predictor

    def _get_polygon_bounding_box(self, polygon: PixelPolygon) -> tuple[int, int]:
        """Get bounding box position from polygon."""
        min_y = min(p[1] for p in polygon)
        min_x = min(p[0] for p in polygon)
        return (min_y, min_x)

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

        return sorted(
            detections, key=lambda det: self._get_polygon_bounding_box(det.polygon)
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

        Args:
            image: PIL Image of the page.
            output_dir: Base output directory.
            state: Question-numbering state, advanced by this call.

        Raises:
            ValueError: If the page has no detections.
        """
        detections = self._detect(image)

        for det in detections:
            if det.label == "option":
                new_option = self._extract_option_number(image, det.polygon)
                if state.on_option(new_option):
                    logger.debug("Option changed", option_counter=state.option)
            elif det.label == "part":
                new_part = self._extract_part_letter(image, det.polygon)
                if state.on_part(new_part):
                    logger.debug("Part changed", part_letter=state.part)
            elif det.label == "question":
                placement = state.next_question()
                if not placement.option or not placement.part:
                    # pathlib drops an empty segment, so this would land one
                    # directory short of {option}/{part}/ and be invisible to
                    # every reader of the output tree.
                    raise ValueError(
                        "Question detected before any option/part marker was read"
                    )
                output_path = (
                    output_dir
                    / str(placement.option)
                    / placement.part
                    / f"{placement.number}.{self.config.image_format}"
                )
                resolved_option = self._crop_and_save(
                    image,
                    det.polygon,
                    output_path,
                    placement.option,
                    output_dir,
                )
                state.commit_question()
                if state.correct_option(resolved_option):
                    logger.info(
                        "Option corrected",
                        from_option=placement.option,
                        to_option=resolved_option,
                    )
                logger.debug(
                    "Extracting question",
                    option=state.option,
                    part=state.part,
                    question=state.question,
                )
