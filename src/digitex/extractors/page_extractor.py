"""Page extractor for extracting question images from a single page."""

from dataclasses import dataclass
from pathlib import Path

import structlog
from PIL import Image

from digitex.core import TextExtractor
from digitex.core.processors import (
    ImageCropper,
    SegmentProcessor,
    resize_image,
)
from digitex.extractors.conflict_resolution import (
    Conflict,
    ConflictResolver,
    keep_current_option,
)
from digitex.ml.predictors import (
    SegmentationPredictionResult,
    YOLO_SegmentationPredictor,
)

logger = structlog.get_logger()

Detection = tuple[str, list[tuple[int, int]]]
OCR_LANGUAGE = "rus"


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
        self.part = "A"
        return True


class PageExtractor:
    """Extract question images from a single page using YOLO segmentation."""

    def __init__(
        self,
        model_path: Path,
        image_format: str = "jpg",
        question_max_width: int = 2000,
        question_max_height: int = 2000,
        predictor: YOLO_SegmentationPredictor | None = None,
        segment_processor: SegmentProcessor | None = None,
        image_cropper: ImageCropper | None = None,
        text_extractor: TextExtractor | None = None,
        on_conflict: ConflictResolver | None = None,
    ) -> None:
        self.model_path = model_path
        self.image_format = image_format
        self.question_max_width = question_max_width
        self.question_max_height = question_max_height

        self._predictor = predictor
        self._segment_processor = segment_processor or SegmentProcessor()
        self._image_cropper = image_cropper or ImageCropper()
        self._text_extractor = text_extractor or TextExtractor(language=OCR_LANGUAGE)
        self._on_conflict = on_conflict or keep_current_option

    @property
    def predictor(self) -> YOLO_SegmentationPredictor:
        """Get or initialize the YOLO predictor."""
        if self._predictor is None:
            self._predictor = YOLO_SegmentationPredictor(str(self.model_path))
        return self._predictor

    def _get_label_name(
        self, result: SegmentationPredictionResult, class_id: int
    ) -> str:
        """Get label name from class ID."""
        return result.id2label.get(class_id, "unknown")

    def _get_polygon_bounding_box(
        self, polygon: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """Get bounding box position from polygon."""
        min_y = min(p[1] for p in polygon)
        min_x = min(p[0] for p in polygon)
        return (min_y, min_x)

    def _crop_and_save(
        self,
        image: Image.Image,
        polygon: list[tuple[int, int]],
        output_path: Path,
        current_option: int,
        source_image_name: str,
        output_dir: Path,
    ) -> int:
        """Crop, process, and save extracted image. Returns resolved option number."""
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        cropped = resize_image(
            cropped, self.question_max_width, self.question_max_height
        )
        processed = self._segment_processor.process(cropped)
        output_path = output_path.with_suffix(f".{self.image_format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            return self._handle_existing_file(
                output_path, processed, current_option, source_image_name, output_dir
            )

        processed.save(output_path)
        return current_option

    def _handle_existing_file(
        self,
        output_path: Path,
        new_image: Image.Image,
        current_option: int,
        source_image_name: str,
        output_dir: Path,
    ) -> int:
        """Handle case when output file already exists. Returns resolved option."""
        resolved_option = self._on_conflict(
            Conflict(
                new_image=new_image,
                existing_path=output_path,
                current_option=current_option,
                source_image_name=source_image_name,
            )
        )

        if resolved_option != current_option:
            correct_path = output_dir / str(resolved_option) / "A" / output_path.name
            correct_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Saving corrected image",
                from_path=str(output_path),
                to_path=str(correct_path),
            )
            new_image.save(str(correct_path))
            output_path.unlink()
            return resolved_option

        return resolved_option

    def _extract_option_number(
        self, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> int | None:
        """Extract option number from image region."""
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        digits = self._text_extractor.extract_digits(cropped)
        if digits:
            remainder = digits[0] % 10
            return 10 if remainder == 0 else remainder
        return None

    def _extract_part_letter(
        self, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> str | None:
        """Extract part letter (A/B) from image region."""
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        text = self._text_extractor.extract_text(cropped)
        text = text.replace("Часть", "").replace("часть", "").strip()
        cyrillic_to_latin = str.maketrans("АБВ", "ABB")
        text_normalized = text.upper().translate(cyrillic_to_latin)

        if "A" in text_normalized:
            return "A"
        if "B" in text_normalized:
            return "B"
        return None

    def _detect(self, image: Image.Image) -> list[Detection]:
        """Run YOLO prediction and return sorted detections.

        Raises:
            ValueError: If no detections are found on the page.
        """
        result = self.predictor.predict(image)

        if not result.ids:
            raise ValueError("No detections found on page")

        class_counts: dict[str, int] = {}
        for class_id in result.ids:
            label = self._get_label_name(result, class_id)
            class_counts[label] = class_counts.get(label, 0) + 1
        logger.debug("Predictions", class_counts=class_counts)

        detections: list[tuple[tuple[int, int], Detection]] = []
        for class_id, polygon in zip(result.ids, result.polygons, strict=False):
            label = self._get_label_name(result, class_id)
            position = self._get_polygon_bounding_box(polygon)
            detections.append((position, (label, polygon)))

        detections.sort(key=lambda x: x[0])
        return [det for _, det in detections]

    def extract(
        self,
        image: Image.Image,
        output_dir: Path,
        state: PageExtractionState | None = None,
        source_image_name: str = "",
    ) -> PageExtractionState:
        """Extract questions from a single page image.

        Args:
            image: PIL Image of the page.
            output_dir: Base output directory.
            state: Extraction state carried across pages. Created fresh if None.
            source_image_name: Source image filename for conflict resolution display.

        Returns:
            Updated extraction state.
        """
        if state is None:
            state = PageExtractionState()

        detections = self._detect(image)

        for label, polygon in detections:
            if label == "option":
                new_option = self._extract_option_number(image, polygon)
                if state.on_option(new_option):
                    logger.debug("Option changed", option_counter=state.option)
            elif label == "part":
                new_part = self._extract_part_letter(image, polygon)
                if state.on_part(new_part):
                    logger.debug("Part changed", part_letter=state.part)
            elif label == "question":
                placement = state.next_question()
                output_path = (
                    output_dir
                    / str(placement.option)
                    / placement.part
                    / f"{placement.number}.{self.image_format}"
                )
                resolved_option = self._crop_and_save(
                    image,
                    polygon,
                    output_path,
                    placement.option,
                    source_image_name,
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

        return state
