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
    AutoConflictResolution,
    ConflictResolutionStrategy,
)
from digitex.ml.predictors import (
    SegmentationPredictionResult,
    YOLO_SegmentationPredictor,
)

logger = structlog.get_logger()

Detection = tuple[str, list[tuple[int, int]]]
OCR_LANGUAGE = "rus"


@dataclass
class PageExtractionState:
    """Mutable state threaded across pages during a book extraction run."""

    option: int = 0
    part: str = ""
    question: int = 0

    def try_advance_option(self, new_option: int | None) -> bool:
        """Advance to the next option if detected. Returns True on change."""
        if new_option is not None and new_option == self.option + 1:
            self.option = new_option
            self.part = "A"
            self.question = 0
            return True
        return False

    def try_advance_part(self, new_part: str | None) -> bool:
        """Switch to a different part if detected. Returns True on change."""
        if new_part is not None and new_part != self.part:
            self.part = new_part
            self.question = 0
            return True
        return False

    def advance_question(self, resolved_option: int) -> int:
        """Increment question counter; apply option correction if needed.

        Returns the option number that was active *before* any correction,
        so callers can log the transition.
        """
        prior_option = self.option
        self.question += 1
        if resolved_option != self.option:
            self.option = resolved_option
            self.part = "A"
        return prior_option


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
        conflict_strategy: ConflictResolutionStrategy | None = None,
    ) -> None:
        self.model_path = model_path
        self.image_format = image_format
        self.question_max_width = question_max_width
        self.question_max_height = question_max_height

        self._predictor = predictor
        self._segment_processor = segment_processor or SegmentProcessor()
        self._image_cropper = image_cropper or ImageCropper()
        self._text_extractor = text_extractor or TextExtractor(language=OCR_LANGUAGE)
        self._conflict_strategy = conflict_strategy or AutoConflictResolution()

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
        resolved_option = self._conflict_strategy.resolve(
            new_image=new_image,
            existing_path=output_path,
            current_option=current_option,
            source_image_name=source_image_name,
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
                if state.try_advance_option(new_option):
                    logger.debug("Option changed", option_counter=state.option)
            elif label == "part":
                new_part = self._extract_part_letter(image, polygon)
                if state.try_advance_part(new_part):
                    logger.debug("Part changed", part_letter=state.part)
            elif label == "question":
                output_path = (
                    output_dir
                    / str(state.option)
                    / state.part
                    / f"{state.question + 1}.{self.image_format}"
                )
                resolved_option = self._crop_and_save(
                    image,
                    polygon,
                    output_path,
                    state.option,
                    source_image_name,
                    output_dir,
                )
                prior_option = state.advance_question(resolved_option)
                logger.debug(
                    "Extracting question",
                    option=state.option,
                    part=state.part,
                    question=state.question,
                )
                if state.option != prior_option:
                    logger.info(
                        "Option corrected",
                        from_option=prior_option,
                        to_option=state.option,
                    )

        return state
