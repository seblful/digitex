"""Page extractor for extracting question images from a single page."""

from pathlib import Path

import structlog
from PIL import Image

from digitex.core import TextExtractor
from digitex.core.processors import (
    ImageCropper,
    SegmentProcessor,
    resize_image,
)
from digitex.ml.predictors import (
    SegmentationPredictionResult,
    YOLO_SegmentationPredictor,
)

logger = structlog.get_logger()

Detection = tuple[str, list[tuple[int, int]]]

OCR_LANGUAGE = "rus"


class PageExtractor:
    """Extract question images from a single page using YOLO segmentation."""

    def __init__(
        self,
        model_path: Path,
        image_format: str,
        question_max_width: int,
        question_max_height: int,
    ) -> None:
        self.model_path = model_path
        self.image_format = image_format
        self.question_max_width = question_max_width
        self.question_max_height = question_max_height

        self._predictor: YOLO_SegmentationPredictor | None = None
        self._segment_processor = SegmentProcessor()
        self._image_cropper = ImageCropper()
        self._text_extractor = TextExtractor(language=OCR_LANGUAGE)

    @property
    def predictor(self) -> YOLO_SegmentationPredictor:
        """Get or initialize the YOLO predictor.

        Returns:
            Initialized YOLO segmentation predictor.
        """
        if self._predictor is None:
            self._predictor = YOLO_SegmentationPredictor(str(self.model_path))
        return self._predictor

    def _get_label_name(
        self, result: SegmentationPredictionResult, class_id: int
    ) -> str:
        return result.id2label.get(class_id, "unknown")

    def _get_polygon_bounding_box(
        self, polygon: list[tuple[int, int]]
    ) -> tuple[int, int]:
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
        existing_option = self._resolve_option_conflict(
            new_image, output_path, current_option, source_image_name
        )
        if existing_option != current_option:
            correct_path = output_dir / str(existing_option) / "A" / output_path.name
            correct_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Saving corrected image",
                from_path=output_path,
                to_path=correct_path,
            )
            new_image.save(str(correct_path))
            output_path.unlink()
            return existing_option
        return existing_option

    def _resolve_option_conflict(
        self,
        new_image: Image.Image,
        existing_path: Path,
        current_option: int,
        source_image_name: str,
    ) -> int:
        new_image.show()
        print(f"Image: {source_image_name}")
        print(f"Current option: {current_option}")

        while True:
            user_input = input(
                f"Enter correct option number (current: {current_option}): "
            ).strip()
            if user_input.isdigit():
                option = int(user_input)
                if 1 <= option <= 10:
                    return option
            print("Please enter a number between 1 and 10")

    def _extract_option_number(
        self, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> int | None:
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        digits = self._text_extractor.extract_digits(cropped)
        if digits:
            remainder = digits[0] % 10
            return 10 if remainder == 0 else remainder
        return None

    def _extract_part_letter(
        self, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> str | None:
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

        Args:
            image: PIL Image to run detection on.

        Returns:
            List of detections sorted by position (top to bottom, left to right).

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
        for class_id, polygon in zip(result.ids, result.polygons):
            label = self._get_label_name(result, class_id)
            position = self._get_polygon_bounding_box(polygon)
            detections.append((position, (label, polygon)))

        detections.sort(key=lambda x: x[0])
        return [det for _, det in detections]

    def extract(
        self,
        image: Image.Image,
        output_dir: Path,
        option_counter: int,
        part_letter: str,
        question_counter: int,
        source_image_name: str,
    ) -> tuple[int, str, int]:
        """Extract questions from a single page image.

        Args:
            image: PIL Image of the page.
            output_dir: Base output directory.
            option_counter: Current option counter.
            part_letter: Current part letter ("A" or "B").
            question_counter: Current question counter.
            source_image_name: Source image filename for display.

        Returns:
            Updated tuple of (option_counter, part_letter, question_counter).
        """
        detections = self._detect(image)

        for label, polygon in detections:
            if label == "option":
                new_option = self._extract_option_number(image, polygon)
                if new_option is not None and new_option == option_counter + 1:
                    option_counter = new_option
                    part_letter = "A"
                    question_counter = 0
                    logger.debug("Option changed", option_counter=option_counter)
            elif label == "part":
                new_part_letter = self._extract_part_letter(image, polygon)
                if new_part_letter is not None and new_part_letter != part_letter:
                    part_letter = new_part_letter
                    question_counter = 0
                    logger.debug("Part changed", part_letter=part_letter)
            elif label == "question":
                output_subdir = output_dir / str(option_counter) / part_letter
                output_path = (
                    output_subdir / f"{question_counter + 1}.{self.image_format}"
                )

                resolved_option = self._crop_and_save(
                    image,
                    polygon,
                    output_path,
                    option_counter,
                    source_image_name,
                    output_dir,
                )
                question_counter += 1
                logger.debug(
                    "Extracting option",
                    option_counter=option_counter,
                    part_letter=part_letter,
                    question_counter=question_counter,
                )
                if resolved_option != option_counter:
                    logger.info(
                        "Option corrected",
                        from_option=option_counter,
                        to_option=resolved_option,
                    )
                    option_counter = resolved_option
                    part_letter = "A"

        return option_counter, part_letter, question_counter
