"""Book extractor for extracting question images from PDF books."""

import logging
import re
from pathlib import Path

import pytesseract
import pypdfium2 as pdfium
from PIL import Image
from tqdm import tqdm

from digitex.core.handlers import PDFHandler
from digitex.core.processors import ImageCropper
from digitex.ml.predictors import (
    SegmentationPredictionResult,
    YOLO_SegmentationPredictor,
)

logger = logging.getLogger(__name__)


class BookExtractor:
    """Extract question images from a single PDF book using YOLO segmentation."""

    def __init__(
        self,
        model_path: Path,
        render_scale: int,
        image_format: str,
    ) -> None:
        """Initialize the book extractor.

        Args:
            model_path: Path to YOLO model.
            render_scale: PDF render scale factor.
            image_format: Output image format.
        """
        self.model_path = model_path
        self.render_scale = render_scale
        self.image_format = image_format

        self._predictor: YOLO_SegmentationPredictor | None = None
        self._pdf_handler = PDFHandler()
        self._image_cropper = ImageCropper()

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
        """Get label name from class ID.

        Args:
            result: Segmentation prediction result.
            class_id: Class ID to look up.

        Returns:
            Label name string.
        """
        return result.id2label.get(class_id, "unknown")

    def _get_polygon_bounding_box(
        self, polygon: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """Get the top-left corner of a polygon's bounding box.

        Args:
            polygon: List of (x, y) tuples.

        Returns:
            Tuple of (y, x) for sorting purposes.
        """
        min_y = min(p[1] for p in polygon)
        min_x = min(p[0] for p in polygon)
        return (min_y, min_x)

    def _extract_option_number(
        self, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> int | None:
        """Extract option number from image using OCR.

        Args:
            image: Source PIL Image.
            polygon: Polygon coordinates for the option region.

        Returns:
            Extracted option number or None if not found.
        """
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)

        text = pytesseract.image_to_string(
            cropped,
            lang="rus",
            config="--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789",
        )
        logger.debug(f"OCR text for option: '{text.strip()}'")

        numbers = re.findall(r"\d+", text)
        logger.debug(f"Extracted numbers from OCR: {numbers}")
        if numbers:
            return int(numbers[0]) % 10

        return None

    def _extract_part_letter(
        self, image: Image.Image, polygon: list[tuple[int, int]]
    ) -> str | None:
        """Extract part letter (A or B) from image using OCR.

        Args:
            image: Source PIL Image.
            polygon: Polygon coordinates for the part region.

        Returns:
            Extracted part letter ("A" or "B") or None if not found.
        """
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)

        text = pytesseract.image_to_string(
            cropped,
            lang="rus",
            config="--psm 7 --oem 1",
        )
        logger.debug(f"OCR text for part: '{text.strip()}'")

        text = text.replace("Часть", "").replace("часть", "").strip()

        cyrillic_to_latin = str.maketrans("АБВ", "ABB")
        text_normalized = text.upper().translate(cyrillic_to_latin)

        if "A" in text_normalized:
            return "A"
        if "B" in text_normalized:
            return "B"

        return None

    def _crop_and_save(
        self,
        image: Image.Image,
        polygon: list[tuple[int, int]],
        output_path: Path,
    ) -> None:
        """Crop image using polygon and save to path.

        Args:
            image: Source PIL Image.
            polygon: Polygon coordinates for cropping.
            output_path: Path to save cropped image.
        """
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)

    def _process_page(
        self,
        image: Image.Image,
        output_dir: Path,
        option_counter: int,
        part_letter: str,
        question_counter: int,
    ) -> tuple[int, str, int]:
        """Process a single page image and extract questions.

        Args:
            image: PIL Image of the page.
            output_dir: Base output directory for this PDF.
            option_counter: Current option counter.
            part_letter: Current part letter ("A" or "B").
            question_counter: Current question counter.

        Returns:
            Updated tuple of (option_counter, part_letter, question_counter).

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
        logger.debug(f"Predictions: {class_counts}")

        detections = []
        for class_id, polygon in zip(result.ids, result.polygons):
            label = self._get_label_name(result, class_id)
            position = self._get_polygon_bounding_box(polygon)
            detections.append((position, label, polygon))

        detections.sort(key=lambda x: x[0])

        for position, label, polygon in detections:
            if label == "option":
                new_option = self._extract_option_number(image, polygon)
                if new_option is not None and new_option == option_counter + 1:
                    option_counter = new_option
                    part_letter = ""
                    question_counter = 0
                    logger.debug(f"Option changed to: {option_counter}")
            elif label == "part":
                new_part_letter = self._extract_part_letter(image, polygon)
                if new_part_letter is not None and new_part_letter != part_letter:
                    part_letter = new_part_letter
                    question_counter = 0
                    logger.debug(f"Part changed to: {part_letter}")
            elif label == "question":
                question_counter += 1

                output_subdir = output_dir / str(option_counter) / part_letter
                output_path = output_subdir / f"{question_counter}.{self.image_format}"

                self._crop_and_save(image, polygon, output_path)

        return option_counter, part_letter, question_counter

    def extract(
        self,
        pdf_path: Path,
        output_dir: Path,
    ) -> None:
        """Extract question images from a single PDF file.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Output directory for extracted images.

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            ValueError: If no detections found on any page.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf = pdfium.PdfDocument(str(pdf_path))
        num_pages = len(pdf)

        option_counter = 0
        part_letter = ""
        question_counter = 0

        for page_idx in tqdm(
            range(num_pages), desc=f"Processing {pdf_path.name}", leave=False
        ):
            page = pdf[page_idx]
            image = self._pdf_handler.get_page_image(page, scale=self.render_scale)

            option_counter, part_letter, question_counter = self._process_page(
                image, output_dir, option_counter, part_letter, question_counter
            )

        pdf.close()
        logger.info(f"Extracted images to {output_dir}")
