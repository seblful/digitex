"""Data extraction utilities."""

import logging
from pathlib import Path
from typing import Dict

import pypdfium2
from PIL import Image

from .handlers import ImageHandler, LabelHandler, PDFHandler
from .processors import ImageProcessor

logger = logging.getLogger(__name__)


class DataCreator:
    """Class for creating training data from PDFs and images.

    This class provides methods to extract and save training data at various
    levels of granularity: pages, questions, parts, and words from PDF documents
    and their annotations.
    """

    def __init__(self) -> None:
        """Initialize the DataCreator with handlers and processors."""
        self.image_processor = ImageProcessor()
        self.pdf_handler = PDFHandler()
        self.image_handler = ImageHandler()
        self.label_handler = LabelHandler()

    @staticmethod
    def _read_classes(classes_path: Path) -> Dict[int, str]:
        """Read class names from a text file.

        Args:
            classes_path: Path to the classes.txt file.

        Returns:
            Dictionary mapping class indices to class names.

        Raises:
            FileNotFoundError: If classes_path does not exist.
            IOError: If the file cannot be read.
        """
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            return {i: cl for i, cl in enumerate(classes)}

    def _save_image(
        self,
        *args: int,
        train_dir: Path,
        image: Image.Image,
        image_name: str,
        num_saved: int,
        num_images: int,
    ) -> int:
        """Save an image to the training directory.

        Args:
            *args: Identifiers to include in the filename.
            train_dir: Directory where the image will be saved.
            image: PIL Image to save.
            image_name: Original name of the image.
            num_saved: Current count of saved images.
            num_images: Total number of images to save.

        Returns:
            Updated count of saved images.

        Raises:
            IOError: If the image cannot be saved.
        """
        image_stem = Path(image_name).stem
        str_ids = "_".join([str(i) for i in args])
        image_path = train_dir / f"{image_stem}_{str_ids}.jpg"

        if not image_path.exists():
            image.save(image_path, "JPEG")
            num_saved += 1
            logger.info(f"{num_saved}/{num_images} images saved.")

            image.close()

        return num_saved

    def extract_pages(
        self,
        raw_dir: str | Path,
        train_dir: str | Path,
        scan_type: str,
        num_images: int,
        max_attempts: int = 1000,
    ) -> None:
        """Extract pages from PDFs and save as training images.

        Args:
            raw_dir: Directory containing raw PDF files.
            train_dir: Directory where extracted pages will be saved.
            scan_type: Type of scan processing ('bw', 'gray', 'color').
            num_images: Number of images to extract.
            max_attempts: Maximum number of extraction attempts before giving up.

        Raises:
            ValueError: If scan_type is invalid or max_attempts is reached.
        """
        raw_dir = Path(raw_dir)
        train_dir = Path(train_dir)
        train_dir.mkdir(parents=True, exist_ok=True)

        if scan_type not in self.image_processor.scan_types:
            raise ValueError(
                f"Scan type must be one of {self.image_processor.scan_types}"
            )

        pdf_list = [
            pdf for pdf in raw_dir.iterdir() if pdf.suffix.lower() == '.pdf'
        ]
        if not pdf_list:
            raise ValueError(f"No PDF files found in {raw_dir}")

        num_saved = 0
        attempts = 0

        while num_images > num_saved:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Failed to extract {num_images} images after {max_attempts} attempts"
                )

            try:
                rand_image, rand_image_name, rand_page_idx = self.pdf_handler.get_random_image(
                    pdf_listdir=[str(pdf) for pdf in pdf_list],
                    pdf_dir=str(raw_dir),
                )

                rand_image = self.image_processor.process(
                    image=rand_image,
                    scan_type=scan_type,
                )

                num_saved = self._save_image(
                    rand_page_idx,
                    train_dir=train_dir,
                    image=rand_image,
                    image_name=rand_image_name,
                    num_saved=num_saved,
                    num_images=num_images,
                )
            except (FileNotFoundError, IOError, ValueError, pypdfium2.PdfiumError) as e:
                logger.warning(f"Failed to extract page: {e}")
                continue

    def extract_questions(
        self,
        page_raw_dir: str | Path,
        train_dir: str | Path,
        num_images: int,
        max_attempts: int = 1000,
    ) -> None:
        """Extract questions from annotated pages.

        Args:
            page_raw_dir: Directory containing page images and labels.
            train_dir: Directory where extracted questions will be saved.
            num_images: Number of questions to extract.
            max_attempts: Maximum number of extraction attempts.

        Raises:
            ValueError: If required directories don't exist or max_attempts reached.
        """
        page_raw_dir = Path(page_raw_dir)
        train_dir = Path(train_dir)
        train_dir.mkdir(parents=True, exist_ok=True)

        images_dir = page_raw_dir / "images"
        labels_dir = page_raw_dir / "labels"
        classes_path = page_raw_dir / "classes.txt"

        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        if not classes_path.exists():
            raise ValueError(f"Classes file not found: {classes_path}")

        classes_dict = self._read_classes(classes_path)
        images_list = list(images_dir.iterdir())

        num_saved = 0
        attempts = 0

        while num_images > num_saved:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Failed to extract {num_images} images after {max_attempts} attempts"
                )

            try:
                rand_image, rand_image_name = self.image_handler.get_random_image(
                    images_listdir=[str(img) for img in images_list],
                    images_dir=str(images_dir),
                )

                rand_points_idx, rand_points = self.label_handler.get_points(
                    image_name=rand_image_name,
                    labels_dir=str(labels_dir),
                    classes_dict=classes_dict,
                    target_classes=["question"],
                )
                if len(rand_points) == 0:
                    continue

                rand_image = self.image_handler.crop_image(
                    image=rand_image,
                    points=rand_points,
                )

                num_saved = self._save_image(
                    rand_points_idx,
                    train_dir=train_dir,
                    image=rand_image,
                    image_name=rand_image_name,
                    num_saved=num_saved,
                    num_images=num_images,
                )
            except (FileNotFoundError, IOError, ValueError) as e:
                logger.warning(f"Failed to extract question: {e}")
                continue

    def extract_parts(
        self,
        question_raw_dir: str | Path,
        train_dir: str | Path,
        num_images: int,
        target_classes: list[str] | None = None,
        max_attempts: int = 1000,
    ) -> None:
        """Extract parts (answers, numbers, options, etc.) from questions.

        Args:
            question_raw_dir: Directory containing question images and labels.
            train_dir: Directory where extracted parts will be saved.
            num_images: Number of parts to extract.
            target_classes: List of class names to extract.
            max_attempts: Maximum number of extraction attempts.

        Raises:
            ValueError: If required directories don't exist or max_attempts reached.
        """
        if target_classes is None:
            target_classes = ["answer", "number", "option", "question", "spec"]

        question_raw_dir = Path(question_raw_dir)
        train_dir = Path(train_dir)
        train_dir.mkdir(parents=True, exist_ok=True)

        images_dir = question_raw_dir / "images"
        labels_dir = question_raw_dir / "labels"
        classes_path = question_raw_dir / "classes.txt"

        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        if not classes_path.exists():
            raise ValueError(f"Classes file not found: {classes_path}")

        classes_dict = self._read_classes(classes_path)
        images_list = list(images_dir.iterdir())

        num_saved = 0
        attempts = 0

        while num_images > num_saved:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Failed to extract {num_images} images after {max_attempts} attempts"
                )

            try:
                rand_image, rand_image_name = self.image_handler.get_random_image(
                    images_listdir=[str(img) for img in images_list],
                    images_dir=str(images_dir),
                )

                rand_points_idx, rand_points = self.label_handler.get_points(
                    image_name=rand_image_name,
                    labels_dir=str(labels_dir),
                    classes_dict=classes_dict,
                    target_classes=target_classes,
                )
                if len(rand_points) == 0:
                    continue

                rand_image = self.image_handler.crop_image(
                    image=rand_image,
                    points=rand_points,
                    offset=0.0,
                )

                num_saved = self._save_image(
                    rand_points_idx,
                    train_dir=train_dir,
                    image=rand_image,
                    image_name=rand_image_name,
                    num_saved=num_saved,
                    num_images=num_images,
                )
            except (FileNotFoundError, IOError, ValueError) as e:
                logger.warning(f"Failed to extract part: {e}")
                continue

    def extract_words(
        self,
        parts_raw_dir: str | Path,
        train_dir: str | Path,
        num_images: int,
        max_attempts: int = 1000,
    ) -> None:
        """Extract words from annotated parts.

        Args:
            parts_raw_dir: Directory containing part images and labels.
            train_dir: Directory where extracted words will be saved.
            num_images: Number of words to extract.
            max_attempts: Maximum number of extraction attempts.

        Raises:
            ValueError: If required directories don't exist or max_attempts reached.
        """
        parts_raw_dir = Path(parts_raw_dir)
        train_dir = Path(train_dir)
        train_dir.mkdir(parents=True, exist_ok=True)

        images_dir = parts_raw_dir / "images"
        labels_dir = parts_raw_dir / "labels"
        classes_path = parts_raw_dir / "classes.txt"

        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        if not classes_path.exists():
            raise ValueError(f"Classes file not found: {classes_path}")

        classes_dict = self._read_classes(classes_path)
        target_classes = ["text"]
        images_list = list(images_dir.iterdir())

        num_saved = 0
        attempts = 0

        while num_images > num_saved:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Failed to extract {num_images} images after {max_attempts} attempts"
                )

            try:
                rand_image, rand_image_name = self.image_handler.get_random_image(
                    images_listdir=[str(img) for img in images_list],
                    images_dir=str(images_dir),
                )

                rand_points_idx, rand_points = self.label_handler.get_points(
                    image_name=rand_image_name,
                    labels_dir=str(labels_dir),
                    classes_dict=classes_dict,
                    target_classes=target_classes,
                )

                rand_image = self.image_handler.crop_image(
                    image=rand_image,
                    points=rand_points,
                    offset=0.0,
                )

                num_saved = self._save_image(
                    rand_points_idx,
                    train_dir=train_dir,
                    image=rand_image,
                    image_name=rand_image_name,
                    num_saved=num_saved,
                    num_images=num_images,
                )
            except (FileNotFoundError, IOError, ValueError) as e:
                logger.warning(f"Failed to extract word: {e}")
                continue
