"""Manual extractor for integrating manually cropped question images."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import structlog
from PIL import Image
from tqdm import tqdm

from digitex.core.corpus import IMAGE_EXTENSIONS, ManualImageName
from digitex.core.processors import SegmentProcessor, resize_image
from digitex.extractors.base import ExtractionResult
from digitex.extractors.exceptions import InvalidFilenameError

logger = structlog.get_logger()


class ManualExtractor:
    """Process manually cropped question images and integrate into extraction output."""

    def __init__(
        self,
        image_format: str = "jpg",
        question_max_width: int = 2000,
        question_max_height: int = 2000,
        manual_dir: Path | None = None,
        output_dir: Path | None = None,
        segment_processor: SegmentProcessor | None = None,
    ) -> None:
        """Initialize the manual extractor.

        Args:
            image_format: Output image format.
            question_max_width: Maximum width for question images.
            question_max_height: Maximum height for question images.
            manual_dir: Directory containing manual images.
            output_dir: Output directory for processed images.
            segment_processor: Optional pre-configured processor (dependency injection).
        """
        self.image_format = image_format
        self.question_max_width = question_max_width
        self.question_max_height = question_max_height
        self.manual_dir = manual_dir
        self.output_dir = output_dir
        self._segment_processor = segment_processor or SegmentProcessor()

    def _parse_filename(self, file_path: Path) -> tuple[int, int, str, int]:
        """Parse manual image filename into components.

        Args:
            file_path: Path to manual image file.

        Returns:
            Tuple of (year, option, part, question_number).

        Raises:
            InvalidFilenameError: If filename doesn't match expected pattern.
        """
        parsed = ManualImageName.parse(file_path.name)
        if parsed is None:
            raise InvalidFilenameError(file_path.name, "YYYY_OPTION_PART_QUESTION.png")
        return parsed.year, parsed.option, parsed.part, parsed.question

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Apply same preprocessing as automated extraction.

        Args:
            image: PIL Image (already cropped manually).

        Returns:
            Preprocessed RGB image ready for saving.
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        img_array = np.array(image)
        if img_array.shape[2] == 4:
            alpha = img_array[:, :, 3]
            if np.all(alpha == 255):
                mask = np.all(img_array[:, :, :3] < 50, axis=2)
                if np.any(mask):
                    img_array[mask, :3] = [255, 255, 255]
                image = Image.fromarray(img_array, mode="RGBA")

        cropped = resize_image(image, self.question_max_width, self.question_max_height)
        return self._segment_processor.process(cropped)

    def _get_existing_images(self, target_dir: Path) -> list[tuple[int, Path]]:
        """Get existing images in directory sorted by question number.

        Args:
            target_dir: Directory containing question images.

        Returns:
            List of (question_number, path) tuples sorted by number.
        """
        if not target_dir.exists():
            return []

        images = []
        for f in target_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    num = int(f.stem)
                    images.append((num, f))
                except ValueError:
                    logger.warning(
                        "Skipping file with non-numeric name",
                        file_path=str(f),
                    )

        return sorted(images, key=lambda x: x[0])

    def _renumber_files(
        self,
        target_dir: Path,
        start_num: int,
        dry_run: bool = False,
    ) -> list[tuple[Path, Path]]:
        """Shift existing files starting from start_num by +1.

        Args:
            target_dir: Directory containing files to renumber.
            start_num: Starting question number to shift from.
            dry_run: If True, only preview changes.

        Returns:
            List of (old_path, new_path) tuples for changed files.
        """
        existing = self._get_existing_images(target_dir)
        to_shift = [(n, p) for n, p in existing if n >= start_num]

        if not to_shift:
            return []

        changes: list[tuple[Path, Path]] = []
        for num, path in reversed(to_shift):
            new_num = num + 1
            new_path = target_dir / f"{new_num}{path.suffix}"
            changes.append((path, new_path))

        if not dry_run and changes:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                for old_path, new_path in changes:
                    temp_path = tmp_dir / new_path.name
                    shutil.move(str(old_path), str(temp_path))
                    shutil.move(str(temp_path), str(new_path))

        return changes

    def _process_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Process single manual image file.

        Args:
            file_path: Path to manual image file.
            dry_run: If True, only preview changes.

        Returns:
            True if processed successfully, False otherwise.
        """
        try:
            year, option, part, question_number = self._parse_filename(file_path)
        except InvalidFilenameError as e:
            logger.error("Skipping invalid filename", error=str(e))
            return False

        subject = file_path.parent.name
        assert self.output_dir is not None
        target_dir = self.output_dir / subject / str(year) / str(option) / part
        target_path = target_dir / f"{question_number}.{self.image_format}"

        logger.info(
            "Processing manual image",
            source=str(file_path),
            subject=subject,
            year=year,
            option=option,
            part=part,
            question=question_number,
            target=str(target_path),
        )

        if dry_run:
            logger.info(
                "[DRY RUN] Would process file",
                source=str(file_path),
                target=str(target_path),
            )
            return True

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.error(
                "Failed to open image",
                file_path=str(file_path),
                error=str(e),
            )
            return False

        processed = self._preprocess(image)

        target_dir.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            logger.info(
                "Target file exists, shifting subsequent files",
                target=str(target_path),
            )
            self._renumber_files(target_dir, question_number, dry_run=False)

        output_path = target_path.with_suffix(f".{self.image_format}")
        processed.save(output_path)
        logger.info("Saved processed image", path=str(output_path))

        file_path.unlink()
        logger.info("Deleted source manual file", path=str(file_path))
        return True

    def extract(self, dry_run: bool = False) -> ExtractionResult:
        """Process all manual images in the manual directory.

        Args:
            dry_run: If True, only preview changes without applying.

        Returns:
            ExtractionResult with statistics.
        """
        if not self.manual_dir or not self.manual_dir.exists():
            logger.warning("Manual directory does not exist", path=str(self.manual_dir))
            return ExtractionResult.success_result(
                processed=0, warnings=["Manual directory does not exist"]
            )

        manual_files = [
            f
            for f in self.manual_dir.rglob("*.png")
            if f.is_file() and f.parent != self.output_dir
        ]

        if not manual_files:
            logger.info("No manual images found", manual_dir=str(self.manual_dir))
            return ExtractionResult.success_result(
                processed=0, warnings=["No manual images found"]
            )

        logger.info(
            "Found manual images to process",
            count=len(manual_files),
            dry_run=dry_run,
        )

        processed_count = 0
        failed_count = 0

        for file_path in tqdm(manual_files, desc="Processing manual images"):
            if self._process_file(file_path, dry_run):
                processed_count += 1
            else:
                failed_count += 1

        if dry_run:
            logger.info("Dry run completed, no changes applied")
            return ExtractionResult.success_result(
                processed=processed_count,
                metadata={"dry_run": True, "failed": failed_count},
            )

        logger.info("Manual image processing completed")
        return ExtractionResult.success_result(
            processed=processed_count,
            metadata={"failed": failed_count},
        )

    def process_all(self, dry_run: bool = False) -> ExtractionResult:
        """Process all manual images (alias for extract).

        Args:
            dry_run: If True, only preview changes without applying.

        Returns:
            ExtractionResult with statistics.
        """
        return self.extract(dry_run=dry_run)
