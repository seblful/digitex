"""Manual extractor for integrating manually cropped question images."""

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import structlog
from PIL import Image
from tqdm import tqdm

from digitex.core.processors import SegmentProcessor, resize_image

logger = structlog.get_logger()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
FILENAME_PATTERN = re.compile(r"^(\d{4})_(\d+)_([AB])_(\d+)\.png$")
VALID_PARTS = {"A", "B"}


class ManualExtractor:
    """Process manually cropped question images and integrate into extraction output."""

    def __init__(
        self,
        image_format: str,
        question_max_width: int,
        question_max_height: int,
        manual_dir: Path,
        output_dir: Path,
    ) -> None:
        self.image_format = image_format
        self.question_max_width = question_max_width
        self.question_max_height = question_max_height
        self.manual_dir = manual_dir
        self.output_dir = output_dir
        self._segment_processor = SegmentProcessor()

    def _parse_filename(self, file_path: Path) -> tuple[int, int, str, int]:
        """Parse manual image filename into components.

        Args:
            file_path: Path to manual image file.

        Returns:
            Tuple of (year, option, part, question_number).

        Raises:
            ValueError: If filename doesn't match expected pattern or has invalid part.
        """
        match = FILENAME_PATTERN.match(file_path.name)
        if not match:
            if file_path.name.endswith(".png"):
                parts = file_path.stem.split("_")
                if len(parts) < 4:
                    raise ValueError(
                        f"Invalid filename format: {file_path.name}. "
                        f"Expected format: YYYY_OPTION_PART_QUESTION.png (e.g., 2016_3_A_20.png)"
                    )
                if len(parts) == 4 and not parts[3]:
                    raise ValueError(
                        f"Question number is missing in filename: {file_path.name}"
                    )
                if len(parts) >= 3 and parts[2].upper() not in VALID_PARTS:
                    raise ValueError(
                        f"Invalid part '{parts[2]}' in filename: {file_path.name}. "
                        f"Part must be one of: {', '.join(sorted(VALID_PARTS))}"
                    )
            raise ValueError(
                f"Invalid filename format: {file_path.name}. "
                f"Expected format: YYYY_OPTION_PART_QUESTION.png (e.g., 2016_3_A_20.png)"
            )

        year = int(match.group(1))
        option = int(match.group(2))
        part = match.group(3).upper()
        question_number = int(match.group(4))

        return year, option, part, question_number

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
        processed = self._segment_processor.process(cropped)
        return processed

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
        dry_run: bool,
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

    def _process_file(self, file_path: Path, dry_run: bool) -> None:
        """Process single manual image file.

        Args:
            file_path: Path to manual image file.
            dry_run: If True, only preview changes.
        """
        try:
            year, option, part, question_number = self._parse_filename(file_path)
        except ValueError as e:
            logger.error("Skipping invalid filename", error=str(e))
            return

        subject = file_path.parent.name
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
            return

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.error(
                "Failed to open image",
                file_path=str(file_path),
                error=str(e),
            )
            return

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

    def process_all(self, dry_run: bool = False) -> None:
        """Process all manual images in the manual directory.

        Args:
            dry_run: If True, only preview changes without applying.
        """
        if not self.manual_dir.exists():
            logger.warning("Manual directory does not exist", path=str(self.manual_dir))
            return

        manual_files = [
            f
            for f in self.manual_dir.rglob("*.png")
            if f.is_file() and f.parent != self.output_dir
        ]

        if not manual_files:
            logger.info("No manual images found", manual_dir=str(self.manual_dir))
            return

        logger.info(
            "Found manual images to process",
            count=len(manual_files),
            dry_run=dry_run,
        )

        for file_path in tqdm(manual_files, desc="Processing manual images"):
            self._process_file(file_path, dry_run)

        if dry_run:
            logger.info("Dry run completed, no changes applied")
        else:
            logger.info("Manual image processing completed")
