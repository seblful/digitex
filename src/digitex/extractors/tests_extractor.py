"""Tests extractor that orchestrates extraction of all image books."""

import json
from pathlib import Path

import structlog
from tqdm import tqdm

from digitex.extractors.book_extractor import BookExtractor

logger = structlog.get_logger()


class TestsExtractor:
    """Orchestrates extraction of question images from all image books."""

    def __init__(
        self,
        model_path: Path,
        image_format: str,
        question_max_width: int,
        question_max_height: int,
        books_dir: Path,
        extraction_dir: Path,
        data_dir: Path | None = None,
    ) -> None:
        self.books_dir = books_dir
        self.extraction_dir = extraction_dir
        self.data_dir = data_dir or extraction_dir.parent / "data"
        self._progress_path = self.data_dir / "progress.json"

        self._book_extractor = BookExtractor(
            model_path=model_path,
            image_format=image_format,
            question_max_width=question_max_width,
            question_max_height=question_max_height,
        )

    def _load_completed(self) -> dict[str, set[str]]:
        if not self._progress_path.exists():
            return {}
        data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        return {k: set(v) for k, v in data.items()}

    def _save_completed(self, completed: dict[str, set[str]]) -> None:
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path.write_text(
            json.dumps({k: sorted(v) for k, v in completed.items()}, indent=2),
            encoding="utf-8",
        )

    def _is_completed(
        self, completed: dict[str, set[str]], subject: str, year: str
    ) -> bool:
        return year in completed.get(subject, set())

    def extract_all(self) -> None:
        """Extract question images from all subjects in the books directory.

        Raises:
            FileNotFoundError: If books directory doesn't exist.
        """
        if not self.books_dir.exists():
            raise FileNotFoundError(f"Books directory not found: {self.books_dir}")

        subject_dirs = [d for d in self.books_dir.iterdir() if d.is_dir()]

        if not subject_dirs:
            logger.warning("No subject folders found", books_dir=self.books_dir)
            return

        completed = self._load_completed()

        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject = subject_dir.name
            images_dir = subject_dir / "images"

            if not images_dir.exists():
                logger.warning("No images folder found", subject_dir=subject_dir)
                continue

            year_dirs = [d for d in images_dir.iterdir() if d.is_dir()]

            if not year_dirs:
                logger.warning("No year folders found", images_dir=images_dir)
                continue

            for year_dir in tqdm(year_dirs, desc=f"Extracting {subject}", leave=False):
                year = year_dir.name

                if self._is_completed(completed, subject, year):
                    logger.info(
                        "Skipping, already extracted", subject=subject, year=year
                    )
                    continue

                output_dir = self.extraction_dir / subject / year
                self._book_extractor.extract(year_dir, output_dir)

                completed.setdefault(subject, set()).add(year)
                self._save_completed(completed)
