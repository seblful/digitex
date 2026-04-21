"""Answers extractor using Mistral OCR API."""

import base64
import json
import os
import re
from pathlib import Path

import structlog
from mistralai.client import Mistral
from tqdm import tqdm

from digitex.config import get_settings
from digitex.extractors.base import BaseExtractor, ExtractionResult
from digitex.extractors.exceptions import APIError, DirectoryNotFoundError

logger = structlog.get_logger()


class AnswersExtractor(BaseExtractor):
    """Extracts answer keys from answer sheet images via Mistral OCR."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Mistral | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            api_key: Mistral API key. If None, uses MISTRAL_API_KEY from environment or settings.
            model: OCR model name. If None, uses settings or default.
            client: Optional pre-configured Mistral client. If None, creates a new client.
        """
        if client is not None:
            self._client = client
            self._api_key = ""
            self._model = ""
            return

        if api_key is None:
            try:
                settings = get_settings()
                api_key = settings.mistral.api_key or os.environ.get("MISTRAL_API_KEY")
            except Exception:
                api_key = os.environ.get("MISTRAL_API_KEY")

        if model is None:
            try:
                settings = get_settings()
                model = settings.mistral.ocr_model
            except Exception:
                model = "mistral-ocr-latest"

        self._api_key = api_key or ""
        self._model = model

        if not self._api_key:
            raise APIError(
                service="Mistral",
                message="API key not set. Set MISTRAL_API_KEY environment variable.",
            )

        self._client = Mistral(api_key=self._api_key)

    def _validate_prerequisites(self) -> None:
        """Validate that client is initialized."""
        if self._client is None:
            raise APIError(
                service="Mistral",
                message="Client not initialized. API key required.",
            )

    def encode_image(self, image_path: Path) -> str:
        """Encode an image file to a data URL (base64).

        Args:
            image_path: Path to the image file.

        Returns:
            Data URL string (e.g. data:image/png;base64,...).
        """
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        suffix = image_path.suffix.lower()
        media_type = (
            "image/jpeg"
            if suffix in (".jpg", ".jpeg")
            else f"image/{suffix[1:] or 'png'}"
        )
        return f"data:{media_type};base64,{b64}"

    def ocr(self, image_path: Path) -> str:
        """Run OCR on an image and return extracted text as markdown.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text as markdown.

        Raises:
            APIError: If OCR API call fails.
        """
        try:
            data_url = self.encode_image(image_path)

            res = self._client.ocr.process(
                model=self._model,
                document={
                    "image_url": {"url": data_url},
                    "type": "image_url",
                },
            )
            if res.pages:
                return res.pages[0].markdown or ""
            return ""
        except Exception as e:
            raise APIError(
                service="Mistral",
                message=f"OCR failed: {str(e)}",
                context={"image_path": str(image_path)},
            ) from e

    def _parse_markdown_table(self, markdown: str) -> list[list[str]]:
        """Parse markdown table into list of rows.

        Args:
            markdown: Markdown text containing tables.

        Returns:
            List of rows, where each row is a list of cell values.
        """
        rows: list[list[str]] = []
        lines = markdown.split("\n")

        in_table = False
        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                if in_table:
                    break
                continue

            in_table = True
            if "|---" in line:
                continue

            cells = [cell.strip() for cell in line.split("|")]
            if cells and (cells[0] == "" or cells[-1] == ""):
                cells = cells[1:-1] if cells[0] == "" else cells
                cells = cells[:-1] if cells and cells[-1] == "" else cells

            if cells and any(cell for cell in cells):
                rows.append(cells)

        return rows

    def _normalize_label(self, label: str) -> str:
        """Normalize question label to use only Latin A or B.

        Converts Cyrillic letters (А, В) to Latin (A, B).

        Args:
            label: Question label (e.g., "А1", "В2").

        Returns:
            Normalized label with Latin letters.
        """
        cyrillic_to_latin = str.maketrans("АВЕС", "ABEC")
        return label.upper().translate(cyrillic_to_latin)

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer to use only Cyrillic letters or digits.

        Converts Latin letters that look like Cyrillic to Cyrillic.

        Args:
            answer: Answer string (e.g., "A1Б2B5", "134").

        Returns:
            Normalized answer with Cyrillic letters.
        """
        latin_to_cyrillic = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")
        return answer.translate(latin_to_cyrillic)

    def _parse_answers_from_markdown(
        self, markdown: str, part: int
    ) -> dict[str, dict[str, str]]:
        """Parse answer keys from OCR markdown output.

        Args:
            markdown: OCR output in markdown format with tables.
            part: Part number (1 or 2) to determine variant mapping.

        Returns:
            Dictionary: {option: {question_label: answer}}
        """
        rows = self._parse_markdown_table(markdown)

        if len(rows) < 2:
            logger.warning("Not enough rows in markdown table", rows=len(rows))
            return {}

        variant_row_idx = 0
        for i, row in enumerate(rows):
            if any(cell.isdigit() for cell in row):
                variant_row_idx = i
                break

        variants: list[int] = []
        for cell in rows[variant_row_idx]:
            if cell.isdigit() and 1 <= int(cell) <= 100:
                variants.append(int(cell))

        if not variants:
            logger.warning("No variants found in answer sheet")
            return {}

        result: dict[str, dict[str, str]] = {}

        for row in rows[variant_row_idx + 1 :]:
            if not row:
                continue

            label = row[0].strip() if row else ""
            if not label or not re.match(r"^[A-ZА-Я]\d+$", label, re.IGNORECASE):
                continue

            label = self._normalize_label(label)

            for i, variant in enumerate(variants):
                if i + 1 >= len(row):
                    break

                answer = row[i + 1].strip()
                if answer:
                    option = str((variant - 1) % 10 + 1)
                    if option not in result:
                        result[option] = {}
                    result[option][label] = self._normalize_answer(answer)

        return result

    def _sort_answers(
        self, answers: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        """Sort answers by option number, then by question label.

        Args:
            answers: Dictionary of {option: {question_label: answer}}.

        Returns:
            Sorted dictionary with options in order, questions sorted within each option.
        """
        sorted_options = sorted(answers.keys(), key=lambda x: int(x))
        result: dict[str, dict[str, str]] = {}
        for option in sorted_options:
            sorted_labels = sorted(
                answers[option].keys(),
                key=lambda x: (x[0], int(x[1:])),
            )
            result[option] = {label: answers[option][label] for label in sorted_labels}
        return result

    def _extract_year_and_part(self, image_path: Path) -> tuple[int, int]:
        """Extract year and part number from image filename.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (year, part_number).

        Raises:
            ValueError: If filename doesn't match expected format.
        """
        match = re.match(r"(\d{4})_(\d+)", image_path.stem)
        if not match:
            raise ValueError(
                f"Invalid filename format: {image_path.name}. Expected format: YYYY_N.jpg"
            )
        return int(match.group(1)), int(match.group(2))

    def extract(
        self,
        subject: str,
    ) -> ExtractionResult:
        """Extract answers from all answer sheet images.

        Args:
            subject: Subject name (e.g., "biology", "chemistry").

        Returns:
            ExtractionResult with statistics.
        """
        try:
            self._validate_prerequisites()
        except APIError as e:
            return ExtractionResult.failure_result(errors=[str(e)])

        settings = get_settings()
        answers_dir = settings.paths.books_dir / subject / "answers"

        if not answers_dir.exists():
            raise DirectoryNotFoundError(answers_dir)

        image_files = sorted(
            [
                p
                for p in answers_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ],
            key=lambda p: p.name,
        )

        if not image_files:
            logger.warning("No answer images found", answers_dir=str(answers_dir))
            return ExtractionResult.success_result(
                processed=0, warnings=["No answer images found"]
            )

        results: dict[int, dict[str, dict[str, str]]] = {}
        year_parts: dict[int, set[int]] = {}

        output_base = (
            settings.paths.extraction_dir
            / settings.extraction.data_dir_name
            / settings.extraction.output_dir_name
        )

        processed_count = 0
        errors: list[str] = []

        for image_path in tqdm(image_files, desc=f"Extracting {subject} answers"):
            try:
                year, part = self._extract_year_and_part(image_path)

                markdown = self.ocr(image_path)
                answers = self._parse_answers_from_markdown(markdown, part)

                if year not in results:
                    results[year] = {}
                    year_parts[year] = set()

                year_parts[year].add(part)

                for label, option_answers in answers.items():
                    if label not in results[year]:
                        results[year][label] = {}
                    results[year][label].update(option_answers)

                if len(year_parts[year]) == 2:
                    year_output_dir = output_base / subject / str(year)
                    year_output_dir.mkdir(parents=True, exist_ok=True)

                    sorted_answers = self._sort_answers(results[year])
                    output_path = year_output_dir / "answers.json"
                    output_path.write_text(
                        json.dumps(sorted_answers, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    tqdm.write(f"Saved answers for {year}")
                    processed_count += 1

            except Exception as e:
                error_msg = f"Failed to process {image_path.name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return ExtractionResult.failure_result(
            errors=errors,
            processed=processed_count,
        ) if errors else ExtractionResult.success_result(
            processed=processed_count,
            metadata={"years_processed": len(results)},
        )
