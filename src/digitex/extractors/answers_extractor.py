"""Answers extractor using Mistral OCR API."""

import base64
import json
import os
import re
from pathlib import Path

from mistralai.client import Mistral
from tqdm import tqdm


class AnswersExtractor:
    """Extracts answer keys from answer sheet images via Mistral OCR."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            api_key: Mistral API key. If None, uses MISTRAL_API_KEY from environment or settings.
            model: OCR model name. If None, uses settings or default.
        """
        if api_key is None:
            try:
                from digitex.config import get_settings
                settings = get_settings()
                api_key = settings.mistral.api_key or os.environ.get("MISTRAL_API_KEY")
            except Exception:
                api_key = os.environ.get("MISTRAL_API_KEY")
        
        if model is None:
            try:
                from digitex.config import get_settings
                settings = get_settings()
                model = settings.mistral.ocr_model
            except Exception:
                model = "mistral-ocr-latest"
        
        self._api_key = api_key or ""
        self._model = model
        self._client: Mistral | None = None

    def _get_client(self) -> Mistral:
        """Return the Mistral client, creating it on first use."""
        if self._api_key is None or self._api_key == "":
            raise ValueError(
                "MISTRAL_API_KEY must be set in environment or passed as api_key"
            )
        if self._client is None:
            self._client = Mistral(api_key=self._api_key)
        return self._client

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
        """
        client = self._get_client()
        data_url = self.encode_image(image_path)

        res = client.ocr.process(
            model=self._model,
            document={
                "image_url": {"url": data_url},
                "type": "image_url",
            },
        )
        if res.pages:
            return res.pages[0].markdown or ""
        return ""

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
                f"Invalid filename format: {image_path.name}. "
                f"Expected format: YYYY_N.jpg"
            )
        return int(match.group(1)), int(match.group(2))

    def extract(
        self,
        subject: str,
    ) -> dict[int, dict[str, dict[str, str]]]:
        """Extract answers from all answer sheet images.

        Args:
            subject: Subject name (e.g., "biology", "chemistry").

        Returns:
            Dictionary: {year: {question_label: {option: answer}}}
        """

        settings = self._get_settings()
        answers_dir = settings.paths.books_dir / subject / "answers"

        if not answers_dir.exists():
            raise FileNotFoundError(f"Answers directory not found: {answers_dir}")

        image_files = sorted(
            [
                p
                for p in answers_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ],
            key=lambda p: p.name,
        )

        results: dict[int, dict[str, dict[str, str]]] = {}
        year_parts: dict[int, set[int]] = {}

        settings = self._get_settings()
        output_base = (
            settings.paths.extraction_dir
            / settings.extraction.data_dir_name
            / settings.extraction.output_dir_name
        )

        for image_path in tqdm(image_files, desc=f"Extracting {subject} answers"):
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
                
                a_count = sum(1 for k in sorted_answers["1"].keys() if k.startswith("A"))
                b_count = sum(1 for k in sorted_answers["1"].keys() if k.startswith("B"))
                
                all_options_same = True
                first_option_questions = set(sorted_answers["1"].keys())
                for opt in sorted_answers:
                    if set(sorted_answers[opt].keys()) != first_option_questions:
                        all_options_same = False
                        break
                
                status = "✓" if all_options_same else "✗"
                tqdm.write(
                    f"Saved answers for {year}: {a_count} A-part, {b_count} B-part [{status}]"
                )

        return results

    def _get_settings(self):
        """Get application settings."""
        from digitex.config import get_settings

        return get_settings()
