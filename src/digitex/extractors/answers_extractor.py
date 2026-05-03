"""Answers extractor using OpenRouter vision API with structured outputs."""

import base64
import json
import re
from pathlib import Path
from typing import Dict

import structlog
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from digitex.extractors.base import BaseExtractor, ExtractionResult
from digitex.extractors.exceptions import APIError, DirectoryNotFoundError

logger = structlog.get_logger()

CYRILLIC_TO_LATIN = str.maketrans("АВЕС", "ABEC")
LATIN_TO_CYRILLIC = str.maketrans("ABCEHKMOPTXYF", "АВСЕНКМОРТХУГ")


class ExamExtraction(BaseModel):
    """Schema for multiple exam options.

    Example: {"1": {"A1": "2", "B1": "ВЕРНАДСКИЙ"}, "2": {...}}
    """

    options: Dict[str, Dict[str, str]]


OCR_SYSTEM_PROMPT = """You are an OCR assistant. Extract the answer table from exam answer sheet images.

Rules:
1. Question labels MUST use Latin letters only — A1, A2, B1, B2 (NOT Cyrillic А, В)
2. Answers MUST use Cyrillic letters where applicable — А1Б2В5 (NOT Latin A, B, B)
3. Digits are always the same in both scripts"""

OCR_USER_PROMPT = "Extract the answer table from this exam answer sheet image."


class AnswersExtractor(BaseExtractor):
    """Extracts answer keys from answer sheet images via OpenRouter vision API."""

    def __init__(
        self,
        api_key: str,
        books_dir: Path,
        output_dir: Path,
        model: str = "moonshotai/kimi-k2.6",
        base_url: str = "https://openrouter.ai/api/v1",
        client: OpenAI | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._books_dir = books_dir
        self._output_dir = output_dir
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _validate_prerequisites(self) -> None:
        if not self._books_dir.exists():
            raise DirectoryNotFoundError(self._books_dir)

    def encode_image(self, image_path: Path) -> str:
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        suffix = image_path.suffix.lower()
        media_type = (
            "image/jpeg"
            if suffix in (".jpg", ".jpeg")
            else f"image/{suffix[1:] or 'png'}"
        )
        return f"data:{media_type};base64,{b64}"

    def ocr(self, image_path: Path) -> dict[str, dict[str, str]]:
        try:
            data_url = self.encode_image(image_path)
            completion = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": OCR_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
                response_format=ExamExtraction,
            )
            extraction = completion.choices[0].message.parsed
            if extraction:
                return extraction.options
            return {}
        except Exception as e:
            raise APIError(
                service="OpenRouter",
                message=f"OCR failed: {str(e)}",
                context={"image_path": str(image_path)},
            ) from e

    def _normalize_label(self, label: str) -> str:
        return label.upper().translate(CYRILLIC_TO_LATIN)

    def _normalize_answer(self, answer: str) -> str:
        return answer.translate(LATIN_TO_CYRILLIC)

    @staticmethod
    def _normalize_option(option: str) -> str:
        return str((int(option) - 1) % 10 + 1)

    def _sort_answers(
        self, answers: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
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
        match = re.match(r"(\d{4})_(\d+)", image_path.stem)
        if not match:
            raise ValueError(
                f"Invalid filename format: {image_path.name}. Expected format: YYYY_N.jpg"
            )
        return int(match.group(1)), int(match.group(2))

    def extract(self, subject: str) -> ExtractionResult:
        answers_dir = self._books_dir / subject / "answers"
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

        years_data: dict[int, dict[str, dict[str, str]]] = {}
        errors: list[str] = []
        skipped_years: set[int] = set()

        for image_path in tqdm(image_files, desc=f"Extracting {subject} answers"):
            try:
                year, _ = self._extract_year_and_part(image_path)
                year_dir = self._output_dir / subject / str(year)
                if (year_dir / "answers.json").exists():
                    if year not in skipped_years:
                        skipped_years.add(year)
                        logger.info("Skipping year, answers.json exists", year=year, subject=subject)
                    continue
                parsed = self.ocr(image_path)
                for option, questions in parsed.items():
                    norm_option = self._normalize_option(option)
                    normalized = {
                        self._normalize_label(k): self._normalize_answer(v)
                        for k, v in questions.items()
                    }
                    years_data.setdefault(year, {}).setdefault(
                        norm_option, {}
                    ).update(normalized)
            except Exception as e:
                msg = f"Failed to process {image_path.name}: {e}"
                logger.error(msg)
                errors.append(msg)

        processed_count = len(skipped_years)
        for year, answers in years_data.items():
            year_dir = self._output_dir / subject / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            (year_dir / "answers.json").write_text(
                json.dumps(self._sort_answers(answers), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            processed_count += 1

        if errors:
            return ExtractionResult.failure_result(
                errors=errors,
                processed=processed_count,
            )
        return ExtractionResult.success_result(
            processed=processed_count,
            metadata={"years_processed": len(years_data)},
        )
