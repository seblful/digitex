"""Answers extractor using OpenRouter vision API with structured outputs."""

import base64
import json
from pathlib import Path

import structlog
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from digitex.domain.corpus import is_image, parse_answer_sheet_stem
from digitex.domain.entities import normalize_option_number
from digitex.pipeline.base import ExtractionResult
from digitex.pipeline.exceptions import APIError, DirectoryNotFoundError

logger = structlog.get_logger()

CYRILLIC_TO_LATIN = str.maketrans("АВЕС", "ABEC")  # noqa: RUF001
LATIN_TO_CYRILLIC = str.maketrans("ABCEHKMOPTXYF", "АВСЕНКМОРТХУГ")


class ExamExtraction(BaseModel):
    """Schema for multiple exam options.

    Example: {"1": {"A1": "2", "B1": "ВЕРНАДСКИЙ"}, "2": {...}}
    """

    options: dict[str, dict[str, str]]


OCR_SYSTEM_PROMPT = (
    "You are an OCR assistant. "
    "Extract the answer table from exam answer sheet images.\n\n"
    "Rules:\n"
    "1. Question labels MUST use Latin letters only — A1, A2, B1, B2 (NOT Cyrillic)\n"
    "2. Answers MUST use Cyrillic letters where applicable (NOT Latin A, B)\n"
    "3. Digits are always the same in both scripts"
)

OCR_USER_PROMPT = "Extract the answer table from this exam answer sheet image."


class AnswersExtractor:
    """Extracts answer keys from answer sheet images via OpenRouter vision API."""

    def __init__(
        self,
        api_key: str,
        books_dir: Path,
        output_dir: Path,
        model: str,
        base_url: str,
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
        # Reading the file is not an API failure — let its own error through so
        # the operator is pointed at the file rather than at their API key.
        data_url = self.encode_image(image_path)
        try:
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
        except Exception as e:
            raise APIError(
                service="OpenRouter",
                message=f"OCR failed for {image_path.name}: {e!s}",
            ) from e

        if not completion.choices:
            raise APIError(
                service="OpenRouter",
                message=f"OCR returned no choices for {image_path.name}",
            )
        extraction = completion.choices[0].message.parsed
        return extraction.options if extraction else {}

    def _normalize_label(self, label: str) -> str:
        """Normalize a question label to ``{A|B}{digits}``.

        The vision model is told to use Latin A/B plus a number, but nothing
        makes it comply. Rejecting a bad shape here puts the failure inside the
        per-sheet handler, which marks the year failed and leaves its
        ``answers.json`` unwritten — rather than crashing the whole run later,
        while sorting keys for output.

        Raises:
            ValueError: If the label is not a part letter followed by digits.
        """
        normalized = label.strip().upper().translate(CYRILLIC_TO_LATIN)
        if (
            len(normalized) < 2
            or normalized[0] not in ("A", "B")
            or not normalized[1:].isdigit()
        ):
            raise ValueError(f"Invalid question label: {label!r}")
        return normalized

    def _normalize_answer(self, answer: str) -> str:
        return answer.translate(LATIN_TO_CYRILLIC)

    @staticmethod
    def _normalize_option(option: str) -> str:
        return str(normalize_option_number(int(option)))

    def _sort_answers(
        self, answers: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        # ``key=int`` widens the inferred element type via ``int``'s union
        # signature; the lambda pins it to ``str``.
        sorted_options = sorted(answers.keys(), key=lambda k: int(k))  # noqa: PLW0108
        result: dict[str, dict[str, str]] = {}
        for option in sorted_options:
            sorted_labels = sorted(
                answers[option].keys(),
                key=lambda x: (x[0], int(x[1:])),
            )
            result[option] = {label: answers[option][label] for label in sorted_labels}
        return result

    def _extract_year(self, image_path: Path) -> int:
        parsed = parse_answer_sheet_stem(image_path.stem)
        if parsed is None:
            raise ValueError(
                f"Invalid filename format: {image_path.name}. "
                "Expected format: YYYY_N.jpg"
            )
        year, _ = parsed
        return year

    def extract(self, subject: str) -> ExtractionResult:
        answers_dir = self._books_dir / subject / "answers"
        if not answers_dir.exists():
            raise DirectoryNotFoundError(answers_dir)

        image_files = sorted(
            [p for p in answers_dir.iterdir() if is_image(p)],
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
        failed_years: set[int] = set()

        for image_path in tqdm(image_files, desc=f"Extracting {subject} answers"):
            year: int | None = None
            try:
                year = self._extract_year(image_path)
                year_dir = self._output_dir / subject / str(year)
                if (year_dir / "answers.json").exists():
                    if year not in skipped_years:
                        skipped_years.add(year)
                        logger.info(
                            "Skipping year, answers.json exists",
                            year=year,
                            subject=subject,
                        )
                    continue
                parsed = self.ocr(image_path)
                for option, questions in parsed.items():
                    norm_option = self._normalize_option(option)
                    normalized = {
                        self._normalize_label(k): self._normalize_answer(v)
                        for k, v in questions.items()
                    }
                    years_data.setdefault(year, {}).setdefault(norm_option, {}).update(
                        normalized
                    )
            except Exception as e:
                msg = f"Failed to process {image_path.name}: {e}"
                logger.error(
                    "Failed to process answer sheet",
                    image_path=str(image_path),
                    error=str(e),
                )
                errors.append(msg)
                if year is not None:
                    failed_years.add(year)

        processed_count = len(skipped_years)
        for year, answers in years_data.items():
            # A year's answers span several sheets. Writing a partial file would
            # make the next run skip the year and never recover the rest.
            if year in failed_years:
                logger.warning(
                    "Not writing answers, some sheets failed",
                    subject=subject,
                    year=year,
                )
                continue
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
