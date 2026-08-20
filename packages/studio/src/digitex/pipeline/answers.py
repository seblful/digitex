"""Answer keys off the back of a book, read by the OpenRouter vision API.

One year's keys are printed across several sheets, and the API is charged per
call, which sets the two rules the whole module is arranged around: a year is
written only when *every* one of its sheets came back, and a year that already
has an ``answers.json`` is never asked for again.

Attribution is the other theme. Reading the file is the operator's problem and
the API failing is the service's, so only the call itself is wrapped in
:class:`APIError`; and a label the model got wrong fails its own sheet rather
than the run, because sorting the output assumes a shape the model was merely
asked for.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import structlog
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from digitex.domain.corpus import (
    PROCESSED,
    book_answers_dir,
    is_image,
    parse_answer_sheet_stem,
)
from digitex.domain.entities import QuestionKey, normalize_option_number
from digitex.pipeline.exceptions import (
    APIError,
    DirectoryNotFoundError,
    InvalidFilenameError,
)
from digitex.pipeline.outcome import AnswersReport

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()

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
        """*image_path* as the ``data:`` URL the vision endpoint takes."""
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
        """Read one answer sheet as ``{option: {label: answer}}``.

        Raises:
            APIError: If the call failed, or came back with no choice to read.
                Nothing else is attributed to the API.
        """
        # Outside the try: reading the file is not an API failure, and its own
        # error points the operator at the file rather than at their API key.
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

        # Checked rather than indexed: `choices[0]` on an empty list raises an
        # IndexError, which reads to an operator like a bad API key.
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
        # The grammar and the Cyrillic fold live on QuestionKey — the same rule
        # ``db.seed`` parses these labels back with.
        try:
            return str(QuestionKey.parse(label))
        except ValueError as e:
            raise ValueError(f"Invalid question label: {label!r}") from e

    def _normalize_answer(self, answer: str) -> str:
        return answer.translate(LATIN_TO_CYRILLIC)

    @staticmethod
    def _normalize_option(option: str) -> str:
        return str(normalize_option_number(int(option)))

    def _normalize_sheet(self, questions: dict[str, str]) -> dict[str, str]:
        """One option's answers, with both scripts settled the way storage wants."""
        return {
            self._normalize_label(label): self._normalize_answer(answer)
            for label, answer in questions.items()
        }

    def _sort_answers(
        self, answers: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        """Answers by option number, then by part letter and question number."""
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
        """The year *image_path*'s name says it belongs to.

        Raises:
            InvalidFilenameError: If the stem is not ``YYYY`` or ``YYYY_N``.
        """
        parsed = parse_answer_sheet_stem(image_path.stem)
        if parsed is None:
            raise InvalidFilenameError(image_path.name, "YYYY.jpg or YYYY_N.jpg")
        year, _ = parsed
        return year

    def _answers_path(self, subject: str, year: int) -> Path:
        """Where one year's answer key is written."""
        return self._output_dir / subject / str(year) / "answers.json"

    def extract(self, subject: str) -> AnswersReport:
        """Read every answer sheet of *subject* and write the years that came whole.

        Raises:
            DirectoryNotFoundError: If the subject has no processed answers
                folder to read.
        """
        answers_dir = book_answers_dir(self._books_dir, subject, PROCESSED)
        if not answers_dir.exists():
            raise DirectoryNotFoundError(answers_dir)

        sheets = sorted(
            (path for path in answers_dir.iterdir() if is_image(path)),
            key=lambda path: path.name,
        )

        if not sheets:
            logger.warning("No answer images found", answers_dir=str(answers_dir))
            return AnswersReport(note="No answer images found")

        read: dict[int, dict[str, dict[str, str]]] = {}
        errors: list[str] = []
        skipped: set[int] = set()
        failed: set[int] = set()

        for sheet in tqdm(sheets, desc=f"Extracting {subject} answers"):
            # Outside the try, so a name that will not parse is attributed to
            # the sheet rather than to whichever year it might have been.
            year: int | None = None
            try:
                year = self._extract_year(sheet)
                if self._answers_path(subject, year).exists():
                    # A written year is never re-read: those API calls are paid
                    # for, and a hand-corrected file must not be overwritten.
                    if year not in skipped:
                        skipped.add(year)
                        logger.info(
                            "Skipping year, answers.json exists",
                            year=year,
                            subject=subject,
                        )
                    continue
                for raw_option, questions in self.ocr(sheet).items():
                    option = self._normalize_option(raw_option)
                    read.setdefault(year, {}).setdefault(option, {}).update(
                        self._normalize_sheet(questions)
                    )
            except Exception as e:
                logger.error(
                    "Failed to process answer sheet",
                    image_path=str(sheet),
                    error=str(e),
                    exc_info=True,
                )
                errors.append(f"Failed to process {sheet.name}: {e}")
                if year is not None:
                    failed.add(year)

        written = len(skipped)
        for year, answers in read.items():
            # A year's answers span several sheets. Writing a partial file would
            # make the next run skip the year and never recover the rest.
            if year in failed:
                logger.warning(
                    "Not writing answers, some sheets failed",
                    subject=subject,
                    year=year,
                )
                continue
            path = self._answers_path(subject, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._sort_answers(answers), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1

        return AnswersReport(
            years=len(read),
            sheets=written,
            failures=tuple(errors),
        )
