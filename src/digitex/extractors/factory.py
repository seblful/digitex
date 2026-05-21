"""Factory for creating extractor instances with proper configuration."""

from pathlib import Path

from digitex.config import Settings
from digitex.extractors.answers_extractor import AnswersExtractor
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.conflict_resolution import ConflictResolver
from digitex.extractors.exceptions import APIError, ModelNotFoundError
from digitex.extractors.manual_extractor import ManualExtractor
from digitex.extractors.page_extractor import PageExtractor
from digitex.extractors.tests_extractor import TestsExtractor


class ExtractorFactory:
    """Factory for creating configured extractor instances.

    Validates that required files and keys exist before constructing, so
    failures surface at construction time rather than mid-extraction.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _resolve_image_params(
        self,
        image_format: str | None,
        question_max_width: int | None,
        question_max_height: int | None,
    ) -> tuple[str, int, int]:
        return (
            image_format or self._settings.extraction.image_format,
            question_max_width or self._settings.extraction.question_max_width,
            question_max_height or self._settings.extraction.question_max_height,
        )

    @staticmethod
    def _require_model(path: Path) -> Path:
        if not path.exists():
            raise ModelNotFoundError(path)
        return path

    def create_page_extractor(
        self,
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
        on_conflict: ConflictResolver | None = None,
    ) -> PageExtractor:
        fmt, max_w, max_h = self._resolve_image_params(
            image_format, question_max_width, question_max_height
        )
        resolved = self._require_model(
            model_path or self._settings.paths.extraction_model_path
        )
        return PageExtractor(
            model_path=resolved,
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
            on_conflict=on_conflict,
        )

    def create_book_extractor(
        self,
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
        on_conflict: ConflictResolver | None = None,
    ) -> BookExtractor:
        fmt, max_w, max_h = self._resolve_image_params(
            image_format, question_max_width, question_max_height
        )
        resolved = self._require_model(
            model_path or self._settings.paths.extraction_model_path
        )
        page_extractor = PageExtractor(
            model_path=resolved,
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
            on_conflict=on_conflict,
        )
        return BookExtractor(
            model_path=resolved,
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
            page_extractor=page_extractor,
        )

    def create_tests_extractor(
        self,
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
        books_dir: Path | None = None,
        extraction_dir: Path | None = None,
    ) -> TestsExtractor:
        fmt, max_w, max_h = self._resolve_image_params(
            image_format, question_max_width, question_max_height
        )
        resolved = self._require_model(
            model_path or self._settings.paths.extraction_model_path
        )
        return TestsExtractor(
            model_path=resolved,
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
            books_dir=books_dir or self._settings.paths.books_dir,
            extraction_dir=extraction_dir or self._settings.paths.extraction_output_dir,
        )

    def create_manual_extractor(
        self,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
        manual_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> ManualExtractor:
        fmt, max_w, max_h = self._resolve_image_params(
            image_format, question_max_width, question_max_height
        )
        return ManualExtractor(
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
            manual_dir=manual_dir or self._settings.paths.extraction_manual_dir,
            output_dir=output_dir or self._settings.paths.extraction_output_dir,
        )

    def create_answers_extractor(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> AnswersExtractor:
        resolved_key = api_key or self._settings.openrouter.api_key
        if not resolved_key:
            raise APIError(
                service="OpenRouter",
                message="API key not set. Set OPENROUTER_API_KEY environment variable.",
            )
        return AnswersExtractor(
            api_key=resolved_key,
            model=model or self._settings.openrouter.model,
            base_url=self._settings.openrouter.base_url,
            books_dir=self._settings.paths.books_dir,
            output_dir=self._settings.paths.extraction_output_dir,
        )
