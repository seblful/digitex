"""Factory for creating extractor instances with proper configuration."""

from pathlib import Path

from digitex.config import get_settings
from digitex.extractors.answers_extractor import AnswersExtractor
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.manual_extractor import ManualExtractor
from digitex.extractors.page_extractor import PageExtractor
from digitex.extractors.tests_extractor import TestsExtractor


class ExtractorFactory:
    """Factory for creating configured extractor instances."""

    @staticmethod
    def _resolve_image_params(
        settings,
        image_format: str | None,
        question_max_width: int | None,
        question_max_height: int | None,
    ) -> tuple[str, int, int]:
        return (
            image_format or settings.extraction.image_format,
            question_max_width or settings.extraction.question_max_width,
            question_max_height or settings.extraction.question_max_height,
        )

    @staticmethod
    def create_page_extractor(
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
    ) -> PageExtractor:
        settings = get_settings()
        fmt, max_w, max_h = ExtractorFactory._resolve_image_params(
            settings, image_format, question_max_width, question_max_height
        )
        return PageExtractor(
            model_path=model_path or settings.paths.extraction_model_path,
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
        )

    @staticmethod
    def create_book_extractor(
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
    ) -> BookExtractor:
        settings = get_settings()
        fmt, max_w, max_h = ExtractorFactory._resolve_image_params(
            settings, image_format, question_max_width, question_max_height
        )
        return BookExtractor(
            model_path=model_path or settings.paths.extraction_model_path,
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
        )

    @staticmethod
    def create_tests_extractor(
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
        books_dir: Path | None = None,
        extraction_dir: Path | None = None,
    ) -> TestsExtractor:
        settings = get_settings()
        fmt, max_w, max_h = ExtractorFactory._resolve_image_params(
            settings, image_format, question_max_width, question_max_height
        )
        return TestsExtractor(
            model_path=model_path or settings.paths.extraction_model_path,
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
            books_dir=books_dir or settings.paths.books_dir,
            extraction_dir=extraction_dir or settings.paths.extraction_output_dir,
        )

    @staticmethod
    def create_manual_extractor(
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
        manual_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> ManualExtractor:
        settings = get_settings()
        fmt, max_w, max_h = ExtractorFactory._resolve_image_params(
            settings, image_format, question_max_width, question_max_height
        )
        return ManualExtractor(
            image_format=fmt,
            question_max_width=max_w,
            question_max_height=max_h,
            manual_dir=manual_dir or settings.paths.extraction_manual_dir,
            output_dir=output_dir or settings.paths.extraction_output_dir,
        )

    @staticmethod
    def create_answers_extractor(
        api_key: str | None = None,
        model: str | None = None,
    ) -> AnswersExtractor:
        settings = get_settings()
        resolved_key = api_key or settings.mistral.api_key
        if not resolved_key:
            from digitex.extractors.exceptions import APIError
            raise APIError(
                service="Mistral",
                message="API key not set. Set MISTRAL_API_KEY environment variable.",
            )
        return AnswersExtractor(
            api_key=resolved_key,
            model=model or settings.mistral.ocr_model,
            books_dir=settings.paths.books_dir,
            output_dir=settings.paths.extraction_output_dir,
        )
