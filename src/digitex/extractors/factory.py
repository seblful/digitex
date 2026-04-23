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
    def create_page_extractor(
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
    ) -> PageExtractor:
        """Create a configured PageExtractor.

        Args:
            model_path: Optional custom model path.
            image_format: Optional custom image format.
            question_max_width: Optional custom max width.
            question_max_height: Optional custom max height.

        Returns:
            Configured PageExtractor instance.
        """
        settings = get_settings()

        return PageExtractor(
            model_path=model_path or settings.paths.extraction_model_path,
            image_format=image_format or settings.extraction.image_format,
            question_max_width=question_max_width
            or settings.extraction.question_max_width,
            question_max_height=question_max_height
            or settings.extraction.question_max_height,
        )

    @staticmethod
    def create_book_extractor(
        model_path: Path | None = None,
        image_format: str | None = None,
        question_max_width: int | None = None,
        question_max_height: int | None = None,
    ) -> BookExtractor:
        """Create a configured BookExtractor.

        Args:
            model_path: Optional custom model path.
            image_format: Optional custom image format.
            question_max_width: Optional custom max width.
            question_max_height: Optional custom max height.

        Returns:
            Configured BookExtractor instance.
        """
        settings = get_settings()

        return BookExtractor(
            model_path=model_path or settings.paths.extraction_model_path,
            image_format=image_format or settings.extraction.image_format,
            question_max_width=question_max_width
            or settings.extraction.question_max_width,
            question_max_height=question_max_height
            or settings.extraction.question_max_height,
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
        """Create a configured TestsExtractor.

        Args:
            model_path: Optional custom model path.
            image_format: Optional custom image format.
            question_max_width: Optional custom max width.
            question_max_height: Optional custom max height.
            books_dir: Optional custom books directory.
            extraction_dir: Optional custom extraction output directory.

        Returns:
            Configured TestsExtractor instance.
        """
        settings = get_settings()

        return TestsExtractor(
            model_path=model_path or settings.paths.extraction_model_path,
            image_format=image_format or settings.extraction.image_format,
            question_max_width=question_max_width
            or settings.extraction.question_max_width,
            question_max_height=question_max_height
            or settings.extraction.question_max_height,
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
        """Create a configured ManualExtractor.

        Args:
            image_format: Optional custom image format.
            question_max_width: Optional custom max width.
            question_max_height: Optional custom max height.
            manual_dir: Optional custom manual images directory.
            output_dir: Optional custom output directory.

        Returns:
            Configured ManualExtractor instance.
        """
        settings = get_settings()

        return ManualExtractor(
            image_format=image_format or settings.extraction.image_format,
            question_max_width=question_max_width
            or settings.extraction.question_max_width,
            question_max_height=question_max_height
            or settings.extraction.question_max_height,
            manual_dir=manual_dir or settings.paths.extraction_manual_dir,
            output_dir=output_dir or settings.paths.extraction_output_dir,
        )

    @staticmethod
    def create_answers_extractor(
        api_key: str | None = None,
        model: str | None = None,
    ) -> AnswersExtractor:
        """Create a configured AnswersExtractor.

        Args:
            api_key: Optional custom API key.
            model: Optional custom OCR model name.

        Returns:
            Configured AnswersExtractor instance.
        """
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
