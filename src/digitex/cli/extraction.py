"""Extraction CLI commands.

Settings are resolved per command and handed to the collaborators the command
builds, so importing this module reads no files and configures no logging —
that happens in the Typer callback, once a command is actually running.
"""

from pathlib import Path
from typing import Annotated, Final

import typer

from digitex.config import Settings, get_settings
from digitex.extractors.answers_extractor import AnswersExtractor
from digitex.extractors.base import ExtractionConfig
from digitex.extractors.exceptions import APIError, ModelNotFoundError
from digitex.extractors.manual_extractor import ManualExtractor
from digitex.extractors.tests_extractor import TestsExtractor
from digitex.extractors.utils import renumber_directory_tree
from digitex.logging import setup_logging
from digitex.services.answer_validator import (
    AnswerValidator,
    PartBCoverage,
    ValidationReport,
    YearReport,
)
from digitex.services.image_census import ImageCensus, SubjectCensus, YearCensus

app = typer.Typer(help="Extraction commands for processing test books.")


@app.callback()
def configure() -> None:
    """Set up logging before any command runs."""
    setup_logging(get_settings())


def _require_model(path: Path) -> Path:
    """Fail on a missing model file now, not when the lazy predictor loads."""
    if not path.exists():
        raise ModelNotFoundError(path)
    return path


def _extraction_config(settings: Settings) -> ExtractionConfig:
    return ExtractionConfig(
        model_path=_require_model(settings.paths.extraction_model_path),
        image_format=settings.extraction.image_format,
        question_max_width=settings.extraction.question_max_width,
        question_max_height=settings.extraction.question_max_height,
    )


def _tests_extractor(settings: Settings) -> TestsExtractor:
    return TestsExtractor(
        config=_extraction_config(settings),
        books_dir=settings.paths.books_dir,
        extraction_dir=settings.paths.extraction_output_dir,
    )


def _manual_extractor(settings: Settings, manual_dir: Path) -> ManualExtractor:
    return ManualExtractor(
        image_format=settings.extraction.image_format,
        question_max_width=settings.extraction.question_max_width,
        question_max_height=settings.extraction.question_max_height,
        manual_dir=manual_dir,
        output_dir=settings.paths.extraction_output_dir,
    )


def _answers_extractor(settings: Settings) -> AnswersExtractor:
    api_key = settings.openrouter.api_key
    if not api_key:
        raise APIError(
            service="OpenRouter",
            message="API key not set. Set OPENROUTER_API_KEY environment variable.",
        )
    return AnswersExtractor(
        api_key=api_key,
        model=settings.openrouter.model,
        base_url=settings.openrouter.base_url,
        books_dir=settings.paths.books_dir,
        output_dir=settings.paths.extraction_output_dir,
    )


def _abort(message: str) -> typer.Exit:
    """Render *message* on stderr and return the exit to raise."""
    typer.echo(typer.style(message, fg="red", bold=True), err=True)
    return typer.Exit(code=1)


def _echo_errors(errors: list[str]) -> None:
    for error in errors:
        typer.echo(f"  - {error}", err=True)


SUBJECT_ARGUMENT = typer.Argument(help="Subject name (e.g., biology, chemistry)")


@app.command(name="extract-questions")
def extract_questions(subject: Annotated[str, SUBJECT_ARGUMENT]) -> None:
    """Extract question images from a specific subject.

    SUBJECT is the name of the subject folder in the books directory.
    """
    try:
        extractor = _tests_extractor(get_settings())
    except ModelNotFoundError as exc:
        raise _abort(f"✗ {exc}") from None

    result = extractor.extract(subject=subject)

    if not result.success:
        typer.echo(typer.style("✗ Extraction failed:", fg="red", bold=True), err=True)
        _echo_errors(result.errors)
        raise typer.Exit(code=1)

    typer.echo(
        typer.style(
            f"✓ Extraction completed: {result.processed} processed,"
            f" {result.skipped} skipped (subject: {subject})",
            fg="green",
        )
    )
    if result.warnings:
        typer.echo(typer.style("\nWarnings:", fg="yellow"))
        for warning in result.warnings:
            typer.echo(f"  - {warning}")
    # A run can succeed with per-page failures; those pages produced no image,
    # so saying so is the difference between "done" and "done, minus four".
    if result.failed:
        typer.echo(
            typer.style(f"\n{result.failed} page(s) failed:", fg="red", bold=True)
        )
        _echo_errors(result.errors)


def _render_year_census(year: YearCensus) -> None:
    label = f"  {year.year}: {year.options} options"
    if year.missing_options:
        label = typer.style(label, fg="red", bold=True)
    elif year.is_complete:
        label = typer.style(label, fg="green")
    typer.echo(label)

    for part in year.parts:
        line = f"    {part.option}/{part.part}: {part.images} images"
        if part.off_mode:
            line = typer.style(line, fg="red", bold=True)
        typer.echo(line)


def _render_census(census: SubjectCensus) -> None:
    for year in census.years:
        _render_year_census(year)
    typer.echo(
        f"\nTotal: {census.images} images in {census.folders} folders"
        f" (subject: {census.subject})"
    )


@app.command(name="count-questions")
def count_questions(subject: Annotated[str, SUBJECT_ARGUMENT]) -> None:
    """Count images in a specific subject's extraction output."""
    census_taker = ImageCensus(get_settings().paths.extraction_output_dir)
    try:
        census = census_taker.take(subject)
    except FileNotFoundError:
        raise _abort(f"Error: Subject '{subject}' not found") from None

    if census.is_empty:
        typer.echo(f"No images found for subject '{subject}'")
        return

    _render_census(census)


@app.command(name="renumber-questions")
def renumber_questions(
    subject: Annotated[str, SUBJECT_ARGUMENT],
    dry_run: Annotated[
        bool, typer.Option(help="Preview changes without renaming")
    ] = True,
) -> None:
    """Renumber images in a specific subject's extraction output to fill gaps."""
    folder = get_settings().paths.extraction_output_dir / subject

    if not folder.is_dir():
        raise _abort(f"Error: Subject '{subject}' not found")

    total = renumber_directory_tree(folder, dry_run=dry_run)

    if dry_run and total:
        typer.echo(f"\n{total} files would be renamed")
    elif total:
        typer.echo(f"Renamed {total} files successfully")
    else:
        typer.echo("All images are already sequential")


@app.command(name="add-questions-manually")
def add_questions_manually(
    subject: Annotated[str, SUBJECT_ARGUMENT],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without applying")
    ] = False,
) -> None:
    """Add manually cropped question images for a specific subject.

    Manual images should be placed in extraction/data/manual/{subject}/
    with filename format: YYYY_OPTION_PART_QUESTION.png
    Example: biology/2016_3_A_20.png
    """
    settings = get_settings()
    manual_dir = settings.paths.extraction_manual_dir / subject

    if not manual_dir.exists():
        raise _abort(f"Error: Manual directory '{subject}' not found")

    result = _manual_extractor(settings, manual_dir).extract(dry_run=dry_run)

    if not result.success:
        typer.echo(
            typer.style("✗ Manual extraction failed:", fg="red", bold=True), err=True
        )
        _echo_errors(result.errors)
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo(
            typer.style(
                f"[DRY RUN] Would process {result.processed} files", fg="yellow"
            )
        )
    else:
        typer.echo(
            typer.style(f"✓ Processed {result.processed} manual images", fg="green")
        )
    if result.failed:
        typer.echo(typer.style(f"  Failed: {result.failed}", fg="red", bold=True))


@app.command(name="extract-answers")
def extract_answers(subject: Annotated[str, SUBJECT_ARGUMENT]) -> None:
    """Extract answer keys from answer sheet images using OpenRouter.

    Answer images should be placed in books/{subject}/answers/
    with filename format: YYYY_N.jpg (e.g., 2016_1.jpg, 2016_2.jpg)

    Results are saved to extraction/data/output/{subject}/{year}/answers.json
    """
    try:
        extractor = _answers_extractor(get_settings())
    except APIError as exc:
        raise _abort(f"✗ {exc}") from None

    result = extractor.extract(subject=subject)

    if not result.success:
        typer.echo(
            typer.style("✗ Answer extraction failed:", fg="red", bold=True), err=True
        )
        _echo_errors(result.errors)
        raise typer.Exit(code=1)

    typer.echo(
        typer.style(
            f"✓ Extracted answers for"
            f" {result.metadata.get('years_processed', 0)} years",
            fg="green",
        )
    )
    if result.errors:
        typer.echo(typer.style("\nErrors:", fg="red"), err=True)
        _echo_errors(result.errors)


# One row per coverage state, keyed by the Literal itself: a new state fails to
# type-check here instead of raising a KeyError at render time.
_PART_B_COVERAGE: Final[dict[PartBCoverage, tuple[str, str, bool]]] = {
    "none": ("red", "NO option has Б", True),
    "partial": ("yellow", "{covered}/{total} options have Б", False),
    "all": ("green", "all options have Б", False),
}


def _render_year_report(year: YearReport) -> None:
    """Emit the colored year-level rendering of a validation outcome."""
    if not year.answers_file_present:
        typer.echo(f"\n{year.year}: ✗ answers.json NOT FOUND")
        return

    if not year.answers_file_valid:
        typer.echo(f"\n{year.year}: ✗ answers.json IS UNREADABLE (bad JSON or shape)")
        return

    if year.has_mismatch:
        status = "❌ MISMATCH"
    elif year.options_differ:
        status = "❌ OPTIONS DIFFER"
    else:
        status = "✅ OK"

    typer.echo(f"\n{year.year}: {status}")
    typer.echo(f"  A-part: {year.a_count}, B-part: {year.b_count}")
    typer.echo(f"  Questions in images: {year.image_question_count}")
    typer.echo(f"  Questions in answers.json: {year.answer_question_count}")

    if year.options_differ:
        typer.echo(
            "  Options with different questions:"
            f" {year.options_with_differing_questions}"
        )
    if year.missing_in_answers:
        typer.echo(f"  Missing in answers.json: {year.missing_in_answers}")
    if year.missing_in_images:
        typer.echo(f"  Missing in images: {year.missing_in_images}")

    color, label, bold = _PART_B_COVERAGE[year.part_b_coverage]
    styled = typer.style(
        label.format(covered=year.options_with_b, total=year.total_options),
        fg=color,
        bold=bold,
    )
    typer.echo(f"  Part B 'Б' check: {styled}")


def _render_validation_report(report: ValidationReport) -> None:
    typer.echo("=" * 60)
    typer.echo(f"CHECKING ANSWERS FOR: {report.subject}")
    typer.echo("=" * 60)

    for year in report.years:
        _render_year_report(year)

    typer.echo("\n" + "=" * 60)
    if report.total_issues == 0:
        typer.echo("RESULT: All years match ✅")
    else:
        typer.echo(f"RESULT: {report.total_issues} issue(s) found ❌")
    typer.echo("=" * 60)


@app.command(name="check-answers")
def check_answers(subject: Annotated[str, SUBJECT_ARGUMENT]) -> None:
    """Check that answers.json files correspond to extracted question images."""
    validator = AnswerValidator(get_settings().paths.extraction_output_dir)
    try:
        report = validator.validate(subject)
    except FileNotFoundError as exc:
        raise _abort(f"Error: {exc} does not exist") from None

    _render_validation_report(report)


if __name__ == "__main__":
    app()
