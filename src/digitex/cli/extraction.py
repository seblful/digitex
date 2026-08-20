"""Extraction CLI commands.

Settings are resolved per command and handed to the collaborators the command
builds, so importing this module reads no files and configures no logging —
that happens in the Typer callback, once a command is actually running.
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer

from digitex.cli._shared import abort
from digitex.config import Settings, get_settings
from digitex.domain.corpus import (
    PROCESSED,
    book_pages_dir,
    is_image,
    natural_sort_key,
)
from digitex.imaging.ocr import TextExtractor
from digitex.logging import setup_logging
from digitex.ml.predictors import YOLO_SegmentationPredictor
from digitex.pipeline.answers import AnswersExtractor
from digitex.pipeline.audit.census import ImageCensus
from digitex.pipeline.audit.validator import AnswerValidator
from digitex.pipeline.base import ExtractionConfig
from digitex.pipeline.book import BookExtractor
from digitex.pipeline.exceptions import (
    APIError,
    DirectoryNotFoundError,
    ModelNotFoundError,
    ReviewAborted,
)
from digitex.pipeline.page import OCR_LANGUAGE, PageExtractor
from digitex.pipeline.recording import (
    Recording,
    RecordingPredictor,
    RecordingTextExtractor,
    as_predictor,
    as_text_extractor,
    directory_digests,
    recorded_output_dir,
    recording_path,
)
from digitex.pipeline.subject import SubjectExtractor

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
        image_format=settings.pipeline.extraction.image_format,
        question_max_width=settings.pipeline.extraction.question_max_width,
        question_max_height=settings.pipeline.extraction.question_max_height,
    )


def _subject_extractor(
    settings: Settings, subject: str = "", *, review: bool = False
) -> SubjectExtractor:
    on_review = None

    if review:
        # Imported here so a machine with no display can still run every
        # other command.
        from digitex.ui.page_review import TkPageReviewer

        output_dir = settings.paths.extraction_output_dir
        on_review = TkPageReviewer(
            subject=subject,
            census=ImageCensus(output_dir),
            validator=AnswerValidator(output_dir),
        )

    # The chain is built from the bottom: the config belongs to PageExtractor,
    # and the runners above it take the configured collaborator.
    page_extractor = PageExtractor(_extraction_config(settings), on_review=on_review)
    return SubjectExtractor(
        books_dir=settings.paths.books_dir,
        extraction_dir=settings.paths.extraction_output_dir,
        book_extractor=BookExtractor(page_extractor),
    )


def _answers_extractor(settings: Settings) -> AnswersExtractor:
    api_key = settings.pipeline.openrouter.api_key
    if not api_key:
        raise APIError(
            service="OpenRouter",
            message="API key not set. Set OPENROUTER_API_KEY environment variable.",
        )
    return AnswersExtractor(
        api_key=api_key,
        model=settings.pipeline.openrouter.model,
        base_url=settings.pipeline.openrouter.base_url,
        books_dir=settings.paths.books_dir,
        output_dir=settings.paths.extraction_output_dir,
    )


def _echo_errors(errors: list[str]) -> None:
    for error in errors:
        typer.echo(f"  - {error}", err=True)


SUBJECT_ARGUMENT = typer.Argument(help="Subject name (e.g., biology, chemistry)")


@app.command(name="rename-pages")
def rename_pages() -> None:
    """Renumber every scanned page to its canonical zero-padded name.

    A scanner names its export after the batch that made it (Химия.001.png) or
    after nothing in particular (10.jpg, which sorts ahead of 2.jpg). This
    renumbers each year's pages in reading order as 001, 002, … keeping the
    file's own format, and moves each page's processed twin with it so the two
    variants never disagree. Answer sheets keep their names — {year}_{n} is
    what says which year and sheet they are.

    Safely re-runnable: pages already correctly named are left alone.
    """
    from digitex.pipeline.preprocessing import rename_pages as run_rename

    settings = get_settings()

    try:
        result = run_rename(settings.paths.books_dir)
    except DirectoryNotFoundError as exc:
        raise abort(f"✗ {exc}") from None

    typer.echo(
        typer.style(
            f"✓ Renamed {result.renamed} page(s), {result.unchanged}"
            f" already named right",
            fg="green",
        )
    )
    if result.errors:
        typer.echo(
            typer.style(f"\n{result.failed} page(s) failed:", fg="red", bold=True)
        )
        _echo_errors(result.errors)


@app.command(name="preprocess-scans")
def preprocess_scans(
    force: Annotated[
        bool,
        typer.Option("--force", help="Reprocess scans that already have an output"),
    ] = False,
) -> None:
    """Correct the raw scans into the processed tree the rest of the pipeline reads.

    Renames every page to its canonical 001, 002, … first — the same pass
    rename-pages runs — so a scanner's own naming never reaches the processed
    tree. Then flattens gutter shadows out of the paper, burns its gray out to
    white, averages the scanner grain away and cuts off the scanner's white
    canvas, writing var/books/{subject}/raw/ to the matching path under
    var/books/{subject}/processed/. Answer sheets keep their {year}_{n} names
    and skip the shadow flatten so their printed shading survives; everything
    else applies to them too.

    Safely re-runnable: scans already processed are skipped unless --force.
    Note that --force can move edges, and annotations drawn on a processed page
    move with them.
    """
    from digitex.pipeline.preprocessing import preprocess_scans as run_preprocessing

    settings = get_settings()

    try:
        result = run_preprocessing(settings.paths.books_dir, force=force)
    except DirectoryNotFoundError as exc:
        raise abort(f"✗ {exc}") from None

    typer.echo(
        typer.style(
            f"✓ Renamed {result.renamed} page(s), processed {result.processed}"
            f" scan(s), skipped {result.skipped} already done"
            f" → {settings.paths.books_dir}",
            fg="green",
        )
    )
    if result.errors:
        typer.echo(
            typer.style(f"\n{result.failed} scan(s) failed:", fg="red", bold=True)
        )
        _echo_errors(result.errors)


@app.command(name="extract-questions")
def extract_questions(
    subject: Annotated[str, SUBJECT_ARGUMENT],
    review: Annotated[
        bool,
        typer.Option(
            "--review",
            help="Check every page in a window before its crops are saved",
        ),
    ] = False,
) -> None:
    """Extract question images from a specific subject.

    SUBJECT is the name of the subject folder in the books directory.

    With --review, each page opens in a window showing its detected polygons
    and the option/part/number every question would be saved as. Correct them
    with the mouse, then approve, skip the page, or abort the run. The window's
    second tab shows the subject's per-year counts and the answers.json check.
    """
    try:
        extractor = _subject_extractor(get_settings(), subject, review=review)
    except ModelNotFoundError as exc:
        raise abort(f"✗ {exc}") from None

    try:
        result = extractor.extract(subject=subject)
    except ReviewAborted as exc:
        # Approved pages keep their images and the year stays unfinished, so
        # re-running picks up where the reviewer left off.
        raise abort(f"✗ {exc}. Re-run to continue.") from None

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


@app.command(name="extract-answers")
def extract_answers(subject: Annotated[str, SUBJECT_ARGUMENT]) -> None:
    """Extract answer keys from answer sheet images using OpenRouter.

    Answer images should be placed in var/books/{subject}/answers/
    with filename format: YYYY_N.jpg (e.g., 2016_1.jpg, 2016_2.jpg)

    Results are saved to var/extraction/output/{subject}/{year}/answers.json
    """
    try:
        extractor = _answers_extractor(get_settings())
    except APIError as exc:
        raise abort(f"✗ {exc}") from None

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


@app.command(name="record-golden")
def record_golden(
    subject: Annotated[str, SUBJECT_ARGUMENT],
    year: Annotated[str, typer.Argument(help="Year folder to record, e.g. 2024")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing recording for this book"),
    ] = False,
) -> None:
    """Record one book's model and OCR answers as a replay fixture.

    Extracts YEAR of SUBJECT into a scratch folder under the data root, keeping
    every answer the segmentation model and OCR gave and the digest of every
    file written. `tests/differential` replays that recording to check a
    restructuring wrote exactly the same images — on a machine with no
    checkpoint, no GPU and no tesseract.

    Slow and rarely run: this is the one command that needs the real model. Its
    output stays under the data root and is never committed, so a checkout
    without it simply skips the differential suite.
    """
    settings = get_settings()
    data_root = settings.paths.data_root

    destination = recording_path(data_root, subject, year)
    if destination.exists() and not force:
        raise abort(f"✗ {destination} already exists. Pass --force to replace it.")

    pages_dir = book_pages_dir(settings.paths.books_dir, subject, PROCESSED) / year
    if not pages_dir.is_dir():
        raise abort(f"✗ No processed pages for {subject}/{year} at {pages_dir}")

    try:
        config = _extraction_config(settings)
    except ModelNotFoundError as exc:
        raise abort(f"✗ {exc}") from None

    recording = Recording(
        source=f"{subject}/{year}",
        image_format=config.image_format,
        question_max_width=config.question_max_width,
        question_max_height=config.question_max_height,
    )
    recording.pages = [
        path.name
        for path in sorted(
            (p for p in pages_dir.iterdir() if is_image(p)), key=natural_sort_key
        )
    ]
    if not recording.pages:
        raise abort(f"✗ No page images in {pages_dir}")

    # Started from empty, so the numbering starts at 1 and the digests below
    # describe a whole book rather than a book plus whatever was there before.
    output_dir = recorded_output_dir(data_root, subject, year)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    page_extractor = PageExtractor(
        config,
        predictor=as_predictor(
            RecordingPredictor(
                YOLO_SegmentationPredictor(str(config.model_path)), recording
            )
        ),
        text_extractor=as_text_extractor(
            RecordingTextExtractor(TextExtractor(language=OCR_LANGUAGE), recording)
        ),
    )
    result = BookExtractor(page_extractor).extract(pages_dir, output_dir)

    recording.outputs = directory_digests(output_dir)
    recording.write(destination)

    typer.echo(
        typer.style(
            f"✓ Recorded {subject}/{year}: {result.processed} pages,"
            f" {len(recording.outputs)} images → {destination}",
            fg="green",
        )
    )
    # A page that failed produced no image, so its answers are missing from the
    # recording too — a replay of it would refuse rather than diverge quietly.
    if result.failed:
        typer.echo(
            typer.style(f"\n{result.failed} page(s) failed:", fg="red", bold=True)
        )
        _echo_errors(result.errors)


if __name__ == "__main__":
    app()
