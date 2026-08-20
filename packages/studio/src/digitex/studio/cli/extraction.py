"""Extraction CLI commands.

Settings are resolved per command and handed to the collaborators the command
builds, so importing this module reads no files and configures no logging — that
happens in the Typer callback, once a command is actually running.
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer

from digitex.config import Settings, get_settings
from digitex.console import abort
from digitex.domain.corpus import (
    PROCESSED,
    book_pages_dir,
    is_image,
    natural_sort_key,
)
from digitex.imaging.ocr import OCR_LANGUAGE, TextExtractor
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
from digitex.pipeline.outcome import SubjectRefused, messages
from digitex.pipeline.page import PageExtractor
from digitex.pipeline.recording import (
    Recording,
    RecordingPredictor,
    RecordingTextExtractor,
    directory_digests,
    recorded_output_dir,
    recording_path,
)
from digitex.pipeline.subject import SubjectExtractor

app = typer.Typer(help="Extraction commands for processing test books.")

SUBJECT_ARGUMENT = typer.Argument(help="Subject name (e.g., biology, chemistry)")


@app.callback()
def configure() -> None:
    """Set up logging before any command runs."""
    setup_logging(get_settings())


# ---------------------------------------------------------------------------
# Reporting
#
# Three colours, three meanings: green finished, yellow happened and is worth
# knowing, red produced nothing. Every command below reports through these, so a
# new one cannot invent a fourth convention.
# ---------------------------------------------------------------------------


def _ok(message: str) -> None:
    """Report a finished command."""
    typer.echo(typer.style(message, fg="green"))


def _note(message: str) -> None:
    """Report something the operator should see that failed nothing."""
    typer.echo(typer.style(message, fg="yellow"))


def _echo_errors(errors: list[str]) -> None:
    for error in errors:
        typer.echo(f"  - {error}", err=True)


def _report_failures(count: int, what: str, lines: list[str]) -> None:
    """Say how many of *what* produced nothing, and name each one.

    A run can finish with per-item failures, and one that swallows them reads as
    "done" when it means "done, minus four". The count is passed rather than
    taken from *lines* because a report may summarise more failures than it
    lists.
    """
    if not lines:
        return
    typer.echo(typer.style(f"\n{count} {what} failed:", fg="red", bold=True))
    _echo_errors(lines)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _require_model(path: Path) -> Path:
    """Fail on a missing model file now, not when the lazy predictor loads.

    Raises:
        ModelNotFoundError: If no checkpoint is at *path*.
    """
    if not path.exists():
        raise ModelNotFoundError(path)
    return path


def _extraction_config(settings: Settings) -> ExtractionConfig:
    return ExtractionConfig(
        image_format=settings.pipeline.extraction.image_format,
        question_max_width=settings.pipeline.extraction.question_max_width,
        question_max_height=settings.pipeline.extraction.question_max_height,
    )


def _subject_extractor(
    settings: Settings, subject: str = "", *, review: bool = False
) -> SubjectExtractor:
    """Build the extraction chain, reviewer included when one is asked for.

    Raises:
        ModelNotFoundError: If the segmentation checkpoint is missing.
    """
    on_review = None

    if review:
        # Imported here so a machine with no display can still run every other
        # command.
        from digitex.ui.page_review import TkPageReviewer

        output_dir = settings.paths.extraction_output_dir
        on_review = TkPageReviewer(
            subject=subject,
            census=ImageCensus(output_dir),
            validator=AnswerValidator(output_dir),
        )

    # The composition root for extraction: the only place a checkpoint path
    # becomes a model and a language becomes an OCR reader. Everything below
    # names both by interface, which is what lets the differential suite replay a
    # run with neither installed.
    #
    # The chain is built from the bottom: the config belongs to PageExtractor,
    # and the runners above it take the configured collaborator.
    page_extractor = PageExtractor(
        _extraction_config(settings),
        detector=YOLO_SegmentationPredictor(
            str(_require_model(settings.paths.extraction_model_path))
        ),
        text_reader=TextExtractor(language=OCR_LANGUAGE),
        on_review=on_review,
    )
    return SubjectExtractor(
        books_dir=settings.paths.books_dir,
        extraction_dir=settings.paths.extraction_output_dir,
        book_extractor=BookExtractor(page_extractor),
    )


def _answers_extractor(settings: Settings) -> AnswersExtractor:
    """Build the answer-key extractor.

    Raises:
        APIError: If no OpenRouter API key is configured.
    """
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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command(name="rename-pages")
def rename_pages() -> None:
    """Renumber every scanned page to its canonical zero-padded name.

    A scanner names its export after the batch that made it (Химия.001.png) or
    after nothing in particular (10.jpg, which sorts ahead of 2.jpg). This
    renumbers each year's pages in reading order as 001, 002, … keeping the
    file's own format, and moves each page's processed twin with it so the two
    variants never disagree. Answer sheets keep their names — {year}_{n} is what
    says which year and sheet they are.

    Safely re-runnable: pages already correctly named are left alone.
    """
    from digitex.pipeline.preprocessing import rename_pages as run_rename

    settings = get_settings()

    try:
        result = run_rename(settings.paths.books_dir)
    except DirectoryNotFoundError as exc:
        raise abort(f"✗ {exc}") from None

    _ok(f"✓ Renamed {result.renamed} page(s), {result.unchanged} already named right")
    _report_failures(result.failed, "page(s)", result.errors)


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
    var/books/{subject}/processed/. Answer sheets keep their {year}_{n} names and
    skip the shadow flatten so their printed shading survives; everything else
    applies to them too.

    Safely re-runnable: scans already processed are skipped unless --force. Note
    that --force can move edges, and annotations drawn on a processed page move
    with them.
    """
    from digitex.pipeline.preprocessing import preprocess_scans as run_preprocessing

    settings = get_settings()

    try:
        result = run_preprocessing(settings.paths.books_dir, force=force)
    except DirectoryNotFoundError as exc:
        raise abort(f"✗ {exc}") from None

    _ok(
        f"✓ Renamed {result.renamed} page(s), processed {result.processed}"
        f" scan(s), skipped {result.skipped} already done"
        f" → {settings.paths.books_dir}"
    )
    _report_failures(result.failed, "scan(s)", result.errors)


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

    With --review, each page opens in a window showing its detected polygons and
    the option/part/number every question would be saved as. Correct them with
    the mouse, then approve, skip the page, or abort the run. The window's second
    tab shows the subject's per-year counts and the answers.json check.
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

    if isinstance(result, SubjectRefused):
        typer.echo(typer.style("✗ Extraction failed:", fg="red", bold=True), err=True)
        _echo_errors([result.reason])
        raise typer.Exit(code=1)

    _ok(
        f"✓ Extraction completed: {result.extracted} processed,"
        f" {len(result.skipped)} skipped (subject: {subject})"
    )
    # Neither of these is a failure, but a run that swallows them silently loses
    # crops — so they are said out loud on the way past.
    for heading, items in (
        ("Kept existing images", result.collisions),
        ("Unfinished question pieces", result.unfinished),
    ):
        if items:
            _note(f"\n{heading}:")
            for line in messages(items):
                typer.echo(f"  - {line}")
    for note in result.notes:
        _note(f"\n{note}")
    _report_failures(len(result.failures), "page(s)", messages(result.failures))


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

    if not result.clean:
        typer.echo(
            typer.style("✗ Answer extraction failed:", fg="red", bold=True), err=True
        )
        _echo_errors(list(result.failures))
        raise typer.Exit(code=1)

    if result.note:
        _note(f"! {result.note}")
    _ok(f"✓ Extracted answers for {result.years} years")


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
    every answer the segmentation model and OCR gave and the digest of every file
    written. `tests/differential` replays that recording to check a restructuring
    wrote exactly the same images — on a machine with no checkpoint, no GPU and
    no tesseract.

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

    config = _extraction_config(settings)
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

    # Guarded here, where the checkpoint is actually resolved. It used to guard
    # `_extraction_config`, which reads settings and cannot raise this — so a
    # machine with no model got a traceback out of the unguarded `_require_model`
    # below, while `extract-questions` exited cleanly for the same reason.
    try:
        model_path = str(_require_model(settings.paths.extraction_model_path))
    except ModelNotFoundError as exc:
        raise abort(f"✗ {exc}") from None

    # Started from empty, so the numbering starts at 1 and the digests below
    # describe a whole book rather than a book plus whatever was there before.
    output_dir = recorded_output_dir(data_root, subject, year)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    page_extractor = PageExtractor(
        config,
        detector=RecordingPredictor(YOLO_SegmentationPredictor(model_path), recording),
        text_reader=RecordingTextExtractor(
            TextExtractor(language=OCR_LANGUAGE), recording
        ),
    )
    result = BookExtractor(page_extractor).extract(pages_dir, output_dir)

    recording.outputs = directory_digests(output_dir)
    recording.write(destination)

    _ok(
        f"✓ Recorded {subject}/{year}: {result.pages} pages,"
        f" {len(recording.outputs)} images → {destination}"
    )
    # A page that failed produced no image, so its answers are missing from the
    # recording too — a replay of it would refuse rather than diverge quietly.
    _report_failures(len(result.failures), "page(s)", messages(result.failures))


if __name__ == "__main__":
    app()
