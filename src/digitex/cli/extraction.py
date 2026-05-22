"""Extraction CLI commands."""

from typing import Annotated

import typer

from digitex.config import get_settings
from digitex.extractors.factory import ExtractorFactory
from digitex.extractors.utils import (
    count_subject_images,
    count_total_images,
    get_mode_values,
    renumber_directory_tree,
)
from digitex.logging import setup_logging
from digitex.services.answer_validator import (
    AnswerValidator,
    ValidationReport,
    YearReport,
)

# Per ADR 0001 — resolve settings once at the CLI boundary and pass them down.
_settings = get_settings()
setup_logging(_settings)

app = typer.Typer(help="Extraction commands for processing test books.")


@app.command(name="extract-questions")
def extract_questions(
    subject: Annotated[
        str, typer.Argument(help="Subject name to extract (e.g., biology, chemistry)")
    ],
) -> None:
    """Extract question images from a specific subject.

    SUBJECT is the name of the subject folder in the books directory.
    """
    extractor = ExtractorFactory(_settings).create_tests_extractor()
    result = extractor.extract(subject=subject)

    if result.success:
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
    else:
        typer.echo(typer.style("✗ Extraction failed:", fg="red", bold=True), err=True)
        for error in result.errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=1)


@app.command(name="count-questions")
def count_questions(  # noqa: PLR0912
    subject: Annotated[
        str, typer.Argument(help="Subject name to count (e.g., biology, chemistry)")
    ],
) -> None:
    """Count images in a specific subject's extraction output."""
    folder = _settings.paths.extraction_output_dir / subject

    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: Subject '{subject}' not found", err=True)
        raise typer.Exit(code=1)

    counts = count_subject_images(folder)

    if not counts:
        typer.echo(f"No images found for subject '{subject}'")
        return

    for year in sorted(counts, key=lambda y: int(y) if y.isdigit() else y):
        options = counts[year]
        num_options = len(options)
        year_label = f"  {year}: {num_options} options"

        all_parts: dict[str, list[int]] = {}
        for opt in options:
            for part, count in options[opt].items():
                if part not in all_parts:
                    all_parts[part] = []
                all_parts[part].append(count)

        part_modes: dict[str, set[int]] = {}
        for part, part_counts in all_parts.items():
            part_modes[part] = get_mode_values(part_counts)

        all_good = True
        for opt in options:
            for part, part_count in options[opt].items():
                if part_count not in part_modes[part]:
                    all_good = False
                    break
            if not all_good:
                break

        if num_options < 10:
            year_label = typer.style(year_label, fg="red", bold=True)
        elif all_good:
            year_label = typer.style(year_label, fg="green")
        typer.echo(year_label)

        for opt in sorted(options, key=lambda o: int(o) if o.isdigit() else o):
            parts_dict = options[opt]
            for part in sorted(parts_dict, key=lambda p: p):
                part_count = parts_dict[part]
                label = f"    {opt}/{part}: {part_count} images"
                if part_count not in part_modes[part]:
                    label = typer.style(label, fg="red", bold=True)
                typer.echo(label)

    total_images, total_folders = count_total_images(folder)
    typer.echo(
        f"\nTotal: {total_images} images in {total_folders} folders"
        f" (subject: {subject})"
    )


@app.command(name="renumber-questions")
def renumber_questions(
    subject: Annotated[
        str, typer.Argument(help="Subject name to renumber (e.g., biology, chemistry)")
    ],
    dry_run: Annotated[
        bool, typer.Option(help="Preview changes without renaming")
    ] = True,
) -> None:
    """Renumber images in a specific subject's extraction output to fill gaps."""
    folder = _settings.paths.extraction_output_dir / subject

    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: Subject '{subject}' not found", err=True)
        raise typer.Exit(code=1)

    total = renumber_directory_tree(folder, dry_run=dry_run)

    if dry_run and total:
        typer.echo(f"\n{total} files would be renamed")
    elif total:
        typer.echo(f"Renamed {total} files successfully")
    else:
        typer.echo("All images are already sequential")


@app.command(name="add-questions-manually")
def add_questions_manually(
    subject: Annotated[
        str, typer.Argument(help="Subject name to process (e.g., biology, chemistry)")
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without applying")
    ] = False,
) -> None:
    """Add manually cropped question images for a specific subject.

    Manual images should be placed in extraction/data/manual/{subject}/
    with filename format: YYYY_OPTION_PART_QUESTION.png
    Example: biology/2016_3_A_20.png
    """
    manual_dir = _settings.paths.extraction_manual_dir / subject

    if not manual_dir.exists():
        typer.echo(f"Error: Manual directory '{subject}' not found", err=True)
        raise typer.Exit(code=1)

    extractor = ExtractorFactory(_settings).create_manual_extractor(
        manual_dir=manual_dir
    )
    result = extractor.process_all(dry_run=dry_run)

    if result.success:
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
        if result.metadata.get("failed"):
            typer.echo(
                typer.style(
                    f"  Failed: {result.metadata['failed']}", fg="red", bold=True
                )
            )
    else:
        typer.echo(
            typer.style("✗ Manual extraction failed:", fg="red", bold=True), err=True
        )
        for error in result.errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=1)


@app.command(name="extract-answers")
def extract_answers(
    subject: Annotated[
        str, typer.Argument(help="Subject name (e.g., biology, chemistry)")
    ],
) -> None:
    """Extract answer keys from answer sheet images using OpenRouter.

    Answer images should be placed in books/{subject}/answers/
    with filename format: YYYY_N.jpg (e.g., 2016_1.jpg, 2016_2.jpg)

    Results are saved to extraction/data/output/{subject}/{year}/answers.json
    """
    extractor = ExtractorFactory(_settings).create_answers_extractor()
    result = extractor.extract(subject=subject)

    if result.success:
        typer.echo(
            typer.style(
                f"✓ Extracted answers for"
                f" {result.metadata.get('years_processed', 0)} years",
                fg="green",
            )
        )
        if result.errors:
            typer.echo(typer.style("\nErrors:", fg="red"))
            for error in result.errors:
                typer.echo(f"  - {error}")
    else:
        typer.echo(
            typer.style("✗ Answer extraction failed:", fg="red", bold=True), err=True
        )
        for error in result.errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=1)


def _render_year_report(year: YearReport) -> None:
    """Emit the colored year-level rendering of a validation outcome."""
    if not year.answers_file_present:
        typer.echo(f"\n{year.year}: ✗ answers.json NOT FOUND")
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

    if year.options_with_b == 0:
        styled = typer.style("NO option has Б", fg="red", bold=True)
    elif year.options_with_b < year.total_options:
        ratio = f"{year.options_with_b}/{year.total_options} options have Б"
        styled = typer.style(ratio, fg="yellow")
    else:
        styled = typer.style("all options have Б", fg="green")
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
def check_answers(
    subject: Annotated[
        str, typer.Argument(help="Subject name (e.g., biology, chemistry)")
    ],
) -> None:
    """Check that answers.json files correspond to extracted question images."""
    validator = AnswerValidator(_settings.paths.extraction_output_dir)
    try:
        report = validator.validate(subject)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc} does not exist", err=True)
        raise typer.Exit(code=1) from None

    _render_validation_report(report)


if __name__ == "__main__":
    app()
