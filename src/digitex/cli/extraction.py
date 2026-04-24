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

setup_logging()

app = typer.Typer(help="Extraction commands for processing test books.")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@app.command(name="extract-questions")
def extract_questions(
    subject: Annotated[
        str, typer.Argument(help="Subject name to extract (e.g., biology, chemistry)")
    ],
) -> None:
    """Extract question images from a specific subject.
    
    SUBJECT is the name of the subject folder in the books directory.
    """
    extractor = ExtractorFactory.create_tests_extractor()
    result = extractor.extract(subject=subject)

    if result.success:
        typer.echo(
            typer.style(
                f"✓ Extraction completed: {result.processed} processed, {result.skipped} skipped (subject: {subject})",
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
def count_questions(
    subject: Annotated[
        str, typer.Argument(help="Subject name to count (e.g., biology, chemistry)")
    ],
) -> None:
    """Count images in a specific subject's extraction output."""
    settings = get_settings()
    folder = settings.paths.extraction_output_dir / subject

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
    typer.echo(f"\nTotal: {total_images} images in {total_folders} folders (subject: {subject})")


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
    settings = get_settings()
    folder = settings.paths.extraction_output_dir / subject

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
    settings = get_settings()
    manual_dir = settings.paths.extraction_manual_dir / subject
    
    if not manual_dir.exists():
        typer.echo(f"Error: Manual directory '{subject}' not found", err=True)
        raise typer.Exit(code=1)
    
    extractor = ExtractorFactory.create_manual_extractor(manual_dir=manual_dir)
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
    """Extract answer keys from answer sheet images using Mistral OCR.

    Answer images should be placed in books/{subject}/answers/
    with filename format: YYYY_N.jpg (e.g., 2016_1.jpg, 2016_2.jpg)

    Results are saved to extraction/data/output/{subject}/{year}/answers.json
    """
    extractor = ExtractorFactory.create_answers_extractor()
    result = extractor.extract(subject=subject)

    if result.success:
        typer.echo(
            typer.style(
                f"✓ Extracted answers for {result.metadata.get('years_processed', 0)} years",
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


@app.command(name="check-answers")
def check_answers(
    subject: Annotated[
        str, typer.Argument(help="Subject name (e.g., biology, chemistry)")
    ],
) -> None:
    """Check that answers.json files correspond to extracted question images.

    Verifies that:
    1. Each year has an answers.json file
    2. Questions in answers.json match the image files in option/part folders
    3. All options have the same questions (validation check)
    4. Reports any mismatches or missing files
    """
    settings = get_settings()
    output_dir = settings.paths.extraction_output_dir / subject

    if not output_dir.exists():
        typer.echo(f"Error: {output_dir} does not exist", err=True)
        raise typer.Exit(code=1)

    years = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])

    typer.echo("=" * 60)
    typer.echo(f"CHECKING ANSWERS FOR: {subject}")
    typer.echo("=" * 60)

    total_issues = 0

    for year in years:
        year_dir = output_dir / year
        answers_file = year_dir / "answers.json"

        if not answers_file.exists():
            typer.echo(f"\n{year}: ✗ answers.json NOT FOUND")
            total_issues += 1
            continue

        with open(answers_file, encoding="utf-8") as f:
            import json

            answers_data = json.load(f)

        answer_questions = set()
        for option_data in answers_data.values():
            answer_questions.update(option_data.keys())

        image_questions = set()
        for option_folder in year_dir.iterdir():
            if not option_folder.is_dir():
                continue
            for part_folder in option_folder.iterdir():
                if not part_folder.is_dir():
                    continue
                for img_file in part_folder.iterdir():
                    if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                        try:
                            q_num = int(img_file.stem)
                            part = part_folder.name.upper()
                            image_questions.add(f"{part}{q_num}")
                        except ValueError:
                            pass

        missing_in_answers = image_questions - answer_questions
        missing_in_images = answer_questions - image_questions

        all_options_same = True
        first_option_questions = set(answers_data.get("1", {}).keys())
        for opt in answers_data:
            if set(answers_data[opt].keys()) != first_option_questions:
                all_options_same = False
                break

        has_mismatch = bool(missing_in_answers or missing_in_images)
        if has_mismatch:
            status = "❌ MISMATCH"
            total_issues += 1
        elif not all_options_same:
            status = "❌ OPTIONS DIFFER"
            total_issues += 1
        else:
            status = "✅ OK"

        a_count = sum(1 for k in answer_questions if k.startswith("A"))
        b_count = sum(1 for k in answer_questions if k.startswith("B"))

        typer.echo(f"\n{year}: {status}")
        typer.echo(f"  A-part: {a_count}, B-part: {b_count}")
        typer.echo(f"  Questions in images: {len(image_questions)}")
        typer.echo(f"  Questions in answers.json: {len(answer_questions)}")

        if not all_options_same:
            different_options = [
                opt
                for opt in answers_data
                if set(answers_data[opt].keys()) != first_option_questions
            ]
            typer.echo(f"  Options with different questions: {different_options}")

        if missing_in_answers:
            typer.echo(f"  Missing in answers.json: {sorted(missing_in_answers)}")
        if missing_in_images:
            typer.echo(f"  Missing in images: {sorted(missing_in_images)}")

    typer.echo("\n" + "=" * 60)
    if total_issues == 0:
        typer.echo("RESULT: All years match ✅")
    else:
        typer.echo(f"RESULT: {total_issues} issue(s) found ❌")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
