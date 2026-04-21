"""Run extraction of question images from image books."""

import platform
import shutil
import tempfile
from pathlib import Path

if platform.system() == "Windows":
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath  # ty: ignore[invalid-assignment]

import typer

from digitex import AnswersExtractor, TestsExtractor
from digitex.config import get_settings
from digitex.extractors.manual_extractor import ManualExtractor
from digitex.logging import setup_logging

setup_logging()

app = typer.Typer()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@app.command()
def extract() -> None:
    """Extract question images from all image books."""
    settings = get_settings()
    model_path = settings.paths.home_dir / settings.extraction.model_path
    extractor = TestsExtractor(
        model_path=model_path,
        image_format=settings.extraction.image_format,
        question_max_width=settings.extraction.question_max_width,
        question_max_height=settings.extraction.question_max_height,
        books_dir=settings.paths.books_dir,
        extraction_dir=settings.paths.extraction_dir
        / settings.extraction.data_dir_name
        / settings.extraction.output_dir_name,
    )
    extractor.extract(subject="all")


@app.command()
def count() -> None:
    """Count images in each subfolder of the extraction output."""
    from collections import Counter

    settings = get_settings()
    folder = (
        settings.paths.extraction_dir
        / settings.extraction.data_dir_name
        / settings.extraction.output_dir_name
    )

    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: {folder} is not a valid directory")
        raise typer.Exit(code=1)

    def count_images(root: Path) -> dict[Path, int]:
        result: dict[Path, int] = {}
        for item in root.iterdir():
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                if root not in result:
                    result[root] = 0
                result[root] += 1
            elif item.is_dir():
                result.update(count_images(item))
        return result

    def get_mode(values: list[int]) -> int:
        if not values:
            return 0
        counter = Counter(values)
        return counter.most_common(1)[0][0]

    def get_all_modes(values: list[int]) -> set[int]:
        if not values:
            return set()
        counter = Counter(values)
        max_count = counter.most_common(1)[0][1]
        return {v for v, c in counter.items() if c == max_count}

    counts = count_images(folder)
    if not counts:
        typer.echo("No images found")
        return

    struct: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for path, img_count in counts.items():
        parts = path.relative_to(folder).parts
        if len(parts) >= 4:
            subject, year, option, part = parts[0], parts[1], parts[2], parts[3]
            if subject not in struct:
                struct[subject] = {}
            if year not in struct[subject]:
                struct[subject][year] = {}
            if option not in struct[subject][year]:
                struct[subject][year][option] = {}
            struct[subject][year][option][part] = img_count

    for subject in sorted(struct):
        typer.echo(subject)
        for year in sorted(struct[subject], key=lambda y: int(y) if y.isdigit() else y):
            options = struct[subject][year]
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
                part_modes[part] = get_all_modes(part_counts)

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

    total = sum(counts.values())
    typer.echo(f"\nTotal: {total} images in {len(counts)} folders")


@app.command()
def renumber(
    dry_run: bool = typer.Option(True, help="Preview changes without renaming"),
) -> None:
    """Renumber images in the extraction output folder to fill gaps (e.g., 1, 2, 4, 5 -> 1, 2, 3, 4)."""
    settings = get_settings()
    folder = (
        settings.paths.extraction_dir
        / settings.extraction.data_dir_name
        / settings.extraction.output_dir_name
    )

    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: {folder} is not a valid directory")
        raise typer.Exit(code=1)

    def find_image_folders(root: Path) -> list[Path]:
        result = []
        for item in root.iterdir():
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                return [root]
            if item.is_dir():
                result.extend(find_image_folders(item))
        return result

    total = 0
    for fp in find_image_folders(folder):
        images = sorted(
            (int(f.stem), f)
            for f in fp.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue

        current = [n for n, _ in images]
        expected = list(range(1, len(images) + 1))
        if current == expected:
            continue

        changes = [
            (f, f.parent / f"{i}{f.suffix}")
            for i, (_, f) in enumerate(images, 1)
            if f.stem != str(i)
        ]

        rel = fp.relative_to(folder)
        if dry_run:
            typer.echo(f"{rel}:")
            for o, n in changes:
                typer.echo(f"  {o.name} -> {n.name}")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                tp = Path(tmp)
                for old, new in changes:
                    shutil.move(str(old), str(tp / new.name))
                    shutil.move(str(tp / new.name), str(new))
        total += len(changes)

    if dry_run and total:
        typer.echo(f"\n{total} files would be renamed")
    elif total:
        typer.echo(f"Renamed {total} files successfully")
    else:
        typer.echo("All images are already sequential")


@app.command()
def add_manual(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying"
    ),
) -> None:
    """Add manually cropped question images to the extraction output.

    Manual images should be placed in extraction/data/manual/{subject}/
    with filename format: YYYY_OPTION_PART_QUESTION.png
    Example: biology/2016_3_A_20.png
    """
    settings = get_settings()
    manual_dir = (
        settings.paths.extraction_dir / settings.extraction.data_dir_name / "manual"
    )
    output_dir = (
        settings.paths.extraction_dir
        / settings.extraction.data_dir_name
        / settings.extraction.output_dir_name
    )

    extractor = ManualExtractor(
        image_format=settings.extraction.image_format,
        question_max_width=settings.extraction.question_max_width,
        question_max_height=settings.extraction.question_max_height,
        manual_dir=manual_dir,
        output_dir=output_dir,
    )
    extractor.process_all(dry_run=dry_run)


@app.command()
def extract_answers(
    subject: str = typer.Argument(..., help="Subject name (e.g., biology, chemistry)"),
) -> None:
    """Extract answer keys from answer sheet images using Mistral OCR.

    Answer images should be placed in books/{subject}/answers/
    with filename format: YYYY_N.jpg (e.g., 2016_1.jpg, 2016_2.jpg)

    Results are saved to extraction/data/output/{subject}/{year}/answers.json
    """
    extractor = AnswersExtractor()
    extractor.extract(subject=subject)


@app.command()
def check_answers(
    subject: str = typer.Argument(..., help="Subject name (e.g., biology, chemistry)"),
) -> None:
    """Check that answers.json files correspond to extracted question images.

    Verifies that:
    1. Each year has an answers.json file
    2. Questions in answers.json match the image files in option/part folders
    3. All options have the same questions (validation check)
    4. Reports any mismatches or missing files
    """
    settings = get_settings()
    output_dir = (
        settings.paths.extraction_dir
        / settings.extraction.data_dir_name
        / settings.extraction.output_dir_name
        / subject
    )

    if not output_dir.exists():
        typer.echo(f"Error: {output_dir} does not exist")
        raise typer.Exit(code=1)

    import json

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
                opt for opt in answers_data 
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
