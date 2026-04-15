"""Run extraction of question images from image books."""

import platform
import shutil
import tempfile
from pathlib import Path

if platform.system() == "Windows":
    import pathlib
    import pathlib._local as _local

    _local.PosixPath = pathlib.WindowsPath

import typer

from digitex import TestsExtractor
from digitex.config import get_settings
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
        / settings.extraction.output_dir_name,
    )
    extractor.extract_all()


@app.command()
def count() -> None:
    """Count images in each subfolder of the extraction output."""
    from collections import Counter

    settings = get_settings()
    folder = settings.paths.extraction_dir / settings.extraction.output_dir_name

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
    folder = settings.paths.extraction_dir / settings.extraction.output_dir_name

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


if __name__ == "__main__":
    app()
