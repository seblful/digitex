"""Try different SegmentProcessor settings on a single page image from books."""

import itertools
from pathlib import Path

import typer
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from digitex.config import get_settings
from digitex.core.processors import ImageCropper, SegmentProcessor, resize_image
from digitex.ml.predictors import YOLO_SegmentationPredictor

app = typer.Typer()

SATURATION_VALUES = [40, 60, 80, 100, 120, 140, 160]
BG_THRESHOLD_VALUES = [160, 180, 200, 220]
GAMMA_VALUES = [0.4, 0.6, 0.8, 1.0]

LABEL_MARGIN = 280
ROW_PADDING = 4

PAGES: list[Path] = [
    Path("books/biology/images/2016/30.jpg"),
    Path("books/biology/images/2024/10.jpg"),
]


def _generate_combinations() -> list[tuple[int, int, float]]:
    return list(itertools.product(SATURATION_VALUES, BG_THRESHOLD_VALUES, GAMMA_VALUES))


def _extract_questions(
    image: Image.Image,
    predictor: YOLO_SegmentationPredictor,
    cropper: ImageCropper,
    max_width: int,
    max_height: int,
) -> list[Image.Image]:
    result = predictor.predict(image)
    if not result.ids:
        return []

    segments: list[Image.Image] = []
    for class_id, polygon in zip(result.ids, result.polygons):
        label = result.id2label.get(class_id, "unknown")
        if label != "question":
            continue
        cropped = cropper.cut_out_image_by_polygon(image, polygon)
        cropped = resize_image(cropped, max_width, max_height)
        segments.append(cropped)

    return segments


def _build_segment_grid(
    raw: Image.Image,
    processor: SegmentProcessor,
    combos: list[tuple[int, int, float]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    label_h = 24
    cell_h = raw.height
    row_height = cell_h + label_h + ROW_PADDING

    rows: list[tuple[str, Image.Image]] = [("000  original", raw.convert("RGB"))]

    for i, (sat, bg, gamma) in enumerate(
        tqdm(combos, desc="Applying settings", leave=False), 1
    ):
        processed = processor.process(
            raw, saturation_threshold=sat, bg_threshold=bg, gamma=gamma
        )
        rows.append((f"{i:03d}  sat={sat}  bg={bg}  gamma={gamma}", processed))

    total_w = LABEL_MARGIN + raw.width
    total_h = row_height * len(rows)

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for i, (desc, img) in enumerate(rows):
        y = i * row_height + ROW_PADDING // 2
        draw.text((8, y + 4), desc, fill=(0, 0, 0), font=font)
        grid.paste(img, (LABEL_MARGIN, y + label_h))

    return grid


def _process_page(
    image_path: Path,
    predictor: YOLO_SegmentationPredictor,
    cropper: ImageCropper,
    processor: SegmentProcessor,
    combos: list[tuple[int, int, float]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_height: int,
    output_dir: Path,
) -> None:
    typer.echo(f"\nProcessing {image_path}")
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    year = image_path.parent.name
    image_num = image_path.stem

    segments = _extract_questions(image, predictor, cropper, max_width, max_height)
    typer.echo(f"Found {len(segments)} questions")

    if not segments:
        typer.echo("No questions detected, skipping")
        return

    for idx, raw in enumerate(tqdm(segments, desc="Building grids", leave=False), 1):
        grid = _build_segment_grid(raw, processor, combos, font)
        grid_path = output_dir / f"{year}_{image_num}_{idx:02d}.png"
        grid.save(grid_path)

    typer.echo(f"Saved {len(segments)} grids to {grid_path.parent}")


@app.callback(invoke_without_command=True)
def tune(
    output_dir: Path = typer.Option(
        Path("extraction/tuning"), help="Output directory for grids"
    ),
) -> None:
    settings = get_settings()
    model_path = settings.extraction.model_path

    if not model_path.exists():
        typer.echo(f"Model not found: {model_path}")
        raise typer.Exit(code=1)

    if not PAGES:
        typer.echo("No pages listed in PAGES, add paths to the script")
        raise typer.Exit(code=1)

    for p in PAGES:
        if not p.exists():
            typer.echo(f"Path not found: {p}")
            raise typer.Exit(code=1)

    predictor = YOLO_SegmentationPredictor(str(model_path))
    cropper = ImageCropper()
    processor = SegmentProcessor()

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    combos = _generate_combinations()
    typer.echo(
        f"Testing {len(combos)} parameter combinations across {len(PAGES)} page(s)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for page_path in PAGES:
        _process_page(
            page_path,
            predictor,
            cropper,
            processor,
            combos,
            font,
            settings.extraction.question_max_width,
            settings.extraction.question_max_height,
            output_dir,
        )


if __name__ == "__main__":
    app()
