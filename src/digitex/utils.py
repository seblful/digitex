import logging
import re
from pathlib import Path

import pypdfium2 as pdfium
import torch
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _natural_sort_key(path: Path) -> list[int | str]:
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.stem):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return parts


def rename_images_to_sequential(base_dir: str | Path) -> None:
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        images = sorted(
            (p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=_natural_sort_key,
        )

        if not images:
            logger.debug(f"No images found in {folder}, skipping")
            continue

        temp_names: list[Path] = []
        for i, img_path in enumerate(images, start=1):
            temp_path = img_path.with_name(f"_temp_{i}{img_path.suffix.lower()}")
            img_path.rename(temp_path)
            temp_names.append(temp_path)

        for i, temp_path in enumerate(temp_names, start=1):
            final_path = temp_path.with_name(f"{i}{temp_path.suffix.lower()}")
            temp_path.rename(final_path)

        logger.info(f"Renamed {len(images)} images in {folder}")


def create_pdf_from_images(
    image_dir: str | Path,
    output_path: str | Path,
) -> None:
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        (p for p in image_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_sort_key,
    )

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument.new()

    for image_path in images:
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            bitmap = pdfium.PdfBitmap.from_pil(image)
            pdf_image = pdfium.PdfImage.new(pdf)
            pdf_image.set_bitmap(bitmap)

            width, height = pdf_image.get_size()
            matrix = pdfium.PdfMatrix().scale(width, height)
            pdf_image.set_matrix(matrix)

            page = pdf.new_page(width, height)
            page.insert_obj(pdf_image)
            page.gen_content()
            bitmap.close()
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Failed to process image {image_path}: {e}")
            continue

    pdf.save(str(output_path), version=17)
    logger.info(f"PDF created successfully: {output_path}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.debug("CUDA device available")
        return torch.device("cuda")
    logger.debug("Using CPU device")
    return torch.device("cpu")


def get_device_count() -> int:
    return torch.cuda.device_count()


def get_device_indices() -> list[int]:
    return list(range(get_device_count()))
