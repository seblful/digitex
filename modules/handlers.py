import logging
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_RENDER_SCALE = 3
DEFAULT_CROP_KERNEL_SIZE = 5
DEFAULT_CROP_OFFSET = 0.025


class PDFHandler:
    """Handler for PDF file operations and page rendering."""

    @staticmethod
    def create_pdf(images: List[Image.Image], output_path: str | Path) -> None:
        """Create a PDF document from a list of PIL images.

        Args:
            images: List of PIL Images to convert to PDF pages.
            output_path: Path where the PDF will be saved.

        Raises:
            IOError: If the PDF cannot be created or saved.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pdf = pdfium.PdfDocument.new()

        for image in images:
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

        pdf.save(str(output_path), version=17)

    def get_page_image(
        self,
        page: pdfium.PdfPage,
        scale: int = DEFAULT_RENDER_SCALE,
    ) -> Image.Image:
        """Render a PDF page to a PIL Image.

        Args:
            page: PDF page object to render.
            scale: Rendering scale factor (higher = better quality).

        Returns:
            PIL Image of the rendered page in RGB format.
        """
        bitmap = page.render(scale=scale, rotation=0)
        image = bitmap.to_pil()
        return image if image.mode == 'RGB' else image.convert('RGB')

    def get_random_image(
        self,
        pdf_listdir: List[str],
        pdf_dir: str | Path,
    ) -> Tuple[Image.Image, str, int]:
        """Get a random page from a random PDF in the directory.

        Args:
            pdf_listdir: List of PDF filenames.
            pdf_dir: Directory containing the PDF files.

        Returns:
            Tuple of (image, image_name, page_index).

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If the PDF cannot be read.
        """
        import random

        pdf_dir = Path(pdf_dir)

        rand_pdf_name = random.choice(pdf_listdir)
        rand_pdf_path = pdf_dir / rand_pdf_name

        if not rand_pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {rand_pdf_path}")

        rand_pdf_obj = pdfium.PdfDocument(str(rand_pdf_path))

        rand_page_idx = random.randint(0, len(rand_pdf_obj) - 1)
        rand_page = rand_pdf_obj[rand_page_idx]

        rand_image = self.get_page_image(page=rand_page)
        rand_image_name = rand_pdf_path.stem + ".jpg"

        rand_pdf_obj.close()

        return rand_image, rand_image_name, rand_page_idx


class ImageHandler:
    """Handler for image operations including cropping and processing."""

    @staticmethod
    def crop_image(
        image: Image.Image,
        points: List[float],
        offset: float = DEFAULT_CROP_OFFSET,
    ) -> Image.Image:
        """Crop an image using a polygon and add white border.

        Args:
            image: Input PIL Image.
            points: Normalized polygon points [x1, y1, x2, y2, ...] (0-1).
            offset: Border size as fraction of image height.

        Returns:
            Cropped PIL Image with white border.

        Raises:
            ValueError: If points list is invalid.
        """
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]

        pts = np.array(points, dtype=np.float32)
        if len(pts) % 2 != 0:
            raise ValueError("Points list must contain an even number of values")

        pts = pts.reshape(-1, 2)

        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        img = img[y:y+h, x:x+w].copy()

        pts = pts - pts.min(axis=0)
        pts_int = pts.astype(np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts_int], -1, (255, 255, 255), -1, cv2.LINE_AA)

        result = cv2.bitwise_and(img, img, mask=mask)
        bg = np.ones_like(img, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        result = bg + result

        border = int(height * offset)
        result = cv2.copyMakeBorder(
            result,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    @staticmethod
    def get_random_image(
        images_listdir: List[str],
        images_dir: str | Path,
    ) -> Tuple[Image.Image, str]:
        """Get a random image from a directory.

        Args:
            images_listdir: List of image filenames.
            images_dir: Directory containing the images.

        Returns:
            Tuple of (image, image_name).

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            IOError: If the image cannot be read.
        """
        import random

        images_dir = Path(images_dir)

        rand_image_name = random.choice(images_listdir)
        rand_image_path = images_dir / rand_image_name

        if not rand_image_path.exists():
            raise FileNotFoundError(f"Image file not found: {rand_image_path}")

        rand_image = Image.open(rand_image_path)

        return rand_image, rand_image_name


class LabelHandler:
    """Handler for reading and processing label annotations."""

    @staticmethod
    def _read_points(label_path: str | Path) -> Dict[int, list[list[float]]]:
        """Read annotation points from a label file.

        Args:
            label_path: Path to the label file.

        Returns:
            Dictionary mapping class indices to lists of point coordinates.

        Raises:
            FileNotFoundError: If the label file doesn't exist.
            IOError: If the file cannot be read.
        """
        points_dict: Dict[int, list[list[float]]] = dict()

        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                if not data:
                    continue

                class_idx = int(data[0])
                points = [float(point) for point in data[1:]]

                if class_idx not in points_dict:
                    points_dict[class_idx] = []
                points_dict[class_idx].append(points)

        return points_dict

    @staticmethod
    def _get_random_points(
        classes_dict: Dict[int, str],
        points_dict: Dict[int, list],
        target_classes: List[str],
    ) -> Tuple[int, list[float]]:
        """Get random points for a target class.

        Args:
            classes_dict: Mapping of class indices to names.
            points_dict: Dictionary of points by class index.
            target_classes: List of class names to filter by.

        Returns:
            Tuple of (points_index, points) or (-1, []) if no match found.
        """
        import random

        points_dict_filtered = {
            k: points_dict[k]
            for k in points_dict
            if classes_dict.get(k) in target_classes
        }

        if not points_dict_filtered:
            return -1, []

        rand_class_idx = random.choice(list(points_dict_filtered.keys()))

        rand_points_idx = random.randint(0, len(points_dict_filtered[rand_class_idx]) - 1)
        rand_points = points_dict_filtered[rand_class_idx][rand_points_idx]

        return rand_points_idx, rand_points

    @staticmethod
    def get_random_label(
        image_name: str,
        labels_dir: str | Path,
    ) -> Tuple[str | None, str | None]:
        """Get the label file path for a given image.

        Args:
            image_name: Name of the image file.
            labels_dir: Directory containing label files.

        Returns:
            Tuple of (label_name, label_path) or (None, None) if not found.
        """
        labels_dir = Path(labels_dir)

        label_name = Path(image_name).stem + '.txt'
        label_path = labels_dir / label_name

        if not label_path.exists():
            return None, None

        return label_name, str(label_path)

    @staticmethod
    def points_to_abs_polygon(
        points: list[float],
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int]]:
        """Convert normalized points to absolute pixel coordinates.

        Args:
            points: Normalized points [x1, y1, x2, y2, ...] (0-1).
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            List of (x, y) tuples with absolute pixel coordinates.

        Raises:
            ValueError: If points list has odd length.
        """
        if len(points) % 2 != 0:
            raise ValueError("Points list must contain an even number of values")

        point_pairs = list(zip(points[::2], points[1::2]))
        abs_points = [
            (int(x * image_width), int(y * image_height))
            for x, y in point_pairs
        ]

        return abs_points

    def get_points(
        self,
        image_name: str,
        labels_dir: str | Path,
        classes_dict: Dict[int, str],
        target_classes: List[str],
    ) -> Tuple[int, list[float]]:
        """Get points for a specific image and target classes.

        Args:
            image_name: Name of the image file.
            labels_dir: Directory containing label files.
            classes_dict: Mapping of class indices to names.
            target_classes: List of class names to filter by.

        Returns:
            Tuple of (points_index, points).

        Raises:
            FileNotFoundError: If the label file doesn't exist.
        """
        _, rand_label_path = self.get_random_label(
            image_name=image_name,
            labels_dir=labels_dir,
        )

        if rand_label_path is None:
            return -1, []

        points_dict = self._read_points(rand_label_path)

        rand_points_idx, rand_points = self._get_random_points(
            classes_dict=classes_dict,
            points_dict=points_dict,
            target_classes=target_classes,
        )

        return rand_points_idx, rand_points
