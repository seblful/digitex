import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import cv2
import numpy as np
from PIL import Image

from modules.processors import FileProcessor

logger = logging.getLogger(__name__)


class AnnsConverter:
    """Base class for converting annotations between different formats."""

    def __init__(self, ls_upload_dir: str | Path) -> None:
        """Initialize the annotation converter.

        Args:
            ls_upload_dir: Base directory for uploaded files.

        Raises:
            ValueError: If ls_upload_dir doesn't exist.
        """
        self.ls_upload_dir = Path(ls_upload_dir).resolve()
        if not self.ls_upload_dir.exists():
            raise ValueError(f"Upload directory does not exist: {self.ls_upload_dir}")

        self.bbox_keys = ["x", "y", "width", "height", "rotation"]

    @staticmethod
    def read_json(json_path: str | Path) -> dict:
        """Read JSON data from a file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            Dictionary containing the JSON data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        return FileProcessor.read_json(json_path)

    @staticmethod
    def write_json(
        json_dicts: dict | list[dict], json_path: str | Path, indent: int = 4
    ) -> None:
        """Write data to a JSON file.

        Args:
            json_dicts: Dictionary or list of dictionaries to write as JSON.
            json_path: Path to the output JSON file.
            indent: Number of spaces for indentation.

        Raises:
            IOError: If the file cannot be written.
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        FileProcessor.write_json(json_dicts, json_path, indent=indent)

    @staticmethod
    def add_filename_index(filename: str, index: int) -> str:
        """Add an index to a filename.

        Args:
            filename: Original filename.
            index: Index to append.

        Returns:
            Filename with index added before extension.
        """
        path = Path(filename)
        return f"{path.stem}_{index}{path.suffix}"

    def create_local_path(self, task_path: str) -> Path:
        """Convert a task path to a local file path.

        Args:
            task_path: Task path from the annotation system.

        Returns:
            Resolved absolute local path.

        Raises:
            ValueError: If the resolved path is outside the upload directory.
        """
        task_path = unquote(task_path)
        path_parts = task_path.split("/")

        if len(path_parts) < 4:
            raise ValueError(f"Invalid task path: {task_path}")

        relative_path = Path(*path_parts[3:])
        local_path = self.ls_upload_dir / relative_path

        try:
            local_path = local_path.resolve()
        except RuntimeError:
            raise ValueError(f"Invalid path: {task_path}")

        if not str(local_path).startswith(str(self.ls_upload_dir)):
            raise ValueError(f"Path traversal attempt detected: {task_path}")

        return local_path

    def create_task_path(
        self,
        local_path: str | Path,
        project_num: str | int | None = None,
        index: int | None = None,
    ) -> str:
        """Convert a local path to a task path.

        Args:
            local_path: Local file path.
            project_num: Optional project number to include.
            index: Optional index to append to filename.

        Returns:
            Task path string.
        """
        local_path = Path(local_path)

        if project_num is not None:
            if len(local_path.parts) < 2:
                raise ValueError(f"Invalid local path for project number: {local_path}")
            parts = list(local_path.parts)
            parts[-2] = str(project_num)
            local_path = Path(*parts)

        filename = local_path.name

        if index is not None:
            filename = self.add_filename_index(filename=filename, index=index)

        filename = quote(filename)

        if len(local_path.parts) < 3:
            raise ValueError(f"Invalid local path: {local_path}")

        task_path = f"/data/{local_path.parts[-3]}/{local_path.parts[-2]}/{filename}"

        return task_path


class OCRAnnsConverter(AnnsConverter):
    """Converter for OCR annotations."""


class OCRBBOXAnnsConverter(OCRAnnsConverter):
    """Converter for OCR bounding box annotations."""

    def __init__(self, ls_upload_dir: str | Path) -> None:
        """Initialize the OCR BBOX converter.

        Args:
            ls_upload_dir: Base directory for uploaded files.
        """
        super().__init__(ls_upload_dir)
        self.output_json_name = "bbox_data.json"

    def get_preds(self, task: dict) -> list[dict]:
        """Extract predictions from a task.

        Args:
            task: Task dictionary containing annotations.

        Returns:
            List of prediction dictionaries.
        """
        preds = [{"result": []}]

        result = task.get('annotations', [{}])[0].get('result', [])

        for entry in result:
            if entry.get('type') == 'textarea':
                output_entry = entry.copy()

                if 'value' in output_entry:
                    output_entry['value'] = {k: v for k, v in output_entry['value'].items() if k != 'text'}
                    output_entry['value']['rectanglelabels'] = ['text']

                output_entry['from_name'] = 'label'
                output_entry['to_name'] = 'image'
                output_entry['type'] = 'rectanglelabels'

                preds[0]['result'].append(output_entry)

        return preds

    def convert(self, input_json_path: str | Path, output_dir: str | Path) -> list[dict]:
        """Convert OCR annotations to bounding box format.

        Args:
            input_json_path: Path to input JSON file.
            output_dir: Directory where output JSON will be saved.

        Returns:
            List of converted annotation dictionaries.

        Raises:
            IOError: If conversion fails.
        """
        input_json_path = Path(input_json_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_json_dicts = self.read_json(input_json_path)

        output_json_dicts = []

        for task in input_json_dicts:
            anns_dict = {}
            anns_dict['data'] = task.get('data', {})

            predictions = self.get_preds(task)
            anns_dict['predictions'] = predictions

            output_json_dicts.append(anns_dict)

        output_json_path = output_dir / self.output_json_name
        self.write_json(json_dicts=output_json_dicts, json_path=output_json_path)

        return output_json_dicts


class OCRCaptionConverter(OCRAnnsConverter):
    """Converter for OCR caption annotations."""

    def __init__(self, ls_upload_dir: str | Path) -> None:
        """Initialize the OCR caption converter.

        Args:
            ls_upload_dir: Base directory for uploaded files.
        """
        super().__init__(ls_upload_dir)
        self.output_json_name = "caption_data.json"

    def cut_rotated_bbox(
        self,
        image: Image.Image,
        image_width: int,
        image_height: int,
        bbox: dict[str, Any],
    ) -> Image.Image:
        """Crop a rotated bounding box from an image.

        Args:
            image: Input PIL Image.
            image_width: Original image width.
            image_height: Original image height.
            bbox: Bounding box dictionary with x, y, width, height, rotation.

        Returns:
            Cropped PIL Image.

        Raises:
            ValueError: If bbox is invalid.
        """
        required_keys = {'x', 'y', 'width', 'height', 'rotation'}
        if not required_keys.issubset(bbox.keys()):
            raise ValueError(f"Invalid bbox, missing keys: {required_keys - bbox.keys()}")

        img = np.array(image)

        x = int(bbox['x'] * image_width / 100)
        y = int(bbox['y'] * image_height / 100)
        width = int(bbox['width'] * image_width / 100)
        height = int(bbox['height'] * image_height / 100)
        angle = bbox['rotation']

        cx = x + (width / 2)
        cy = y + (height / 2)
        rect = ((cx, cy), (width, height), angle)

        rect_points = cv2.boxPoints(rect)

        src_pts = rect_points.astype(np.float32)
        dst_pts = np.array(
            [[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        result = cv2.warpPerspective(img, M, (width, height))

        return Image.fromarray(result)

    def create_output_path(
        self,
        output_dir: str | Path,
        input_image_path: str | Path,
        index: int,
    ) -> Path:
        """Create output path for a cropped image.

        Args:
            output_dir: Directory for output images.
            input_image_path: Path to original image.
            index: Index to append to filename.

        Returns:
            Path for saving the cropped image.
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "caption-images"
        images_dir.mkdir(parents=True, exist_ok=True)

        input_image_path = Path(input_image_path)
        image_name = self.add_filename_index(filename=input_image_path.name, index=index)

        output_image_path = images_dir / image_name

        return output_image_path

    def get_preds(self, entry: dict) -> list[dict]:
        """Extract predictions from an annotation entry.

        Args:
            entry: Annotation entry dictionary.

        Returns:
            List of prediction dictionaries.
        """
        preds = [{"result": []}]

        output_entry = {}
        output_entry['from_name'] = "caption"
        output_entry['to_name'] = "image"
        output_entry['type'] = "textarea"
        output_entry['origin'] = "manual"

        value = entry.get('value', {})
        text = value.get('text', [])
        if text:
            output_entry['value'] = {'text': [text[0]]}
        else:
            output_entry['value'] = {'text': [""]}

        preds[0]['result'].append(output_entry)

        return preds

    def convert(
        self,
        input_json_path: str | Path,
        output_project_num: int,
        output_dir: str | Path,
    ) -> None:
        """Convert OCR annotations to caption format.

        Args:
            input_json_path: Path to input JSON file.
            output_project_num: Project number for output.
            output_dir: Directory where outputs will be saved.

        Raises:
            IOError: If conversion fails.
        """
        input_json_path = Path(input_json_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ocr_json_dicts = self.read_json(input_json_path)

        output_json_dicts = []

        for task in ocr_json_dicts:
            task_image_path = task.get('data', {}).get('image', '')
            if not task_image_path:
                logger.warning("Task missing image path, skipping")
                continue

            input_image_path = self.create_local_path(task_path=task_image_path)

            if not input_image_path.exists():
                logger.warning(f"Image not found: {input_image_path}, skipping")
                continue

            try:
                image = Image.open(input_image_path)
            except (FileNotFoundError, IOError) as e:
                logger.warning(f"Failed to open image {input_image_path}: {e}")
                continue

            result = task.get('annotations', [{}])[0].get('result', [])

            for i, entry in enumerate(result):
                if entry.get('type') == 'textarea':
                    bbox = {
                        k: v
                        for k, v in entry.get('value', {}).items()
                        if k in self.bbox_keys
                    }

                    try:
                        cropped_image = self.cut_rotated_bbox(
                            image=image,
                            image_width=entry.get('original_width', image.width),
                            image_height=entry.get('original_height', image.height),
                            bbox=bbox,
                        )
                    except (ValueError, IOError) as e:
                        logger.warning(f"Failed to crop image: {e}")
                        continue

                    caption_image_path = self.create_task_path(
                        local_path=input_image_path,
                        project_num=output_project_num,
                        index=i,
                    )
                    output_image_path = self.create_output_path(
                        output_dir=output_dir,
                        input_image_path=input_image_path,
                        index=i,
                    )

                    try:
                        cropped_image.save(output_image_path)
                    except (IOError, ValueError) as e:
                        logger.warning(f"Failed to save image {output_image_path}: {e}")
                        continue

                    anns_dict = {}
                    anns_dict['data'] = {'captioning': caption_image_path}

                    predictions = self.get_preds(entry=entry)
                    anns_dict['predictions'] = predictions
                    output_json_dicts.append(anns_dict)

            image.close()

        output_json_path = output_dir / self.output_json_name
        self.write_json(json_dicts=output_json_dicts, json_path=output_json_path)
