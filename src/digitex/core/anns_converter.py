import os
from urllib.parse import quote, unquote
from PIL import Image
import numpy as np
import cv2
from .processors.file import FileProcessor


class AnnsConverter:
    def __init__(self, ls_upload_dir: str) -> None:
        self.ls_upload_dir = ls_upload_dir
        self.bbox_keys = ["x", "y", "width", "height", "rotation"]

    def add_filename_index(self, filename: str, index: int) -> str:
        name, ext = os.path.splitext(filename)
        return f"{name}_{index}{ext}"

    def normalize_task_path(self, task_path: str) -> str:
        task_path = unquote(task_path)
        task_path = os.path.normpath(task_path)
        return "/".join(task_path.split(os.sep)[3:])

    def create_local_path(self, task_path: str) -> str:
        normalized_path = self.normalize_task_path(task_path)
        local_path = os.path.join(self.ls_upload_dir, normalized_path)
        return os.path.normpath(local_path)

    def create_task_path(
        self, local_path: str, project_num: str = None, index: int = None
    ) -> str:
        path_split = local_path.split(os.sep)
        if project_num is not None:
            path_split[-2] = str(project_num)
        filename = path_split[-1]
        if index is not None:
            filename = self.add_filename_index(filename, index)
        return f"/data/{path_split[-3]}/{path_split[-2]}/{quote(filename)}"


class OCRAnnsConverter(AnnsConverter):
    def __init__(self, ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)


class OCRBBOXAnnsConverter(OCRAnnsConverter):
    def __init__(self, ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)
        self.output_json_name = "bbox_data.json"

    def extract_bbox_predictions(self, task: dict) -> list[dict]:
        predictions = [{"result": []}]
        for entry in task["annotations"][0]["result"]:
            if entry["type"] == "textarea":
                bbox_entry = self.create_bbox_entry(entry)
                predictions[0]["result"].append(bbox_entry)
        return predictions

    def create_bbox_entry(self, entry: dict) -> dict:
        bbox_entry = entry.copy()
        del bbox_entry["value"]["text"]
        bbox_entry["value"]["rectanglelabels"] = ["text"]
        bbox_entry.update(
            {"from_name": "label", "to_name": "image", "type": "rectanglelabels"}
        )
        return bbox_entry

    def convert(self, input_json_path: str, output_dir: str) -> list[dict]:
        input_json_dicts = FileProcessor.read_json(input_json_path)
        output_json_dicts = []

        for task in input_json_dicts:
            anns_dict = {
                "data": task["data"],
                "predictions": self.extract_bbox_predictions(task),
            }
            output_json_dicts.append(anns_dict)

        output_json_path = os.path.join(output_dir, self.output_json_name)
        FileProcessor.write_json(
            json_dicts=output_json_dicts, json_path=output_json_path
        )
        return output_json_dicts


class OCRCaptionConverter(OCRAnnsConverter):
    def __init__(self, ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)
        self.output_json_name = "caption_data.json"

    def cut_rotated_bbox(
        self, image: Image, image_width: int, image_height: int, bbox: dict
    ) -> Image:
        img = np.array(image)
        x, y, width, height, angle = self.calculate_absolute_bbox(
            image_width, image_height, bbox
        )
        rect = ((x + width / 2, y + height / 2), (width, height), angle)
        return self.extract_rotated_region(img, rect, width, height)

    def calculate_absolute_bbox(
        self, image_width: int, image_height: int, bbox: dict
    ) -> tuple:
        x = int(bbox["x"] * image_width / 100)
        y = int(bbox["y"] * image_height / 100)
        width = int(bbox["width"] * image_width / 100)
        height = int(bbox["height"] * image_height / 100)
        angle = bbox["rotation"]
        return x, y, width, height, angle

    def extract_rotated_region(
        self, img: np.ndarray, rect: tuple, width: int, height: int
    ) -> Image:
        rect_points = cv2.boxPoints(rect).astype(np.float32)
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect_points, dst_pts)
        result = cv2.warpPerspective(img, M, (width, height))
        return Image.fromarray(result)

    def create_output_path(
        self, output_dir: str, input_image_path: str, index: int
    ) -> str:
        image_name = self.add_filename_index(os.path.basename(input_image_path), index)
        return os.path.normpath(os.path.join(output_dir, image_name))

    def extract_caption_predictions(self, entry: dict) -> list[dict]:
        predictions = [{"result": []}]
        predictions[0]["result"].append(
            {
                "from_name": "caption",
                "to_name": "image",
                "type": "textarea",
                "origin": "manual",
                "value": {"text": [entry["value"]["text"][0]]},
            }
        )
        return predictions

    def convert(
        self, input_json_path: str, output_project_num: int, output_dir: str
    ) -> None:
        ocr_json_dicts = FileProcessor.read_json(input_json_path)
        caption_images_dir = os.path.join(output_dir, "caption-images")
        os.makedirs(caption_images_dir, exist_ok=True)

        output_json_dicts = []
        for task in ocr_json_dicts:
            input_image_path = self.create_local_path(task["data"]["ocr"])
            image = Image.open(input_image_path)

            i = 0
            for entry in task["annotations"][0]["result"]:
                if entry["type"] == "textarea":
                    bbox = {
                        k: v for k, v in entry["value"].items() if k in self.bbox_keys
                    }
                    cropped_image = self.cut_rotated_bbox(
                        image=image,
                        image_width=entry["original_width"],
                        image_height=entry["original_height"],
                        bbox=bbox,
                    )
                    caption_image_path = self.create_task_path(
                        input_image_path, output_project_num, i
                    )
                    output_image_path = self.create_output_path(
                        caption_images_dir, input_image_path, i
                    )
                    cropped_image.save(output_image_path)

                    anns_dict = {
                        "data": {"captioning": caption_image_path},
                        "predictions": self.extract_caption_predictions(entry),
                    }
                    output_json_dicts.append(anns_dict)

                    i += 1

        output_json_path = os.path.join(output_dir, self.output_json_name)
        FileProcessor.write_json(
            json_dict=output_json_dicts, json_path=output_json_path
        )
