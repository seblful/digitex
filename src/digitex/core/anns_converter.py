import os
from urllib.parse import quote, unquote
from PIL import Image
import numpy as np
from .processors.img import ImgProcessor, ImgWarper
from .processors.file import FileProcessor


class AnnsConverter:
    def __init__(self, ls_upload_dir: str) -> None:
        self.ls_upload_dir = ls_upload_dir
        self.bbox_keys = ["x", "y", "width", "height", "rotation"]

    def unquote_path(self, path: str) -> str:
        return unquote(path)

    def normalize_path(self, path: str) -> str:
        return os.path.normpath(path)

    def standardize_path(self, path: str) -> str:
        path = self.unquote_path(path)
        path = self.normalize_path(path)

        return path

    def parse_path(self, path: str) -> tuple[int, str]:
        path = self.standardize_path(path)
        project_num, filename = path.split(os.sep)[-2:]
        return int(project_num), filename

    def normalize_task_path(self, task_path: str) -> str:
        task_path = unquote(task_path)
        task_path = os.path.normpath(task_path)
        return "/".join(task_path.split(os.sep)[3:])

    def add_filename_index(self, filename: str, index: int) -> str:
        name, ext = os.path.splitext(filename)
        return f"{name}_{index}{ext}"

    def remove_last_filename_index(self, filename: str) -> str:
        name, ext = os.path.splitext(filename)
        if "_" in name:
            name = "_".join(name.split("_")[:-1])
        return f"{name}{ext}"

    def remove_prefixes(self, filename: str) -> str:
        return filename.split("-")[-1].strip()

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
        self.output_json_name = "converted_data.json"

    def get_abs_box(self, entry: dict) -> tuple[int, int, int, int]:
        rel_box = {k: v for k, v in entry["value"].items() if k in self.bbox_keys}
        x = int(rel_box["x"] * entry["original_width"] / 100)
        y = int(rel_box["y"] * entry["original_height"] / 100)
        width = int(rel_box["width"] * entry["original_width"] / 100)
        height = int(rel_box["height"] * entry["original_height"] / 100)

        return x, y, width, height

    def _crop_image(
        self, image: Image.Image, box: tuple[int, int, int, int], angle: int
    ) -> np.ndarray:
        img = ImgProcessor.image2img(image)
        cropped_img = ImgWarper.warp_img_by_box(img, box, angle)
        cropped_image = ImgProcessor.img2image(cropped_img)

        return cropped_image

    def get_present_image_filenames(self, caption_json_path: str) -> set[str]:
        caption_json_dicts = FileProcessor.read_json(caption_json_path)
        present_image_filenames = set()
        for task in caption_json_dicts:
            _, image_filename = self.parse_path(task["data"]["captioning"])
            image_filename = self.remove_last_filename_index(image_filename)
            image_filename = self.remove_prefixes(image_filename)
            present_image_filenames.add(image_filename)
        return present_image_filenames

    def image_is_presented(
        self, image_filename: str, present_image_filenames: set[str]
    ) -> bool:
        image_filename = self.remove_prefixes(image_filename)

        if image_filename in present_image_filenames:
            return True
        return False

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
        self,
        ocr_json_path: str,
        caption_json_path: str,
        caption_project_num: int,
        output_dir: str,
    ) -> None:
        output_images_dir = os.path.join(output_dir, "converted-images")
        os.makedirs(output_images_dir, exist_ok=True)

        present_image_filenames = self.get_present_image_filenames(caption_json_path)

        output_json_dicts = []
        ocr_json_dicts = FileProcessor.read_json(ocr_json_path)
        for task in ocr_json_dicts:
            ocr_project_num, image_filename = self.parse_path(task["data"]["ocr"])

            # Check if the image is already presented in the caption project
            if self.image_is_presented(image_filename, present_image_filenames):
                continue

            input_image_path = self.create_local_path(task["data"]["ocr"])

            image = Image.open(input_image_path)

            i = 0
            for entry in task["annotations"][0]["result"]:
                if entry["type"] == "textarea":
                    box = self.get_abs_box(entry)

                    cropped_image = self._crop_image(
                        image, box, entry["value"]["rotation"]
                    )
                    task_image_path = self.create_task_path(
                        input_image_path, caption_project_num, i
                    )
                    output_image_path = self.create_output_path(
                        output_images_dir, input_image_path, i
                    )
                    cropped_image.save(output_image_path)

                    anns_dict = {
                        "data": {"captioning": task_image_path},
                        "predictions": self.extract_caption_predictions(entry),
                    }
                    output_json_dicts.append(anns_dict)

                    i += 1

        output_json_path = os.path.join(output_dir, self.output_json_name)
        FileProcessor.write_json(
            json_dict=output_json_dicts, json_path=output_json_path
        )
