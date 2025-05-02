import os
from urllib.parse import quote, unquote
from PIL import Image
import numpy as np
from .processors.img import ImgProcessor, ImgWarper
from .processors.file import FileProcessor


class AnnsConverter:
    def __init__(self, ls_local_storage_path: str) -> None:
        self.ls_local_storage_path = ls_local_storage_path
        self.ls_local_storage_prefix = "/data/local-files/?d="

        self.bbox_keys = ["x", "y", "width", "height", "rotation"]

    def quote_path(self, path: str) -> str:
        return quote(path)

    def unquote_path(self, path: str) -> str:
        return unquote(path)

    def normalize_path(self, path: str) -> str:
        return os.path.normpath(path)

    def standardize_path(self, path: str) -> str:
        path = self.unquote_path(path)
        path = self.normalize_path(path)

        return path

    def get_filename(self, path: str) -> str:
        path = self.standardize_path(path)
        filename = os.path.basename(path)
        return filename

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


class OCRAnnsConverter(AnnsConverter):
    def __init__(self, ls_local_storage_path: str) -> None:
        super().__init__(ls_local_storage_path)


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
    def __init__(self, ls_local_storage_path: str) -> None:
        super().__init__(ls_local_storage_path)
        self.convertation_name = "conv_ocr_to_caption"
        self.output_json_name = "converted_data.json"

    def _get_abs_box(self, entry: dict) -> tuple[int, int, int, int]:
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

    def _get_exist_image_filenames(self, caption_json_path: str) -> set[str]:
        caption_json_dicts = FileProcessor.read_json(caption_json_path)
        exist_image_filenames = set()
        for task in caption_json_dicts:
            image_filename = self.get_filename(task["data"]["captioning"])
            image_filename = self.remove_last_filename_index(image_filename)
            image_filename = self.remove_prefixes(image_filename)
            exist_image_filenames.add(image_filename)
        return exist_image_filenames

    def _image_is_exist(
        self, image_filename: str, exist_image_filenames: set[str]
    ) -> bool:
        image_filename = self.remove_prefixes(image_filename)

        if image_filename in exist_image_filenames:
            return True
        return False

    def create_local_path(self, dir: str, filename: str) -> str:
        local_path = os.path.join(dir, filename)
        local_path = self.normalize_path(local_path)
        return local_path

    def create_output_path(self, filename: str, i: int) -> str:
        output_filename = self.add_filename_index(filename, i)
        output_path = os.path.join(self.ls_local_storage_path, output_filename)
        output_path = self.normalize_path(output_path)
        return output_path

    def create_task_path(self, path: str) -> str:
        task_path = self.ls_local_storage_prefix + path
        task_path = self.normalize_path(task_path).replace("\\", "/")
        return task_path

    def _get_caption_preds(self, entry: dict) -> list[dict]:
        predictions = [{"model_version": self.convertation_name, "result": []}]
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
        ocr_images_dir: str,
        ocr_json_path: str,
        caption_json_path: str,
        output_dir: str,
    ) -> None:
        exist_image_filenames = self._get_exist_image_filenames(caption_json_path)

        output_json_dicts = []
        ocr_json_dicts = FileProcessor.read_json(ocr_json_path)
        for task in ocr_json_dicts:
            input_image_filename = self.get_filename(task["data"]["ocr"])

            # Check if the image is already exist in the caption project
            if self._image_is_exist(input_image_filename, exist_image_filenames):
                continue

            # Open image
            input_image_path = self.create_local_path(
                ocr_images_dir, input_image_filename
            )
            input_image = Image.open(input_image_path)

            # Iterate through each entry in the task and crop the image
            i = 0
            for entry in task["annotations"][0]["result"]:
                if entry["type"] == "textarea":
                    box = self._get_abs_box(entry)
                    cropped_image = self._crop_image(
                        input_image, box, entry["value"]["rotation"]
                    )

                    # Save image
                    output_image_path = self.create_output_path(input_image_filename, i)
                    cropped_image.save(output_image_path)

                    # Add task to output json dict
                    task_image_path = self.create_task_path(output_image_path)
                    anns_dict = {
                        "data": {"captioning": task_image_path},
                        "predictions": self._get_caption_preds(entry),
                    }
                    output_json_dicts.append(anns_dict)

                    i += 1

        output_json_path = os.path.join(output_dir, self.output_json_name)
        FileProcessor.write_json(
            json_dict=output_json_dicts, json_path=output_json_path
        )
