import os
import json

from urllib.parse import quote, unquote
from PIL import Image

import numpy as np
import cv2


class AnnsConverter:
    def __init__(self,
                 ls_upload_dir: str) -> None:
        self.ls_upload_dir = ls_upload_dir

        self.bbox_keys = ["x", "y", "width", "height", "rotation"]

    @staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def write_json(json_dicts: dict,
                   json_path: str,
                   indent: int = 4) -> None:
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dicts, json_file,
                      indent=indent,
                      ensure_ascii=False)

        return None

    def add_filename_index(self,
                           filename: str,
                           index: int) -> str:
        filename_split = os.path.splitext(filename)
        filename = f"{filename_split[0]}_{str(index)}{filename_split[1]}"

        return filename

    def create_local_path(self,
                          task_path: str) -> str:
        task_path = unquote(task_path)
        task_path = os.path.normpath(task_path)
        path_split = task_path.split(os.sep)
        task_path = "/".join(path_split[3:])

        local_path = os.path.join(self.ls_upload_dir, task_path)
        local_path = os.path.normpath(local_path)

        return local_path

    def create_task_path(self,
                         local_path: str,
                         project_num: str = None,
                         index: int = None) -> str:
        path_split = local_path.split(os.sep)

        # Change project_num
        if project_num is not None:
            path_split[-2] = str(project_num)

        filename = path_split[-1]

        # Add index to filename
        if index is not None:
            filename = self.add_filename_index(filename=filename,
                                               index=index)

        filename = quote(filename)
        task_path = f"/data/{path_split[-3]}/{path_split[-2]}/{filename}"

        return task_path


class OCRAnnsConverter(AnnsConverter):
    def __init__(self,
                 ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)


class OCRBBOXAnnsConverter(OCRAnnsConverter):
    def __init__(self,
                 ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)

        self.output_json_name = "bbox_data.json"

    def get_preds(self, task: dict) -> list[dict]:
        # Create empty list to results
        preds = [{"result": []}]

        result = task['annotations'][0]['result']

        for entry in result:
            if entry['type'] == 'textarea':
                output_entry = entry.copy()

                del output_entry['value']['text']

                output_entry['value']['rectanglelabels'] = ['text']

                output_entry['from_name'] = 'label'
                output_entry['to_name'] = 'image'
                output_entry['type'] = 'rectanglelabels'

                preds[0]['result'].append(output_entry)

        return preds

    def convert(self,
                input_json_path: str,
                output_dir: str) -> list[dict]:
        # Read json_path
        input_json_dicts = self.read_json(input_json_path)

        output_json_dicts = []

        # Iterating through tasks
        for task in input_json_dicts:
            anns_dict = {}

            anns_dict['data'] = task['data']

            predictions = self.get_preds(task)
            anns_dict['predictions'] = predictions

            output_json_dicts.append(anns_dict)

        # Write to json file
        output_json_path = os.path.join(output_dir, self.output_json_name)
        self.write_json(json_dicts=output_json_dicts,
                        json_path=output_json_path)

        return output_json_dicts


class OCRCaptionConverter(OCRAnnsConverter):
    def __init__(self,
                 ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)

        self.output_json_name = "caption_data.json"

    def cut_rotated_bbox(self,
                         image: Image,
                         image_width: int,
                         image_height: int,
                         bbox: dict) -> Image:
        img = np.array(image)

        # Convert relative values to absolute pixels
        x = int(bbox['x'] * image_width / 100)
        y = int(bbox['y'] * image_height / 100)
        width = int(bbox['width'] * image_width / 100)
        height = int(bbox['height'] * image_height / 100)
        angle = bbox['rotation']

        # Define the rectangle points
        cx = x + (width / 2)
        cy = y + (height / 2)
        rect = ((cx, cy), (width, height), angle)

        # Get the rotation matrix
        rect_points = cv2.boxPoints(rect)
        # rect_points = np.int0(rect_points)

        # Get rotated rectangle ROI
        src_pts = rect_points.astype(np.float32)
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype=np.float32)

        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply perspective transformation
        result = cv2.warpPerspective(img, M, (width, height))

        image = Image.fromarray(result)

        return image

    def create_output_path(self,
                           output_dir: str,
                           input_image_path: str,
                           index: int) -> str:

        # Create path to save subimages
        images_dir = os.path.join(output_dir, "caption-images")
        # os.mkdir(images_dir)

        # Split
        path_split = input_image_path.split(os.sep)

        # Add index to filename
        image_name = path_split[-1]
        image_name = self.add_filename_index(filename=image_name,
                                             index=index)

        # Create output path
        output_image_path = os.path.join(images_dir, image_name)
        output_image_path = os.path.normpath(output_image_path)

        return output_image_path

    def get_preds(self, entry: dict) -> list[dict]:
        preds = [{"result": []}]

        # Fill result
        output_entry = {}
        output_entry['from_name'] = "caption"
        output_entry['to_name'] = "image"
        output_entry['type'] = "textarea"
        output_entry['origin'] = "manual"

        text = entry['value']['text'][0]
        output_entry['value'] = {'text': [text]}
        preds[0]['result'].append(output_entry)

        return preds

    def convert(self,
                input_json_path: str,
                output_project_num: int,
<<<<<<< HEAD
                output_dir: str) -> None:
=======
                output_dir: str):
>>>>>>> e9fbb901d89caeb0b8506c331c7aa28d18a4cb8a
        # Read json_path
        ocr_json_dicts = self.read_json(input_json_path)

        output_json_dicts = []

        # Iterating through tasks
        for task in ocr_json_dicts:
            # Open image
            task_image_path = task['data']['image']
            input_image_path = self.create_local_path(
                task_path=task_image_path)
            image = Image.open(input_image_path)

            result = task['annotations'][0]['result']
            for i, entry in enumerate(result):
                if entry['type'] == 'textarea':
                    # Crop image
                    bbox = {k: v for k,
                            v in entry['value'].items() if k in self.bbox_keys}
                    cropped_image = self.cut_rotated_bbox(image=image,
                                                          image_width=entry['original_width'],
                                                          image_height=entry['original_height'],
                                                          bbox=bbox)

                    # Create caption and output image paths and save image
                    caption_image_path = self.create_task_path(local_path=input_image_path,
                                                               project_num=output_project_num,
                                                               index=i)
                    output_image_path = self.create_output_path(output_dir=output_dir,
                                                                input_image_path=input_image_path,
                                                                index=i)
                    cropped_image.save(output_image_path)

                    # Fill annotations with predictions for each cropped image
                    anns_dict = {}
                    anns_dict['data'] = {'captioning': caption_image_path}

                    predictions = self.get_preds(entry=entry)
                    anns_dict['predictions'] = predictions
                    output_json_dicts.append(anns_dict)

        # Write to json file
        output_json_path = os.path.join(output_dir, self.output_json_name)
        self.write_json(json_dicts=output_json_dicts,
                        json_path=output_json_path)
