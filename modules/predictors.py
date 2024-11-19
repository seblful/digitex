from abc import ABC, abstractmethod
from typing import List, Dict

from PIL import Image

from ultralytics import YOLO
from ultralytics.engine.results import Results


class Predictor(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def get_points(self):
        pass


class YoloPredictor(Predictor):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.__model = None
        self.classes_dict = self.model.names

    @property
    def model(self) -> YOLO:
        if self.__model is None:
            self.__model = YOLO(self.model_path, verbose=False)

        return self.__model

    def predict(self,
                image: Image.Image) -> Results:
        result = self.model.predict(image, verbose=False)[0]

        return result

    def get_points(self,
                   image: Image.Image) -> Dict[int, List[float]]:
        result = self.predict(image)

        # Get points
        points_dict = dict()
        for box, points in zip(result.boxes, result.masks.xyn):

            points = points.reshape(-1).tolist()
            class_idx = int(box.cls.item())

            # Append points to the list in dict
            if class_idx not in points_dict:
                points_dict[class_idx] = []
            points_dict[class_idx].append(points)

        return points_dict

    # @staticmethod
    # def __detect_text(image: Image.Image,
    #                   det_processor: SegformerImageProcessor,
    #                   det_model: PreTrainedModel) -> list[Image.Image]:
    #     det_result = batch_text_detection(images=[image],
    #                                       processor=det_processor,
    #                                       model=det_model)
    #     # Convert image to numpy
    #     img = np.array(image)

    #     # Iterate through detected bboxes and add cropped images to list
    #     det_images = []
    #     for polygon_box in det_result[0].bboxes:
    #         x_min, y_min, x_max, y_max = polygon_box.bbox

    #         det_img = img[y_min:y_max, x_min:x_max]

    #         if not np.sum(det_img) == 0:
    #             det_image = Image.fromarray(det_img)
    #             det_images.append(det_image)

    #     return det_images
