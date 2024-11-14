from abc import ABC, abstractmethod


class Predictor(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class YoloPredictor(Predictor):
    def __init__(self, model_path: str):
        self.model_path = model_path

    @property
    def model(self):
        pass

    def predict(self):
        pass

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
