import os
import logging

from PIL import Image

import pypdfium2 as pdfium

from ultralytics import YOLO

from surya.model.detection.model import load_model as load_text_det_model, load_processor as load_text_det_processor
from surya.model.recognition.processor import load_processor as load_ocr_rec_processor
from surya.model.recognition.model import load_model as load_ocr_rec_model

from modules.processors import ImageProcessor


class TestExtractor:
    def __init__(self,
                 pdf_dir: str,
                 models_dir: str,
                 outputs_dir: str,
                 langs: list = ["ru", "en"],
                 log_level=logging.INFO) -> None:

        # Paths
        self.pdf_dir = pdf_dir
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir

        # Language
        self.langs = langs

        # Image processor
        self.image_processor = ImageProcessor()

        # Setup logging
        self.__setup_logging(log_level=log_level)

        # Load models and processors
        self.__load_models()

    def __load_models(self) -> None:
        # YOLO models
        self.page_det_model = YOLO(os.path.join(self.models_dir, "page.pt"))
        self.question_det_model = YOLO(
            os.path.join(self.models_dir, "question.pt"))

        # Surya models
        self.text_det_processor = load_text_det_processor()
        self.text_det_model = load_text_det_model()

        self.ocr_rec_processor = load_ocr_rec_processor()
        self.ocr_rec_model = load_ocr_rec_model()

        logging.debug("All models and processors were loaded.")

    def __setup_logging(self, log_level) -> None:
        logging.basicConfig(level=log_level,
                            format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%d/%m/%Y %H:%M:%S",
                            filename="basic.log")
        logging.info("Logging is configured.")

    def __get_page_image(self,
                         page: pdfium.PdfPage,
                         scale: int = 3) -> Image.Image:
        # Get image from pdf
        bitmap = page.render(scale=scale,
                             rotation=0)
        image = bitmap.to_pil()

        # Check image mode and convert if not RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    def __predict_page(self,
                       image: Image) -> tuple[Image.Image, list, list]:
        # Preprocess image
        image = self.image_processor.process(image=image,
                                             scan_type="color",
                                             resize=True)

        # Make predictions and retrieve polygons and labels
        results = self.page_det_model.predict(source=image,
                                              verbose=True)

        polygons = [ann.astype(int) for ann in results[0].masks.xy]
        labels = [results[0].names[i.item()] for i in results[0].boxes.cls]

        # Postprocess image
        image = self.image_processor.process(image=image,
                                             scan_type="color",
                                             remove_ink=True)

        return image, polygons, labels

    def extract(self) -> None:
        # Iterate through each pdf
        for pdf_name in os.listdir(self.pdf_dir):
            full_pdf_path = os.path.join(self.pdf_dir, pdf_name)
            pdf_obj = pdfium.PdfDocument(full_pdf_path)

            # Iterate through each page in pdf
            for pdf_page in pdf_obj:
                # Get image
                page_image = self.__get_page_image(pdf_page)

                # Predict page
                page_image, page_polygons, page_labels = self.__predict_page(
                    image=page_image)

                # Predict questions

                break

            break
