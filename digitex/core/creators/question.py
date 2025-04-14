import os
from .base import BaseDataCreator
from ..predictors import YOLO_SegmentationPredictor


class QuestionDataCreator(BaseDataCreator):
    def extract_questions(
        self, page_raw_dir: str, train_dir: str, num_images: int
    ) -> None:
        images_dir = os.path.join(page_raw_dir, "images")
        labels_dir = os.path.join(page_raw_dir, "labels")
        classes_path = os.path.join(page_raw_dir, "classes.txt")
        classes_dict = self._read_classes(classes_path)
        images_listdir = os.listdir(images_dir)
        num_saved = 0

        while num_images != num_saved:
            rand_image, rand_image_name = self.image_handler.get_random_image(
                images_listdir=images_listdir, images_dir=images_dir
            )
            rand_points_idx, rand_points = self.label_handler.get_points(
                image_name=rand_image_name,
                labels_dir=labels_dir,
                classes_dict=classes_dict,
                target_classes=["question"],
            )
            rand_points = self.label_handler.points_to_abs_polygon(
                points=rand_points,
                image_width=rand_image.width,
                image_height=rand_image.height,
            )
            rand_image = self.image_handler.crop_image(
                image=rand_image, points=rand_points
            )
            num_saved = self._save_image(
                rand_points_idx,
                output_dir=train_dir,
                image=rand_image,
                image_name=rand_image_name,
                num_saved=num_saved,
                num_images=num_images,
            )

    def predict_questions(
        self,
        raw_dir: str,
        train_dir: str,
        yolo_model_path: str,
        scan_type: str,
        num_images: int,
    ) -> None:
        yolo_predictor = YOLO_SegmentationPredictor(model_path=yolo_model_path)
        pdf_listdir = [pdf for pdf in os.listdir(raw_dir) if pdf.endswith("pdf")]
        num_saved = 0

        while num_images != num_saved:
            rand_image, rand_image_name, rand_page_idx = (
                self.pdf_handler.get_random_image(
                    pdf_listdir=pdf_listdir, pdf_dir=raw_dir
                )
            )
            rand_image = self.img_processor.process(
                image=rand_image, scan_type=scan_type
            )
            pred_result = yolo_predictor(image=rand_image)
            points_dict = pred_result.id2polygons
            rand_points_idx, rand_points = self.label_handler._get_random_points(
                classes_dict=pred_result.id2label,
                points_dict=points_dict,
                target_classes=["question"],
            )
            rand_image = self.image_handler.crop_image(
                image=rand_image, points=rand_points
            )
            num_saved = self._save_image(
                rand_page_idx,
                rand_points_idx,
                output_dir=train_dir,
                image=rand_image,
                image_name=rand_image_name,
                num_saved=num_saved,
                num_images=num_images,
            )
