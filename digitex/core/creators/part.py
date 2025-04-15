import os
from .base import BaseDataCreator
from ..predictors.segmentation import YOLO_SegmentationPredictor


class PartDataCreator(BaseDataCreator):
    def extract_part_from_annotations(
        self,
        annotated_question_dir: str,
        output_dir: str,
        num_samples: int,
        target_classes: list[str] = ["answer", "number", "option", "question", "spec"],
    ) -> None:
        source_images_dir = os.path.join(annotated_question_dir, "images")
        annotation_dir = os.path.join(annotated_question_dir, "labels")
        classes_file = os.path.join(annotated_question_dir, "classes.txt")
        class_mapping = self._read_classes(classes_file)
        available_images = os.listdir(source_images_dir)
        processed_count = 0

        while num_samples != processed_count:
            source_image, image_filename = self.get_listdir_random_image(
                available_images, source_images_dir
            )
            part_idx, part_coords = self._get_points(
                image_name=image_filename,
                labels_dir=annotation_dir,
                classes_dict=class_mapping,
                target_classes=target_classes,
            )
            if not part_coords:
                continue
            part_absolute_coords = self._convert_points_to_polygon(
                points=part_coords,
                image_width=source_image.width,
                image_height=source_image.height,
            )
            cropped_part = self._crop_image(
                image=source_image, points=part_absolute_coords
            )
            processed_count = self._save_image(
                part_idx,
                output_dir=output_dir,
                image=cropped_part,
                image_name=image_filename,
                num_saved=processed_count,
                num_images=num_samples,
            )

    def predict_parts(
        self,
        raw_dir: str,
        train_dir: str,
        yolo_page_model_path: str,
        yolo_question_model_path: str,
        scan_type: str,
        num_images: int,
        target_classes: list[str] = ["answer", "number", "option", "question", "spec"],
    ) -> None:
        yolo_page_predictor = YOLO_SegmentationPredictor(yolo_page_model_path)
        yolo_question_predictor = YOLO_SegmentationPredictor(yolo_question_model_path)
        pdf_listdir = [pdf for pdf in os.listdir(raw_dir) if pdf.endswith("pdf")]
        num_saved = 0

        while num_images != num_saved:
            page_rand_image, rand_image_name, rand_page_idx = self.get_pdf_random_image(
                pdf_listdir, raw_dir
            )
            page_rand_image = self._process_image(
                image=page_rand_image, scan_type=scan_type
            )
            page_pred_result = yolo_page_predictor(page_rand_image)
            page_points_dict = page_pred_result.id2polygons
            question_rand_points_idx, question_rand_points = (
                self.label_handler._get_random_points(
                    classes_dict=page_pred_result.id2label,
                    points_dict=page_points_dict,
                    target_classes=["question"],
                )
            )
            question_rand_image = self._crop_image(
                image=page_rand_image, points=question_rand_points
            )
            question_pred_result = yolo_question_predictor(question_rand_image)
            question_points_dict = question_pred_result.id2polygons
            part_rand_points_idx, part_rand_points = (
                self.label_handler._get_random_points(
                    classes_dict=question_pred_result.id2label,
                    points_dict=question_points_dict,
                    target_classes=target_classes,
                )
            )
            if not part_rand_points:
                continue
            part_rand_image = self._crop_image(
                image=question_rand_image, points=part_rand_points
            )
            num_saved = self._save_image(
                rand_page_idx,
                question_rand_points_idx,
                part_rand_points_idx,
                output_dir=train_dir,
                image=part_rand_image,
                image_name=rand_image_name,
                num_saved=num_saved,
                num_images=num_images,
            )
