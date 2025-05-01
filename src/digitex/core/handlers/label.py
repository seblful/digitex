import os
import random


class LabelHandler:
    @staticmethod
    def _read_points(label_path: str) -> dict[int, list[list[float]]]:
        points_dict = dict()
        with open(label_path, "r") as f:
            for line in f:
                # Get points
                data = line.strip().split()
                class_idx = int(data[0])
                points = [float(point) for point in data[1:]]

                # Append points to the list in dict
                if class_idx not in points_dict:
                    points_dict[class_idx] = []
                points_dict[class_idx].append(points)

        return points_dict

    @staticmethod
    def _get_random_points(
        classes_dict: dict[int, str],
        points_dict: dict[int, list],
        target_classes: list[str],
    ) -> tuple[int, list[float]]:
        # Create subset of dict with target classes
        points_dict = {
            k: points_dict[k] for k in points_dict if classes_dict[k] in target_classes
        }

        if not points_dict:
            return -1, []

        # Get random label
        rand_class_idx = random.choice(list(points_dict.keys()))

        # Get random points
        rand_points_idx = random.randint(0, len(points_dict[rand_class_idx]) - 1)
        rand_points = points_dict[rand_class_idx][rand_points_idx]

        return rand_points_idx, rand_points

    @staticmethod
    def get_random_label(image_name: str, labels_dir: str) -> str:
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        # Return None if label_path doesn't exist
        if not os.path.exists(label_path):
            return (None, None)

        return label_name, label_path

    @staticmethod
    def points_to_abs_polygon(
        points: list[float], image_width: int, image_height: int
    ) -> list[tuple[int, int]]:
        points = list(zip(points[::2], points[1::2]))
        points = [(int(x * image_width), int(y * image_height)) for x, y in points]

        return points

    def get_points(
        self,
        image_name: str,
        labels_dir: str,
        classes_dict: dict[int, str],
        target_classes: list[str],
    ) -> tuple[int, list[float]]:
        _, rand_label_path = self.get_random_label(
            image_name=image_name, labels_dir=labels_dir
        )
        points_dict = self._read_points(rand_label_path)

        rand_points_idx, rand_points = self._get_random_points(
            classes_dict=classes_dict,
            points_dict=points_dict,
            target_classes=target_classes,
        )

        return rand_points_idx, rand_points
