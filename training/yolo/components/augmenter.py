import os
from PIL import Image

import numpy as np
import cv2

from tqdm import tqdm

import supervision as sv
import albumentations as A

from modules.handlers import LabelHandler
from modules.processors import FileProcessor

from .dataset import DatasetCreator
from .converter import Converter
from .annotation import Keypoint, KeypointsObject, AnnotationCreator
from .utils import get_random_img


class Augmenter:
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.classes_path = os.path.join(raw_dir, 'classes.txt')

        self.img_ext = ".jpg"
        self.anns_ext = ".txt"

        self._transforms = None
        self._augmenter = None

        self.__id2label = None
        self.__label2id = None

    @property
    def transforms(self) -> A.Compose:
        if self._transforms is None:
            self._transforms = [
                A.AdditiveNoise(p=0.3),
                A.Downscale(scale_range=[0.4, 0.9], p=0.3),
                A.RGBShift(p=0.3),
                A.RingingOvershoot(p=0.3),
                A.Spatter(mean=[0.5, 0.6], p=0.2),
                A.ToGray(p=0.4),
                A.ChannelShuffle(p=0.3),
                A.Emboss(p=0.3),
                A.GaussNoise(std_range=[0.05, 0.15], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.MedianBlur(p=0.3),
                A.PlanckianJitter(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomShadow(shadow_intensity_range=[0.1, 0.4], p=0.3),
                A.SaltAndPepper(amount=[0.01, 0.03], p=0.2),
                A.GaussianBlur(blur_limit=6, p=0.3),
                A.ISONoise(p=0.2),
                A.MotionBlur(p=0.3),
                A.PlasmaBrightnessContrast(p=0.3),
                A.RandomFog(p=0.3),
                A.Sharpen(p=0.4),
                A.Blur(p=0.3),
                A.Illumination(p=0.3),
                A.CLAHE(p=0.3),
                A.Posterize(p=0.3),
                A.Affine(scale=[0.92, 1.08], fill=255, p=0.4),
                A.CoarseDropout(fill=255, p=0.1),
                A.Pad(padding=[15, 15], fill=255, p=0.4),
                A.RandomScale(p=0.4),
                A.SafeRotate(limit=(-3, 3), fill=255, p=0.4)
            ]

        return self._transforms

    @property
    def augmenter(self) -> None:
        return self._augmenter

    @augmenter.setter
    def augmenter(self, value) -> None:
        self._augmenter = value

    @property
    def id2label(self) -> dict[int, str]:
        if self.__id2label is None:
            classes = DatasetCreator.read_classes_file(self.classes_path)
            self.__id2label = {k: v for k, v in enumerate(classes)}

        return self.__id2label

    @property
    def label2id(self) -> dict[str, int]:
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def find_name(self, img_name: str) -> str:
        name = os.path.splitext(img_name)[0]

        increment = 1
        while True:
            aug_name = f"{name}_aug_{increment}"
            filename = f"{aug_name}{self.img_ext}"
            filepath = os.path.join(self.train_dir, filename)
            if not os.path.exists(filepath):
                return aug_name
            increment += 1

    def save_anns(self) -> None:
        pass

    def save_image(self,
                   name: str,
                   img: np.ndarray) -> None:
        filename = f"{name}{self.img_ext}"
        filepath = os.path.join(self.train_dir, filename)

        # Save image
        image = Image.fromarray(img)
        image.save(filepath)

    def save(self,
             img_name,
             img: np.ndarray,
             points_dict: dict[int, list]) -> None:

        name = self.find_name(img_name)
        self.save_image(name, img)
        self.save_anns(name, points_dict)


class OBB_PolygonAugmenter(Augmenter):
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 anns_type: str) -> None:
        super().__init__(raw_dir, dataset_dir)
        self.anns_type = anns_type

        self.preprocess_funcs = {"polygon": Converter.point_to_polygon,
                                 "obb": Converter.xyxyxyxy_to_polygon}
        self.postprocess_funcs = {"polygon": Converter.polygon_to_point,
                                  "obb": Converter.polygon_to_xyxyxyxy}

        if anns_type not in self.preprocess_funcs:
            raise ValueError(
                f"anns_type must be one of {list(self.preprocess_funcs.keys())}.")

        self.preprocess_func = self.preprocess_funcs[anns_type]
        self.postprocess_func = self.postprocess_funcs[anns_type]

    @property
    def augmenter(self) -> A.Compose:
        if self._augmenter is None:
            augmenter = A.Compose(self.transforms)

        return augmenter

    def save_anns(self,
                  name: str,
                  points_dict: dict[int, list]) -> None:
        filename = f"{name}{self.anns_ext}"
        filepath = os.path.join(self.train_dir, filename)

        # Write each class and anns to txt
        with open(filepath, 'w') as file:
            if points_dict is None:
                return

            for class_idx, points in points_dict.items():
                for point in points:
                    point = [str(pts) for pts in point]
                    pts = " ".join(point)
                    line = f"{class_idx} {pts}\n"
                    file.write(line)

    def create_masks(self,
                     img_name: str,
                     img_width: int,
                     img_height: int) -> None | dict[int, list]:
        anns_name = os.path.splitext(img_name)[0] + '.txt'
        anns_path = os.path.join(self.train_dir, anns_name)

        points_dict = LabelHandler._read_points(anns_path)

        if not points_dict:
            return None

        masks_dict = {key: [] for key in points_dict.keys()}

        # Iterate through points, preprocess and convert to mask
        for class_idx, points in points_dict.items():
            for point in points:
                polygon = self.preprocess_func(point, img_width, img_height)

                # Convert polygon to mask
                mask = sv.polygon_to_mask(polygon, (img_width, img_height))

                masks_dict[class_idx].append(mask)

        return masks_dict

    def create_anns(self,
                    masks_dict: dict[int, list],
                    img_width: int,
                    img_height: int) -> None | dict[int, list]:
        if masks_dict is None:
            return None

        points_dict = {key: [] for key in masks_dict.keys()}

        # Iterate through masks and convert to anns
        for class_idx, masks in masks_dict.items():
            for mask in masks:
                polygons = sv.mask_to_polygons(mask)
                polygon = max(polygons, key=cv2.contourArea)
                anns = self.postprocess_func(
                    polygon, img_width, img_height)
                points_dict[class_idx].append(anns)

        return points_dict

    def augment_img(self,
                    img: np.ndarray,
                    masks_dict: dict[int, list] = None) -> tuple[np.ndarray, None] | tuple[np.ndarray, dict[int, list]]:

        # Case if no masks_dict
        if masks_dict is None:
            transf = self.augmenter(image=img)
            transf_img = transf['image']

            return (transf_img, None)

        # Obtain masks
        masks = []
        for v in masks_dict.values():
            masks.extend(v)

        # Transform
        transf = self.augmenter(image=img, masks=masks)
        transf_img = transf['image']
        transf_masks = transf['masks']

        # Create transf_masks_dict
        transf_masks_dict = {key: [] for key in masks_dict.keys()}
        i = 0
        for class_idx, masks in masks_dict.items():
            for _ in range(len(masks)):
                transf_masks_dict[class_idx].append(transf_masks[i])
                i += 1

        return transf_img, transf_masks_dict

    def augment(self,
                num_images: int) -> None:
        images_listdir = [img_name for img_name in os.listdir(
            self.train_dir) if img_name.endswith(".jpg")]

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            # Get random img
            img_name, img = get_random_img(self.train_dir, images_listdir)

            # Create masks
            orig_height, orig_width = img.shape[:2]
            masks_dict = self.create_masks(
                img_name, orig_width, orig_height)

            # Augment
            transf_img, transf_masks_dict = self.augment_img(img, masks_dict)
            transf_height, transf_width = transf_img.shape[:2]

            # Create anns
            transf_points_dict = self.create_anns(
                transf_masks_dict, transf_width, transf_height)
            self.save(img_name, transf_img, transf_points_dict)


class KeypointAugmenter(Augmenter):
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 anns_type: str) -> None:
        super().__init__(raw_dir, dataset_dir)

        self.anns_type = anns_type

        if anns_type != "keypoint":
            raise ValueError(
                f"anns_type must be 'keypoint'.")

    @property
    def augmenter(self) -> A.Compose:
        if self._augmenter is None:
            augmenter = A.Compose(self.transforms,
                                  keypoint_params=A.KeypointParams(format='xy',
                                                                   remove_invisible=False))

        return augmenter

    def save_anns(self,
                  name: str,
                  kps_objs: list[KeypointsObject]) -> None:
        filename = f"{name}{self.anns_ext}"
        filepath = os.path.join(self.train_dir, filename)

        keypoints_strs = []

        for kps_obj in kps_objs:
            keypoints_strs.append(kps_obj.to_string())

        FileProcessor.write_txt(filepath, lines=keypoints_strs)

    def create_kps_from_nums(self, nums: list[float]) -> list[Keypoint]:
        points = nums[5:]
        kps = []

        for i in range(0, len(points), 3):
            pts = points[i:i+3]
            kp = Keypoint(pts[0], pts[1], int(pts[2]))
            kps.append(kp)

        return kps

    def create_kps_from_coords(self,
                               coords: list[tuple],
                               img_width: int,
                               img_height: int) -> list[Keypoint]:
        kps = []
        for coord in coords:
            kp = Keypoint(int(coord[0]), int(coord[1]), 1)
            kp.clip(img_width, img_height)
            kps.append(kp)

        return kps

    def create_kps_objs_from_file(self,
                                  img_name: str) -> list[KeypointsObject]:
        anns_name = os.path.splitext(img_name)[0] + '.txt'
        anns_path = os.path.join(self.train_dir, anns_name)

        lines = FileProcessor.read_txt(anns_path)

        if not lines:
            return []

        kps_objs = []
        for line in lines:
            nums = line.strip().split()
            nums = list(map(float, nums))

            # Create keypoints
            kps = self.create_kps_from_nums(nums)

            # Create keypoints object
            kps_obj = KeypointsObject(class_idx=int(nums[0]),
                                      keypoints=kps,
                                      num_keypoints=len(kps),
                                      bbox_center=(nums[1], nums[2]),
                                      bbox_width=nums[3],
                                      bbox_height=nums[4])

            kps_objs.append(kps_obj)

        return kps_objs

    def create_kps_objs_from_coords(self,
                                    kps_objs: list[KeypointsObject],
                                    transf_coords: list[list],
                                    img_width: int,
                                    img_height: int) -> list[KeypointsObject]:
        if not transf_coords:
            return []

        transf_kps_objs = []

        coords_i = 0
        for kps_obj in kps_objs:
            num_vis = len(kps_obj.get_vis_coords())
            transf_coords = transf_coords[coords_i:num_vis]

            # Create kps from transformed and from original non-visible
            nonvis_kps = [kp for kp in kps_obj.keypoints[num_vis:]]
            transf_kps = self.create_kps_from_coords(
                transf_coords, img_width, img_height)
            kps = transf_kps + nonvis_kps

            # Create keypoints object
            transf_kps_obj = KeypointsObject(class_idx=kps_obj.class_idx,
                                             keypoints=kps,
                                             num_keypoints=kps_obj.num_keypoints)
            transf_kps_objs.append(transf_kps_obj)

            coords_i += num_vis

        return transf_kps_objs

    def augment_img(self,
                    img: np.ndarray,
                    kps_objs: list[KeypointsObject]) -> tuple[np.ndarray, list]:
        # Transform without keypoints
        if not kps_objs:
            transf = self.augmenter(image=img)
            transf_img = transf["image"]

            return (transf_img, [])

        # Retrieve all visible coordinates and save map
        coords = []
        for kps_obj in kps_objs:
            vis_coords = kps_obj.get_vis_coords()
            coords.extend(vis_coords)

        # Transform with keypoints
        transf = self.augmenter(image=img,
                                keypoints=coords)
        transf_img = transf['image']
        transf_coords = transf['keypoints']

        return transf_img, transf_coords

    def augment(self,
                num_images) -> None:
        images_listdir = [img_name for img_name in os.listdir(
            self.train_dir) if img_name.endswith(".jpg")]

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            # Get random img
            img_name, img = get_random_img(self.train_dir, images_listdir)
            orig_height, orig_width = img.shape[:2]

            # Create keypoints objects from file
            abs_kps_objs = self.create_kps_objs_from_file(img_name)
            rel_kps_objs = [kps_obj.to_relative(
                orig_width, orig_height, clip=True) for kps_obj in abs_kps_objs]

            # Augment
            transf_img, transf_coords = self.augment_img(img, rel_kps_objs)
            transf_height, transf_width = transf_img.shape[:2]

            # Create transformed keypoints objects from transf_coords
            transf_rel_kps_objs = self.create_kps_objs_from_coords(
                rel_kps_objs, transf_coords, transf_width, transf_height)
            transf_abs_kps_objs = [kps_obj.to_absolute(
                transf_width, transf_height) for kps_obj in transf_rel_kps_objs]

            # Save annotation
            self.save(img_name, transf_img, transf_abs_kps_objs)
