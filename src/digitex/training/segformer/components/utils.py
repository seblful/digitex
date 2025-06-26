import yaml

import torch
from transformers import SegformerForSemanticSegmentation

from digitex.settings import settings

from digitex.training.superpoint.components.annotation import (
    RelativeKeypoint,
    AbsoluteKeypoint,
    RelativeKeypointsObject,
    AbsoluteKeypointsObject,
)


def create_abs_kps_from_label(
    label: list[list],
    clip: bool,
    img_width: int = None,
    img_height: int = None,
) -> list[AbsoluteKeypoint]:
    kps = []

    for value in label:
        kp = AbsoluteKeypoint(value[0], value[1], int(value[2]))
        if clip:
            kp.clip(img_width, img_height)
        kps.append(kp)

    return kps


def create_rel_kps_from_label(label: list[list], clip: bool) -> list[RelativeKeypoint]:
    kps = []

    for value in label:
        kp = RelativeKeypoint(value[0], value[1], int(value[2]))
        if clip:
            kp.clip()
        kps.append(kp)

    return kps


def create_abs_kps_obj_from_label(
    label: list[list],
    clip: bool,
    img_width: int = None,
    img_height: int = None,
    num_keypoints: int = None,
) -> AbsoluteKeypointsObject:
    if not label:
        return AbsoluteKeypointsObject(0, [], 0)

    kps = create_abs_kps_from_label(label, clip, img_width, img_height)
    num_keypoints = num_keypoints if num_keypoints is not None else len(kps)
    kps_obj = AbsoluteKeypointsObject(0, kps, num_keypoints)

    return kps_obj


def create_rel_kps_obj_from_label(
    label: list[list], clip: bool
) -> RelativeKeypointsObject:
    if not label:
        return RelativeKeypointsObject(0, [], 0)

    kps = create_rel_kps_from_label(label, clip)
    kps_obj = RelativeKeypointsObject(
        0,
        kps,
        len(kps),
    )

    return kps_obj


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(config: dict) -> SegformerForSemanticSegmentation:
    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["model_name"],
        num_labels=config["dataset"]["num_classes"],
        id2label=config["model"]["id2label"],
        label2id=config["model"]["label2id"],
        ignore_mismatched_sizes=True,
    )

    # Load checkpoint
    checkpoint = torch.load(
        config["trainer"]["checkpoint_path"],
        map_location=settings.DEVICE,
        weights_only=False,
    )

    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(settings.DEVICE)
    model.eval()

    return model


def save_model(model: SegformerForSemanticSegmentation, config: dict) -> None:
    model.save_pretrained(config["trainer"]["checkpoint_dir"])


def convert_model(config: dict) -> None:
    model = load_model(config)
    save_model(model, config)
