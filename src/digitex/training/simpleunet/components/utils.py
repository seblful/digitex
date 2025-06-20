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
