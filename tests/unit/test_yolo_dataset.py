"""Tests for the YOLO DatasetCreator — ``create()`` is the test surface.

The creator takes labelled images and a directory of image files, so every test
drives the real build and asserts on the returned :class:`Dataset` and the tree
it emitted. Where those annotations came from is `test_labeling_export`'s
business; nothing here mentions Label Studio.
"""

from pathlib import Path

import pytest
import yaml
from structlog.testing import capture_logs

from digitex.domain.annotations import AnnotatedImage, LabelledRegion
from digitex.domain.entities import NormalizedPolygon
from digitex.ml.yolo.dataset import DatasetCreator

LABEL_LINE = "0 0.100000 0.200000 0.500000 0.200000 0.500000 0.800000 0.100000 0.800000"


_POLYGON = NormalizedPolygon([(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)])


def _annotation(image_name: str, label: str = "question") -> AnnotatedImage:
    return AnnotatedImage(
        filename=image_name,
        regions=(LabelledRegion(label=label, polygon=_POLYGON),),
    )


def _creator(
    tmp_path: Path,
    annotations: list[AnnotatedImage],
    *,
    train_split: float = 0.8,
    images: tuple[str, ...] = (),
    seed: int = 0,
    dataset_dir_name: str = "dataset",
) -> DatasetCreator:
    """Build a creator rooted in tmp_path, seeding the image files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    for name in images:
        (images_dir / name).write_bytes(b"fake image data")

    return DatasetCreator(
        annotations=annotations,
        images_dir=images_dir,
        dataset_dir=tmp_path / dataset_dir_name,
        train_split=train_split,
        seed=seed,
    )


def _all_files(dataset_dir: Path, suffix: str) -> set[str]:
    return {p.name for p in dataset_dir.rglob(f"*{suffix}")}


class TestCreate:
    def test_creates_all_three_splits_and_data_yaml(self, tmp_path: Path) -> None:
        names = tuple(f"image{i}.jpg" for i in range(10))
        creator = _creator(
            tmp_path,
            [_annotation(n) for n in names],
            train_split=0.7,
            images=names,
        )

        dataset = creator.create()

        assert dataset.dataset_dir == tmp_path / "dataset"
        for split in ("train", "val", "test"):
            assert (tmp_path / "dataset" / split).is_dir()
        assert (tmp_path / "dataset" / "data.yaml").exists()
        assert dataset.total == 10

    def test_split_counts_divide_the_remainder_sixty_forty(
        self, tmp_path: Path
    ) -> None:
        names = tuple(f"image{i}.jpg" for i in range(10))
        creator = _creator(
            tmp_path,
            [_annotation(n) for n in names],
            train_split=0.8,
            images=names,
        )

        dataset = creator.create()

        # 80% train; the remaining 20% splits 60/40 into val/test.
        assert (dataset.train, dataset.val, dataset.test) == (8, 1, 1)

    @staticmethod
    def _split_of(dataset_dir: Path) -> dict[str, str]:
        """Which split each image landed in."""
        return {path.name: path.parent.name for path in dataset_dir.rglob("*.jpg")}

    def test_the_same_seed_deals_the_same_split(self, tmp_path: Path) -> None:
        """An unseeded shuffle re-deals on every build.

        A model trained before the rebuild has then seen part of the test split
        it is about to be scored against, and no two runs are comparable.
        """
        names = tuple(f"image{i}.jpg" for i in range(20))
        annotations = [_annotation(n) for n in names]

        first = _creator(tmp_path, annotations, images=names).create()
        second = _creator(
            tmp_path, annotations, images=names, dataset_dir_name="rebuild"
        ).create()

        assert self._split_of(first.dataset_dir) == self._split_of(second.dataset_dir)

    def test_another_seed_deals_another_split(self, tmp_path: Path) -> None:
        """The split is fixed, not hardcoded — a new seed reshuffles on purpose."""
        names = tuple(f"image{i}.jpg" for i in range(20))
        annotations = [_annotation(n) for n in names]

        first = _creator(tmp_path, annotations, images=names).create()
        other = _creator(
            tmp_path, annotations, images=names, seed=7, dataset_dir_name="rebuild"
        ).create()

        assert self._split_of(first.dataset_dir) != self._split_of(other.dataset_dir)

    def test_every_annotated_image_is_copied_exactly_once(self, tmp_path: Path) -> None:
        names = tuple(f"image{i}.jpg" for i in range(6))
        creator = _creator(tmp_path, [_annotation(n) for n in names], images=names)

        creator.create()

        assert _all_files(tmp_path / "dataset", ".jpg") == set(names)

    def test_a_duplicate_basename_is_reported_not_silent(self, tmp_path: Path) -> None:
        """Two batches can both hold an image1.jpg; the later one wins the key.

        The overwrite itself is unchanged — what must not happen is the first
        one's polygons vanishing with no trace in the log. The export reader
        hands both over precisely so this is the layer that notices.
        """
        entries = [
            _annotation("image1.jpg", label="question"),
            _annotation("image1.jpg", label="option"),
        ]
        creator = _creator(tmp_path, entries, images=("image1.jpg",))

        with capture_logs() as logs:
            dataset = creator.create()

        assert "duplicate_image_basename" in [log["event"] for log in logs]
        assert dataset.total == 1

    def test_an_image_with_no_regions_ships_without_a_label_file(
        self, tmp_path: Path
    ) -> None:
        """A page an annotator opened and drew nothing on is a negative example.

        It must still be copied — the model needs pages with nothing on them —
        but a label file holding no lines is not a valid YOLO instance.
        """
        creator = _creator(
            tmp_path,
            [AnnotatedImage(filename="blank.jpg")],
            train_split=1.0,
            images=("blank.jpg",),
        )

        dataset = creator.create()

        assert dataset.train == 1
        assert (tmp_path / "dataset" / "train" / "blank.jpg").exists()
        assert not (tmp_path / "dataset" / "train" / "blank.txt").exists()

    def test_derives_a_sorted_class_map(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [
                _annotation("image1.jpg", label="question"),
                _annotation("image2.jpg", label="option"),
            ],
            images=("image1.jpg", "image2.jpg"),
        )

        dataset = creator.create()

        assert dataset.classes == {0: "option", 1: "question"}

    def test_writes_a_yolo_label_file_beside_each_image(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [_annotation("image0.jpg")],
            train_split=1.0,
            images=("image0.jpg",),
        )

        creator.create()

        label_path = tmp_path / "dataset" / "train" / "image0.txt"
        assert label_path.read_text() == LABEL_LINE

    def test_reports_annotated_images_missing_from_disk(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [_annotation("present.jpg"), _annotation("absent.jpg")],
            train_split=1.0,
            images=("present.jpg",),
        )

        dataset = creator.create()

        assert dataset.train == 1
        assert dataset.missing_images == ("absent.jpg",)
        assert not (tmp_path / "dataset" / "train" / "absent.jpg").exists()
        assert not (tmp_path / "dataset" / "train" / "absent.txt").exists()

    def test_no_annotations_still_produces_the_tree(self, tmp_path: Path) -> None:
        dataset = _creator(tmp_path, []).create()

        assert dataset.total == 0
        assert dataset.classes == {}
        assert dataset.missing_images == ()
        assert (tmp_path / "dataset" / "data.yaml").exists()

    def test_data_yaml_names_the_class_map_and_split_dirs(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [
                _annotation("image1.jpg", label="question"),
                _annotation("image2.jpg", label="option"),
            ],
            images=("image1.jpg", "image2.jpg"),
        )

        creator.create()

        data = yaml.safe_load((tmp_path / "dataset" / "data.yaml").read_text())
        assert data["names"] == {0: "option", 1: "question"}
        assert (data["train"], data["val"], data["test"]) == ("train", "val", "test")

    def test_data_yaml_path_is_absolute_when_outside_cwd(self, tmp_path: Path) -> None:
        """tmp_path sits outside cwd, so ``path`` must not be a failed relative_to."""
        _creator(tmp_path, []).create()

        data = yaml.safe_load((tmp_path / "dataset" / "data.yaml").read_text())
        assert Path(data["path"]) == tmp_path / "dataset"


class TestDatasetValue:
    def test_total_sums_the_splits(self, tmp_path: Path) -> None:
        names = tuple(f"image{i}.jpg" for i in range(5))
        dataset = _creator(
            tmp_path,
            [_annotation(n) for n in names],
            train_split=0.6,
            images=names,
        ).create()

        assert dataset.total == dataset.train + dataset.val + dataset.test
        assert dataset.total == 5

    def test_is_immutable(self, tmp_path: Path) -> None:
        dataset = _creator(tmp_path, []).create()

        with pytest.raises(AttributeError):
            dataset.train = 99  # ty: ignore[invalid-assignment]
