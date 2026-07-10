"""Tests for the ManualExtractor class."""

from pathlib import Path

import pytest
from PIL import Image

from digitex.extractors.exceptions import InvalidFilenameError
from digitex.extractors.manual_extractor import ManualExtractor


@pytest.fixture
def extractor(tmp_path: Path) -> ManualExtractor:
    manual_dir = tmp_path / "manual"
    output_dir = tmp_path / "output"
    manual_dir.mkdir()
    output_dir.mkdir()
    return ManualExtractor(
        image_format="jpg",
        question_max_width=2000,
        question_max_height=2000,
        manual_dir=manual_dir,
        output_dir=output_dir,
    )


class TestManualExtractorFilename:
    """Filename parsing and validation."""

    def test_parse_valid_filename(self, extractor: ManualExtractor) -> None:
        result = extractor._parse_filename(Path("2016_3_A_20.png"))
        assert result == (2016, 3, "A", 20)

    def test_parse_valid_filename_part_b(self, extractor: ManualExtractor) -> None:
        result = extractor._parse_filename(Path("2020_5_B_15.png"))
        assert result == (2020, 5, "B", 15)

    @pytest.mark.parametrize(
        "name",
        ["2016_3_C_20.png", "2016_3_A_.png", "invalid.png", "2016_3_20.png"],
        ids=["bad-part", "missing-question", "no-structure", "missing-part"],
    )
    def test_parse_invalid_filename_raises(
        self, extractor: ManualExtractor, name: str
    ) -> None:
        with pytest.raises(InvalidFilenameError, match="Invalid filename format"):
            extractor._parse_filename(Path(name))


class TestManualExtractorPreprocessing:
    """Image preprocessing."""

    def test_preprocess_returns_rgb(self, extractor: ManualExtractor) -> None:
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        result = extractor._preprocess(img)
        assert result.mode == "RGB"

    def test_preprocess_resizes_large_image(self, extractor: ManualExtractor) -> None:
        img = Image.new("RGBA", (3000, 3000), (255, 255, 255, 255))
        result = extractor._preprocess(img)
        assert result.width <= 2000
        assert result.height <= 2000

    def test_preprocess_preserves_aspect_ratio(
        self, extractor: ManualExtractor
    ) -> None:
        img = Image.new("RGBA", (1000, 500), (255, 255, 255, 255))
        result = extractor._preprocess(img)
        assert result.width == 2000
        assert result.height == 1000

    def test_preprocess_converts_black_background_to_white(
        self, extractor: ManualExtractor
    ) -> None:
        img = Image.new("RGB", (500, 500), (0, 0, 0))
        result = extractor._preprocess(img)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)
        assert result.getpixel((250, 250)) == (255, 255, 255)

    def test_preprocess_converts_transparent_to_white(
        self, extractor: ManualExtractor
    ) -> None:
        img = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
        result = extractor._preprocess(img)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)


class TestManualExtractorRenumbering:
    """File renumbering when a manual image is inserted mid-sequence."""

    @pytest.fixture
    def target_dir(self, extractor: ManualExtractor) -> Path:
        """Seed the output tree with numbered images 1-3 and 20-22."""
        assert extractor.output_dir is not None
        target_dir = extractor.output_dir / "biology" / "2016" / "3" / "A"
        target_dir.mkdir(parents=True)
        for i in [1, 2, 3, 20, 21, 22]:
            Image.new("RGB", (100, 100), (255, 255, 255)).save(target_dir / f"{i}.jpg")
        return target_dir

    def test_get_existing_images_sorted_by_number(
        self, extractor: ManualExtractor, target_dir: Path
    ) -> None:
        images = extractor._get_existing_images(target_dir)
        assert len(images) == 6
        assert images[0] == (1, target_dir / "1.jpg")
        assert images[-1] == (22, target_dir / "22.jpg")

    def test_renumber_files_shifts_from_start_number(
        self, extractor: ManualExtractor, target_dir: Path
    ) -> None:
        changes = extractor._renumber_files(target_dir, start_num=20, dry_run=False)

        assert len(changes) == 3
        # 20->21, 21->22, 22->23; 1-3 untouched
        assert not (target_dir / "20.jpg").exists()
        assert (target_dir / "21.jpg").exists()
        assert (target_dir / "22.jpg").exists()
        assert (target_dir / "23.jpg").exists()
        assert (target_dir / "1.jpg").exists()
        assert (target_dir / "2.jpg").exists()
        assert (target_dir / "3.jpg").exists()

    def test_renumber_files_dry_run_previews_without_moving(
        self, extractor: ManualExtractor, target_dir: Path
    ) -> None:
        changes = extractor._renumber_files(target_dir, start_num=20, dry_run=True)

        assert len(changes) == 3
        assert (target_dir / "20.jpg").exists()
        assert (target_dir / "21.jpg").exists()
        assert (target_dir / "22.jpg").exists()


class TestManualExtractorProcessAll:
    """The full processing pipeline."""

    @pytest.fixture
    def manual_image(self, extractor: ManualExtractor) -> Path:
        """Drop one manual image into the extractor's manual directory."""
        assert extractor.manual_dir is not None
        subject_dir = extractor.manual_dir / "biology"
        subject_dir.mkdir()
        manual_file = subject_dir / "2016_3_A_20.png"
        Image.new("RGBA", (500, 500), (255, 255, 255, 255)).save(manual_file)
        return manual_file

    def test_process_all_moves_image_into_output_tree(
        self, extractor: ManualExtractor, manual_image: Path
    ) -> None:
        extractor.process_all(dry_run=False)

        assert not manual_image.exists()
        assert extractor.output_dir is not None
        output_file = extractor.output_dir / "biology" / "2016" / "3" / "A" / "20.jpg"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_process_all_dry_run_leaves_files_alone(
        self, extractor: ManualExtractor, manual_image: Path
    ) -> None:
        extractor.process_all(dry_run=True)

        assert manual_image.exists()
        assert extractor.output_dir is not None
        output_file = extractor.output_dir / "biology" / "2016" / "3" / "A" / "20.jpg"
        assert not output_file.exists()
