"""Tests for the ManualExtractor class."""

from pathlib import Path

import pytest
from PIL import Image

from digitex.extractors.manual_extractor import ManualExtractor


class TestManualExtractorFilename:
    """Test filename parsing and validation."""

    @pytest.fixture
    def extractor(self, tmp_path: Path) -> ManualExtractor:
        """Create a ManualExtractor instance."""
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

    def test_parse_valid_filename(self, extractor: ManualExtractor) -> None:
        """Test parsing a valid filename."""
        result = extractor._parse_filename(Path("2016_3_A_20.png"))
        assert result == (2016, 3, "A", 20)

    def test_parse_valid_filename_part_b(self, extractor: ManualExtractor) -> None:
        """Test parsing a valid filename with part B."""
        result = extractor._parse_filename(Path("2020_5_B_15.png"))
        assert result == (2020, 5, "B", 15)

    def test_parse_invalid_part(self, extractor: ManualExtractor) -> None:
        """Test that invalid part raises InvalidFilenameError."""
        from digitex.extractors.exceptions import InvalidFilenameError
        
        with pytest.raises(InvalidFilenameError, match="Invalid filename format"):
            extractor._parse_filename(Path("2016_3_C_20.png"))

    def test_parse_missing_question(self, extractor: ManualExtractor) -> None:
        """Test that missing question number raises InvalidFilenameError."""
        from digitex.extractors.exceptions import InvalidFilenameError
        
        with pytest.raises(InvalidFilenameError, match="Invalid filename format"):
            extractor._parse_filename(Path("2016_3_A_.png"))

    def test_parse_invalid_format(self, extractor: ManualExtractor) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid filename format"):
            extractor._parse_filename(Path("invalid.png"))

    def test_parse_missing_part(self, extractor: ManualExtractor) -> None:
        """Test that missing part raises ValueError."""
        with pytest.raises(ValueError, match="Invalid filename format"):
            extractor._parse_filename(Path("2016_3_20.png"))


class TestManualExtractorPreprocessing:
    """Test image preprocessing."""

    @pytest.fixture
    def extractor(self, tmp_path: Path) -> ManualExtractor:
        """Create a ManualExtractor instance."""
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

    def test_preprocess_returns_rgb(self, extractor: ManualExtractor) -> None:
        """Test that preprocessing returns an RGB image."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        result = extractor._preprocess(img)
        assert result.mode == "RGB"

    def test_preprocess_resizes_large_image(self, extractor: ManualExtractor) -> None:
        """Test that preprocessing resizes images larger than max dimensions."""
        img = Image.new("RGBA", (3000, 3000), (255, 255, 255, 255))
        result = extractor._preprocess(img)
        assert result.width <= 2000
        assert result.height <= 2000

    def test_preprocess_preserves_aspect_ratio(
        self, extractor: ManualExtractor
    ) -> None:
        """Test that preprocessing preserves aspect ratio when resizing."""
        img = Image.new("RGBA", (1000, 500), (255, 255, 255, 255))
        result = extractor._preprocess(img)
        assert result.width == 2000
        assert result.height == 1000

    def test_preprocess_converts_black_background_to_white(
        self, extractor: ManualExtractor
    ) -> None:
        """Test that preprocessing converts black backgrounds to white."""
        img = Image.new("RGB", (500, 500), (0, 0, 0))
        result = extractor._preprocess(img)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)
        assert result.getpixel((250, 250)) == (255, 255, 255)

    def test_preprocess_converts_transparent_to_white(
        self, extractor: ManualExtractor
    ) -> None:
        """Test that preprocessing converts transparent backgrounds to white."""
        img = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
        result = extractor._preprocess(img)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)


class TestManualExtractorRenumbering:
    """Test file renumbering functionality."""

    @pytest.fixture
    def extractor(self, tmp_path: Path) -> ManualExtractor:
        """Create a ManualExtractor instance with test files."""
        manual_dir = tmp_path / "manual"
        base_output_dir = tmp_path / "output"
        target_dir = base_output_dir / "biology" / "2016" / "3" / "A"
        manual_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)

        # Create test files
        for i in [1, 2, 3, 20, 21, 22]:
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            img.save(target_dir / f"{i}.jpg")

        return ManualExtractor(
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            manual_dir=manual_dir,
            output_dir=base_output_dir,
        )

    def test_get_existing_images(self, extractor: ManualExtractor) -> None:
        """Test getting existing images from directory."""
        assert extractor.output_dir is not None
        target_dir = extractor.output_dir / "biology" / "2016" / "3" / "A"
        images = extractor._get_existing_images(target_dir)
        assert len(images) == 6
        assert images[0] == (1, target_dir / "1.jpg")
        assert images[-1] == (22, target_dir / "22.jpg")

    def test_renumber_files_shifts_correctly(self, extractor: ManualExtractor) -> None:
        """Test that renumbering shifts files correctly."""
        assert extractor.output_dir is not None
        target_dir = extractor.output_dir / "biology" / "2016" / "3" / "A"
        changes = extractor._renumber_files(target_dir, start_num=20, dry_run=False)

        # Check that changes were returned
        assert len(changes) == 3

        # Check that files were renamed: 20->21, 21->22, 22->23
        assert not (target_dir / "20.jpg").exists()
        assert (target_dir / "21.jpg").exists()
        assert (target_dir / "22.jpg").exists()
        assert (target_dir / "23.jpg").exists()

        # Check that non-shifted files are unchanged
        assert (target_dir / "1.jpg").exists()
        assert (target_dir / "2.jpg").exists()
        assert (target_dir / "3.jpg").exists()

    def test_renumber_files_dry_run(self, extractor: ManualExtractor) -> None:
        """Test that dry run doesn't modify files."""
        assert extractor.output_dir is not None
        target_dir = extractor.output_dir / "biology" / "2016" / "3" / "A"
        changes = extractor._renumber_files(target_dir, start_num=20, dry_run=True)

        # Check that changes were returned but files weren't modified
        assert len(changes) == 3
        assert (target_dir / "20.jpg").exists()
        assert (target_dir / "21.jpg").exists()
        assert (target_dir / "22.jpg").exists()


class TestManualExtractorProcessAll:
    """Test the full processing pipeline."""

    @pytest.fixture
    def setup_test_files(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Set up test files for processing."""
        manual_dir = tmp_path / "manual" / "biology"
        output_dir = tmp_path / "output"
        manual_dir.mkdir(parents=True)
        output_dir.mkdir()

        # Create a test manual image
        img = Image.new("RGBA", (500, 500), (255, 255, 255, 255))
        manual_file = manual_dir / "2016_3_A_20.png"
        img.save(manual_file)

        return tmp_path, manual_dir, output_dir

    def test_process_all_processes_files(
        self, setup_test_files: tuple[Path, Path, Path]
    ) -> None:
        """Test that process_all processes all manual files."""
        tmp_path, manual_dir, output_dir = setup_test_files

        extractor = ManualExtractor(
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            manual_dir=manual_dir.parent,
            output_dir=output_dir,
        )
        extractor.process_all(dry_run=False)

        # Check that manual file was deleted
        assert not (manual_dir / "2016_3_A_20.png").exists()

        # Check that output file was created
        output_file = output_dir / "biology" / "2016" / "3" / "A" / "20.jpg"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_process_all_dry_run(
        self, setup_test_files: tuple[Path, Path, Path]
    ) -> None:
        """Test that dry run doesn't modify files."""
        tmp_path, manual_dir, output_dir = setup_test_files

        extractor = ManualExtractor(
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            manual_dir=manual_dir.parent,
            output_dir=output_dir,
        )
        extractor.process_all(dry_run=True)

        # Check that manual file still exists
        assert (manual_dir / "2016_3_A_20.png").exists()

        # Check that output file was not created
        output_file = output_dir / "biology" / "2016" / "3" / "A" / "20.jpg"
        assert not output_file.exists()
