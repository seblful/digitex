"""Tests for configuration settings module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from digitex.config.settings import (
    AppSettings,
    DatabaseSettings,
    TrainingSettings,
    PathsSettings,
    Settings,
    get_settings,
)


class TestAppSettings:
    """Test AppSettings class."""

    def test_default_values(self) -> None:
        """Test that AppSettings has correct default values."""
        settings = AppSettings()
        assert settings.render_scale == 3
        assert settings.crop_offset == 0.025
        assert settings.log_level == "INFO"

    def test_render_scale_validation(self) -> None:
        """Test that render_scale validation works correctly."""
        settings = AppSettings(render_scale=5)
        assert settings.render_scale == 5

    def test_render_scale_below_minimum(self) -> None:
        """Test that render_scale below minimum raises validation error."""
        with pytest.raises(Exception):
            AppSettings(render_scale=0)

    def test_render_scale_above_maximum(self) -> None:
        """Test that render_scale above maximum raises validation error."""
        with pytest.raises(Exception):
            AppSettings(render_scale=11)

    def test_crop_offset_validation(self) -> None:
        """Test that crop_offset validation works correctly."""
        settings = AppSettings(crop_offset=0.5)
        assert settings.crop_offset == 0.5

    def test_crop_offset_negative(self) -> None:
        """Test that negative crop_offset raises validation error."""
        with pytest.raises(Exception):
            AppSettings(crop_offset=-0.1)

    def test_crop_offset_above_one(self) -> None:
        """Test that crop_offset above 1.0 raises validation error."""
        with pytest.raises(Exception):
            AppSettings(crop_offset=1.1)


class TestDatabaseSettings:
    """Test DatabaseSettings class."""

    def test_default_database_path(self) -> None:
        """Test default database path."""
        settings = DatabaseSettings()
        assert settings.path == "data/tests.db"

    def test_custom_database_path(self) -> None:
        """Test custom database path."""
        settings = DatabaseSettings(path="custom/path.db")
        assert settings.path == "custom/path.db"


class TestTrainingSettings:
    """Test TrainingSettings class."""

    def test_default_training_values(self) -> None:
        """Test that TrainingSettings has correct default values."""
        settings = TrainingSettings()
        assert settings.num_epochs == 100
        assert settings.image_size == 640
        assert settings.batch_size == 4
        assert settings.overlap_mask is False
        assert settings.patience == 50
        assert settings.seed == 42
        assert settings.data_subdir == "page"
        assert settings.model_type == "seg"
        assert settings.model_size == "m"

    def test_custom_training_values(self) -> None:
        """Test custom training values."""
        settings = TrainingSettings(
            num_epochs=50,
            batch_size=32,
            image_size=512,
        )
        assert settings.num_epochs == 50
        assert settings.batch_size == 32
        assert settings.image_size == 512

    def test_image_size_multiple_of_32(self) -> None:
        """Test that image_size must be a multiple of 32."""
        settings = TrainingSettings(image_size=640)
        assert settings.image_size == 640

    def test_image_size_not_multiple_of_32(self) -> None:
        """Test that image_size not multiple of 32 raises validation error."""
        with pytest.raises(Exception):
            TrainingSettings(image_size=500)

    def test_positive_validation(self) -> None:
        """Test that positive validation works for various fields."""
        with pytest.raises(Exception):
            TrainingSettings(num_epochs=0)

        with pytest.raises(Exception):
            TrainingSettings(batch_size=0)

        with pytest.raises(Exception):
            TrainingSettings(patience=0)


class TestPathsSettings:
    """Test PathsSettings class."""

    def test_home_dir(self) -> None:
        """Test that home_dir returns current working directory."""
        settings = PathsSettings()
        assert isinstance(settings.home_dir, Path)
        assert settings.home_dir.exists()

    def test_data_dir(self) -> None:
        """Test that data_dir is computed correctly."""
        settings = PathsSettings()
        assert settings.data_dir == settings.home_dir / "data"

    def test_dataset_dir(self) -> None:
        """Test that dataset_dir is computed correctly."""
        settings = PathsSettings()
        assert settings.dataset_dir == settings.data_dir / "dataset"

    def test_model_dir(self) -> None:
        """Test that model_dir is computed correctly."""
        settings = PathsSettings()
        assert settings.model_dir == settings.home_dir / "models"

    def test_raw_data_dir(self) -> None:
        """Test that raw_data_dir is computed correctly."""
        settings = PathsSettings()
        assert settings.raw_data_dir == settings.home_dir / "raw-data"


class TestSettings:
    """Test main Settings class."""

    def test_settings_composition(self) -> None:
        """Test that Settings composes all sub-settings correctly."""
        settings = Settings()
        assert isinstance(settings.app, AppSettings)
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.training, TrainingSettings)
        assert isinstance(settings.paths, PathsSettings)

    def test_settings_load_method(self) -> None:
        """Test Settings.load() class method."""
        settings = Settings.load()
        assert isinstance(settings, Settings)

    @patch.dict("os.environ", {"APP_LOG_LEVEL": "DEBUG"})
    def test_environment_variable_loading(self) -> None:
        """Test that settings can be loaded from environment variables."""
        settings = Settings.load()
        assert settings.app.log_level == "DEBUG"


class TestGetSettings:
    """Test get_settings singleton function."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_singleton(self) -> None:
        """Test that get_settings returns the same instance on multiple calls."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert id(settings1) == id(settings2)

    def test_get_settings_has_all_categories(self) -> None:
        """Test that get_settings returns settings with all categories."""
        settings = get_settings()
        assert hasattr(settings, "app")
        assert hasattr(settings, "database")
        assert hasattr(settings, "training")
        assert hasattr(settings, "paths")
