"""Tests for configuration settings module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from digitex.config.settings import (
    DatabaseSettings,
    DataSettings,
    ExtractionSettings,
    LabelStudioSettings,
    PathsSettings,
    Settings,
    TrainingSettings,
    get_settings,
)


class TestDatabaseSettings:
    """Test DatabaseSettings class."""

    def test_default_dsn(self) -> None:
        """Default DSN points at the conventional local Postgres."""
        settings = DatabaseSettings()
        assert str(settings.dsn).startswith("postgresql://")
        assert "localhost" in str(settings.dsn)

    def test_custom_dsn(self) -> None:
        settings = DatabaseSettings(
            dsn="postgresql://u:p@db.example:5432/x"  # type: ignore[arg-type]
        )
        assert str(settings.dsn).startswith("postgresql://")
        assert "db.example" in str(settings.dsn)

    def test_conninfo_appends_sslmode(self) -> None:
        settings = DatabaseSettings(sslmode="require")
        assert "sslmode=require" in settings.conninfo

    def test_server_options_includes_timeouts(self) -> None:
        settings = DatabaseSettings(
            statement_timeout_ms=1234, idle_in_transaction_timeout_ms=5678
        )
        assert "statement_timeout=1234" in settings.server_options
        assert "idle_in_transaction_session_timeout=5678" in settings.server_options


class TestTrainingSettings:
    """Test TrainingSettings class."""

    def test_default_training_values(self) -> None:
        """Test that TrainingSettings has correct default values."""
        settings = TrainingSettings()
        assert settings.runs_dir_name == "runs"

    def test_custom_training_values(self) -> None:
        """Test custom training values."""
        settings = TrainingSettings(runs_dir_name="custom_runs")
        assert settings.runs_dir_name == "custom_runs"


class TestDataSettings:
    """Test DataSettings class."""

    def test_default_data_values(self) -> None:
        """Test that DataSettings has correct default values."""
        settings = DataSettings()
        assert settings.image_size == 1280
        assert settings.dataset_dir_name == "dataset"
        assert settings.images_dir_name == "images"

    def test_custom_data_values(self) -> None:
        """Test custom data values."""
        settings = DataSettings(image_size=512)
        assert settings.image_size == 512

    def test_image_size_multiple_of_32(self) -> None:
        """Test that image_size must be a multiple of 32."""
        settings = DataSettings(image_size=640)
        assert settings.image_size == 640

    def test_image_size_not_multiple_of_32(self) -> None:
        """Test that image_size not multiple of 32 raises validation error."""
        with pytest.raises(ValidationError):
            DataSettings(image_size=500)


class TestExtractionSettings:
    """Test ExtractionSettings class."""

    def test_default_extraction_values(self) -> None:
        """Test that ExtractionSettings has correct default values."""
        settings = ExtractionSettings()
        assert settings.question_max_width == 2000
        assert settings.question_max_height == 2000
        assert settings.image_format == "jpg"

    def test_custom_extraction_values(self) -> None:
        """Test custom extraction values."""
        settings = ExtractionSettings(
            question_max_width=1000,
            question_max_height=1500,
            image_format="png",
        )
        assert settings.question_max_width == 1000
        assert settings.question_max_height == 1500
        assert settings.image_format == "png"

    def test_positive_validation(self) -> None:
        """Test that positive validation works for dimensions."""
        with pytest.raises(ValidationError):
            ExtractionSettings(question_max_width=0)

        with pytest.raises(ValidationError):
            ExtractionSettings(question_max_height=0)


class TestLabelStudioSettings:
    """Test LabelStudioSettings class."""

    def test_default_label_studio_values(self) -> None:
        """Test that LabelStudioSettings has correct default URL."""
        settings = LabelStudioSettings()
        assert settings.url == "http://localhost:8080"

    def test_custom_label_studio_values(self) -> None:
        """Test custom Label Studio values."""
        settings = LabelStudioSettings(
            url="http://custom:9000",
            api_key="test-key",
        )
        assert settings.url == "http://custom:9000"
        assert settings.api_key == "test-key"


class TestPathsSettings:
    """Test PathsSettings class."""

    def test_root_dir(self) -> None:
        """Test that root_dir returns current working directory."""
        settings = PathsSettings()
        assert isinstance(settings.root_dir, Path)
        assert settings.root_dir.exists()

    def test_training_dir(self) -> None:
        """Test that training_dir is computed correctly."""
        settings = PathsSettings()
        assert settings.training_dir == settings.root_dir / "training"

    def test_books_dir(self) -> None:
        """Test that books_dir is computed correctly."""
        settings = PathsSettings()
        assert settings.books_dir == settings.root_dir / "books"

    def test_extraction_dir(self) -> None:
        """Test that extraction_dir is computed correctly."""
        settings = PathsSettings()
        assert settings.extraction_dir == settings.root_dir / "extraction"


class TestSettings:
    """Test main Settings class."""

    def test_settings_composition(self) -> None:
        """Test that Settings composes all sub-settings correctly."""
        settings = Settings()
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.training, TrainingSettings)
        assert isinstance(settings.data, DataSettings)
        assert isinstance(settings.paths, PathsSettings)
        assert isinstance(settings.extraction, ExtractionSettings)
        assert isinstance(settings.label_studio, LabelStudioSettings)

    def test_settings_load_method(self) -> None:
        """Test Settings.load() class method."""
        settings = Settings.load()
        assert isinstance(settings, Settings)

    @patch.dict(
        "os.environ",
        {"DATABASE_URL": "postgresql://u:p@example.test:5432/d"},
    )
    def test_environment_variable_loading(self) -> None:
        """DATABASE_URL feeds DatabaseSettings.dsn."""
        settings = Settings.load()
        assert "example.test" in str(settings.database.dsn)


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
        assert hasattr(settings, "database")
        assert hasattr(settings, "training")
        assert hasattr(settings, "data")
        assert hasattr(settings, "paths")
        assert hasattr(settings, "extraction")
        assert hasattr(settings, "label_studio")
