"""Tests for configuration settings module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from digitex.config import (
    DatabaseSettings,
    DataSettings,
    ExtractionSettings,
    LabelStudioSettings,
    OpenRouterSettings,
    PathsSettings,
    PipelineSettings,
    Settings,
    get_settings,
    reset_settings_cache,
)


class TestDatabaseSettings:
    """Test DatabaseSettings class."""

    def test_default_dsn(self) -> None:
        """Default DSN points at the conventional local Postgres."""
        settings = DatabaseSettings()
        assert str(settings.dsn).startswith("postgresql://")
        assert "localhost" in str(settings.dsn)

    def test_custom_dsn(self) -> None:
        settings = DatabaseSettings.model_validate(
            {"dsn": "postgresql://u:p@db.example:5432/x"}
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

    def test_data_root_defaults_beside_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``var/`` under the cwd — never derived from the package's location."""
        monkeypatch.chdir(tmp_path)
        assert PathsSettings().data_root == tmp_path.resolve() / "var"

    def test_relative_data_root_is_resolved_at_load_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So errors name an absolute path, and a later chdir cannot move it."""
        monkeypatch.chdir(tmp_path)
        settings = PathsSettings(data_root=Path("corpus"))

        assert settings.data_root.is_absolute()
        assert settings.books_dir == tmp_path.resolve() / "corpus" / "books"

    def test_books_dir(self, tmp_path: Path) -> None:
        settings = PathsSettings(data_root=tmp_path)
        assert settings.books_dir == tmp_path / "books"

    def test_extraction_output_dir(self, tmp_path: Path) -> None:
        settings = PathsSettings(data_root=tmp_path)
        assert settings.extraction_output_dir == tmp_path / "extraction" / "output"

    def test_extraction_model_path(self, tmp_path: Path) -> None:
        settings = PathsSettings(data_root=tmp_path)
        assert settings.extraction_model_path == tmp_path / "models" / "page.pt"

    def test_training_data_dir(self, tmp_path: Path) -> None:
        settings = PathsSettings(data_root=tmp_path)
        assert settings.training_data_dir == tmp_path / "training" / "data"

    def test_question_images_dir_defaults_to_the_extraction_output(
        self, tmp_path: Path
    ) -> None:
        """A laptop serves the tree the object_keys were written from."""
        settings = PathsSettings(data_root=tmp_path)
        assert settings.question_images_dir == settings.extraction_output_dir

    def test_questions_dir_overrides_it(self, tmp_path: Path) -> None:
        """Production bind-mounts the corpus somewhere data_root cannot reach."""
        corpus = tmp_path / "mnt" / "questions"
        settings = PathsSettings(data_root=tmp_path, questions_dir=corpus)

        assert settings.question_images_dir == corpus


class TestSettings:
    """Test main Settings class."""

    def test_settings_composition(self) -> None:
        """Test that Settings composes all sub-settings correctly."""
        settings = Settings()
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.paths, PathsSettings)
        assert isinstance(settings.pipeline, PipelineSettings)

    def test_pipeline_only_settings_sit_behind_pipeline(self) -> None:
        """The groups no deployed code reads are grouped, not flat on Settings."""
        settings = Settings()
        assert isinstance(settings.pipeline.data, DataSettings)
        assert isinstance(settings.pipeline.extraction, ExtractionSettings)
        assert isinstance(settings.pipeline.label_studio, LabelStudioSettings)
        assert isinstance(settings.pipeline.openrouter, OpenRouterSettings)

        for name in ("data", "extraction", "label_studio", "openrouter"):
            assert not hasattr(settings, name), (
                f"{name} is still reachable flat on Settings"
            )

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

    @pytest.mark.parametrize("var", ["ENVIRONMENT", "APP_ENVIRONMENT"])
    def test_either_spelling_reaches_app_environment(self, var: str) -> None:
        """docker-compose sets ENVIRONMENT; the JSON log renderer reads this."""
        with patch.dict("os.environ", {var: "production"}):
            assert Settings.load().app.environment == "production"


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
        for name in ("app", "bot", "database", "logging", "paths", "timezone"):
            assert hasattr(settings, name)
        assert hasattr(settings.pipeline, "extraction")
        assert hasattr(settings.pipeline, "label_studio")

    def test_reset_clears_the_cache(self) -> None:
        """Tests that repoint the environment need the next call to re-read."""
        first = get_settings()
        reset_settings_cache()

        assert get_settings() is not first
