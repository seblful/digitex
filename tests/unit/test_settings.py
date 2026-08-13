"""Tests for the settings tree: what a machine with no configuration gets.

Every group defaults to something usable, so most of what matters here is the
default itself. The rest is the handful of values that must fail loudly rather
than default quietly — an out-of-range dimension, and an environment name the
log renderer would not recognize.

``conftest`` clears the settings environment per test, so these assert the
code's own defaults rather than whatever this machine's ``.env`` carries.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable


class TestDatabaseSettings:
    def test_the_default_dsn_names_a_local_postgres(self) -> None:
        assert (
            str(DatabaseSettings().dsn)
            == "postgresql://digitex:digitex@localhost:5432/digitex"
        )

    def test_database_url_is_what_supplies_the_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 12-factor name, which is what Compose and CI both set."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example:5432/x")

        assert "db.example" in str(Settings.load().database.dsn)

    def test_sslmode_reaches_the_conninfo_when_set(self) -> None:
        assert "sslmode=require" in DatabaseSettings(sslmode="require").conninfo

    def test_an_unset_sslmode_stays_out_of_the_conninfo(self) -> None:
        """Libpq applies its own default; naming it here would override that."""
        assert "sslmode" not in DatabaseSettings().conninfo

    def test_both_timeouts_are_passed_to_the_server(self) -> None:
        settings = DatabaseSettings(
            statement_timeout_ms=1234, idle_in_transaction_timeout_ms=5678
        )

        assert "statement_timeout=1234" in settings.server_options
        assert "idle_in_transaction_session_timeout=5678" in settings.server_options


class TestPipelineValidation:
    """The values that are refused, rather than defaulted around."""

    def test_a_training_image_size_off_the_stride_is_refused(self) -> None:
        """YOLO strides by 32, so a size between two multiples cannot be used."""
        with pytest.raises(ValidationError):
            DataSettings(image_size=500)

    @pytest.mark.parametrize("size", [640, 1280], ids=["640", "1280"])
    def test_a_multiple_of_the_stride_is_accepted(self, size: int) -> None:
        assert DataSettings(image_size=size).image_size == size

    @pytest.mark.parametrize(
        "build",
        [
            lambda: ExtractionSettings(question_max_width=0),
            lambda: ExtractionSettings(question_max_height=0),
        ],
        ids=["width", "height"],
    )
    def test_a_question_image_cannot_be_zero_sized(
        self, build: Callable[[], ExtractionSettings]
    ) -> None:
        """A zero-sized crop would be written as an unopenable file."""
        with pytest.raises(ValidationError):
            build()


class TestPathsSettings:
    def test_data_root_defaults_beside_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``var/`` under the cwd — never derived from the package's location."""
        monkeypatch.chdir(tmp_path)

        assert PathsSettings().data_root == tmp_path.resolve() / "var"

    def test_a_relative_data_root_is_resolved_at_load_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So errors name an absolute path, and a later chdir cannot move it."""
        monkeypatch.chdir(tmp_path)

        settings = PathsSettings(data_root=Path("corpus"))

        assert settings.data_root.is_absolute()
        assert settings.books_dir == tmp_path.resolve() / "corpus" / "books"

    @pytest.mark.parametrize(
        ("attribute", "relative"),
        [
            ("books_dir", "books"),
            ("models_dir", "models"),
            ("extraction_dir", "extraction"),
            ("extraction_output_dir", "extraction/output"),
            ("extraction_model_path", "models/page.pt"),
            ("training_data_dir", "training/data"),
        ],
        ids=[
            "books",
            "models",
            "extraction",
            "extraction-output",
            "extraction-model",
            "training-data",
        ],
    )
    def test_every_path_hangs_off_the_data_root(
        self, tmp_path: Path, attribute: str, relative: str
    ) -> None:
        settings = PathsSettings(data_root=tmp_path)

        assert getattr(settings, attribute) == tmp_path.joinpath(*relative.split("/"))

    def test_question_images_default_to_the_extraction_output(
        self, tmp_path: Path
    ) -> None:
        """A laptop serves the tree the object_keys were written from."""
        settings = PathsSettings(data_root=tmp_path)

        assert settings.question_images_dir == settings.extraction_output_dir

    def test_questions_dir_overrides_where_images_are_served_from(
        self, tmp_path: Path
    ) -> None:
        """Production bind-mounts the corpus somewhere data_root cannot reach."""
        corpus = tmp_path / "mnt" / "questions"

        settings = PathsSettings(data_root=tmp_path, questions_dir=corpus)

        assert settings.question_images_dir == corpus


class TestSettingsComposition:
    def test_the_groups_no_deployed_code_reads_sit_behind_pipeline(self) -> None:
        """Grouped, not flat: reading the path says which layer owns the value."""
        settings = Settings()

        assert isinstance(settings.pipeline, PipelineSettings)
        assert isinstance(settings.pipeline.data, DataSettings)
        assert isinstance(settings.pipeline.extraction, ExtractionSettings)
        assert isinstance(settings.pipeline.label_studio, LabelStudioSettings)
        assert isinstance(settings.pipeline.openrouter, OpenRouterSettings)

        for name in ("data", "extraction", "label_studio", "openrouter"):
            assert not hasattr(settings, name), (
                f"{name} is still reachable flat on Settings"
            )


class TestEnvironmentName:
    @pytest.mark.parametrize("var", ["ENVIRONMENT", "APP_ENVIRONMENT"])
    def test_either_spelling_selects_the_environment(
        self, var: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """docker-compose sets ENVIRONMENT; the JSON log renderer reads this."""
        monkeypatch.setenv(var, "production")

        assert Settings.load().app.environment == "production"

    def test_a_near_miss_fails_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A value like "prod" must be refused, not defaulted around.

        As a plain string it would validate fine and silently ship console
        logs to the production collector — the one failure this field exists
        to prevent.
        """
        monkeypatch.setenv("ENVIRONMENT", "prod")

        with pytest.raises(ValidationError):
            Settings.load()

    def test_an_unconfigured_machine_is_a_development_one(self) -> None:
        assert Settings().app.environment == "development"


class TestGetSettings:
    def test_the_same_instance_is_handed_out_every_time(self) -> None:
        """Entry points resolve once and thread the result down."""
        assert get_settings() is get_settings()

    def test_reset_clears_the_cache(self) -> None:
        """Tests that repoint the environment need the next call to re-read."""
        first = get_settings()

        reset_settings_cache()

        assert get_settings() is not first
