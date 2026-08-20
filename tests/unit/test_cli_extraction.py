"""Tests for the extraction CLI.

The commands are adapters: settings in, a collaborator built, its result
rendered. These tests drive them through Typer's runner with a Settings whose
paths point into ``tmp_path``, which is only possible because the module reads
nothing at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from digitex.config import (
    OpenRouterSettings,
    PathsSettings,
    PipelineSettings,
    Settings,
)
from digitex.studio.cli import extraction

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A Settings rooted in tmp_path, installed for the CLI's get_settings.

    Every value a test asserts on is set here rather than defaulted, so the
    paths point somewhere disposable. ``conftest`` keeps the environment out of
    it; this fixture decides what goes in.
    """
    resolved = Settings(
        paths=PathsSettings(data_root=tmp_path),
        pipeline=PipelineSettings(openrouter=OpenRouterSettings(api_key="")),
    )
    monkeypatch.setattr(extraction, "get_settings", lambda: resolved)
    # The Typer callback configures logging from the real settings; keep it
    # from touching the developer's logs/ directory.
    monkeypatch.setattr(extraction, "setup_logging", lambda _settings: None)
    return resolved


def _seed_output(settings: Settings, subject: str, year: str, options: int) -> None:
    root = settings.paths.extraction_output_dir / subject / year
    for option in range(1, options + 1):
        part_dir = root / str(option) / "A"
        part_dir.mkdir(parents=True)
        (part_dir / "1.jpg").write_bytes(b"x")


class TestExtractQuestions:
    def test_missing_model_is_reported_not_raised(self, settings: Settings) -> None:
        """A missing model exits like every other failure, not with a traceback."""
        result = runner.invoke(extraction.app, ["extract-questions", "biology"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "Model" in result.output or "model" in result.output


class TestExtractAnswers:
    def test_missing_api_key_is_reported_not_raised(self, settings: Settings) -> None:
        result = runner.invoke(extraction.app, ["extract-answers", "biology"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "API key" in result.output


class TestCommandSurface:
    def test_counting_and_renumbering_are_the_review_window_s_job_now(self) -> None:
        """They were removed, not renamed — a stale habit should fail loudly."""
        for gone in ("count-questions", "check-answers", "renumber-questions"):
            assert runner.invoke(extraction.app, [gone, "biology"]).exit_code != 0
