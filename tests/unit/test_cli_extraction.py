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

from digitex.cli import extraction
from digitex.config import OpenRouterSettings, PathsSettings, Settings
from digitex.core.domain import OPTIONS_PER_BOOK

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A Settings rooted in tmp_path, installed for the CLI's get_settings.

    Every value a test asserts on is set here rather than inherited: something
    earlier in the session may have called ``Settings.load()``, which pulls the
    developer's ``.env`` into ``os.environ`` where pydantic-settings finds it.
    """
    resolved = Settings(
        paths=PathsSettings(root_dir=tmp_path),
        openrouter=OpenRouterSettings(api_key=""),
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


class TestCountQuestions:
    def test_unknown_subject_exits_nonzero(self, settings: Settings) -> None:
        result = runner.invoke(extraction.app, ["count-questions", "biology"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_empty_subject_says_so(self, settings: Settings) -> None:
        (settings.paths.extraction_output_dir / "biology").mkdir(parents=True)

        result = runner.invoke(extraction.app, ["count-questions", "biology"])

        assert result.exit_code == 0
        assert "No images found" in result.output

    def test_reports_options_and_totals(self, settings: Settings) -> None:
        _seed_output(settings, "biology", "2020", OPTIONS_PER_BOOK)

        result = runner.invoke(extraction.app, ["count-questions", "biology"])

        assert result.exit_code == 0
        assert f"2020: {OPTIONS_PER_BOOK} options" in result.output
        assert f"Total: {OPTIONS_PER_BOOK} images" in result.output

    def test_flags_a_year_missing_options(self, settings: Settings) -> None:
        _seed_output(settings, "biology", "2020", 3)

        result = runner.invoke(extraction.app, ["count-questions", "biology"])

        assert result.exit_code == 0
        assert "2020: 3 options" in result.output


class TestRenumberQuestions:
    def test_unknown_subject_exits_nonzero(self, settings: Settings) -> None:
        result = runner.invoke(extraction.app, ["renumber-questions", "biology"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_already_sequential_says_so(self, settings: Settings) -> None:
        _seed_output(settings, "biology", "2020", 1)

        result = runner.invoke(extraction.app, ["renumber-questions", "biology"])

        assert result.exit_code == 0
        assert "already sequential" in result.output


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


class TestCheckAnswers:
    def test_unknown_subject_exits_nonzero(self, settings: Settings) -> None:
        result = runner.invoke(extraction.app, ["check-answers", "biology"])

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_reports_a_year_without_answers(self, settings: Settings) -> None:
        _seed_output(settings, "biology", "2020", 1)

        result = runner.invoke(extraction.app, ["check-answers", "biology"])

        assert result.exit_code == 0
        assert "CHECKING ANSWERS FOR: biology" in result.output
        assert "answers.json NOT FOUND" in result.output
        assert "issue(s) found" in result.output


class TestAddQuestionsManually:
    def test_missing_manual_directory_exits_nonzero(self, settings: Settings) -> None:
        result = runner.invoke(extraction.app, ["add-questions-manually", "biology"])

        assert result.exit_code == 1
        assert "Manual directory" in result.output
