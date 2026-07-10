"""Tests for the answers.json / image-tree alignment validator."""

import json
from pathlib import Path

import pytest

from digitex.services.answer_validator import AnswerValidator


def _write_year(
    year_dir: Path,
    answers: dict[str, dict[str, str]] | None,
    images: list[str],
) -> None:
    """Lay out one year of extraction output: images plus optional answers.json."""
    for rel in images:
        image_path = year_dir / rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
    year_dir.mkdir(parents=True, exist_ok=True)
    if answers is not None:
        (year_dir / "answers.json").write_text(
            json.dumps(answers, ensure_ascii=False), encoding="utf-8"
        )


class TestAnswerValidator:
    def test_missing_subject_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AnswerValidator(tmp_path).validate("biology")

    def test_clean_year(self, tmp_path: Path) -> None:
        _write_year(
            tmp_path / "biology" / "2016",
            answers={"1": {"A1": "3", "B1": "Б"}},
            images=["1/A/1.jpg", "1/B/1.jpg"],
        )

        report = AnswerValidator(tmp_path).validate("biology")

        assert report.subject == "biology"
        assert report.total_issues == 0
        (year,) = report.years
        assert year.is_clean
        assert year.year == "2016"
        assert year.a_count == 1
        assert year.b_count == 1
        assert year.image_question_count == 2
        assert year.answer_question_count == 2

    def test_missing_answers_file_flags_year(self, tmp_path: Path) -> None:
        _write_year(
            tmp_path / "biology" / "2016",
            answers=None,
            images=["1/A/1.jpg"],
        )

        report = AnswerValidator(tmp_path).validate("biology")

        (year,) = report.years
        assert year.answers_file_present is False
        assert not year.is_clean
        assert report.total_issues == 1

    def test_mismatched_images_and_answers(self, tmp_path: Path) -> None:
        _write_year(
            tmp_path / "biology" / "2016",
            answers={"1": {"A1": "1", "A3": "3"}},
            images=["1/A/1.jpg", "1/A/2.jpg"],
        )

        report = AnswerValidator(tmp_path).validate("biology")

        (year,) = report.years
        assert year.missing_in_answers == ["A2"]
        assert year.missing_in_images == ["A3"]
        assert year.has_mismatch
        assert not year.is_clean

    def test_options_with_differing_question_sets(self, tmp_path: Path) -> None:
        _write_year(
            tmp_path / "biology" / "2016",
            answers={"1": {"A1": "1"}, "2": {"A2": "2"}},
            images=["1/A/1.jpg", "1/A/2.jpg"],
        )

        report = AnswerValidator(tmp_path).validate("biology")

        (year,) = report.years
        assert year.options_with_differing_questions == ["2"]
        assert year.options_differ
        assert not year.is_clean

    def test_part_b_answers_without_cyrillic_marker_flagged(
        self, tmp_path: Path
    ) -> None:
        _write_year(
            tmp_path / "biology" / "2016",
            answers={"1": {"A1": "1", "B1": "123"}},
            images=["1/A/1.jpg", "1/B/1.jpg"],
        )

        report = AnswerValidator(tmp_path).validate("biology")

        (year,) = report.years
        assert year.options_with_b == 0
        assert year.total_options == 1
        assert not year.is_clean

    def test_years_sorted_and_counted_across_report(self, tmp_path: Path) -> None:
        _write_year(
            tmp_path / "biology" / "2017",
            answers={"1": {"A1": "1", "B1": "Б"}},
            images=["1/A/1.jpg", "1/B/1.jpg"],
        )
        _write_year(
            tmp_path / "biology" / "2016",
            answers=None,
            images=["1/A/1.jpg"],
        )

        report = AnswerValidator(tmp_path).validate("biology")

        assert [y.year for y in report.years] == ["2016", "2017"]
        assert report.total_issues == 1
