"""Tests for the review window's stats tab.

`format_validation` is what the removed ``check-answers`` command used to
print. It is pure text over a report, so it is checked without a display.
The panel's own contract — ``show`` recounts only when something moved — is
checked against a real Tk widget and skips where there is no display.
"""

from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from digitex.pipeline.audit.validator import ValidationReport, YearReport
from digitex.ui.stats_panel import MAX_LISTED, StatsPanel, format_validation

if TYPE_CHECKING:
    from collections.abc import Iterator


def _year(
    *,
    answers_file_present: bool = True,
    answers_file_valid: bool = True,
    missing_in_answers: list[str] | None = None,
    missing_in_images: list[str] | None = None,
    options_with_differing_questions: list[str] | None = None,
) -> YearReport:
    """A year that lines up, unless the caller says which way it does not."""
    return YearReport(
        year="2016",
        answers_file_present=answers_file_present,
        answers_file_valid=answers_file_valid,
        a_count=30,
        b_count=10,
        image_question_count=40,
        answer_question_count=40,
        missing_in_answers=missing_in_answers or [],
        missing_in_images=missing_in_images or [],
        options_with_differing_questions=options_with_differing_questions or [],
        options_with_b=10,
        total_options=10,
    )


class TestFormatValidation:
    def test_a_matching_year_reads_as_ok(self) -> None:
        text = format_validation(ValidationReport(subject="biology", years=[_year()]))

        assert "2016: OK" in text
        assert "All years match" in text

    def test_a_missing_answers_file_is_named_before_anything_else(self) -> None:
        text = format_validation(
            ValidationReport(
                subject="biology", years=[_year(answers_file_present=False)]
            )
        )

        assert "answers.json NOT FOUND" in text
        assert "A-part" not in text  # nothing to count against

    def test_an_unreadable_answers_file_says_so(self) -> None:
        text = format_validation(
            ValidationReport(subject="biology", years=[_year(answers_file_valid=False)])
        )

        assert "answers.json IS UNREADABLE" in text

    def test_a_mismatch_lists_what_is_missing_on_each_side(self) -> None:
        text = format_validation(
            ValidationReport(
                subject="biology",
                years=[
                    _year(missing_in_answers=["1/A/3"], missing_in_images=["2/B/7"])
                ],
            )
        )

        assert "2016: MISMATCH" in text
        assert "missing in answers: 1/A/3" in text
        assert "missing in images: 2/B/7" in text
        assert "1 issue(s) found" in text

    def test_a_long_list_of_keys_is_cut_short(self) -> None:
        """A year whose answers.json never arrived would print hundreds."""
        missing = [f"1/A/{n}" for n in range(1, 40)]

        text = format_validation(
            ValidationReport(
                subject="biology", years=[_year(missing_in_answers=missing)]
            )
        )

        assert f"+{len(missing) - MAX_LISTED} more" in text
        assert "1/A/13" not in text

    def test_options_that_differ_are_reported_without_a_count_mismatch(self) -> None:
        text = format_validation(
            ValidationReport(
                subject="biology",
                years=[_year(options_with_differing_questions=["3", "7"])],
            )
        )

        assert "2016: OPTIONS DIFFER" in text
        assert "options differing: 3, 7" in text


class _FakeCensus:
    """Records take() calls; the panel's staleness protocol is under test."""

    def __init__(self) -> None:
        self.takes: list[str] = []

    def take(self, subject: str) -> Any:
        self.takes.append(subject)
        return SimpleNamespace(subject=subject, images=0, folders=0, years=[])


@pytest.fixture(scope="module")
def root() -> Iterator[tk.Tk]:
    """One interpreter for the module — same reasoning as test_ui_page_review."""
    try:
        made = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    made.withdraw()
    yield made
    made.destroy()


class TestShowRecountsOnlyWhenSomethingMoved:
    """``show`` is the whole protocol: when to pay a recount is the panel's own."""

    @staticmethod
    def _panel(root: tk.Tk) -> tuple[StatsPanel, _FakeCensus]:
        census = _FakeCensus()
        return StatsPanel(root, census=cast("Any", census)), census

    def test_the_first_show_recounts(self, root: tk.Tk) -> None:
        panel, census = self._panel(root)

        panel.show("biology", "2016")

        assert census.takes == ["biology"]

    def test_showing_an_unchanged_target_recounts_nothing(self, root: tk.Tk) -> None:
        """Flipping tabs over an unchanged corpus must not walk the tree again."""
        panel, census = self._panel(root)

        panel.show("biology", "2016")
        panel.show("biology", "2016")

        assert census.takes == ["biology"]

    def test_a_reported_write_makes_the_next_show_recount(self, root: tk.Tk) -> None:
        panel, census = self._panel(root)
        panel.show("biology", "2016")

        panel.page_written()
        panel.show("biology", "2016")

        assert census.takes == ["biology", "biology"]

    def test_a_new_target_recounts(self, root: tk.Tk) -> None:
        panel, census = self._panel(root)
        panel.show("biology", "2016")

        panel.show("biology", "2017")

        assert census.takes == ["biology", "biology"]
