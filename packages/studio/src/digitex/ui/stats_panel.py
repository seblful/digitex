"""The review window's second tab: what the subject has produced so far.

The same `ImageCensus` and `AnswerValidator` the removed ``count-questions``
and ``check-answers`` commands rendered, shown beside the page that is adding
to the tally rather than in a terminal after the run.

Counting walks the whole output tree, so the panel only recounts when it is
actually on screen and something has moved — see :meth:`StatsPanel.show`.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digitex.pipeline.audit.census import ImageCensus
    from digitex.pipeline.audit.validator import AnswerValidator, ValidationReport

# How many question keys to name before saying "and N more" — a year whose
# answers.json never arrived would otherwise print hundreds.
MAX_LISTED = 12

OK_COLOR = "#1f9d55"
OFF_COLOR = "#c00000"
# Matches the window's own secondary ink — light enough to rank below the
# counts, dark enough to still read as text.
MUTED = "#4b5563"


def _listed(keys: list[str]) -> str:
    shown = ", ".join(keys[:MAX_LISTED])
    rest = len(keys) - MAX_LISTED
    return f"{shown}, +{rest} more" if rest > 0 else shown


def format_validation(report: ValidationReport) -> str:
    """Render an answers report as plain text, the way check-answers did."""
    lines = [f"Answers for {report.subject}"]

    for year in report.years:
        if not year.answers_file_present:
            lines.append(f"\n{year.year}: answers.json NOT FOUND")
            continue
        if not year.answers_file_valid:
            lines.append(f"\n{year.year}: answers.json IS UNREADABLE")
            continue

        if year.has_mismatch:
            status = "MISMATCH"
        elif year.options_differ:
            status = "OPTIONS DIFFER"
        else:
            status = "OK"

        lines.append(f"\n{year.year}: {status}")
        lines.append(f"  A-part {year.a_count}, B-part {year.b_count}")
        lines.append(
            f"  in images {year.image_question_count},"
            f" in answers.json {year.answer_question_count}"
        )
        if year.options_with_differing_questions:
            lines.append(
                f"  options differing: {_listed(year.options_with_differing_questions)}"
            )
        if year.missing_in_answers:
            lines.append(f"  missing in answers: {_listed(year.missing_in_answers)}")
        if year.missing_in_images:
            lines.append(f"  missing in images: {_listed(year.missing_in_images)}")
        lines.append(
            f"  Part B 'Б': {year.part_b_coverage}"
            f" ({year.options_with_b}/{year.total_options} options)"
        )

    issues = report.total_issues
    lines.append(f"\n{issues} issue(s) found" if issues else "\nAll years match")
    return "\n".join(lines)


class StatsPanel(ttk.Frame):
    """Per-year option/part counts, plus the answer-key check.

    ``census`` and ``validator`` are both optional: without them the panel says
    so rather than pretending the corpus is empty.
    """

    def __init__(
        self,
        master: tk.Misc,
        subject: str = "",
        census: ImageCensus | None = None,
        validator: AnswerValidator | None = None,
    ) -> None:
        super().__init__(master, padding=8)
        self._subject = subject
        self._census = census
        self._validator = validator
        self._year = ""
        self._stale = True

        self.rowconfigure(1, weight=3)
        self.rowconfigure(4, weight=2)
        self.columnconfigure(0, weight=1)
        self._build()

    def _build(self) -> None:
        head = ttk.Frame(self)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)

        self._title = ttk.Label(head, text="", font=("TkDefaultFont", 10, "bold"))
        self._title.grid(row=0, column=0, sticky="w")
        ttk.Button(head, text="Recount", width=9, command=self._recount).grid(
            row=0, column=1, sticky="e"
        )

        tree_frame = ttk.Frame(self)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 8))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            tree_frame, columns=("images",), show="tree headings", height=12
        )
        self._tree.heading("#0", text="year / option / part")
        self._tree.heading("images", text="images")
        self._tree.column("#0", width=210, anchor="w")
        self._tree.column("images", width=70, anchor="e", stretch=False)
        self._tree.tag_configure("complete", foreground=OK_COLOR)
        self._tree.tag_configure("off", foreground=OFF_COLOR)
        self._tree.tag_configure("here", font=("TkDefaultFont", 9, "bold"))
        self._tree.grid(row=0, column=0, sticky="nsew")

        bar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=bar.set)

        ttk.Separator(self, orient="horizontal").grid(row=2, column=0, sticky="ew")

        answers = ttk.Frame(self)
        answers.grid(row=3, column=0, sticky="ew", pady=(8, 4))
        answers.columnconfigure(0, weight=1)
        ttk.Label(answers, text="answers.json vs the images", foreground=MUTED).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(answers, text="Check answers", command=self.check_answers).grid(
            row=0, column=1, sticky="e"
        )

        text_frame = ttk.Frame(self)
        text_frame.grid(row=4, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        self._text = tk.Text(
            text_frame,
            width=40,
            height=10,
            wrap="word",
            font=("TkFixedFont", 9),
            relief="flat",
            background="#f6f6f6",
            state="disabled",
        )
        self._text.grid(row=0, column=0, sticky="nsew")
        text_bar = ttk.Scrollbar(
            text_frame, orient="vertical", command=self._text.yview
        )
        text_bar.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=text_bar.set)

    # --- what the window drives ---

    def show(self, subject: str, year: str) -> None:
        """Bring the panel up to date for *subject*/*year*, recounting if needed.

        The window says what should be on screen; when to pay for a recount is
        this panel's own business. Counting walks the whole output tree, so it
        runs only when the target moved or a write was reported since the last
        look — flipping tabs over an unchanged corpus recounts nothing.
        """
        if (subject, year) != (self._subject, self._year):
            self._subject = subject
            self._year = year
            self._stale = True
        if self._stale:
            self._recount()

    def page_written(self) -> None:
        """Note that crops landed on disk; the next ``show`` recounts."""
        self._stale = True

    def _recount(self) -> None:
        """Recount the subject's output tree — what is on disk, this page aside."""
        self._stale = False
        self._tree.delete(*self._tree.get_children())

        if self._census is None:
            self._title.configure(text="no census available")
            return

        try:
            census = self._census.take(self._subject)
        except FileNotFoundError:
            self._title.configure(text=f"{self._subject}: nothing extracted yet")
            return

        self._title.configure(
            text=(
                f"{census.subject}: {census.images} images in {census.folders} folders"
            )
        )

        for year in census.years:
            here = year.year == self._year
            node = self._tree.insert(
                "",
                "end",
                text=f"{year.year} — {year.options} options",
                values=(year.images,),
                tags=("complete" if year.is_complete else "off",)
                + (("here",) if here else ()),
                open=here,
            )
            for part in year.parts:
                self._tree.insert(
                    node,
                    "end",
                    text=f"{part.option} / {part.part}",
                    values=(part.images,),
                    tags=("off",) if part.off_mode else (),
                )

    def check_answers(self) -> None:
        """Validate answers.json against the images, and show the report."""
        if self._validator is None:
            self._show("no validator available")
            return

        try:
            report = self._validator.validate(self._subject)
        except FileNotFoundError:
            self._show(f"{self._subject} has no extraction output yet")
            return

        self._show(format_validation(report))

    def _show(self, text: str) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", text)
        self._text.configure(state="disabled")
