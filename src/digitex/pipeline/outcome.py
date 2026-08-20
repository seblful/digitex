"""What an extraction run produced, as values a caller can act on.

The type this replaces carried ``success: bool``, three counters, two lists of
formatted strings and a ``dict[str, Any]``, and its own docstring had to warn
that ``success=True`` did not mean everything succeeded. Nothing downstream
could ask *which* pages failed or *what* was kept without parsing prose, and
merging two runs shallow-merged their metadata dicts, so two books sharing a
key silently lost one.

Three things happen during a book that are not "a page was extracted", and
each is now a value rather than a sentence:

- a **collision** — the slot was taken, so the existing file was kept. Not a
  failure: replaying an interrupted year meets its own output on every page.
- a **failure** — the page raised and produced nothing.
- **unfinished pieces** — a question was still open when the book ended, so
  nothing was written for it.

A report holds those, and answers questions about itself. Formatting them for
a terminal is the CLI's job, which is the point: the report says what happened,
the renderer says how it reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from digitex.domain.placement import QuestionPlacement


@dataclass(frozen=True)
class Collision:
    """A question whose slot was already taken, so the existing file stayed."""

    page: str
    placement: QuestionPlacement

    def __str__(self) -> str:
        return (
            f"{self.page}: {self.placement} already extracted, kept the existing image"
        )


@dataclass(frozen=True)
class PageFailure:
    """A page that raised. Nothing was written for it."""

    page: str
    cause: str

    def __str__(self) -> str:
        return f"Failed to process {self.page}: {self.cause}"


@dataclass(frozen=True)
class UnfinishedPieces:
    """Question pieces still held when the book ended, joined to nothing."""

    page: str
    count: int

    def __str__(self) -> str:
        return (
            f"{self.page}: a question piece was left unfinished,"
            " nothing was written for it"
        )


@dataclass(frozen=True)
class BookReport:
    """One book's run.

    ``pages`` counts pages that came through, not questions written — a page
    of nothing but markers is processed and writes no file.
    """

    pages: int = 0
    collisions: tuple[Collision, ...] = ()
    failures: tuple[PageFailure, ...] = ()
    unfinished: tuple[UnfinishedPieces, ...] = ()
    note: str = ""
    """Why a book produced nothing, when that is not an error — an empty
    directory, say. Empty for every book that had pages to read."""

    @property
    def clean(self) -> bool:
        """True when every page came through. Collisions do not spoil a run."""
        return not self.failures

    @property
    def complete(self) -> bool:
        """True when this year may be recorded as finished.

        A clean run over at least one page. An empty book directory reports a
        clean run over nothing and must not be marked done, or the year is
        never retried.
        """
        return self.clean and self.pages > 0


@dataclass(frozen=True)
class YearReport:
    """One year of a subject, and how its book went."""

    year: str
    book: BookReport


@dataclass(frozen=True)
class SubjectReport:
    """Every year of one subject that this run touched.

    ``skipped`` names years already recorded complete, which are not opened at
    all — distinct from a year that ran and wrote nothing.
    """

    years: tuple[YearReport, ...] = ()
    skipped: tuple[str, ...] = ()

    @property
    def extracted(self) -> int:
        return len(self.years)

    @property
    def clean(self) -> bool:
        return all(year.book.clean for year in self.years)

    @property
    def collisions(self) -> list[Collision]:
        return [c for year in self.years for c in year.book.collisions]

    @property
    def failures(self) -> list[PageFailure]:
        return [f for year in self.years for f in year.book.failures]

    @property
    def unfinished(self) -> list[UnfinishedPieces]:
        return [u for year in self.years for u in year.book.unfinished]

    @property
    def notes(self) -> list[str]:
        return [year.book.note for year in self.years if year.book.note]


@dataclass(frozen=True)
class SubjectRefused:
    """The run never started: there was nothing to extract, or nowhere to look.

    Separate from a report with no years, because the two mean opposite things
    to a caller — one is "your archive is missing", the other "nothing left to
    do". The old type spelled both ``success=False`` and ``success=True`` with
    zero counters respectively, and callers had to know which was which.
    """

    reason: str


SubjectOutcome = SubjectReport | SubjectRefused
"""What extracting a subject produced, or why it could not begin."""


@dataclass(frozen=True)
class AnswersReport:
    """One subject's answer-key extraction.

    ``years`` is how many years' keys were written — the number that used to
    live in a ``metadata`` dict under a string key every caller had to guess,
    with a default in case it was not there.
    """

    years: int = 0
    sheets: int = 0
    failures: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def clean(self) -> bool:
        return not self.failures


def messages(items: Iterable[object]) -> list[str]:
    """Render outcome values for a terminal, in the order they happened.

    The one place a report becomes prose. Kept here beside the values so the
    wording of a collision lives next to what a collision is, rather than
    being reinvented at each of the three commands that report one.
    """
    return [str(item) for item in items]
