"""Domain types — the single home for entities and value objects.

`ExamType` and `QuestionKey` are value objects (immutable, no identity).
`Question`, `Session`, `Student`, `TestResult` are repository return-shapes
(Pydantic). Everything that crosses a module boundary should import from this
file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 — Pydantic needs runtime type
from typing import Final, Literal, NamedTuple, NewType

from pydantic import BaseModel, Field

ExamType = Literal["CE", "CT"]
Part = Literal["A", "B"]
RegistrationStatus = Literal["pending", "approved", "rejected"]

# A polygon crosses several coordinate spaces between the YOLO mask it starts
# as and the crop or Label Studio annotation it ends as. Percent and normalized
# points are both pairs of floats, so without distinct types a percent polygon
# converted a second time type-checks and silently divides by 10 000. Each
# space is its own type: whoever produces it wraps once, and every hop
# downstream then says which space it is in.
PixelPolygon = NewType("PixelPolygon", list[tuple[int, int]])
"""Points in source-image pixels — what a :class:`Detection` carries."""

PercentPolygon = NewType("PercentPolygon", list[list[float]])
"""Points as percentages (0-100) of the image size — Label Studio's space."""

NormalizedPolygon = NewType("NormalizedPolygon", list[tuple[float, float]])
"""Points scaled to 0-1 — the space YOLO label files are written in."""

_EXAM_TYPES: Final = ("CE", "CT")

# From 2023 each year's books exist in two exam variants: CE (options 1-5)
# and CT (the rest). Earlier years are CT only.
EXAM_TYPE_INTRO_YEAR: Final = 2023
_CE_MAX_OPTION: Final = 5

# A Book carries this many Options, interleaved across its Pages.
OPTIONS_PER_BOOK: Final = 10

# A Part A question offers this many numbered answers, and the option keyboard
# is built from it. Fixed across the corpus: ``books`` used to carry a per-book
# override that nothing ever wrote, so every question already had five.
PART_A_OPTION_COUNT: Final = 5

# The corpus is Russian throughout, so a hand-typed question key can carry the
# Cyrillic letters A and VE — indistinguishable on screen from Latin A and B,
# and rejected as a bad part letter without this fold.
_CYRILLIC_PART_LETTERS: Final = str.maketrans("АВ", "AB")  # noqa: RUF001


def normalize_option_number(raw: int) -> int:
    """Fold an absolute option number onto the 1..``OPTIONS_PER_BOOK`` range.

    Answer sheets and page markers number options in blocks — 1-10, then 11-20,
    then 21-30 — and every block is the same ten Options, so 11 and 21 both
    mean Option 1.
    """
    return (raw - 1) % OPTIONS_PER_BOOK + 1


def year_has_exam_types(year: int) -> bool:
    """Return True if the year's books split into CE and CT variants."""
    return year >= EXAM_TYPE_INTRO_YEAR


def exam_type_for(year: int, option_number: int) -> ExamType:
    """Return the exam type an option belongs to."""
    if year_has_exam_types(year) and option_number <= _CE_MAX_OPTION:
        return "CE"
    return "CT"


def parse_exam_type(raw: str) -> ExamType:
    """Narrow a string to an ``ExamType``, or raise.

    Raises:
        ValueError: If *raw* is not one of the two exam types.
    """
    if raw not in _EXAM_TYPES:
        raise ValueError(f"Unknown exam type {raw!r}; expected 'CE' or 'CT'")
    # No cast: membership in `_EXAM_TYPES` is what narrows `raw` to the literal,
    # so the checker already knows. A cast here would only hide it going wrong.
    return raw


@dataclass(frozen=True)
class Detection:
    """One thing the segmentation model found on a page.

    The label is already resolved against the model's class map and the polygon
    is in source-image pixels, so a consumer needs neither the class id nor the
    id-to-label mapping. One detection is one record, which is why there is no
    "same number of labels as polygons" invariant to check.

    The score is the model's confidence in this one region. Extraction ignores
    it — a region either placed or it didn't — but a pre-annotation carries it
    to Label Studio, where it is what an annotator sorts a review queue by.
    """

    label: str
    polygon: PixelPolygon
    score: float


@dataclass(frozen=True)
class QuestionKey:
    """Identifies a question within an option by part and number.

    Corresponds to keys in answers.json (e.g. "A1", "B12") and the
    filesystem path segment {part}/{number}.jpg.
    """

    part: Part
    number: int

    @classmethod
    def parse(cls, raw: str) -> QuestionKey:
        raw = raw.strip().upper().translate(_CYRILLIC_PART_LETTERS)
        if len(raw) < 2 or raw[0] not in ("A", "B") or not raw[1:].isdigit():
            raise ValueError(f"Invalid question key: {raw!r}")
        part: Part = "A" if raw[0] == "A" else "B"
        return cls(part=part, number=int(raw[1:]))

    def __str__(self) -> str:
        return f"{self.part}{self.number}"


class Student(BaseModel):
    """A Telegram user, and whether they may use the bot.

    Identity and authorization are one record because a student has exactly one
    status: there is no state in which a person is registered twice, or approved
    without existing.

    ``telegram_name`` is the display name Telegram reports. ``full_name`` is
    what the student typed when applying, so it is None until they have.
    """

    telegram_id: int
    telegram_name: str
    telegram_username: str | None = None
    full_name: str | None = None
    status: RegistrationStatus
    created_at: datetime
    handled_at: datetime | None = None
    handled_by: int | None = None


class Question(BaseModel):
    question_id: int
    part: Part
    question_number: int
    # Where the image lives, relative to the corpus root the process serves from
    # — the renderer joins the two. It rides along with the metadata because it
    # is a short string; only a question with no image row at all has None.
    image_key: str | None = None
    telegram_file_id: str | None = None


class Session(BaseModel):
    session_id: int
    student_telegram_id: int
    option_id: int
    started_at: datetime
    completed_at: datetime | None = None


class TestResult(BaseModel):
    session_id: int
    exam_type: ExamType = "CT"
    part_a_score: int
    part_b_score: int
    total_score: int
    max_score: int
    time_spent: float = Field(description="Total time in seconds")
    completed_at: datetime


# Narrow read-shapes the repositories hand back. They live here rather than
# beside the SQL because they cross module boundaries — the bot's question round
# and results screen both read them.


class SubjectRow(NamedTuple):
    id: int
    name: str


class SessionInfo(NamedTuple):
    subject_name: str
    year: int
    option_number: int


class WrongAnswer(NamedTuple):
    """One question a student got wrong, as it was scored at the time.

    ``correct_answer`` is the snapshot taken when the answer was recorded, not
    the current key — a later correction to the corpus must not rewrite what a
    finished test reported. It is None when the question had no key at all.
    """

    question_number: int
    part: Part
    student_answer: str
    correct_answer: str | None


class QuestionOrigin(NamedTuple):
    """Which book and option a randomly drawn question came from."""

    year: int
    option_number: int
    exam_type: ExamType


__all__ = [
    "EXAM_TYPE_INTRO_YEAR",
    "OPTIONS_PER_BOOK",
    "PART_A_OPTION_COUNT",
    "Detection",
    "ExamType",
    "NormalizedPolygon",
    "Part",
    "PercentPolygon",
    "PixelPolygon",
    "Question",
    "QuestionKey",
    "QuestionOrigin",
    "RegistrationStatus",
    "Session",
    "SessionInfo",
    "Student",
    "SubjectRow",
    "TestResult",
    "WrongAnswer",
    "exam_type_for",
    "normalize_option_number",
    "parse_exam_type",
    "year_has_exam_types",
]
