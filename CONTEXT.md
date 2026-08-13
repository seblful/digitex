# Digitex — Domain Glossary

Definitions for the project's core domain terms. Use these names exactly in
code and review discussions — drift creates communication overhead and
muddies architectural decisions.

For the *architectural* vocabulary (Module, Interface, Depth, Seam, Adapter,
Leverage, Locality, Deletion test), see the language reference shipped with
the `/refactoring:improve-codebase-architecture` skill.

______________________________________________________________________

## Domain entities

- **Book** — a directory of scanned page images for one exam subject and year
  (`books/<subject>/<year>/`). The raw input to extraction.
- **Page** — a single image inside a Book. May contain multiple Questions.
- **Question** — one exam question. Identified by `(subject, year, option, part, number)`. Stored as a cropped image plus optional OCR text.
- **Option** — a numbered variant of an exam (1–10). A Book contains several
  Options interleaved across its Pages.
- **Part** — `"A"` (multiple-choice with numbered answers) or `"B"` (free-text
  answer). Every Question belongs to exactly one Part.
- **Answer** — the student's response. Part A answers are integers; Part B
  answers are strings normalized via `core.answer.check_answer`.
- **Answer key** — the correct answer to a Question, or None when the year
  shipped without one. A Question with no key is still stored so its image is
  servable, and nothing a Student sends can match it.
- **Topic** — a named subject-level tag on Questions (`topics`), mapped to
  Questions through `question_topics`. Two subjects may use the same topic name
  without sharing questions.
- **TestResult** — a record of a Student's attempt at a set of Questions
  during one Session.
- **Session** — a single Telegram-bot run-through of a Test by a Student.
- **Student** — a Telegram user, keyed by their `telegram_id`. The row is the
  person: it exists from their first contact and carries their registration
  status, so "authorized" is a state of a Student rather than a separate record.
- **ExamType** — `"CE"` (Централизованный экзамен) or `"CT"`
  (Централизованное тестирование). Carried as `Literal["CE", "CT"]`.

## Processes

- **Extraction** — turning a Book into Question images on disk. Several
  named flavors:
  - **Page extraction** — one Page → multiple Question crops. Driven by
    `PageExtractor` using YOLO segmentation.
  - **Book extraction** — every Page in a Book.
  - **Tests extraction** — every Book in the books directory.
  - **Manual extraction** — integrating hand-cropped Question images that
    YOLO missed.
  - **Answers extraction** — pulling the answer key off the back of a Book
    via the OpenRouter vision API.
- **Conflict** — an extraction collision: a new Question image would overwrite
  an existing file. Resolved by a `ConflictResolver` — a callable
  `(Conflict) -> int`, not a Protocol class.
- **Renumbering** — adjusting Question file numbers within an Option/Part to
  fill gaps left after manual additions.

## Bot conversation shapes

- **Standard testing** — the Student answers a fixed queue of Questions; each
  answer is recorded to a Session.
- **Random testing** — one Question at a time, drawn at random, with
  immediate correct/wrong feedback. No Session is recorded.
- **Topic mode** — Random testing restricted to a topic name.

## Infrastructure terms

- **UnitOfWork (UoW)** — an async context manager that borrows one connection
  from the application's `AsyncConnectionPool` (psycopg 3) and wraps it in a
  single transaction. Every DB write goes through a UoW. Handlers acquire the
  pool from aiogram's `workflow_data` (injected by `cli/bot.py`).
- **Schema migrations** — Alembic, hand-written raw SQL (no ORM, no
  autogenerate). The `digitex-db` CLI is the entry point.
- **Repository** — the only layer that touches raw SQL. One per aggregate
  (`QuestionRepository`, `StudentRepository`, `SessionRepository`,
  `BookRepository`). The shapes they return live in `core/domain.py`, because
  callers outside the DB layer read them.
- **`questions` table** — one table with a `part` column, not a table per Part.
  The part is always a bound parameter, never interpolated into SQL, and
  `question_id` is unique across both Parts. Because the id alone names a
  question, `images`, `question_topics` and `session_answers` reference it
  alone — they do not carry a `part`, and neither do the repository methods that
  address a question.
- **Answer history** — `session_answers` rows are immutable records, not a view
  of the corpus. Each stores the `correct_answer` it was judged against and
  references its Question `ON DELETE RESTRICT`, so re-loading or correcting the
  corpus can neither rewrite nor erase a finished test.
- **Settings** — Pydantic-settings tree loaded once via `get_settings()`.
  Composed of `PathsSettings`, `BotSettings`, `DatabaseSettings`,
  `ExtractionSettings`, `OpenRouterSettings`, `LabelStudioSettings`,
  `LoggingSettings`, `DataSettings`, `TimezoneSettings`, `AppSettings`.
  Resolved per command inside the CLI entrypoints and threaded in — never
  imported deep in the call stack, and never at module import.

## ML terms

- **Detection** — one thing the segmentation model found on a page: a resolved
  `label` and a `PixelPolygon`. `YOLO_SegmentationPredictor.predict` returns a
  `list[Detection]`; PageExtractor sorts them top-to-bottom by polygon bounding
  box before assembling Questions. There is no predictor abstraction — the one
  concrete predictor is named at each of its three call sites.
- **Polygon spaces** — `PixelPolygon` (source-image pixels), `PercentPolygon`
  (Label Studio's 0-100), `NormalizedPolygon` (YOLO label files' 0-1). Distinct
  types, so a conversion cannot be applied twice by accident.

## Naming conventions worth preserving

- `on_conflict` (not `conflict_strategy`) for a `ConflictResolver` callable
  parameter — matches the callable-not-class shape.
- `ask_question` (not `send_question_with_cache`) for the bot's "render a
  Question and surface any new file_id" recipe. It returns the new file_id
  to the caller, which folds the cache write into the next UoW via the
  `pending_file_id_cache` FSM field. `send_question` is the lower-level
  primitive in `bot.renderer`.
- `extract` (the verb) is reserved for the top-level operation of an
  Extractor; internal helpers use `_crop_and_save`, `_detect`, etc.
