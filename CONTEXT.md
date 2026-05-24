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
- **TestResult** — a record of a Student's attempt at a set of Questions
  during one Session.
- **Session** — a single Telegram-bot run-through of a Test by a Student.
- **Student** — a Telegram user authorized to use the bot.
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
  `AuthorizedUserRepository`, `BookRepository`).
- **Settings** — Pydantic-settings tree loaded once via `get_settings()`.
  Composed of `PathsSettings`, `BotSettings`, `DatabaseSettings`,
  `ExtractionSettings`, `TrainingSettings`, `OpenRouterSettings`,
  `LabelStudioSettings`, `LoggingSettings`, `DataSettings`, `AppSettings`.
  Resolved once at module boundaries (the CLI entrypoints) and threaded
  in — never imported deep in the call stack.

## ML terms

- **Predictor** — a model wrapper that turns a PIL Image into a
  `SegmentationPredictionResult` (class IDs + polygons). Currently only
  `YOLO_SegmentationPredictor`.
- **Detection** — one item in a Predictor's output: a `(label, polygon)`
  pair. PageExtractor sorts detections top-to-bottom by polygon bounding
  box before assembling Questions.

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
