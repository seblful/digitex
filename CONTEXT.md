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
  (`books/<subject>/raw/pages/<year>/`, mirrored corrected under
  `books/<subject>/processed/`). The raw input to extraction.
- **Page** — a single image inside a Book. May contain multiple Questions.
- **Question** — one exam question. Identified by `(subject, year, option, part, number)`. Stored as a cropped image plus optional OCR text.
- **Option** — a numbered variant of an exam (1–10). A Book contains several
  Options interleaved across its Pages.
- **Part** — `"A"` (multiple-choice with numbered answers) or `"B"` (free-text
  answer). Every Question belongs to exactly one Part.
- **Answer** — the student's response. Part A answers are integers; Part B
  answers are strings, scored via `domain.answer.AnswerKey.matches`.
- **AnswerKey** — the correct answer to a Question as a value object
  (`domain.answer.AnswerKey`): the part travels with the value, so matching,
  storage form (`stored`) and the None-key rule live in one place. Built by
  `QuestionRepository.get_correct_answer` and carried across every seam from
  there.
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
    `PageExtractor` (`pipeline/page.py`) using YOLO segmentation.
  - **Book extraction** — every Page in a Book (`pipeline/book.py`).
  - **Subject extraction** — every Book of one subject, skipping years already
    recorded complete. Driven by `SubjectExtractor` (`pipeline/subject.py`).
  - **Answers extraction** — pulling the answer key off the back of a Book
    via the OpenRouter vision API (`pipeline/answers.py`).
- **Review** — checking a Page's detected regions by hand before its crops are
  written. A `PageReviewer` is a callable
  `(PageProposal) -> ReviewedPage | None` — a callable, not a Protocol class,
  and the interactive one is a Tk window. The Proposal carries the extractor's
  own `crop` callable, so a reviewer previews the file that would be written
  rather than its own likeness of it — the same rule the numbering preview
  follows by replaying `place_questions`.
- **Placement** — the Option/Part/number one detected Question is written as.
  Handed out by `PageExtractionState` and applied by `place_questions`, the one
  walk shared by the review preview and the write.
- **Numbering fault** — a Placement that does not continue its Option/Part
  folder: its number is already on disk (the crop would overwrite an extracted
  Question) or past the end (the folder would keep a hole). `numbering_fault`
  finds the first one and is the rule on both sides of the reviewer seam: the
  review window refuses to approve a fault, and the extractor replays every
  page through the same check before writing — a gap refuses the page, a
  collision keeps the existing file (that is what lets a resumed year replay
  its pages). The output tree is never allowed out of order, which is why
  there is no renumbering pass.

## Bot conversation shapes

- **Standard testing** — the Student answers a fixed queue of Questions; each
  answer is recorded to a Session.
- **Random testing** — one Question at a time, drawn at random, with
  immediate correct/wrong feedback. No Session is recorded.
- **Topic mode** — Random testing restricted to a topic name.
- **Round** — the per-update handle on a question round
  (`bot.answer_flow.Round`): it owns the bot, the FSM context, the questions
  directory and the transaction seam (`open_uow`). Its interface is the three
  things a handler does to a round — `show_testing_question` /
  `show_random_question`, and `end`, which pays the parked `file_id` debt and
  clears the conversation state together. `end` holds the bot's only
  `state.clear()`; no handler names `pending_file_id_cache`.

## Infrastructure terms

- **UnitOfWork (UoW)** — an async context manager that borrows one connection
  from the application's `AsyncConnectionPool` (psycopg 3) and wraps it in a
  single transaction. Every DB write goes through a UoW. Handlers acquire the
  pool from aiogram's `workflow_data` (injected by `cli/bot.py`).
- **Schema migrations** — Alembic, hand-written raw SQL (no ORM, no
  autogenerate). The `digitex-db` CLI is the entry point. The scripts and
  `alembic.ini` live *inside* the package at `db/migrations/`, resolved through
  `importlib.resources` by `db.schema.alembic_config()` — which is what lets the
  image install the project as an ordinary wheel with no copy of `src/`.
- **Repository** — the only layer that touches raw SQL. One per aggregate
  (`QuestionRepository`, `StudentRepository`, `SessionRepository`,
  `BookRepository`). The shapes they return live in `domain/entities.py`, because
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
- **Settings** — Pydantic-settings tree loaded once via `get_settings()`, one
  module per layer under `config/`. Top level: `app`, `bot`, `database`,
  `logging`, `paths`, `timezone`, and `pipeline`. Resolved per command inside
  the CLI entrypoints and threaded in — never imported deep in the call stack,
  and never at module import.
- **`Settings.pipeline`** — the groups only the local workflows read
  (`extraction`, `openrouter`, `label_studio`, `data`), grouped so the call site
  says which layer owns the value. Nothing the deployed bot runs reads it.
- **Data root** — `PathsSettings.data_root` (`PATH_DATA_ROOT`, default `var/`).
  Every non-code input and output hangs off it. No path is ever derived from the
  package's own location, which is why there is no `BASE_DIR`.
- **Deploy boundary** — only `bot`, `db`, `domain` and `config` ship. Enforced
  two ways, because neither covers the other: `[tool.importlinter]` states which
  packages and which third-party distributions the deploy layer may reach, and
  `tests/contracts/` imports every deployed module in an environment built the
  way production is. Adding an import from `bot` to `imaging`, `ml`, `labeling`,
  `pipeline` or `ui` fails both.

## ML terms

- **Detection** — one thing the segmentation model found on a page: a resolved
  `label` and a `PixelPolygon`. `YOLO_SegmentationPredictor.predict` returns a
  `list[Detection]`; PageExtractor sorts them top-to-bottom by polygon bounding
  box before assembling Questions. There is no predictor abstraction — the one
  concrete predictor is named at each of its three call sites.
- **Polygon spaces** — `PixelPolygon` (source-image pixels), `PercentPolygon`
  (Label Studio's 0-100), `NormalizedPolygon` (YOLO label files' 0-1). Distinct
  types, so a conversion cannot be applied twice by accident.

## GUI terms

- **Review window** — the one Tk window a `--review` run uses. Built once and
  loaded with one Page after another, which is what carries the zoom, the pan
  and the selected tab across pages. `digitex.ui` is the only package that
  imports tkinter.
- **Snapshot** — one entry in the review window's undo timeline: a copy of the
  Page's Regions, its entry state and the selection. Undo is a stack of copies
  rather than of inverse operations, because the window edits Regions in place.
- **DPI scale** — the factor the display is scaled by, read once before the Tk
  root exists. Sizes in `gui` are written for a 100% display and passed through
  `scaled()`; without the awareness call Windows stretches the whole window and
  its text renders soft.

## Naming conventions worth preserving

- `on_review` (not `review_strategy`) for a `PageReviewer` callable parameter —
  matches the callable-not-class shape.
- `show_testing_question` / `show_random_question` (not
  `send_question_with_cache`) for the Round's "render a Question and park any
  new file_id" recipe — the debt is settled inside the next round's UoW via
  the `pending_file_id_cache` FSM field. `send_question` is the lower-level
  primitive in `bot.renderer`.
- `extract` (the verb) is reserved for the top-level operation of an
  Extractor; internal helpers use `_crop`, `_detect`, `read_page`, etc.
