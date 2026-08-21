# Domain Glossary

Every domain term Digitex uses, defined once. Use these names exactly in code
and review discussions — drift creates communication overhead and muddies
architectural decisions.

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
  `QuestionCatalog.get_correct_answer` and carried across every seam from
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

- **Extraction** — turning a Book into Question images on disk. `extract` (the
  verb) names only the top-level operation of an Extractor; internal helpers
  are `_crop`, `_detect`, `read_page`. Several named flavors:
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
  which is why the parameter is `on_review` and not `review_strategy`; the
  interactive one is a Tk window. The Proposal carries the extractor's
  own `crop` callable, so a reviewer previews the file that would be written
  rather than its own likeness of it — the same rule the numbering preview
  follows by replaying `place_questions`.
- **Outline alignment** — rebuilding a hand-traced region outline from the print
  it already contains, so every region on every page sits the same distance from
  its own text (`imaging/outlines.py`). Its own vocabulary:
  - **Own print** — the ink a region owns. Every blob on the page goes to the
    region already holding most of it, decided once before any outline moves, so
    a rebuilt outline can neither be held out by a neighbour's descender nor
    reach across and take a neighbour's line. A blob no outline holds half of
    belongs to nobody.
  - **Band** — one line of a region's print, read off the row profile rather
    than by chasing letters: a run of inked rows is one line however many pieces
    its letters come in. Padded and stacked, the bands are the rebuilt outline.
  - **Claim** — what one region asks for before the others are consulted: its
    padded bands, clipped to the licence its original outline gives it.
  - **Territory** — the settled partition of the claims. White space two regions
    both reach for goes to whichever one's print is nearer, and neighbours are
    then pushed a further two pixels apart, because a boundary two regions share
    survives the rotation back onto the scan's axes only to be rounded onto the
    same pixel line and filled by both.
  - **Level frame** — the page turned so its text runs horizontal. All of the
    above happens there: a margin measured on a tilted scan's own axes is a
    different margin at each end of a line.
- **Carrying a page** — copying one task's annotation from one Label Studio
  project into another with its outlines aligned (`labeling/transfer.py`,
  `digitex-label copy-aligned`). The destination project is the only record of
  what has been carried — a page it already holds an annotation for is passed
  over — which is what makes the command rerunnable and safe to interrupt.
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

## Extraction outcomes

- **BookReport** — what one book's run produced (`pipeline/outcome.py`): pages
  that came through, and three things that are not "a page was extracted".
  `clean` means nothing failed; `complete` means the year may be recorded as
  finished, which additionally requires at least one page — an empty book
  directory is clean over nothing and must stay retryable.
- **Collision** — a question whose slot was already taken, so the existing file
  was kept. Not a failure: a year resumed after an interruption meets its own
  earlier output on every page it replays.
- **PageFailure** — a page that raised and produced nothing.
- **UnfinishedPieces** — question pieces still held when the book ended, joined
  to nothing, so no file carries them.
- **SubjectRefused** — the run never began: no archive, no subject, no
  processed pages. Distinct from a `SubjectReport` holding no years, which means
  there was nothing left to do.

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
  `show_random_question` (not `send_question_with_cache`; `send_question` in
  `bot.renderer` is the lower-level primitive), and `end`, which pays the
  parked `file_id` debt and clears the conversation state together. `end` holds
  the bot's only `state.clear()`; no handler names `pending_file_id_cache`.

## Infrastructure terms

- **UnitOfWork (UoW)** — an async context manager that borrows one connection
  from the application's `AsyncConnectionPool` (psycopg 3) and wraps it in a
  single transaction. Every DB write goes through a UoW. Nothing in `bot` names
  it: handlers take an **OpenUow** and open the transaction through that.
- **OpenUow** — the transaction seam (`domain.ports.OpenUow`): a factory that
  starts a transaction and hands back the **Repositories** inside it. Injected
  through aiogram's `workflow_data` by `service/cli/bot.py`, which is the one
  module that turns a psycopg pool into one. A factory rather than an open
  transaction, because a round renders a question in one and settles the parked
  `file_id` in the next.
- **Ports** — the protocols in `domain/ports.py` that the bot is written
  against: `QuestionCatalog`, `QuestionDraw`, `TopicIndex`, `FileIdCache`,
  `CatalogIndex`, `SessionLog`, `StudentDirectory`, and `Repositories` over
  them. Deliberately narrower than the classes that satisfy them — `TopicIndex`
  has one method where the Postgres class has four, because the bot only reads a
  subject's topics. Nothing in `db` imports or mentions them; the fit is
  structural, and `ty` is what checks the signatures.
- **Schema migrations** — Alembic, hand-written raw SQL (no ORM, no
  autogenerate). The `digitex-db` CLI is the entry point. The scripts and
  `alembic.ini` live *inside* the package at `db/migrations/`, resolved through
  `importlib.resources` by `db.schema.alembic_config()` — which is what lets the
  image install `digitex-service` as an ordinary wheel carrying no source tree.
- **Repository** — the only layer that touches raw SQL. One class per *role*
  rather than per aggregate, because a question is addressed five different
  ways and no caller wanted more than three of them: `QuestionCatalog` (reading
  one to serve), `QuestionDraw` (picking one at random), `TopicIndex` (the topic
  map), `FileIdCache` (the Telegram `file_id` debt), `QuestionCorpus` (loading
  the extraction output in), plus `StudentRepository`, `SessionRepository` and
  `BookRepository`. All share the connection the UoW opened. The shapes they
  return live in `domain/entities.py`, because callers outside the DB layer read
  them.
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
- **Workspace member** — one of the three distributions the repo builds:
  `digitex-core` (`domain`, `config`, `logging`, `console`), `digitex-service`
  (`bot`, `db`, `service`) and `digitex-studio` (`imaging`, `ml`, `labeling`,
  `pipeline`, `ui`, `studio`). `digitex` is a PEP 420 namespace, so no member
  owns a `digitex/__init__.py` and each contributes subpackages to the same
  import root. Which member a module lives in is the load-bearing fact about
  it; the directory prefix (`packages/<member>/src/`) is just where that is
  written down.
- **Deploy boundary** — only `digitex-core` and `digitex-service` ship, and
  that is a property of the dependency graph rather than a rule about it: the
  service member does not depend on the studio member, so the production image
  never resolves OpenCV, torch or Tesseract. They are absent, not forbidden.
  Two things still have jobs packaging cannot do. `[tool.importlinter]` states
  the direction of imports *inside* a member — chiefly the inversion, that
  `bot` may not reach `db`, `psycopg` or `psycopg_pool` at all — plus one
  belt-and-braces contract that catches a studio import on a dev machine, where
  the studio *is* installed. `tests/contracts/` imports every deployed module
  in an environment built the way production is, which proves the declared
  dependency list is *sufficient* to run the bot rather than merely minimal.

## ML terms

- **Detection** — one thing the segmentation model found on a page: a resolved
  `label` and a `PixelPolygon`. PageExtractor sorts them top-to-bottom by
  polygon bounding box before assembling Questions.
- **Smoothing** — the one pass every prediction leaves through, on its way out
  of `detections_from`: Visvalingam-Whyatt drops the vertex spanning the least
  area with its neighbours until none spans less than `SMOOTH_AREA` of the page's
  diagonal squared and no more than `SMOOTH_BUDGET` of them are left. A mask
  contour arrives with a median of 57 points and a tail past 380; a hand-drawn
  label has 10. Both thresholds are relative to the page because a scan reaches
  the model at anything from 640 to 3400 pixels tall. Not Douglas-Peucker: at
  the same point count it drifts three times as far from the mask and cuts three
  times as much print out of a crop, because a staircase tread is far from the
  edge that replaces it while spanning almost no area.
- **RegionDetector** — what page extraction calls a detector
  (`pipeline.ports.RegionDetector`): `predict(image) -> list[Detection]` and
  nothing else. `YOLO_SegmentationPredictor` is one; the differential harness
  supplies another that answers from a recording. Importing
  `pipeline.page` therefore loads neither torch nor ultralytics, which is the
  point — extraction after the detections is arithmetic on pixels.
- **TextReader** — the same idea for OCR (`extract_text`, `extract_digits`,
  `detect_skew`). Narrower than `TextExtractor`, which also takes a tesseract
  config and a language override that extraction never passes.
- **Polygon spaces** — `PixelPolygon` (source-image pixels), `PercentPolygon`
  (Label Studio's 0-100), `NormalizedPolygon` (YOLO label files' 0-1). Distinct
  types, so a conversion cannot be applied twice by accident. The conversions
  live in `domain/geometry.py`; reading Label Studio's own URIs does not, and
  is in `labeling/uris.py`.
- **AnnotatedImage** — labelled training data in nobody's format
  (`domain/annotations.py`): a filename and its `LabelledRegion`s in normalized
  space. `labeling.export.read_export` produces them and holds every assumption
  about the tool's JSON; `DatasetCreator` consumes them and knows no vendor.

## GUI terms

- **ReviewController** — what a page review *is*, minus the pixels
  (`ui/controller.py`): the loaded page, the selection, the hover, the verdict,
  and every question that can be asked about them — what each row says, whether
  the page may be approved, what the crop preview shows. Builds no widget and
  imports no tkinter, so all of it is testable without a display. Refusing to
  approve a page whose numbering would not continue the output tree lives here,
  because it is the same rule the extractor applies before writing.
- **Review window** — the one Tk window a `--review` run uses, and a view over
  the controller. Built once and loaded with one Page after another, which is
  what carries the zoom, the pan and the selected tab across pages.
  `digitex.ui.page_review` is the only module that imports tkinter.
- **The three review layers** — `PageEdits` decides what an edit *means*,
  `ReviewController` what a review *is*, `_ReviewWindow` what it *looks like*.
  Each is testable without the one above it.
- **Snapshot** — one entry in the undo timeline: a copy of the Page's Regions,
  its entry state and the selection. Undo is a stack of copies rather than of
  inverse operations, because Regions are edited in place.
- **DPI scale** — the factor the display is scaled by, read once before the Tk
  root exists. Sizes in `gui` are written for a 100% display and passed through
  `scaled()`; without the awareness call Windows stretches the whole window and
  its text renders soft.
