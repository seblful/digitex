# Digitex

Document digitization toolkit for processing images, ML-based document segmentation, and centralized testing via Telegram bot.

## Project Structure

Digitex is two products over one domain, and the split is a **uv workspace of
three distributions** rather than a rule about one package. `digitex` itself is
a PEP 420 namespace: no member owns a `digitex/__init__.py`, and each
contributes its own subpackages to it.

```
digitex/
├── packages/
│   ├── core/src/digitex/            # digitex-core — importable by anything
│   │   ├── domain/                  # pure: entities, answer rules, numbering, ports
│   │   ├── config/                  # settings, one module per layer
│   │   ├── logging.py               # structlog setup
│   │   └── console.py               # the two things every entry point needs
│   ├── service/src/digitex/         # digitex-service — the only thing that deploys
│   │   ├── bot/                     # Telegram bot (aiogram)
│   │   ├── db/                      # Postgres: pool, unit of work, role repositories
│   │   │   └── migrations/          # Alembic scripts, shipped inside the package
│   │   └── service/cli/             # digitex-bot, digitex-db — the composition roots
│   └── studio/src/digitex/          # digitex-studio — laptop only, never shipped
│       ├── imaging/                 # scale · layout · levels · denoise · document · crop
│       ├── ml/                      # YOLO segmentation and training
│       ├── labeling/                # Label Studio client, export reader, URIs
│       ├── pipeline/                # books → question images (+ audit/)
│       ├── ui/                      # the review controller, and the Tk view over it
│       └── studio/cli/              # digitex-extract, digitex-train, digitex-label
├── configs/training/         # YOLO hyperparameter YAMLs
├── deploy/                   # Dockerfile, deploy.sh, seed_prod.ps1
├── docker-compose.yml        # stays at the root: its relative paths are $APP_DIR
├── tests/                    # unit, characterization, differential,
│                             #   integration, contracts
├── docs/                     # setup, deployment, workflows (see docs/README.md)
├── var/                      # the data root — gitignored, PATH_DATA_ROOT
└── CLAUDE.md                 # AI agent instructions
```

Dependencies point one way, and the two branches never meet:

```
studio ──▶ ui ──▶ pipeline ──▶ labeling ──▶ ml ──▶ imaging ──┐
                                                             ├──▶ domain
service ──▶ bot ───────────────────────────────────────────  ┘
    └─────▶ db  ───────────────────────────────────────────  ┘
```

Most of that is now a **packaging fact rather than a policy**:
`digitex-service` does not depend on `digitex-studio`, so in the production
image OpenCV, torch and Tesseract are not forbidden — they were never resolved.
[`[tool.importlinter]`](pyproject.toml) is down to the rules packaging cannot
state, chiefly the direction of imports *inside* a member.

The one that matters most: `bot` and `db` are siblings, not a stack. The bot is
written against the protocols in `domain/ports.py`; `db` provides classes that
answer to them; and `service/cli/bot.py` is the only module that names both. An
import from `bot` to `db` fails a contract rather than surfacing as an
ImportError on the VPS.

### The data root

Nothing outside `packages/` is code. The book archive, extraction output, model
weights and training data all live under one directory — `var/` by default,
or wherever `PATH_DATA_ROOT` points — so the checkout stays small and an
installed package never guesses where a corpus is.

```
var/books/{subject}/raw/pages/{year}/…       scanned pages, the raw input
var/books/{subject}/processed/pages/{year}/… the same pages, corrected
var/books/{subject}/topics.json              the topic map, seeded into the db
var/extraction/output/…               question images, the corpus the bot serves
var/models/page.pt                    the segmentation checkpoint
var/training/{data,runs}/             YOLO datasets and run outputs
```

## Features

- **Image Extraction**: Extract and process question images from book images
- **Image Processing**: Crop, transform, resize with aspect ratio preservation
- **YOLO Segmentation**: Detect and segment document regions
- **Telegram Bot**: Take centralized tests via Telegram with automatic grading
- **Configuration Management**: Pydantic-based settings with environment variable support

## CLI Commands

```bash
# Prepare raw scans into var/books/{subject}/processed/ (renames, then corrects)
digitex-extract preprocess-scans

# Extract question images from books
digitex-extract extract-questions <subject>

# Train YOLO segmentation model
digitex-train create-dataset
digitex-train train

# Start Telegram bot
digitex-bot

# Manage schema migrations
uv run digitex-db upgrade

# Populate database from extraction output
uv run digitex-db populate

# Check the image rows still match the files on disk
uv run digitex-db check-images

# Record a book's model and OCR answers as a replay fixture
uv run digitex-extract record-golden <subject> <year>
```

## Telegram Bot

The bot allows students to take centralized tests via Telegram:

1. **Start** — `/start` to register and select a subject
1. **Navigate** — Choose subject → year → option number
1. **Test** — Answer Part A (multiple choice 1-5) and Part B (text) questions
1. **Results** — Get instant score and mistake review

### Setup & deployment

- **Run locally**: see [docs/local-setup.md](docs/local-setup.md)
- **Deploy to a VPS / day-2 ops**: see [docs/production.md](docs/production.md)
- **Migration CLI & schema conventions**: see [docs/database-reference.md](docs/database-reference.md)

Full doc index: [docs/README.md](docs/README.md).

## Configuration

This project uses Pydantic Settings with one `.env` file per machine:

```
.env                  # This machine's config (gitignored)
.env.example          # Reference template (committed)
```

Your laptop's `.env` holds development values, the server's holds production
ones — there is no second file to keep in sync. Real environment variables win
over the file, which is how Compose injects `DATABASE_URL` and sets
`ENVIRONMENT=production` to select the JSON log renderer.

See `.env.example` for all available variables and their defaults.

## Setup

This project uses `uv` for dependency management.

```bash
# Everything (--no-extra cu130 on a machine without an NVIDIA GPU)
uv sync --all-extras --no-extra cpu

# The studio half alone — scans, training, annotation
uv sync --extra studio --extra cpu

# The service half alone — bot and database, no gigabytes
uv sync

# Run extraction
uv run digitex-extract --help

# Run training
uv run digitex-train --help
```

The bot needs no extra at all: `uv sync` on its own installs `digitex-core` and
`digitex-service`, which is exactly what the production image gets.

### Requirements

- Python 3.13+
- uv package manager

## Modules

- **domain**: exam entities, answer matching, question numbering, corpus
  layout, polygon spaces, and the ports the bot is written against — pure
  Python and pydantic, importable by anything
- **db**: the only layer that writes SQL; one class per role rather than per
  aggregate, all writes through a `UnitOfWork`
- **bot**: the aiogram bot — the only thing that deploys, and it names no
  database
- **pipeline**: question images out of book scans, page → book → subject, plus
  `audit/` to check what came out
- **ml**: YOLO segmentation, dataset building and training
- **labeling**: Label Studio client and prediction upload
- **imaging**: cropping, deskewing, resizing, OCR
- **ui**: `ReviewController` — a page review with no widget in it — and the
  Tk window that draws it, the only module that imports tkinter

## Development

See [CLAUDE.md](CLAUDE.md) for code standards, type-hinting requirements,
testing guidelines, and git workflow.

## Testing

```bash
# Everything: unit, characterization, differential, integration, contracts
uv run pytest

# The unit suite alone — no Docker required
uv run pytest tests/unit

# One file
uv run pytest tests/unit/test_bot_handlers.py

# Layering and the bot/db inversion, statically
uv run lint-imports
```

`tests/differential/` replays a recorded book and checks the bytes of every
question image it writes — the guarantee any change to extraction is measured
against. It needs a recording under the data root (`record-golden` above) and
skips without one, the way the integration suite skips without Docker.

`tests/contracts/` is only meaningful in an environment built the way
production is, which is what the `deploy boundary` CI job does:

```bash
UV_PROJECT_ENVIRONMENT=.venv-prod uv sync --locked --no-dev --group contracts
UV_PROJECT_ENVIRONMENT=.venv-prod uv run --no-sync pytest tests/contracts -o addopts=""
```
