# Local Setup

Run Digitex on your laptop: dependencies, Postgres, schema, seed data, bot, tests.

For VPS deployment, see [production.md](production.md).
For migration internals and schema conventions, see [database-reference.md](database-reference.md).

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose
- A Telegram bot token from [@BotFather](https://t.me/BotFather) (only if you'll run the bot)

## 1. Install dependencies

```bash
uv sync --all-extras --no-extra cpu
```

Plain `uv sync` installs only what production runs — the bot, its database
layer, and the migration CLI. The rest is grouped by workflow:

| Extra | For | Notable weight |
| ------------ | ---------------------------------- | ------------------------- |
| `extraction` | books → question images | OpenCV, Pillow, Tesseract |
| `ml` | YOLO training and prediction | ultralytics (needs torch) |
| `labeling` | the Label Studio annotation server | a whole Django app |
| `cu130` | torch from the CUDA wheel index | ~3GB |
| `cpu` | torch from the CPU wheel index | ~200MB |

`cpu` and `cu130` conflict, so exactly one can be active — which is why the
command above takes everything *except* `cpu`. On a machine without an NVIDIA
GPU, swap it: `uv sync --all-extras --no-extra cu130`. Working on one thing
only? `uv sync --extra pipeline` is enough.

## 2. Configure environment

```powershell
cp .env.example .env
# Edit .env — set BOT_TOKEN, BOT_ADMIN_USER_ID, POSTGRES_PASSWORD
```

One file per machine: this one holds your development values, the server's
holds production ones. Both the app and Docker Compose read `.env` and nothing
else, so there is no second copy to keep in sync.

Minimum required values:

| Variable | Value |
| ------------------- | --------------------------------- |
| `BOT_TOKEN` | Your token from @BotFather |
| `BOT_ADMIN_USER_ID` | Your Telegram user ID |
| `POSTGRES_PASSWORD` | Any value (e.g. `digitex` is fine for local) |

See `.env.example` for the full list of available variables.

## 3. Start Postgres

```bash
docker compose up -d postgres
```

Postgres listens on `127.0.0.1:5433` (port 5433 to avoid clashing with any
native Postgres on 5432).

## 4. Apply schema

```bash
uv run digitex-db upgrade
```

## 5. Seed data

```bash
uv run digitex-db populate
```

Idempotent — re-running is safe (`get_or_create`).

## 6. Run the bot

```bash
uv run digitex-bot
```

## Tests

```bash
uv run pytest              # all tests
uv run pytest -v           # verbose
uv run pytest tests/test_handlers.py
```

## Related

- [Label Studio](label-studio.md) — annotation server for training data
- [Training](training.md) — YOLO model training workflow
- [Extraction](extraction.md) — extracting question images from books
