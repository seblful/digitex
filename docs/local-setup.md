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
uv sync
```

## 2. Configure environment

```powershell
cp .env.example .env.development
# Edit .env.development — set BOT_TOKEN, BOT_ADMIN_USER_ID, POSTGRES_PASSWORD
cp .env.development .env
```

The second copy gives Docker Compose a `.env` to auto-load — so you never need
the `--env-file .env.development` flag. Keep `.env` and `.env.development` in
sync (re-copy whenever you edit one).

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
uv run python scripts/populate_db.py
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
