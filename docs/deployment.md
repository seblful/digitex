# Deployment Guide

Deploy the Digitex Telegram bot to Railway with a persistent SQLite database.

## Prerequisites

- GitHub repository with your code pushed
- Railway account (<https://railway.app>)
- Telegram bot token from [@BotFather](https://t.me/BotFather)

## Quick Start

```bash
# Build and test locally
docker build -t digitex-bot .
docker run --rm --env-file .env digitex-bot

# Push to GitHub (Railway auto-deploys from there)
git push
```

## Step by Step

### 1. Push to GitHub

Railway links directly to your repo for automatic deployments:

```bash
git add .
git commit -m "chore: ready for deployment"
git push
```

### 2. Create Railway Project

1. Log in to [Railway.app](https://railway.app)
2. Click **+ New Project** → **Deploy from GitHub repo**
3. Select your repository
4. The first build may fail — that's fine, you need to configure volumes and variables first

### 3. Add Persistent Volume

SQLite data is wiped on every Railway deploy unless stored on a volume:

1. In your project dashboard, right-click the canvas → **Volume**
2. Set **Mount Path** to `/app/data`
3. Connect the volume to your bot service

### 4. Set Environment Variables

Go to **Variables** in your bot service and add:

| Variable | Value | Required |
|----------|-------|----------|
| `BOT__TOKEN` | Your token from @BotFather | Yes |
| `BOT__ADMIN_USER_ID` | Your Telegram user ID | Yes |
| `DB_PATH` | `/app/data/production.db` | Yes |
| `APP_ENVIRONMENT` | `production` | Yes |
| `LOGGING_CONSOLE_LEVEL` | `INFO` | No |
| `RAILWAY_RUN_UID` | `0` | Yes |

`RAILWAY_RUN_UID=0` runs the container as root so it can write to the volume.

### 5. Configure Start Command

Railway auto-detects the Dockerfile. No manual start command is needed.
The entrypoint automatically creates the database schema and seeds question
data on first boot.

### 6. Redeploy

1. Railway triggers a redeploy when you save variables
2. Wait for the status to turn green
3. Open your bot on Telegram and send `/start`

### 7. Verify Persistence

1. Use the bot to generate some data (registration, test session)
2. Go to **Deployments** → **Redeploy**
3. After the bot comes back, check that your data is still there

## Architecture

```
Docker Image                    Railway Volume
┌─────────────────┐            ┌──────────────────┐
│ /app/seed/      │  first     │ /app/data/       │
│   seed.db       │──deploy──▶│   production.db  │
│                 │            │                  │
│ /app/scripts/   │            │ (persists across │
│   script.sql    │            │  redeploys)      │
└─────────────────┘            └──────────────────┘
```

- **First deploy**: entrypoint copies `seed.db` (your question data) to the volume
- **Schema fallback**: if no seed is present, creates empty tables from `script.sql`
- **Subsequent deploys**: volume already has `production.db` — entrypoint skips init

## Local Development

For local development, create a `.env` file:

```env
APP_ENVIRONMENT=development
DB_PATH=data/development.db
BOT__TOKEN=your_test_bot_token
BOT__ADMIN_USER_ID=your_telegram_id
LOGGING_CONSOLE_LEVEL=DEBUG
```

Then run:

```bash
uv run digitex-bot run
```

The development database (`data/development.db`) is separate from the production
database (`/app/data/production.db` on Railway). They never interfere.
