# Deployment Guide

Deploy the Digitex Telegram bot on a VPS with Docker.

## Prerequisites

- A VPS (Ubuntu/Debian recommended, ~$5/mo is enough)
- Docker and Docker Compose installed
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- Extraction output (`extraction/data/output/`) on your local machine, ready
  to seed the production database

## Quick Start

```bash
# 1. SSH into the VPS and clone
ssh root@45.129.186.187
git clone https://github.com/seblful/digitex /opt/digitex
cd /opt/digitex

# 2. Configure environment
mkdir -p logs
cp .env.production .env
nano .env  # set BOT_TOKEN, BOT_ADMIN_USER_ID, DATABASE_URL, DB_SSLMODE

# 3. Start Postgres + bot (postgres is in docker-compose.yml)
docker compose up -d postgres

# 4. Apply schema and seed data
docker compose run --rm bot uv run digitex-db upgrade
docker compose run --rm bot uv run python scripts/populate_db.py

# 5. Start the bot
docker compose up -d bot

# 6. Check logs
docker compose logs -f bot
```

## Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y

# Install micro
sudo apt-get install micro

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in

# Install Docker Compose plugin
sudo apt-get install -y docker-compose-plugin
```

### 2. Clone and Configure

```bash
git clone https://github.com/seblful/digitex.git /opt/digitex
cd /opt/digitex

mkdir -p logs

cp .env.production .env
micro .env
```

Required variables in `.env`:

| Variable | Value | Required |
| ----------------------- | --------------------------------------------------- | -------- |
| `BOT_TOKEN` | Your token from @BotFather | Yes |
| `BOT_ADMIN_USER_ID` | Your Telegram user ID | Yes |
| `DATABASE_URL` | `postgresql://digitex:<password>@postgres:5432/digitex` | Yes |
| `DB_SSLMODE` | `disable` (in-cluster) or `require` (external DB) | Yes |
| `LOGGING_CONSOLE_LEVEL` | `INFO` | No |

### 3. Apply schema and seed data

```bash
docker compose up -d postgres
docker compose run --rm bot uv run digitex-db upgrade
docker compose run --rm bot uv run python scripts/populate_db.py
```

### 4. Start the Bot

```bash
docker compose up -d bot
```

### 5. Updates

```bash
cd /opt/digitex
git pull
docker compose build --no-cache
docker compose up -d
```

### 6. Manage

```bash
# View logs
docker-compose logs -f

# Stop
docker-compose down

# Restart
docker-compose restart

# Rebuild and restart after code changes
docker-compose build && docker-compose up -d
```
