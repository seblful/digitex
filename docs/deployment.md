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
nano .env  # set BOT_TOKEN, BOT_ADMIN_USER_ID, POSTGRES_PASSWORD, DATABASE_URL, DB_SSLMODE

# 3. Start Postgres
docker compose up -d postgres

# 4. Apply schema (data load happens separately — see step 3 below)
docker compose run --rm bot uv run digitex-db upgrade

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
| `POSTGRES_PASSWORD` | Strong password (e.g. `openssl rand -base64 24`) — read by the postgres container | Yes |
| `DATABASE_URL` | `postgresql://digitex:<password>@postgres:5432/digitex` (must match `POSTGRES_PASSWORD`) | Yes |
| `DB_SSLMODE` | `disable` (in-cluster) or `require` (external DB) | Yes |
| `LOGGING_CONSOLE_LEVEL` | `INFO` | No |

See [database.md](database.md) for the full DB setup story — password
generation, SSH-tunnel access from your PC, and backups.

### 3. Apply schema and seed data

```bash
docker compose up -d postgres
docker compose run --rm bot uv run digitex-db upgrade
```

For the initial data load, `populate_db.py` reads from
`extraction/data/output/` — this directory is not part of the production
image. Run it locally through an SSH tunnel (recommended):

```powershell
# Terminal 1 — keep open
ssh -L 5433:localhost:5432 root@<vps-ip>
```

```bash
# Terminal 2 — seed through the tunnel
DATABASE_URL="postgresql://digitex:<password>@localhost:5433/digitex" \
    uv run python scripts/populate_db.py
```

Always pass `DATABASE_URL` explicitly when targeting production — a bare
`uv run python scripts/populate_db.py` should always hit local, never the
VPS. See [database.md](database.md) for the full tunnel setup.

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
