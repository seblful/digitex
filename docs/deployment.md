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
ssh root@<vps-ip>
git clone https://github.com/seblful/digitex /opt/digitex
cd /opt/digitex

# 2. Configure environment
mkdir -p logs
cp .env.example .env.production
nano .env.production  # set BOT_TOKEN, BOT_ADMIN_USER_ID, POSTGRES_PASSWORD

# 3. Start Postgres (compose only auto-loads `.env` — point it at the
#    production file so POSTGRES_PASSWORD resolves)
docker compose --env-file .env.production up -d postgres

# 4. Apply schema (data load happens separately — see step 3 below)
docker compose --env-file .env.production run --rm bot uv run digitex-db upgrade

# 5. Start the bot
docker compose --env-file .env.production up -d bot

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

cp .env.example .env.production
micro .env.production
```

Required variables in `.env.production`:

| Variable | Value | Required |
| ----------------------- | --------------------------------------------------- | -------- |
| `BOT_TOKEN` | Your token from @BotFather | Yes |
| `BOT_ADMIN_USER_ID` | Your Telegram user ID | Yes |
| `POSTGRES_PASSWORD` | Strong password (`openssl rand -base64 24`) | Yes |
| `DB_SSLMODE` | `disable` (in-cluster) or `require` (external DB) | Yes |
| `LOGGING_CONSOLE_LEVEL` | `INFO` | No |

`DATABASE_URL` is derived automatically from `POSTGRES_PASSWORD` by Compose —
no need to set it manually. See [database.md](database.md) for the full DB
setup story — password generation, SSH-tunnel access from your PC, and backups.

### 3. Apply schema and seed data

```bash
docker compose --env-file .env.production up -d postgres
docker compose --env-file .env.production run --rm bot uv run digitex-db upgrade
```

For the initial data load, seed from your local machine through an SSH
tunnel — see [data-update.md](data-update.md) for the exact commands.

### 4. Start the Bot

```bash
docker compose --env-file .env.production up -d bot
```

### 5. Updates

```bash
cd /opt/digitex
git pull
docker compose --env-file .env.production build --no-cache
docker compose --env-file .env.production up -d
```

### 6. Manage

```bash
# View logs
docker compose logs -f

# Stop
docker compose --env-file .env.production down

# Restart
docker compose --env-file .env.production restart

# Rebuild and restart after code changes
docker compose --env-file .env.production build \
  && docker compose --env-file .env.production up -d
```
