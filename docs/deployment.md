# Deployment Guide

Deploy the Digitex Telegram bot on a VPS with Docker.

## Prerequisites

- A VPS (Ubuntu/Debian recommended, ~$5/mo is enough)
- Docker and Docker Compose installed
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- Your `seed.db` file ready

## Quick Start

```bash
# 1. Clone the repo on your VPS
git clone <your-repo-url> /opt/digitex
cd /opt/digitex

# 2. Place your seed database
cp /path/to/your/seed.db ./seed/seed.db

# 3. Create .env file
cp .env.example .env
# Edit .env with your BOT__TOKEN and BOT__ADMIN_USER_ID

# 4. Start the bot
docker compose up -d

# 5. Check logs
docker compose logs -f
```

## Setup

### 1. Server Preparation

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in

# Install Docker Compose plugin
sudo apt-get install -y docker-compose-plugin
```

### 2. Clone and Configure

```bash
git clone https://github.com/seblful/digitex /opt/digitex
cd /opt/digitex

# Create directories for persistent data
mkdir -p data logs seed

# Place your seed database (questions, etc.)
cp /path/to/local/development.db ./seed/seed.db

# Configure environment
cp .env.example .env
nano .env
```

Required variables in `.env`:

| Variable | Value | Required |
|----------|-------|----------|
| `BOT__TOKEN` | Your token from @BotFather | Yes |
| `BOT__ADMIN_USER_ID` | Your Telegram user ID | Yes |
| `APP_ENVIRONMENT` | `production` | Yes |
| `LOGGING_CONSOLE_LEVEL` | `INFO` | No |

### 3. Start the Bot

```bash
docker compose up -d
```

The entrypoint automatically:
1. Checks if `/app/data/production.db` exists
2. If not, copies `seed.db` from `/app/seed/` (or creates schema from `script.sql`)
3. Starts the bot

### 4. Updates

```bash
cd /opt/digitex
git pull
docker compose build --no-cache
docker compose up -d
```

### 5. Manage

```bash
# View logs
docker compose logs -f

# Stop
docker compose down

# Restart
docker compose restart

# Rebuild and restart after code changes
docker compose build && docker compose up -d
```

## Architecture

```
Host filesystem              Docker Container
┌─────────────────┐         ┌──────────────────┐
│ ./data/         │ mount   │ /app/data/       │
│   production.db │◀────────│   production.db  │
│                 │         │                  │
│ ./seed/         │ mount   │ /app/seed/       │
│   seed.db       │◀────────│   seed.db        │
│                 │         │                  │
│ ./logs/         │ mount   │ /app/logs/       │
└─────────────────┘         └──────────────────┘
```

- **Seed**: place your `.db` file in `./seed/seed.db` before first start
- **Data**: production database persists in `./data/`
- **Logs**: application logs in `./logs/`

## Local Development

```bash
# Install deps
uv sync

# Create .env
cp .env.example .env

# Run directly
uv run python -m digitex.cli.bot

# Or with Docker
docker compose up
```
