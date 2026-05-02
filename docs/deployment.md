# Deployment Guide

Deploy the Digitex Telegram bot on a VPS with Docker.

## Prerequisites

- A VPS (Ubuntu/Debian recommended, ~$5/mo is enough)
- Docker and Docker Compose installed
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- Your `seed.db` file ready

## Quick Start

```bash
# 1. Copy seed database from your LOCAL machine
scp ./data/seed.db root@45.129.186.187:/opt/digitex/data/seed.db

# 2. SSH into the VPS and set up
ssh root@45.129.186.187
git clone https://github.com/seblful/digitex /opt/digitex
cd /opt/digitex

# 3. Move seed into place and configure
mkdir -p data logs
mv /opt/digitex/data/seed.db ./data/production.db
cp .env.example .env
nano .env

# 4. Start the bot
docker-compose up -d

# 5. Check logs
docker-compose logs -f
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

### 2. Copy seed database from local PC

```bash
# Run this on your LOCAL machine
scp ./data/seed.db root@45.129.186.187:/opt/digitex/data/seed.db
```

### 3. Clone and Configure

```bash
git clone https://github.com/seblful/digitex /opt/digitex
cd /opt/digitex

# Create directories for persistent data
mkdir -p data logs

# Move seed into place
mv /opt/digitex/data/seed.db ./data/production.db

# Configure environment
cp .env.example .env
micro .env
```

Required variables in `.env`:

| Variable                | Value                      | Required |
| ----------------------- | -------------------------- | -------- |
| `BOT__TOKEN`            | Your token from @BotFather | Yes      |
| `BOT__ADMIN_USER_ID`    | Your Telegram user ID      | Yes      |
| `APP_ENVIRONMENT`       | `production`               | Yes      |
| `LOGGING_CONSOLE_LEVEL` | `INFO`                     | No       |

### 4. Start the Bot

```bash
docker compose up -d
```

### 5. Updates

```bash
cd /opt/digitex
git pull
docker-compose build --no-cache
docker-compose up -d
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
