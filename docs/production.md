# Production Runbook

Single source of truth for everything VPS-side: first-time deploy, day-2 ops
(code/schema/data updates), database access, backups, troubleshooting.

For laptop dev setup, see [local-setup.md](local-setup.md).
For schema/migration internals, see [database-reference.md](database-reference.md).

______________________________________________________________________

## 1. First-time deploy

### 1.1 Prerequisites

- A VPS (Ubuntu/Debian, ~$5/mo is enough)
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- Extraction output (`extraction/data/output/`) on your local machine, ready
  to seed the production database

### 1.2 Server preparation

```bash
ssh root@<vps-ip>

# Update system
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y

# Editor
sudo apt-get install -y micro

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# log out, log back in

# Compose plugin
sudo apt-get install -y docker-compose-plugin
```

### 1.3 Clone and configure

```bash
git clone https://github.com/seblful/digitex.git /opt/digitex
cd /opt/digitex
mkdir -p logs

cp .env.example .env.production
micro .env.production            # fill in the values below
ln -s .env.production .env       # so docker compose auto-loads it
```

Required in `.env.production`:

| Variable | Value | Required |
| ----------------------- | --------------------------------------------------- | -------- |
| `BOT_TOKEN` | Your token from @BotFather | Yes |
| `BOT_ADMIN_USER_ID` | Your Telegram user ID | Yes |
| `POSTGRES_PASSWORD` | Strong password (`openssl rand -base64 24`) | Yes |
| `DB_SSLMODE` | `disable` (in-cluster) or `require` (external DB) | Yes |
| `LOGGING_CONSOLE_LEVEL` | `INFO` | No |

`DATABASE_URL` is derived automatically from `POSTGRES_PASSWORD` by Compose —
no need to set it manually. Full env reference: `.env.example`.

> The `ln -s .env.production .env` step is what lets every subsequent
> `docker compose …` command work without `--env-file .env.production`.

### 1.4 Start Postgres + apply schema

```bash
docker compose up -d postgres
docker compose run --rm bot uv run digitex-db upgrade
```

### 1.5 Seed the database (from your laptop)

Open an SSH tunnel from your PC (see [§3 Database access](#3-database-access-from-your-pc)
for why a tunnel), then seed through it:

```powershell
# Terminal 1 — PC, keep open
ssh -L 5433:localhost:5432 root@<vps-ip>
```

```powershell
# Terminal 2 — PC
$env:DATABASE_URL = "postgresql://digitex:<password>@localhost:5433/digitex"
uv run python scripts/populate_db.py
```

`populate_db.py` is idempotent (`get_or_create`), so re-running is safe.

### 1.6 Start the bot

```bash
# on the VPS
docker compose up -d bot
docker compose logs -f bot
```

______________________________________________________________________

## 2. Day-2 operations

Pick the scenario that matches what changed. **If multiple things changed,
follow the order in §2.4.**

### 2.1 Code change only (no schema, no data)

```bash
# on the VPS
cd /opt/digitex
git pull
docker compose build --no-cache bot
docker compose up -d bot
```

### 2.2 New migration (schema change)

```bash
# on the VPS
cd /opt/digitex
git pull
docker compose run --rm bot uv run digitex-db upgrade
docker compose up -d bot          # restart picks up any code changes too
```

### 2.3 New extracted data (no schema change)

Seed through the SSH tunnel — same flow as [§1.5](#15-seed-the-database-from-your-laptop).

### 2.4 Combined: code + schema + data

Order matters: **schema before data**, code last.

```bash
# VPS — pull and migrate
cd /opt/digitex
git pull
docker compose run --rm bot uv run digitex-db upgrade
```

```powershell
# PC — seed through the tunnel
ssh -L 5433:localhost:5432 root@<vps-ip>           # terminal 1
$env:DATABASE_URL = "postgresql://digitex:<password>@localhost:5433/digitex"
uv run python scripts/populate_db.py               # terminal 2
```

```bash
# VPS — rebuild + restart bot
docker compose build --no-cache bot
docker compose up -d bot
```

### 2.5 Manage / inspect

```bash
docker compose logs -f bot       # follow bot logs
docker compose ps                # status
docker compose restart bot       # restart bot only
docker compose down              # stop everything (preserves the pgdata volume)
```

______________________________________________________________________

## 3. Database access from your PC

Use an **SSH tunnel** — your existing SSH key is the auth layer, no extra
ports get exposed.

```powershell
ssh -L 5433:localhost:5432 root@<vps-ip>
```

This forwards your PC's `localhost:5433` to the VPS's `localhost:5432`. Point
any client (psql, DBeaver, pgAdmin, DataGrip) at `localhost:5433`:

```powershell
psql "postgresql://digitex:<password>@localhost:5433/digitex"
```

### Why an SSH tunnel and not exposing 5432 publicly

- No port 5432 on the public internet — scanners can't reach Postgres at all.
- Auth is your SSH key *plus* the DB password. Two independent layers.
- Encrypted in transit (SSH handles it).
- No firewall changes, no VPN.
- The bot's own connection is unchanged — it talks to `postgres:5432` over
  the internal docker network, never touching the host's loopback.

______________________________________________________________________

## 4. Backups & restore

### 4.1 Daily backup via cron

```cron
# /etc/cron.d/digitex-db-backup  (on the VPS)
0 3 * * * root docker exec digitex-postgres pg_dump -U digitex -Fc digitex \
    > /opt/digitex/backups/$(date +\%F).dump
```

Rotate weekly:

```cron
0 4 * * 0 root find /opt/digitex/backups -mtime +14 -delete
```

### 4.2 Restore

```bash
# on the VPS
docker exec -i digitex-postgres pg_restore -U digitex -d digitex -c < <file>.dump
```

The `-c` flag drops existing objects before recreating them. To restore into
a fresh DB instead, drop & recreate the database first:

```bash
docker exec -it digitex-postgres psql -U digitex -d postgres -c \
  "DROP DATABASE digitex; CREATE DATABASE digitex OWNER digitex;"
docker exec -i digitex-postgres pg_restore -U digitex -d digitex < <file>.dump
```

______________________________________________________________________

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `POSTGRES_PASSWORD must be set` | `.env` symlink missing or empty | `ln -s .env.production .env`, verify with `cat .env` |
| Bot exits immediately, no logs | `BOT_TOKEN` missing/invalid | Check `.env.production`, `docker compose logs bot` |
| `relation "…" does not exist` | Migrations not applied after `git pull` | `docker compose run --rm bot uv run digitex-db upgrade` |
| `populate_db.py` says "connection refused" | SSH tunnel closed | Reopen terminal 1; check tunnel command is still running |
| Bot can't reach DB inside container | Postgres healthcheck failing | `docker compose logs postgres` — usually wrong `POSTGRES_PASSWORD` |
| Disk full on VPS | Old backups + docker images piling up | `docker system prune -a`, prune `/opt/digitex/backups` |

### Rollback a bad deploy

```bash
cd /opt/digitex
git log --oneline -n 10               # find the previous good SHA
git checkout <sha>
docker compose build --no-cache bot
docker compose up -d bot
```

If the bad deploy included a migration, you'll also need a backup restore
(§4.2) — Alembic has no automated downgrade for hand-written SQL revisions.
