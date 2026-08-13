# Production Runbook

Single source of truth for everything VPS-side: first-time deploy, day-2 ops
(code/schema/data updates), database access, backups, troubleshooting.

For laptop dev setup, see [local-setup.md](local-setup.md).
For the pipeline that does most of this for you, see [ci-cd.md](ci-cd.md).
For schema/migration internals, see [database-reference.md](database-reference.md).

______________________________________________________________________

## 1. First-time deploy

### 1.1 Prerequisites

- A VPS (Ubuntu/Debian, ~$5/mo is enough)
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- Extraction output (`extraction/data/output/`) on your local machine, ready
  to seed the production database
- `rsync` on your PC (`choco install rsync`) — question images are files, and
  this is what ships them

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

cp .env.example .env
micro .env                       # fill in the values below
```

Required in `.env`:

| Variable | Value | Required |
| ----------------------- | --------------------------------------------------- | -------- |
| `BOT_TOKEN` | Your token from @BotFather | Yes |
| `BOT_ADMIN_USER_ID` | Your Telegram user ID | Yes |
| `POSTGRES_PASSWORD` | Strong password (`openssl rand -base64 24`) | Yes |
| `DB_SSLMODE` | `disable` (in-cluster) or `require` (external DB) | Yes |
| `LOGGING_CONSOLE_LEVEL` | `INFO` | No |

`DATABASE_URL` is derived automatically from `POSTGRES_PASSWORD` by Compose —
no need to set it manually. Full env reference: `.env.example`.

> This is the only env file on the box. Compose auto-loads `.env` for both
> variable substitution and the bot's environment, and the deploy pins the
> released image tag in it.

### 1.4 Start Postgres + apply schema

```bash
docker compose up -d postgres
docker compose run --rm bot digitex-db upgrade
```

### 1.5 Seed the data (from your laptop)

Data ships in two parts, because question images are files on disk and only
their keys are rows: **the images sync first, then the database is seeded**. A
row names a file, so seeding first would point the bot at images that have not
arrived yet.

`./scripts/seed_prod.ps1 -VpsHost <vps-ip>` does both in one command. The manual
equivalent, for when you want the tunnel open anyway (see
[§3 Database access](#3-database-access-from-your-pc) for why a tunnel):

```powershell
# PC — 1. the images. --delete is what removes what you deleted locally.
ssh root@<vps-ip> "mkdir -p /opt/digitex/data/questions"
cd extraction/data/output
rsync -rlt --delete --chmod=D755,F644 --info=progress2 `
    ./ root@<vps-ip>:/opt/digitex/data/questions/
cd ../../..
```

```powershell
# Terminal 1 — PC, keep open
ssh -L 5433:localhost:5432 root@<vps-ip>
```

```powershell
# Terminal 2 — PC, 2. the rows
$env:DATABASE_URL = "postgresql://digitex:<password>@localhost:5433/digitex"
uv run digitex-db populate
```

`populate` is idempotent (`get_or_create`), so re-running is safe. It
migrates the schema first, so it is also a valid way to apply a pending
migration.

> Run rsync from *inside* `extraction/data/output` and pass `./`. A Windows
> rsync reads a `C:\…` source as a remote host spec; a relative path has no
> drive letter to misread.

Confirm the two halves agree before starting the bot:

```bash
# on the VPS
docker compose run --rm bot digitex-db check-images
```

### 1.6 Start the bot

```bash
# on the VPS
docker compose up -d bot
docker compose logs -f bot
```

______________________________________________________________________

## 2. Day-2 operations

**Code and schema ship by merging to `main`** — GitHub Actions builds the image
and releases it. See [ci-cd.md](ci-cd.md) for the pipeline, its secrets, and
rollback. Data still ships from your laptop, because the images never enter git.

| What changed | How it ships |
| ------------------ | ---------------------------------------------------------- |
| Code | merge to `main` |
| Schema (migration) | merge to `main` — the release migrates before restarting |
| Extracted data | `./scripts/seed_prod.ps1` from your PC ([§2.2](#22-new-extracted-data)) |

If all three changed: merge to `main`, wait for the release, then seed.

Question images are the one thing that is *not* in the database: they live at
`/opt/digitex/data/questions` on the VPS, bind-mounted read-only into the bot,
and the `images` table stores each one's path and content hash. That keeps the
corpus (~400MB) out of every `pg_dump`, and it is why "extracted data" ships
files as well as rows.

### 2.1 Release by hand

Only needed when Actions is unavailable. `scripts/deploy.sh` is the same script
CI runs — it pins the tag, migrates, restarts, and rolls back if the bot never
becomes healthy:

```bash
# on the VPS
cd /opt/digitex
TAG=sha-abc1234 bash scripts/deploy.sh      # a tag from the GHCR package page
```

To build on the VPS instead of pulling a published image (last resort — it
compiles on the box):

```bash
cd /opt/digitex
git fetch origin main && git checkout -f origin/main
docker compose build --no-cache bot
docker compose run --rm bot digitex-db upgrade
docker compose up -d bot
```

> A release leaves the checkout detached at the deployed commit, so `git pull`
> reports "not currently on a branch" — fetch and check out explicitly, as
> above. `docker compose build` also overwrites the tagged image the pinned
> `TAG` names, which is what makes the locally built one run.

### 2.2 New extracted data

```powershell
# on your PC, from the repo root
$env:VPS_HOST = "<vps-ip>"
./scripts/seed_prod.ps1                     # or -Subject biology
```

Syncs the images, then opens the SSH tunnel, migrates, seeds, and closes it.
Idempotent (`get_or_create`), so re-running is safe. The manual equivalent is
[§1.5](#15-seed-the-data-from-your-laptop).

The image sync always covers the whole corpus even with `-Subject`: it deletes
remote files that no longer exist locally, and a partial sync could not tell a
subject you did not seed from a subject you deleted.

A re-extracted image keeps its path, so the row's content hash is what marks it
changed — and that drops the cached Telegram `file_id`, which is what makes the
bot upload the corrected image instead of re-serving the old one. Sync without
seeding and the two fall out of step; `digitex-db check-images` says so:

```bash
# on the VPS
docker compose run --rm bot digitex-db check-images
```

It lists keys whose file is missing (sync), files whose bytes no longer match
the seeded hash (populate), and files nothing points at, then exits non-zero.

### 2.3 Manage / inspect

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

The dump covers the database only. Question images are files under
`/opt/digitex/data/questions`, and they are not backed up here on purpose —
your laptop's `extraction/data/output/` is their source of truth, and
`scripts/seed_prod.ps1` puts them back. A restore is therefore: restore the
dump, re-run the seed script, then `digitex-db check-images`.

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
| `POSTGRES_PASSWORD must be set` | `.env` missing or empty | `cp .env.example .env` and fill it in, verify with `cat .env` |
| Bot exits immediately, no logs | `BOT_TOKEN` missing/invalid | Check `.env`, `docker compose logs bot` |
| `relation "…" does not exist` | Migrations not applied | `docker compose run --rm bot digitex-db upgrade` |
| `digitex-db populate` says "connection refused" | SSH tunnel closed | Reopen terminal 1; check tunnel command is still running |
| Bot logs "Question images directory not found" and exits | `./data/questions` not on the VPS | Run `scripts/seed_prod.ps1` from your PC to sync it |
| One question fails to send, the rest work | Its file never synced | `digitex-db check-images`, then re-run the seed script |
| Bot serves an old image after re-extraction | Images synced, rows not re-seeded | `digitex-db populate` — the hash change drops the stale `file_id` |
| Bot can't reach DB inside container | Postgres healthcheck failing | `docker compose logs postgres` — usually wrong `POSTGRES_PASSWORD` |
| Disk full on VPS | Old backups + docker images piling up | `docker system prune -a`, prune `/opt/digitex/backups` |

### Rollback a bad deploy

A release that never becomes healthy rolls itself back. To undo one that came up
healthy but misbehaves, re-release the previous tag — `Actions → Deploy → Run workflow`, or on the VPS:

```bash
cd /opt/digitex
grep '^TAG=' .env                     # what is deployed now
TAG=sha-abc1234 bash scripts/deploy.sh
```

Published tags are listed on the GHCR package page; each deploy run's summary
names the one it released.

If the bad deploy included a migration, you'll also need a backup restore
(§4.2) — Alembic has no automated downgrade for hand-written SQL revisions.
