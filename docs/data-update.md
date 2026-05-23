# Data Update Guide

## New questions extracted (extraction output changed)

```powershell
# Terminal 1 — open tunnel, keep it open
ssh -L 5433:localhost:5432 root@<vps-ip>
```

```bash
# Terminal 2 — seed new data through the tunnel
DATABASE_URL="postgresql://digitex:<password>@localhost:5433/digitex" \
    uv run python scripts/populate_db.py
```

`populate_db.py` is idempotent — uses `get_or_create`, so re-running is safe.

## Schema changed (new migration added)

```bash
# on the VPS
cd /opt/digitex
git pull
docker compose run --rm bot uv run digitex-db upgrade
docker compose up -d bot  # restart if code also changed
```

## Both (new migration + new data)

Schema always before data.

```bash
# VPS — apply schema first
git pull
docker compose run --rm bot uv run digitex-db upgrade
docker compose up -d bot
```

```bash
# Local — then seed through the tunnel
DATABASE_URL="postgresql://digitex:<password>@localhost:5433/digitex" \
    uv run python scripts/populate_db.py
```
