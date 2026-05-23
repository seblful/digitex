# Database Guide

PostgreSQL 17 with psycopg 3 (async) + connection pool, Alembic-managed
schema migrations. This guide covers running Postgres locally for dev,
deploying it alongside the bot on a single VPS, connecting from your PC,
backups, and optional hardening.

## Schema migrations

```bash
uv run digitex-db upgrade        # apply all pending migrations
uv run digitex-db current        # show what's applied
uv run digitex-db history        # list revisions
uv run digitex-db revision "msg" # scaffold a new empty revision
```

Migration files live in `migrations/versions/`. They are hand-written raw
SQL — there is no ORM, so autogenerate is disabled. Every schema change is
one new file added to `migrations/versions/`, applied with
`digitex-db upgrade`.

## Local development

`docker-compose.yml` ships a `postgres:17-alpine` service. `POSTGRES_PASSWORD`
is required — set it in `.env.development` (already set to `digitex` for local
dev). The host port is `5433` to avoid clashing with any native Postgres install.

```bash
docker compose up -d postgres
uv run digitex-db upgrade
uv run python scripts/populate_db.py
```

`.env.development` points `DATABASE_URL` at this local instance.

## Deploying Postgres on the same VPS as the bot

The compose file runs Postgres alongside the bot. Two things to get right
before going live:

### 1. Bind Postgres to localhost only

The compose file already does this (`127.0.0.1:5433:5432`). Port 5433 is
*not* exposed to the public internet — scanners cannot reach it. The bot
container still connects through the internal docker network using the
service name `postgres`.

### 2. Set a real password

Generate one and add it to `.env.production` on the VPS:

```bash
openssl rand -base64 24
```

```env
POSTGRES_PASSWORD=<the generated value>
```

`DATABASE_URL` is derived automatically from `POSTGRES_PASSWORD` by
`docker-compose.yml` — you do not need to set it manually.

## Connecting to the VPS database from your PC

Use an **SSH tunnel** — your existing SSH key is the auth layer, no extra
ports get exposed. Use port `5433` on the local side to avoid colliding with
any local Postgres instance:

```powershell
# in one PowerShell window — keep it open while you work
ssh -L 5433:localhost:5432 root@<vps-ip>
```

That forwards your PC's `localhost:5433` to the VPS's `localhost:5432`. Then
point any client (psql, DBeaver, pgAdmin, DataGrip) at `localhost:5433` on
your PC with the credentials from `.env`. The connection is encrypted by SSH
end-to-end; no extra DB SSL config is needed for the tunnel.

```powershell
psql "postgresql://digitex:<password>@localhost:5433/digitex"
```

For seeding the production database through the tunnel, see
[data-update.md](data-update.md).

### Why an SSH tunnel and not exposing 5432 publicly

- No port 5432 on the public internet — scanners can't reach Postgres at all.
- Auth is your SSH key *plus* the DB password. Two independent layers.
- Encrypted in transit automatically (the SSH tunnel does it).
- No firewall rule changes, no VPN to set up.
- The bot's connection is unchanged — it talks to `postgres:5432` over the
  internal docker network, never touching the host's loopback.

## Backups (self-hosted = your job)

Add a daily `pg_dump` to cron on the VPS:

```cron
# /etc/cron.d/digitex-db-backup  (on the VPS)
0 3 * * * root docker exec digitex-postgres pg_dump -U digitex -Fc digitex \
    > /opt/digitex/backups/$(date +\%F).dump
```

Rotate old dumps weekly: `find /opt/digitex/backups -mtime +14 -delete`.

Restore with `pg_restore`:

```bash
docker exec -i digitex-postgres pg_restore -U digitex -d digitex -c < <file>.dump
```

## Optional: least-privileged app role

For an extra layer, run app queries as a role that lacks DDL:

```sql
CREATE ROLE digitex_app LOGIN PASSWORD '<app-secret>';
GRANT CONNECT ON DATABASE digitex TO digitex_app;
GRANT USAGE ON SCHEMA public TO digitex_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO digitex_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO digitex_app;
```

Point the bot at `digitex_app` and keep migrations running as the
superuser `digitex`. If the DB is ever on a different host than the bot,
also set `DB_SSLMODE=require`.
