# Database Reference

Schema-migration CLI, migration conventions, and optional hardening.

For local Postgres setup, see [local-setup.md](local-setup.md).
For VPS Postgres deployment, SSH tunnels, backups, and restore, see
[production.md](production.md).

## Stack

PostgreSQL 17 with psycopg 3 (async) + connection pool. Alembic-managed
schema migrations using **hand-written raw SQL**. No ORM, no autogenerate.

## Migration CLI

```bash
uv run digitex-db upgrade        # apply all pending migrations
uv run digitex-db current        # show what's applied
uv run digitex-db history        # list revisions
uv run digitex-db revision "msg" # scaffold a new empty revision
uv run digitex-db check-images   # rows vs. question image files on this machine
```

## Question images are files

`images` stores each question image's `object_key` (its path below the corpus
root, e.g. `biology/2016/1/A/3.jpg`) and its `content_hash` — not the bytes.
The corpus itself is synced to the server separately; see
[production.md §1.5](production.md#15-seed-the-data-from-your-laptop).

Two consequences worth knowing before you touch this table:

- The hash is not redundant with the key. Re-extracting a question rewrites the
  *same* path, and the hash is the only thing that shows it changed — which is
  what drops the cached `telegram_file_id` so the bot stops serving the
  superseded upload.
- A row and its file can drift apart, which no constraint can catch.
  `check-images` is the reconcile, and it exits non-zero so a deploy can gate
  on it.

## Authoring a migration

The scripts and `alembic.ini` live *inside* the package, at
`packages/service/src/digitex/db/`, and are found through `importlib.resources`
— see `digitex.db.schema.alembic_config()`. That is what lets the production
image install `digitex-service` as an ordinary wheel with no copy of the source
tree: nothing resolves a migration path by walking up from a source file. A dev
checkout is an editable install, so `revision` still writes into the tree below.

1. `uv run digitex-db revision "short description"` — scaffolds a new file
   under `packages/service/src/digitex/db/migrations/versions/`.
1. Write raw SQL in the `upgrade()` and `downgrade()` functions. Use
   `op.execute("…")`.
1. Commit the new file alongside any code that depends on it.

Conventions:

- One concern per migration. Don't bundle unrelated changes.
- Always provide a `downgrade()` — even if it's only `op.execute("…")` to
  drop what `upgrade()` created. Note that for destructive changes downgrade
  is necessarily lossy; a backup restore (see
  [production.md §4.2](production.md#42-restore)) is the real safety net.
- Names match the domain glossary in [CONTEXT.md](../CONTEXT.md). Don't
  invent new terms in SQL.

## Optional: least-privileged app role

For an extra layer of defence, run app queries as a role that lacks DDL:

```sql
CREATE ROLE digitex_app LOGIN PASSWORD '<app-secret>';
GRANT CONNECT ON DATABASE digitex TO digitex_app;
GRANT USAGE ON SCHEMA public TO digitex_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO digitex_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO digitex_app;
```

Point the bot at `digitex_app` and keep migrations running as the
superuser `digitex`. If the DB ever lives on a different host than the bot,
also set `DB_SSLMODE=require`.
