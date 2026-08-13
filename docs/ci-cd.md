# CI/CD

How code, schema, and data reach production. GitHub Actions runs the checks and
the release; the VPS only pulls a prebuilt image.

For the VPS itself — first-time setup, DB access, backups — see
[production.md](production.md).

______________________________________________________________________

## 1. Branch flow

| Branch | On push | Deploys? |
| ---------------- | ---------------------------------------------- | -------- |
| feature branches | nothing (open a PR to get checks) | no |
| `develop` | `ci.yml` — lint, types, tests, docker build | no |
| `main` | `deploy.yml` — the same checks, then a release | **yes** |

Day-to-day: branch off `develop`, PR into `develop`, and when it's ready
PR `develop` → `main`. Merging to `main` is the deploy.

```
feature ──PR──> develop ──PR──> main ──> GHCR image ──> VPS
                   │               │
                 checks       checks + release
```

## 2. What runs

### ci.yml

Four jobs in parallel — `lint` (the pre-commit hooks: ruff, ruff-format,
mdformat), `types` (`ty check`), `test` (the full pytest suite against a real
Postgres 17 service container), and `docker build` (builds the bot image and
proves its entrypoint imports).

The suite includes the integration tests: `DIGITEX_TEST_DSN` points them at the
service container, so they don't start one through testcontainers.

### deploy.yml

Runs `ci.yml` as a gate, then:

1. **publish** — builds the image and pushes
   `ghcr.io/seblful/digitex-bot:sha-<short>` plus `:latest`.
1. **release** — SSHes to the VPS and runs `scripts/deploy.sh`, which pins the
   tag, pulls it, applies migrations **with the new image**, restarts the bot,
   and waits for the healthcheck.

A failed migration aborts before the running bot is touched. A bot that never
reports healthy is rolled back to the previous tag automatically.

## 3. One-time setup

### 3.1 Repository secrets

`Settings → Secrets and variables → Actions → Secrets`:

| Secret | Value |
| ----------------- | ------------------------------------------------------- |
| `VPS_HOST` | VPS IP or hostname |
| `VPS_USER` | SSH user (`root`, per the current runbook) |
| `VPS_SSH_KEY` | **Private** key with access to the VPS (full PEM text) |
| `VPS_KNOWN_HOSTS` | Output of `ssh-keyscan -H <vps-ip>` — pins the host key |

Optional variables (same page, `Variables` tab) — defaults in parentheses:
`VPS_PORT` (`22`), `VPS_APP_DIR` (`/opt/digitex`).

`GITHUB_TOKEN` needs no setup: the workflow uses it to push to GHCR and, over
the SSH session, to authenticate that one `docker pull`.

Generate a deploy-only keypair rather than reusing your laptop's:

```bash
ssh-keygen -t ed25519 -C "github-actions-digitex" -f ~/.ssh/digitex_deploy
ssh-copy-id -i ~/.ssh/digitex_deploy.pub root@<vps-ip>
ssh-keyscan -H <vps-ip>                 # → VPS_KNOWN_HOSTS
cat ~/.ssh/digitex_deploy               # → VPS_SSH_KEY
```

### 3.2 VPS

The VPS keeps the git checkout (for `docker-compose.yml` and the scripts) but
no longer builds anything. If it was set up per production.md §1, it is already
ready. Confirm:

```bash
cd /opt/digitex
git remote -v          # must point at the GitHub repo
readlink -f .env       # must resolve to .env.production
docker compose ps
```

> `scripts/deploy.sh` checks out the exact deployed commit, so **local edits
> under `/opt/digitex` other than `.env.production` are discarded** on release.

Optional: make the GHCR package public
(`github.com/users/seblful/packages` → `digitex-bot` → visibility) so manual
`docker compose pull` on the VPS works without logging in.

### 3.3 Branch protection (recommended)

`Settings → Branches → Add rule` for `main`: require a pull request, and require
the `lint` / `types` / `test` / `docker build` checks to pass. That's what keeps
`main` deployable by definition.

## 4. Shipping changes

### Code

Merge to `main`. Nothing to do by hand.

### Schema (a new migration)

Same — merge to `main`. `scripts/deploy.sh` runs `digitex-db upgrade` with the
new image before starting it. Write the migration per
[database-reference.md](database-reference.md).

### Data (new extracted questions)

Images live only on your machine (`extraction/data/` is gitignored), so this
one path stays laptop-driven:

```powershell
$env:VPS_HOST = "<vps-ip>"
./scripts/seed_prod.ps1                 # every subject
./scripts/seed_prod.ps1 -Subject biology
```

The script opens the SSH tunnel, migrates, seeds through it, and closes the
tunnel. It prompts for the production `POSTGRES_PASSWORD` unless
`$env:PROD_DB_PASSWORD` is set.

Order when code, schema, and data all change: merge to `main` first (schema and
code land together), then seed.

## 5. Rollback

Re-run the deploy with an older tag — `Actions → Deploy → Run workflow` and
enter e.g. `sha-abc1234`. Tags are listed on the package page, and each deploy's
summary names the tag it released.

By hand on the VPS, equivalently:

```bash
cd /opt/digitex
TAG=sha-abc1234 bash scripts/deploy.sh
```

A rollback restores the image, not the database. If the bad deploy applied a
migration, restore from a backup — production.md §4.2.

## 6. Troubleshooting

| Symptom | Cause | Fix |
| ------------------------------------------- | ------------------------------------ | ------------------------------------------------------------- |
| `Host key verification failed` | `VPS_KNOWN_HOSTS` missing or stale | Re-run `ssh-keyscan -H <vps-ip>` and update the secret |
| `Permission denied (publickey)` | `VPS_SSH_KEY` wrong or not installed | Re-run `ssh-copy-id`; the secret needs the whole PEM |
| `denied` / `unauthorized` on `docker pull` | GHCR package not visible to the VPS | Make the package public, or check the release job's login step |
| `no .env in /opt/digitex` | `.env` symlink missing | `ln -s .env.production .env` (production.md §1.3) |
| Release rolled itself back | Bot never reported healthy | Read the job log's `docker compose logs` tail — usually config |
| `uv.lock` out of date in CI | Dependency change not locked | `uv lock` locally and commit the result |
| Local `import torch` breaks after a `uv sync` | torch now lives in an extra | `uv sync --extra cu130` (local-setup.md §1) |
