#!/usr/bin/env bash
# Release one image tag on the VPS. Run by .github/workflows/deploy.yml over
# SSH; safe to run by hand for a rollback:
#
#     TAG=sha-abc1234 bash deploy/deploy.sh
#
# Order matters: migrate with the new image *before* the new bot starts, so a
# failed migration leaves the old bot serving. A GHCR password on stdin is
# consumed by `docker login`; without one the image must be public.
#
# Everything lives in main() because the release checks out the very repo this
# file sits in — bash reads a script incrementally, so a plain top-level body
# could be rewritten mid-run. A function is parsed before it runs.
set -Eeuo pipefail

main() {
  local TAG=${TAG:?TAG is required (e.g. sha-abc1234)}
  local APP_DIR=${APP_DIR:-/opt/digitex}
  local GHCR_USER=${GHCR_USER:-}
  local HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-150}

  # Image tags carry the commit's short SHA, so the tag alone says which
  # commit's compose file belongs with the image. A tag without that prefix
  # (`latest`) names no commit, and the checkout is left alone.
  local SHA=${SHA:-}
  if [ -z "$SHA" ] && [ "${TAG#sha-}" != "$TAG" ]; then
    SHA=${TAG#sha-}
  fi

  cd "$APP_DIR"

  # Compose reads ./.env both for ${VAR} substitution and as the bot's
  # env_file, and this script pins the released tag in it.
  if [ ! -e .env ]; then
    echo "error: no .env in $APP_DIR — see docs/production.md §1.3" >&2
    return 1
  fi
  local env_file
  env_file=$(readlink -f .env)

  local previous_tag previous_sha
  previous_tag=$(sed -n 's/^TAG=//p' "$env_file" | tail -1)
  previous_sha=$(git rev-parse HEAD)

  if [ -n "$GHCR_USER" ] && [ ! -t 0 ]; then
    log "authenticating to ghcr.io"
    docker login ghcr.io --username "$GHCR_USER" --password-stdin
  fi

  # The compose file and scripts come from git; the app itself comes from the
  # image. Local edits under $APP_DIR outside .env are discarded.
  if [ -n "$SHA" ]; then
    log "syncing repo to $SHA"
    git fetch --quiet origin
    git checkout -f --quiet "$SHA"
  fi

  # The container runs as uid 10001, so the bind-mounted log directory has to
  # belong to it or the first write fails. Idempotent, and cheaper than a
  # one-time setup step nobody remembers.
  mkdir -p logs
  chown -R 10001:10001 logs

  log "pulling $TAG"
  set_tag "$env_file" "$TAG"
  docker compose pull bot

  log "applying migrations"
  docker compose run --rm bot digitex-db upgrade

  log "starting bot"
  docker compose up -d --no-build bot

  log "waiting for healthy (${HEALTH_TIMEOUT}s)"
  if wait_healthy "$HEALTH_TIMEOUT"; then
    log "deployed $TAG"
    docker image prune -f >/dev/null 2>&1 || true
    return 0
  fi

  echo "error: bot did not become healthy" >&2
  docker compose logs --tail 50 bot || true

  if [ -z "$previous_tag" ] || [ "$previous_tag" = "$TAG" ]; then
    echo "error: no previous tag recorded — not rolling back" >&2
    return 1
  fi

  log "rolling back to $previous_tag"
  set_tag "$env_file" "$previous_tag"
  git checkout -f --quiet "$previous_sha"
  docker compose up -d --no-build bot
  echo "warning: rolled the image back; any migration this deploy applied is" \
    "still in place — see docs/production.md §4.2 to restore from a backup" >&2
  return 1
}

log() { printf '\n=== %s\n' "$*"; }

# Pin TAG in the env file rather than passing it per command, so every later
# `docker compose …` on this box — including a hand-run one — sees the tag that
# is actually deployed.
set_tag() {
  local file=$1 value=$2
  if grep -q '^TAG=' "$file"; then
    sed -i "s|^TAG=.*|TAG=$value|" "$file"
  else
    printf 'TAG=%s\n' "$value" >>"$file"
  fi
}

# A crash-looping bot keeps flapping back to "running" thanks to
# restart: unless-stopped, so the deadline — not the state — is what decides.
wait_healthy() {
  local deadline=$((SECONDS + $1)) status
  while ((SECONDS < deadline)); do
    status=$(docker inspect --format '{{.State.Health.Status}}' digitex-bot 2>/dev/null || echo missing)
    case "$status" in
      healthy) return 0 ;;
      unhealthy) return 1 ;;
    esac
    sleep 5
  done
  return 1
}

main "$@"
