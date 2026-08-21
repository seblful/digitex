# Docs Index

Operational and workflow docs for Digitex. Source files live under `docs/`.

## Run it

- [Local setup](local-setup.md) — laptop dev: deps, Postgres, schema, seed, bot, tests
- [CI/CD](ci-cd.md) — branch flow, what ships on a merge to `main`, secrets, rollback
- [Production runbook](production.md) — VPS deploy, day-2 ops, DB access, backups, troubleshooting

## Reference

- [Database](database-reference.md) — migration CLI, schema conventions, hardening
- [`.env.example`](../.env.example) — every supported env var, with defaults
- [Glossary](glossary.md) — every domain term, the settings tree, the deploy boundary
- [CONTEXT.md](../CONTEXT.md) — the map: overview, architecture, decisions, non-goals
- [README](../README.md#project-structure) — the three packages, their layers, and the data root

## Workflows

- [Extraction](extraction.md) — books → question images via YOLO, with the review window
- [Training](training.md) — YOLO model training
- [Label Studio](label-studio.md) — annotation server

## Project

- [todo](todo.md) — open tasks
