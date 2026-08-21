# AI Coding Guidelines

## Core Principles

### Honest Pushback

**Flag bad ideas. Don't silently comply.**

Before executing, raise a concern if the request:

- Conflicts with an established project standard or convention
- Is an antipattern for this stack, context, or scale
- Would introduce security, correctness, or maintainability risk

State the concern in one sentence, name the specific tension, then proceed or
ask — don't lecture. If the user confirms anyway, comply and move on.

### Goal-Driven Execution

**Every task needs a check that can fail.**

Restate the request as a condition something can verify — a test, a command, an
observable output — then work until it holds. If a task admits no such check,
say so before starting instead of calling it done by inspection.

______________________________________________________________________

## Project Context

`CONTEXT.md` is the authoritative reference. Create it if missing; read it
before any non-trivial task. Five sections:

- **Overview** — what it does, who for, what it replaces
- **Architecture** — one diagram: components and data flow
- **Vocabulary** — terms whose meaning here differs from the plain-English or
  library one
- **Decisions** — X over Y because Z
- **Non-Goals** — what it deliberately will not do

It is a map, not documentation: anything a reader could get from the code or
`ls` stays out. Respect the caps in its HTML comments — at a cap, replace the
weakest entry rather than growing the list. Update it when a non-goal,
decision, domain term, or component boundary changes, not when code changes.

The rest of the domain vocabulary — every entity, process, outcome and
infrastructure term — lives in [docs/glossary.md](docs/glossary.md), which the
caps do not bind.

______________________________________________________________________

## Project Standards

**All AI agents must strictly adhere to these rules.**

- **Code**: Follow patterns in `python-patterns` — see `python-testing` for tests
- **Tests**: Every feature and bug fix requires tests
- **Type checking**: Run `uv run ty check` after adding or modifying any Python
  code; fix all errors before proceeding
- **Test suite**: Run `uv run pytest` after writing or changing code covered by
  tests; all tests must pass
- **Commits**: Commit only when explicitly asked one message before, for
  commits use [Conventional Commits](https://www.conventionalcommits.org/).
- **Logging**: Use `structlog` — log at `DEBUG` for internal state, `INFO` for
  significant lifecycle events, `WARNING` for recoverable anomalies, `ERROR`
  for failures that need attention; never log secrets or PII; prefer structured
  key-value pairs over interpolated strings
- **Layers**: Only `core` and `service` deploy. Never import `imaging`, `ml`,
  `labeling`, `pipeline` or `ui` from them — the production image does not
  install those distributions at all. `bot` may not reach `db` or `psycopg`
  either: handlers take an `OpenUow` from `domain.ports`, and
  `service/cli/bot.py` is the only module that names both sides. A new module
  belongs in the workspace member that already holds its dependencies; adding a
  studio dependency to core or service is the one change that breaks the
  deploy. `uv run lint-imports` checks the imports; `[tool.importlinter]` in
  `pyproject.toml` is the contract, `CONTEXT.md` records the decision
- **Paths**: Non-code paths derive from `settings.paths.data_root`, never from
  a module's `__file__`

| Tool | Purpose |
| :--- | :------ |
| [uv](https://docs.astral.sh/uv/) | Package manager — never use `pip` or `venv` |
| [Ruff](https://docs.astral.sh/ruff/) | Linting and formatting |
| [ty](https://github.com/astral-sh/ty) | Type checking |
| [pytest](https://pytest.org/) | Testing + coverage |
| [pre-commit](https://pre-commit.com/) | Git hooks |
| [mdformat](https://mdformat.readthedocs.io/) | Markdown formatting |
| [structlog](https://www.structlog.org/) | Structured logging |
| [pydantic](https://docs.pydantic.dev/) | Data validation and settings |
| [typer](https://typer.tiangolo.com/) | CLI entry points |

______________________________________________________________________

## Security

- **Never** hardcode secrets (API keys, passwords, tokens)
- Store secrets in environment variables or `.env` — never commit `.env`
- Use `pydantic-settings` for secret management
- Never auto-run destructive commands (`rm -rf`, `del /s`, `curl | sh`)
- Respect `.ignore` paths (`.env*`, `.ssh/`, `secrets/`)
