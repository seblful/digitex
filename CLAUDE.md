# AI Coding Guidelines

## Core Principles

### 1. Think Before Coding

**Don't assume. Surface tradeoffs. Ask when uncertain.**

- State assumptions explicitly
- Present multiple interpretations when they exist
- Push back if a simpler approach exists
- Name what's confusing before proceeding

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features, abstractions, or configurability beyond what was asked
- No error handling for impossible scenarios
- If it could be 50 lines but is 200, rewrite it

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting
- Match existing style even if you'd do it differently
- Remove only the orphans your changes create
- Mention unrelated dead code; don't delete it

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → Write tests for invalid inputs, then make them pass
- "Fix the bug" → Write a test that reproduces it, then make it pass

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

______________________________________________________________________

## Project Standards

**All AI agents must strictly adhere to these rules.**

- **Code**: Follow patterns in `python-patterns` — see `python-testing` for tests, `commits` for git
- **Tests**: All features and bug fixes require tests
- **Commits**: Only commit when explicitly asked

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

______________________________________________________________________

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:

- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
