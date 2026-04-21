# Agent Instructions & Project Standards

**All AI agents must strictly adhere to these rules.**

---

## Dependencies & Package Manager

- **Python**: `3.13`
- **Package Manager**: [uv](https://docs.astral.sh/uv/) — **never** use `pip` or `venv`

| Command                  | Purpose                   |
| :----------------------- | :------------------------ |
| `uv add <package>`       | Add production dependency |
| `uv add --dev <package>` | Add dev dependency        |
| `uv sync`                | Lock dependencies         |
| `uv run <cmd>`           | Run commands              |

---

## Code Standards

Follow **PEP 8**. Simplicity is key—avoid over-engineering.

### Naming

- Variables/Functions: `snake_case`
- Classes: `CamelCase`
- Strings: f-strings (`f"..."`)

### Docstrings

- Use Google-style docstrings for all public functions and classes

### Imports

- **Order** (enforced by Ruff): stdlib → third-party → local
- Sort alphabetically within groups, use absolute imports
- Organize with: `uv run ruff check --select I --fix .`

### Type Hints

- Mandatory for all function signatures (arguments and return types)
- Use Python 3.10+ syntax (`list`, `str | None`)
- Validate with: `uvx ty` — must pass with zero errors

### File System

- **Always** use `pathlib.Path`
- **Never** use `os.path.join` or string concatenation

### Logging

- Use **structlog** only (never `logging` directly)
- Configure via `setup_logging()` from `digitex.logging`
- Create loggers with: `structlog.get_logger()`

### Settings

- Use **Pydantic Settings** with structured groups
- Location: `src/digitex/settings.py`
- Import: `from digitex.settings import settings`
- Nest with `__` delimiter: `APP__DEBUG`, `LOGGING__LEVEL`

### Data Classes vs Pydantic

| Use Case                    | Choice                             | Reason                           |
| :-------------------------- | :--------------------------------- | :------------------------------- |
| Configuration from env vars | `BaseSettings`                     | Built-in env var loading         |
| Simple data containers      | `pydantic.dataclass`               | Validation with dataclass syntax |
| External data validation    | `BaseModel`                        | Runtime validation               |
| Immutable DTOs              | `@pydantic.dataclass(frozen=True)` | Immutability                     |

**Best Practices:**

- Use **`pydantic.dataclass`** for simple data containers that benefit from light validation
- Use **`BaseModel`** for data that comes from external sources and needs full validation
- Use **`BaseSettings`** for configuration that should be customizable via environment variables
- Keep settings in `settings.py` - don't mix data models with configuration

### CLI & Scripts

- Use **typer** for all entry points

### Progress Bars

- Use `tqdm` with `desc` parameter for clarity

### Security

- **Never** hardcode secrets (API keys, passwords, tokens)
- Store secrets in environment variables or `.env` files (never commit `.env`)
- Use `pydantic-settings` for secret management

### Datetime

- **Always** use timezone-aware datetimes
- Use `datetime.now(timezone.utc)` instead of `datetime.utcnow()`
- Serialize with `.isoformat()` for JSON; parse with `datetime.fromisoformat()`

### ABC & Protocols

- Use `abc.ABC` for explicit inheritance-based polymorphism
- Use `typing.Protocol` for structural typing (duck typing)
- Mark methods with `@abstractmethod` when requiring implementation

---

## Testing

- **Framework**: pytest
- **Location**: `tests/` mirroring project structure
- **Naming**: Functions start with `test_`
- **Fixtures**: Use pytest fixtures (not class-based setup)
- **Rule**: All new features and bug fixes require tests

Run tests with: `uv run pytest`

---

## Security & Safety

- **Commands:** Never auto-run destructive ones (`rm -rf`, `del /s`, `curl | sh`, etc.).
- **Secrets:** Never put API keys, passwords, or tokens in documentation or code. Use env vars.
- **.cursorignore:** Respect ignored paths. Don't insist on indexing `.env*`, `.ssh/`, `secrets/`.
- **MCP:** Prefer read-only access. Avoid broad tokens or full admin scopes.

---

## Git Workflow

### Rule

Only create commits when the user **explicitly asks**. Never commit proactively.

### Pre-commit

```bash
uv run pre-commit install
```

### Conventional Commits

```
<type>: <description>
```

- Lowercase, concise, no trailing period

### Types

| Type       | Purpose                                |
| :--------- | :------------------------------------- |
| `feat`     | New feature                            |
| `fix`      | Bug fix                                |
| `refactor` | Code change without feature/bug change |
| `test`     | Test updates                           |
| `docs`     | Documentation only                     |
| `chore`    | Routine tasks, dependency updates      |

### AI-Generated Code

- Mark AI-generated or AI-assisted commits with `Co-authored-by: Cursor/opencode/Claude Code`

---
