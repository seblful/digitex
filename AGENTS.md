# Agent Instructions & Project Standards

This document outlines the architecture, environment setup, and coding standards for this project. **All AI agents contributing to this repository must strictly adhere to these rules.**

## Environment Setup

- **Python Version**: `3.13`
- **Package Manager**:(https://docs.astral.sh/uv/)

**Workflow:**
Do not rely on standard `pip` or `venv` commands. Consistently use `uv` for dependency management and execution:

```bash
uv sync
uv run main.py
```

# Code Standards & Conventions

We strictly adhere to **PEP 8** standards. Simplicity is key—avoid over-engineering solutions.

## Formatting & Naming

- **Variables/Functions**: Use `snake_case`.
- **Classes**: Use `CamelCase` (PascalCase).
- **Docstrings**: Follow the **Google Python Style Guide**.
- **Strings**: Use f-strings (`f"…"`) for all strings that include variables or expressions.

## Imports

Group imports in this exact order, with a blank line between groups:

1. **Standard library** (e.g., `pathlib`, `logging`)
2. **Third-party** (e.g., `typer`, `chromadb`)
3. **Local / Project** (e.g., `from my_project.config import …`)

_Note: Sort imports alphabetically within each group. Use absolute imports for project code._

## Type Hinting

- **Strict Rule**: Type hints are mandatory for **all** function signatures (arguments and return types).
- **Syntax**: Use modern Python 3.10+ syntax (e.g., `list`, `str | None`).
- **Validation**: We use **ty** for static type checking and LSP integrations. Your code must pass `uvx ty` check with zero errors.

## File System Operations

- **Strict Rule**: **NEVER** use `os.path.join` or string concatenation for paths.
- **Requirement**: **ALWAYS** use `pathlib.Path`.

## CLI & Scripts

- Use `typer` for all scripts and CLI entry points. It natively integrates with our type hints to produce clear help documentation and argument parsing.

## Progress Bars

- Use `tqdm` for progress feedback when iterating over long-running loops.
- Set the `desc` parameter for clarity (e.g., `tqdm(items, desc="Processing files")`).

# Testing Standards

We use `pytest` as our testing framework. All new features and bug fixes must be accompanied by tests.

- **Location**: Place all tests in the `tests/` directory. Mirror the project structure (e.g., tests for `src/parser.py` go in `tests/test_parser.py`).
- **Naming**: Test functions must begin with `test_` (e.g., `test_extract_text_returns_string()`).
- **Fixtures**: Use `pytest` fixtures for setup and teardown instead of standard class-based setups.
- **Execution**: Always run tests using `uv`:

  ```bash
  uv run pytest
  ```

# Git Workflow

We follow a **Simplified Conventional Commits** format. This keeps history clean and machine-readable without unnecessary complexity.

## Commit Message Format

```text
<type>: <description>
```

_(Keep the description lowercase, concise, and do not end with a period)._

## Allowed Types

Use only the following core types:

| Type           | Description                                                               |
| :------------- | :------------------------------------------------------------------------ |
| **`feat`**     | A new feature or functionality                                            |
| **`fix`**      | A bug fix                                                                 |
| **`refactor`** | A code change that neither fixes a bug nor adds a feature (e.g., cleanup) |
| **`test`**     | Adding missing tests or correcting existing tests                         |
| **`docs`**     | Documentation-only changes (like updating README.md)                      |
| **`chore`**    | Routine tasks, dependency updates (`uv.lock`), or tool configurations     |
