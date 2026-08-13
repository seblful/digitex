"""PostgreSQL connection settings — shared by the bot and the migrations."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings.

    The DSN is read from the ``DATABASE_URL`` env var (12-factor convention)
    or ``DB_DSN`` as a fallback.
    """

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    dsn: PostgresDsn = Field(
        default=PostgresDsn("postgresql://digitex:digitex@localhost:5432/digitex"),
        validation_alias=AliasChoices("dsn", "DB_DSN", "DATABASE_URL"),
        description="PostgreSQL connection string.",
    )
    pool_min_size: int = Field(default=2, ge=1)
    pool_max_size: int = Field(default=10, ge=1)
    pool_timeout: float = Field(default=10.0, gt=0)
    connect_timeout: int = Field(
        default=10, gt=0, description="TCP connect timeout in seconds."
    )
    statement_timeout_ms: int = Field(
        default=5000,
        ge=0,
        description="Server-side statement timeout in milliseconds.",
    )
    idle_in_transaction_timeout_ms: int = Field(
        default=10000,
        ge=0,
        description="Server-side idle-in-transaction timeout in milliseconds.",
    )
    sslmode: (
        Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        | None
    ) = Field(
        default=None,
        description="If set, appended to the DSN as ?sslmode=...",
    )

    @computed_field
    @property
    def conninfo(self) -> str:
        """DSN as a libpq conninfo string suitable for psycopg/AsyncConnectionPool."""
        dsn_str = str(self.dsn)
        params: list[str] = [f"connect_timeout={self.connect_timeout}"]
        if self.sslmode is not None:
            params.append(f"sslmode={self.sslmode}")
        sep = "&" if "?" in dsn_str else "?"
        return f"{dsn_str}{sep}{'&'.join(params)}"

    @computed_field
    @property
    def server_options(self) -> str:
        """Libpq ``options`` parameter setting statement + idle-in-tx timeouts."""
        idle_ms = self.idle_in_transaction_timeout_ms
        return (
            f"-c statement_timeout={self.statement_timeout_ms}"
            f" -c idle_in_transaction_session_timeout={idle_ms}"
        )
