"""Alembic environment.

Migrations are hand-written raw SQL (`op.execute(...)`); autogenerate is not
supported because the project intentionally has no ORM models — repositories
own the SQL directly. `target_metadata = None` enforces this.
"""

from __future__ import annotations

from alembic import context

from digitex.config import get_settings

config = context.config

# Inject the DSN from pydantic-settings so a single source of truth governs
# both runtime connections and migrations.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database.conninfo)

target_metadata = None


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection (sync psycopg)."""
    from sqlalchemy import create_engine

    url = config.get_main_option("sqlalchemy.url")
    # Use SQLAlchemy + psycopg3 sync driver for the alembic engine.
    # Migrations are short-lived, sync is fine.
    sa_url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(sa_url, future=True)

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()

    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
