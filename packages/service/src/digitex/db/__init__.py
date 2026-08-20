"""Async PostgreSQL data access — pools, transactions, and the SQL behind them.

Two things are exported: how to get a pool, and how to open a transaction on
one. The repositories are reached through a :class:`UnitOfWork`, and the
migrations through :mod:`digitex.db.schema`, so importing this package costs
neither Alembic nor the corpus loader.
"""

from .pool import build_pool, null_pool_lifespan, pool_lifespan
from .unit_of_work import UnitOfWork

__all__ = ["UnitOfWork", "build_pool", "null_pool_lifespan", "pool_lifespan"]
