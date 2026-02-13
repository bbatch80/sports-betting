"""
SQLAlchemy Engine factory with connection pooling.

This module creates and manages the SQLAlchemy engine singleton.
Configured for PostgreSQL only (no SQLite fallback).

Key concepts:
- Singleton pattern: One engine instance shared across the application
- Connection pooling: Reuses database connections for efficiency
- NullPool option: For AWS Lambda + RDS Proxy (let proxy handle pooling)

Usage:
    from src.db.engine import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
"""

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool, QueuePool

from src.config import DatabaseConfig

# Module-level singleton
_engine: Optional[Engine] = None


def get_engine(config: Optional[DatabaseConfig] = None) -> Engine:
    """
    Get or create the SQLAlchemy engine.

    Uses singleton pattern to reuse connections across requests.
    For Lambda: engine is created once per container lifetime.

    Args:
        config: Optional database configuration. If not provided,
                loads from environment variables.

    Returns:
        SQLAlchemy Engine instance

    Why singleton?
        Creating a database connection is expensive. By reusing the same
        engine (and its connection pool), we avoid the overhead of
        establishing new connections for each request.

        In AWS Lambda, the engine persists across "warm" invocations
        within the same container, making subsequent requests faster.
    """
    global _engine

    if _engine is not None:
        return _engine

    if config is None:
        config = DatabaseConfig.from_env()

    # Choose pooling strategy based on environment
    if config.use_null_pool:
        # For Lambda + RDS Proxy: let proxy handle connection pooling
        poolclass = NullPool
        pool_kwargs = {}
    else:
        # Direct PostgreSQL connection: use QueuePool
        poolclass = QueuePool
        pool_kwargs = {
            "pool_size": config.pool_size,
            "max_overflow": config.max_overflow,
            "pool_recycle": config.pool_recycle,
            "pool_pre_ping": config.pool_pre_ping,
        }

    _engine = create_engine(
        config.url,
        poolclass=poolclass,
        **pool_kwargs
    )

    return _engine


def reset_engine() -> None:
    """
    Reset the engine singleton. Useful for testing.

    This disposes all connections in the pool and clears the singleton,
    allowing a fresh engine to be created on the next get_engine() call.
    """
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
