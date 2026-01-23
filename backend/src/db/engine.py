"""
SQLAlchemy Engine factory with connection pooling.

This module creates and manages the SQLAlchemy engine singleton.
The engine is configured based on the database backend (SQLite or PostgreSQL).

Key concepts:
- Singleton pattern: One engine instance shared across the application
- Connection pooling: Reuses database connections for efficiency
- NullPool option: For AWS Lambda + RDS Proxy (let proxy handle pooling)

Usage:
    from src.database.engine import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
"""

from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

from src.config import DatabaseBackend, DatabaseConfig

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
        # NullPool creates a new connection for each request and closes it after
        # This is efficient when RDS Proxy manages the actual pool
        poolclass = NullPool
        pool_kwargs = {}
    elif config.is_sqlite:
        # For SQLite: use QueuePool with small pool
        # Size > 1 needed because dashboard mixes old (conn) and new (repository) patterns
        poolclass = QueuePool
        pool_kwargs = {
            "pool_size": 5,
            "max_overflow": 2,
            "pool_timeout": 30,
        }
    else:
        # For PostgreSQL (direct connection, not via proxy):
        # Use QueuePool to maintain a pool of connections
        poolclass = QueuePool
        pool_kwargs = {
            "pool_size": config.pool_size,
            "max_overflow": config.max_overflow,
            "pool_recycle": config.pool_recycle,
            "pool_pre_ping": config.pool_pre_ping,
        }

    # SQLite-specific connection arguments
    connect_args = {}
    if config.is_sqlite:
        # Allow SQLite to be used across threads (needed for Streamlit)
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        config.url,
        poolclass=poolclass,
        connect_args=connect_args,
        **pool_kwargs
    )

    # SQLite: Enable foreign key enforcement
    # By default, SQLite doesn't enforce foreign keys - we need to enable it
    if config.is_sqlite:
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

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
