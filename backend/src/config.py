"""
Configuration settings for sports betting analytics.

Contains:
- Sport definitions for The Odds API
- Database configuration (SQLite/PostgreSQL)
- API settings
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

# Sport definitions
SPORTS = {
    'nfl': {
        'key': 'americanfootball_nfl',
        'name': 'NFL',
    },
    'nba': {
        'key': 'basketball_nba',
        'name': 'NBA',
    },
    'nhl': {
        'key': 'icehockey_nhl',
        'name': 'NHL',
    },
    'ncaam': {
        'key': 'basketball_ncaab',
        'name': 'NCAAM',
    },
    'ncaaf': {
        'key': 'americanfootball_ncaaf',
        'name': 'NCAAF',
    },
}

# Default settings
DEFAULT_MARKETS = ['spreads']
DEFAULT_REGIONS = ['us']
DEFAULT_BOOKMAKER = ['draftkings']

# API settings
API_RATE_LIMIT_DELAY = 1.0  # Seconds between requests


def get_sport_api_key(sport_key: str) -> str:
    """Get API key for a sport"""
    if sport_key not in SPORTS:
        raise ValueError(f"Unknown sport: {sport_key}")
    return SPORTS[sport_key]['key']


def get_all_sport_keys() -> list:
    """Get list of all sport keys"""
    return list(SPORTS.keys())


# =============================================================================
# Database Configuration
# =============================================================================

class DatabaseBackend(Enum):
    """Supported database backends."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


# Default paths
DEFAULT_SQLITE_PATH = Path(__file__).parent.parent / "data" / "analytics.db"


@dataclass
class DatabaseConfig:
    """
    Database configuration that supports both SQLite and PostgreSQL.

    Automatically detects backend type from DATABASE_URL environment variable.
    Falls back to SQLite when DATABASE_URL is not set (local development).

    Example usage:
        config = DatabaseConfig.from_env()
        print(f"Using {config.backend.value} database")
    """
    backend: DatabaseBackend
    url: str

    # Connection pool settings (PostgreSQL only)
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    pool_pre_ping: bool = True  # Verify connection is alive before using

    # Lambda-specific settings
    use_null_pool: bool = False  # True when using RDS Proxy

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        Create configuration from environment variables.

        Environment variables:
            DATABASE_URL: PostgreSQL connection string (if not set, uses SQLite)
            SQLITE_DB_PATH: Path to SQLite database (default: backend/data/analytics.db)
            DB_POOL_SIZE: Connection pool size (default: 5)
            DB_MAX_OVERFLOW: Max overflow connections (default: 10)
            DB_POOL_RECYCLE: Seconds before recycling connections (default: 3600)
            DB_PRE_PING: Verify connections before use (default: true)
            USE_NULL_POOL: Use NullPool for RDS Proxy (default: false)
        """
        database_url = os.getenv("DATABASE_URL")

        if database_url and database_url.startswith("postgresql"):
            # PostgreSQL configuration
            return cls(
                backend=DatabaseBackend.POSTGRESQL,
                url=database_url,
                pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
                pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
                pool_pre_ping=os.getenv("DB_PRE_PING", "true").lower() == "true",
                use_null_pool=os.getenv("USE_NULL_POOL", "false").lower() == "true",
            )
        else:
            # SQLite configuration (local development)
            db_path = os.getenv("SQLITE_DB_PATH")
            if db_path:
                sqlite_path = Path(db_path)
            else:
                sqlite_path = DEFAULT_SQLITE_PATH

            return cls(
                backend=DatabaseBackend.SQLITE,
                url=f"sqlite:///{sqlite_path}",
                pool_size=5,  # Allow multiple connections for mixed old/new API usage
                max_overflow=2,
            )

    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL backend."""
        return self.backend == DatabaseBackend.POSTGRESQL

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite backend."""
        return self.backend == DatabaseBackend.SQLITE
