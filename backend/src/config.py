"""
Configuration settings for sports betting analytics.

Contains:
- Sport definitions for The Odds API
- Database configuration (PostgreSQL)
- API settings
"""

import os
from dataclasses import dataclass
from datetime import datetime
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

@dataclass
class DatabaseConfig:
    """
    Database configuration for PostgreSQL.

    Loads DATABASE_URL from environment (auto-loads .env if present).
    Errors immediately if DATABASE_URL is not set — no silent SQLite fallback.

    Example usage:
        config = DatabaseConfig.from_env()
        engine = create_engine(config.url)
    """
    url: str

    # Connection pool settings
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

        Automatically loads .env file from the backend directory if present.

        Environment variables:
            DATABASE_URL: PostgreSQL connection string (REQUIRED)
            DB_POOL_SIZE: Connection pool size (default: 5)
            DB_MAX_OVERFLOW: Max overflow connections (default: 10)
            DB_POOL_RECYCLE: Seconds before recycling connections (default: 3600)
            DB_PRE_PING: Verify connections before use (default: true)
            USE_NULL_POOL: Use NullPool for RDS Proxy (default: false)

        Raises:
            RuntimeError: If DATABASE_URL is not set
        """
        # Auto-load .env file if present (covers dashboard, scripts, etc.)
        _load_dotenv_once()

        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Set it in your environment or in backend/.env file. "
                "Example: DATABASE_URL=postgresql://user:pass@host:5432/dbname"
            )

        return cls(
            url=database_url,
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            pool_pre_ping=os.getenv("DB_PRE_PING", "true").lower() == "true",
            use_null_pool=os.getenv("USE_NULL_POOL", "false").lower() == "true",
        )

    @property
    def is_postgresql(self) -> bool:
        return True


# .env auto-loading (runs once)
_dotenv_loaded = False

def _load_dotenv_once():
    """Load .env file from backend directory if present. Runs only once."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True

    # Look for .env in the backend directory (parent of src/)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            # python-dotenv not installed — .env must be loaded externally
            pass
