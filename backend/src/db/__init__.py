"""
Database package for sports betting analytics.

This package provides a database abstraction layer that supports both
SQLite (for local development) and PostgreSQL (for production on AWS RDS).

Usage:
    from src.database import AnalyticsRepository

    repo = AnalyticsRepository()
    games_df = repo.get_games(sport="NFL")

The backend is automatically selected based on the DATABASE_URL environment variable:
- If DATABASE_URL is set and starts with "postgresql": uses PostgreSQL
- Otherwise: uses SQLite at backend/data/analytics.db

For backward compatibility, the original database.py facade module still works:
    from src.database import get_games, insert_games
"""

from src.db.engine import get_engine, reset_engine
from src.db.models import games, historical_ratings, metadata, create_tables
from src.db.repository import AnalyticsRepository

__all__ = [
    # Engine
    "get_engine",
    "reset_engine",
    # Models
    "games",
    "historical_ratings",
    "metadata",
    "create_tables",
    # Repository
    "AnalyticsRepository",
]
