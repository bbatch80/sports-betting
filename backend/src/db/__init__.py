"""
Database package for sports betting analytics (PostgreSQL).

Usage:
    from src.database import AnalyticsRepository

    repo = AnalyticsRepository()
    games_df = repo.get_games(sport="NFL")

Requires DATABASE_URL environment variable (auto-loaded from backend/.env).

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
