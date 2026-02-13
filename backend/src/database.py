"""
Database facade for backward compatibility.

This module maintains the original function signatures so existing code
(15+ files) doesn't need to change. All operations delegate to the new
repository pattern internally.

MIGRATION NOTE:
    New code should import from src.database (the package) directly:

        from src.database import AnalyticsRepository
        repo = AnalyticsRepository()
        df = repo.get_games(sport="NFL")

    This facade module is preserved for backward compatibility with
    existing code that uses the old pattern:

        from src.database import get_connection, get_games
        conn = get_connection()
        df = get_games(conn, sport="NFL")
"""
from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

# Import from the new db package
from src.db import AnalyticsRepository, get_engine
from src.db.models import create_tables


# Module-level repository singleton
_repository: Optional[AnalyticsRepository] = None


def _get_repository() -> AnalyticsRepository:
    """Get or create the repository singleton."""
    global _repository
    if _repository is None:
        _repository = AnalyticsRepository()
        _repository.init_schema()
    return _repository


# =============================================================================
# Backward Compatible Connection Functions
# =============================================================================


def get_connection(db_path: Optional[Path] = None, check_same_thread: bool = True):
    """
    Get a database connection.

    DEPRECATED: This returns a SQLAlchemy connection instead of sqlite3.
    New code should use AnalyticsRepository instead.

    Note: The db_path and check_same_thread parameters are ignored.
    Database configuration is now controlled via environment variables.
    """
    warnings.warn(
        "get_connection() is deprecated. Use AnalyticsRepository instead. "
        "Example: from src.database import AnalyticsRepository; repo = AnalyticsRepository()",
        DeprecationWarning,
        stacklevel=2,
    )
    engine = get_engine()
    return engine.connect()


def init_schema(conn) -> None:
    """
    Initialize the database schema.

    DEPRECATED: Schema is now initialized automatically.
    The conn parameter is ignored.
    """
    _get_repository().init_schema()


def init_database(db_path: Optional[Path] = None):
    """
    Initialize database and return connection.

    DEPRECATED: Returns a SQLAlchemy connection.
    New code should use AnalyticsRepository instead.
    """
    warnings.warn(
        "init_database() is deprecated. Use AnalyticsRepository instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    repo = _get_repository()
    repo.init_schema()
    return get_engine().connect()


# =============================================================================
# Data Loading (Backward Compatible)
# =============================================================================


def insert_games(conn, games_df: pd.DataFrame, sport: str) -> int:
    """
    Insert games from a DataFrame into the database.

    The conn parameter is accepted for backward compatibility but ignored.
    Operations use the repository pattern internally.

    Args:
        conn: Ignored (kept for backward compatibility)
        games_df: DataFrame with game data
        sport: Sport identifier (NFL, NBA, NCAAM)

    Returns:
        Number of games inserted/updated
    """
    return _get_repository().insert_games(games_df, sport)


# =============================================================================
# Query Functions (Backward Compatible)
# =============================================================================


def get_games(
    conn,
    sport: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_spread: Optional[float] = None,
    max_spread: Optional[float] = None,
) -> pd.DataFrame:
    """
    Query games with flexible filtering.

    The conn parameter is accepted for backward compatibility but ignored.

    Args:
        conn: Ignored (kept for backward compatibility)
        sport: Filter by sport
        team: Filter by team name
        venue: 'home', 'away', or None for both
        start_date: Include games on or after this date
        end_date: Include games on or before this date
        min_spread: Minimum closing spread
        max_spread: Maximum closing spread

    Returns:
        DataFrame with game data
    """
    return _get_repository().get_games(
        sport=sport,
        team=team,
        venue=venue,
        start_date=start_date,
        end_date=end_date,
        min_spread=min_spread,
        max_spread=max_spread,
    )


def get_all_teams(conn, sport: Optional[str] = None) -> List[str]:
    """
    Get list of all teams in the database.

    The conn parameter is accepted for backward compatibility but ignored.
    """
    return _get_repository().get_all_teams(sport)


def get_sports(conn) -> List[str]:
    """
    Get list of all sports in the database.

    The conn parameter is accepted for backward compatibility but ignored.
    """
    return _get_repository().get_sports()


def get_date_range(conn, sport: Optional[str] = None) -> Tuple[date, date]:
    """
    Get the date range of games in the database.

    The conn parameter is accepted for backward compatibility but ignored.
    """
    return _get_repository().get_date_range(sport)


def get_game_count(conn, sport: Optional[str] = None) -> int:
    """
    Get total number of games in the database.

    The conn parameter is accepted for backward compatibility but ignored.
    """
    return _get_repository().get_game_count(sport)


# =============================================================================
# CLI for testing (preserved from original)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database utilities")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    args = parser.parse_args()

    repo = _get_repository()

    if args.init:
        print(f"Initializing database...")
        repo.init_schema()
        print("Database initialized successfully")

    if args.stats:
        print(f"\nDatabase Statistics:")
        print(f"Backend: {repo.config.backend.value}")
        print(f"Total games: {repo.get_game_count()}")
        for sport in repo.get_sports():
            count = repo.get_game_count(sport)
            teams = len(repo.get_all_teams(sport))
            dates = repo.get_date_range(sport)
            print(f"  {sport}: {count} games, {teams} teams, {dates[0]} to {dates[1]}")
