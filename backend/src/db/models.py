"""
SQLAlchemy Table definitions using Core (not ORM).

Why Core instead of ORM?
    The existing codebase uses pandas DataFrames for data manipulation.
    SQLAlchemy Core lets us:
    - Define tables once, works for both SQLite and PostgreSQL
    - Use pd.read_sql() directly with SQLAlchemy connections
    - Avoid ORM overhead when we just need DataFrames

Tables:
    games: Historical game results with spreads and scores
    historical_ratings: Daily team rating snapshots for backtesting
    todays_games: Scheduled games fetched from Odds API (pre-computed daily)
"""

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)

# Metadata container for all tables
# Used by create_all() to create tables and by migrations
metadata = MetaData()


# =============================================================================
# Games Table
# =============================================================================

games = Table(
    "games",
    metadata,
    # Primary key - auto-incrementing ID
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Game identification
    Column("sport", String(50), nullable=False),  # NFL, NBA, NCAAM
    Column("game_date", Date, nullable=False),
    Column("home_team", String(100), nullable=False),
    Column("away_team", String(100), nullable=False),

    # Betting data
    Column("closing_spread", Float),  # Spread for home team (negative = favored)

    # Results
    Column("home_score", Integer),
    Column("away_score", Integer),
    Column("spread_result", Float),  # home_score - away_score - closing_spread

    # Ensure no duplicate games
    UniqueConstraint(
        "sport", "game_date", "home_team", "away_team",
        name="uq_game"
    ),
)

# Indexes for common query patterns
Index("idx_games_sport", games.c.sport)
Index("idx_games_date", games.c.game_date)
Index("idx_games_home_team", games.c.home_team)
Index("idx_games_away_team", games.c.away_team)
Index("idx_games_sport_date", games.c.sport, games.c.game_date)


# =============================================================================
# Historical Ratings Table
# =============================================================================

historical_ratings = Table(
    "historical_ratings",
    metadata,
    # Primary key
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Rating identification
    Column("sport", String(50), nullable=False),
    Column("snapshot_date", Date, nullable=False),  # Date this rating was calculated
    Column("team", String(100), nullable=False),

    # Power ratings
    Column("win_rating", Float),  # Team strength based on wins
    Column("ats_rating", Float),  # Team strength based on spread coverage
    Column("market_gap", Float),  # ats_rating - win_rating (market perception gap)

    # Context
    Column("games_analyzed", Integer),  # Number of games in the rating calculation
    Column("win_rank", Integer),  # Rank by win_rating
    Column("ats_rank", Integer),  # Rank by ats_rating

    # Ensure one rating per team per day per sport
    UniqueConstraint(
        "sport", "snapshot_date", "team",
        name="uq_rating"
    ),
)

# Indexes for efficient rating lookups
Index(
    "idx_historical_ratings_lookup",
    historical_ratings.c.sport,
    historical_ratings.c.snapshot_date
)
Index(
    "idx_historical_ratings_team",
    historical_ratings.c.sport,
    historical_ratings.c.team,
    historical_ratings.c.snapshot_date
)


# =============================================================================
# Today's Games Table (scheduled games fetched from Odds API)
# =============================================================================

todays_games = Table(
    "todays_games",
    metadata,
    # Primary key
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Game identification
    Column("sport", String(50), nullable=False),  # NFL, NBA, NCAAM
    Column("game_date", Date, nullable=False),
    Column("commence_time", DateTime, nullable=False),  # Full timestamp with time
    Column("home_team", String(100), nullable=False),
    Column("away_team", String(100), nullable=False),

    # Betting data
    Column("spread", Float),  # Spread for home team (negative = favored)
    Column("spread_source", String(50)),  # Bookmaker name (e.g., DraftKings)

    # Timestamps
    Column("created_at", DateTime),
    Column("updated_at", DateTime),

    # Ensure no duplicate games
    UniqueConstraint(
        "sport", "game_date", "home_team", "away_team",
        name="uq_todays_game"
    ),
)

# Indexes for efficient lookups
Index("idx_todays_games_sport_date", todays_games.c.sport, todays_games.c.game_date)


# =============================================================================
# Schema Management
# =============================================================================

def create_tables(engine) -> None:
    """
    Create all tables if they don't exist.

    This is idempotent - safe to call multiple times.
    Uses SQLAlchemy's metadata.create_all() which checks for existing tables.

    Args:
        engine: SQLAlchemy Engine instance
    """
    metadata.create_all(engine)


def drop_tables(engine) -> None:
    """
    Drop all tables. USE WITH CAUTION.

    Primarily useful for testing. Will delete all data.

    Args:
        engine: SQLAlchemy Engine instance
    """
    metadata.drop_all(engine)
