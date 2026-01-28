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

    # Betting data - Spreads
    Column("closing_spread", Float),  # Spread for home team (negative = favored)

    # Betting data - Totals (Over/Under)
    Column("closing_total", Float),   # Over/under line at game start

    # Results
    Column("home_score", Integer),
    Column("away_score", Integer),
    Column("spread_result", Float),  # home_score - away_score - closing_spread
    Column("total_result", Float),   # (home_score + away_score) - closing_total
                                     # Positive = OVER, Negative = UNDER

    # Team Totals - Individual team O/U lines (e.g., "Lakers O/U 115.5")
    Column("home_team_total", Float),        # O/U line for home team
    Column("away_team_total", Float),        # O/U line for away team
    Column("home_team_total_result", Float), # home_score - home_team_total
                                             # Positive = OVER, Negative = UNDER
    Column("away_team_total_result", Float), # away_score - away_team_total
                                             # Positive = OVER, Negative = UNDER

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

    # Betting data - Spreads
    Column("spread", Float),  # Spread for home team (negative = favored)
    Column("spread_source", String(50)),  # Bookmaker name (e.g., DraftKings)

    # Betting data - Totals (Over/Under)
    Column("total", Float),           # Over/under line
    Column("total_source", String(50)),  # Bookmaker name

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
# Current Rankings Table (pre-computed daily for fast dashboard loads)
# =============================================================================

current_rankings = Table(
    "current_rankings",
    metadata,
    # Primary key
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Team identification
    Column("sport", String(50), nullable=False),
    Column("team", String(100), nullable=False),

    # Power ratings
    Column("win_rating", Float),
    Column("ats_rating", Float),
    Column("market_gap", Float),  # ats_rating - win_rating

    # Rankings
    Column("win_rank", Integer),
    Column("ats_rank", Integer),

    # Records
    Column("win_record", String(20)),   # "18-7"
    Column("ats_record", String(20)),   # "15-10"

    # Context
    Column("games_analyzed", Integer),
    Column("is_reliable", Integer),  # 1 or 0 (boolean as int for SQLite compat)

    # Timestamp
    Column("computed_at", DateTime),

    # Ensure one record per team per sport
    UniqueConstraint("sport", "team", name="uq_current_ranking"),
)

# Index for efficient lookups
Index("idx_current_rankings_sport", current_rankings.c.sport)


# =============================================================================
# Current Streaks Table (pre-computed daily for fast dashboard loads)
# =============================================================================

current_streaks = Table(
    "current_streaks",
    metadata,
    # Primary key
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Team identification
    Column("sport", String(50), nullable=False),
    Column("team", String(100), nullable=False),

    # Streak info
    Column("streak_length", Integer),
    Column("streak_type", String(10)),  # 'WIN' or 'LOSS'

    # Timestamp
    Column("computed_at", DateTime),

    # Ensure one record per team per sport
    UniqueConstraint("sport", "team", name="uq_current_streak"),
)

# Index for efficient lookups
Index("idx_current_streaks_sport", current_streaks.c.sport)


# =============================================================================
# Detected Patterns Table (pre-computed daily for fast dashboard loads)
# =============================================================================

detected_patterns = Table(
    "detected_patterns",
    metadata,
    # Primary key
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Pattern identification
    Column("sport", String(50), nullable=False),
    Column("pattern_type", String(20), nullable=False),  # 'streak_fade' or 'streak_ride'
    Column("streak_type", String(10), nullable=False),   # 'WIN' or 'LOSS'
    Column("streak_length", Integer, nullable=False),
    Column("handicap", Integer, nullable=False),

    # Statistics
    Column("cover_rate", Float),
    Column("baseline_rate", Float),
    Column("edge", Float),
    Column("sample_size", Integer),
    Column("confidence", String(10)),  # 'high', 'medium', 'low'

    # Timestamp
    Column("computed_at", DateTime),

    # Ensure one pattern per combination
    UniqueConstraint(
        "sport", "pattern_type", "streak_type", "streak_length", "handicap",
        name="uq_detected_pattern"
    ),
)

# Index for efficient lookups
Index("idx_detected_patterns_sport", detected_patterns.c.sport)


# =============================================================================
# Today's Recommendations Table (pre-computed daily for fast dashboard loads)
# =============================================================================

todays_recommendations = Table(
    "todays_recommendations",
    metadata,
    # Primary key
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Game identification
    Column("sport", String(50), nullable=False),
    Column("game_date", Date, nullable=False),
    Column("home_team", String(100), nullable=False),
    Column("away_team", String(100), nullable=False),
    Column("game_time", String(50)),

    # Spread info
    Column("spread", Float),
    Column("spread_source", String(100)),

    # Team tiers and ratings
    Column("home_tier", String(20)),
    Column("away_tier", String(20)),
    Column("home_ats_rating", Float),
    Column("away_ats_rating", Float),

    # Streak info
    Column("home_streak_length", Integer),
    Column("home_streak_type", String(10)),
    Column("away_streak_length", Integer),
    Column("away_streak_type", String(10)),

    # Recommendations stored as JSON
    Column("recommendations_json", String(4000)),  # JSON array of BetRecommendation objects

    # Timestamp
    Column("computed_at", DateTime),

    # Ensure one record per game per day
    UniqueConstraint("sport", "game_date", "home_team", "away_team", name="uq_todays_recommendation"),
)

# Index for efficient lookups
Index("idx_todays_rec_sport_date", todays_recommendations.c.sport, todays_recommendations.c.game_date)


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
