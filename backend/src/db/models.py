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
    Text,
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

    # Opening team totals (captured at 6:30 AM by collect_todays_games Lambda)
    Column("home_team_total", Float),  # O/U line for home team
    Column("away_team_total", Float),  # O/U line for away team

    # Closing lines (captured 30 min before game by n8n workflow)
    Column("closing_spread", Float),       # Closing spread
    Column("closing_total", Float),        # Closing game total
    Column("closing_home_tt", Float),      # Closing home team total
    Column("closing_away_tt", Float),      # Closing away team total
    Column("closing_captured_at", DateTime),  # When closing lines were captured

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
# Current O/U Streaks Table (pre-computed daily for fast dashboard loads)
# =============================================================================

current_ou_streaks = Table(
    "current_ou_streaks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sport", String(50), nullable=False),
    Column("team", String(100), nullable=False),
    Column("streak_length", Integer),
    Column("streak_type", String(10)),  # 'OVER' or 'UNDER'
    Column("computed_at", DateTime),
    UniqueConstraint("sport", "team", name="uq_current_ou_streak"),
)

Index("idx_current_ou_streaks_sport", current_ou_streaks.c.sport)


# =============================================================================
# Current TT Streaks Table (pre-computed daily for fast dashboard loads)
# =============================================================================

current_tt_streaks = Table(
    "current_tt_streaks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sport", String(50), nullable=False),
    Column("team", String(100), nullable=False),
    Column("streak_length", Integer),
    Column("streak_type", String(10)),  # 'OVER' or 'UNDER'
    Column("computed_at", DateTime),
    UniqueConstraint("sport", "team", name="uq_current_tt_streak"),
)

Index("idx_current_tt_streaks_sport", current_tt_streaks.c.sport)


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
    Column("market_type", String(10), server_default="ats"),  # 'ats', 'ou', 'tt'
    Column("pattern_type", String(20), nullable=False),  # 'streak_fade' or 'streak_ride'
    Column("streak_type", String(10), nullable=False),   # 'WIN'/'LOSS' (ATS) or 'OVER'/'UNDER' (O/U, TT)
    Column("streak_length", Integer, nullable=False),
    Column("handicap", Integer, nullable=False),

    # Statistics
    Column("cover_rate", Float),
    Column("baseline_rate", Float),
    Column("edge", Float),
    Column("sample_size", Integer),
    Column("confidence", String(10)),  # 'high', 'medium', 'low'

    # Full coverage profile across all handicap levels for this (streak_type, streak_length)
    Column("coverage_profile_json", Text),  # JSON: {"0": {"cover_rate": 0.42, ...}, "1": {...}, ...}

    # Timestamp
    Column("computed_at", DateTime),

    # Ensure one pattern per combination (includes market_type)
    UniqueConstraint(
        "sport", "market_type", "pattern_type", "streak_type", "streak_length", "handicap",
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

    # Totals info
    Column("total", Float),
    Column("total_source", String(100)),
    Column("home_team_total", Float),
    Column("away_team_total", Float),

    # Team tiers and ratings
    Column("home_tier", String(20)),
    Column("away_tier", String(20)),
    Column("home_ats_rating", Float),
    Column("away_ats_rating", Float),

    # ATS Streak info
    Column("home_streak_length", Integer),
    Column("home_streak_type", String(10)),
    Column("away_streak_length", Integer),
    Column("away_streak_type", String(10)),

    # O/U Streak info
    Column("home_ou_streak_length", Integer),
    Column("home_ou_streak_type", String(10)),
    Column("away_ou_streak_length", Integer),
    Column("away_ou_streak_type", String(10)),

    # TT Streak info
    Column("home_tt_streak_length", Integer),
    Column("home_tt_streak_type", String(10)),
    Column("away_tt_streak_length", Integer),
    Column("away_tt_streak_type", String(10)),

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
# Prediction Results Table (individual bet tracking with outcomes)
# =============================================================================

prediction_results = Table(
    "prediction_results",
    metadata,
    # Primary key
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Game identification
    Column("sport", String(50), nullable=False),
    Column("game_date", Date, nullable=False),
    Column("home_team", String(100), nullable=False),
    Column("away_team", String(100), nullable=False),
    Column("game_time", String(50)),

    # Prediction-time lines (snapshot at ~6:30 AM when recommendation was generated)
    Column("spread", Float),
    Column("total", Float),
    Column("home_team_total", Float),
    Column("away_team_total", Float),

    # Closing lines (DraftKings closing lines, filled at resolution from games table)
    Column("closing_spread", Float),
    Column("closing_total", Float),
    Column("closing_home_tt", Float),
    Column("closing_away_tt", Float),

    # Team strength at recommendation time
    Column("home_tier", String(20)),
    Column("away_tier", String(20)),
    Column("home_ats_rating", Float),
    Column("away_ats_rating", Float),

    # ATS streaks
    Column("home_streak_length", Integer),
    Column("home_streak_type", String(10)),
    Column("away_streak_length", Integer),
    Column("away_streak_type", String(10)),

    # O/U streaks
    Column("home_ou_streak_length", Integer),
    Column("home_ou_streak_type", String(10)),
    Column("away_ou_streak_length", Integer),
    Column("away_ou_streak_type", String(10)),

    # TT streaks
    Column("home_tt_streak_length", Integer),
    Column("home_tt_streak_type", String(10)),
    Column("away_tt_streak_length", Integer),
    Column("away_tt_streak_type", String(10)),

    # Recommendation details
    Column("bet_on", String(200), nullable=False),   # team name, "Game Total OVER", "Team TT OVER", etc.
    Column("source", String(20), nullable=False),     # 'streak', 'ou_streak', 'tt_streak'
    Column("edge", Float),
    Column("confidence", String(10)),                 # 'high', 'medium', 'low'
    Column("rationale", String(500)),
    Column("handicap", Integer),

    # Pattern details (from detected_patterns table)
    Column("market_type", String(10)),                # 'ats', 'ou', 'tt'
    Column("pattern_type", String(20)),               # 'streak_ride' or 'streak_fade'
    Column("cover_rate", Float),
    Column("baseline_rate", Float),
    Column("sample_size", Integer),

    # Derived columns (pre-computed for analysis)
    Column("bet_is_home", Integer),                   # 1=home, 0=away, NULL=totals bet
    Column("tier_matchup", String(40)),               # e.g. "Elite vs Weak"
    Column("spread_bucket", String(20)),              # "PK-1", "1.5-3.5", "4-7", "7.5-10", "10+"
    Column("rating_diff", Float),                     # home_ats_rating - away_ats_rating

    # Outcome (filled at resolution)
    Column("outcome", String(10)),                    # 'WIN', 'LOSS', 'PUSH', or NULL (unresolved)
    Column("margin", Float),                          # how much we won/lost by (positive = good)
    Column("home_score", Integer),
    Column("away_score", Integer),

    # Timestamps
    Column("captured_at", DateTime),
    Column("resolved_at", DateTime),

    # Prevent duplicates if Lambda reruns
    UniqueConstraint(
        "sport", "game_date", "home_team", "away_team", "bet_on", "source",
        name="uq_prediction_result"
    ),
)

# Indexes for prediction results
Index("idx_pred_results_sport_date", prediction_results.c.sport, prediction_results.c.game_date)
Index("idx_pred_results_outcome", prediction_results.c.outcome)
Index("idx_pred_results_source", prediction_results.c.source)


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
