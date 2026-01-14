"""
Database operations repository.

Provides a clean interface for all database operations, returning pandas DataFrames.
Handles SQLite vs PostgreSQL differences internally (especially for upserts).

Usage:
    from src.database import AnalyticsRepository

    repo = AnalyticsRepository()

    # Query games
    df = repo.get_games(sport="NFL", start_date=date(2024, 9, 1))

    # Insert games (handles duplicates via upsert)
    repo.insert_games(games_df, sport="NFL")

Why Repository Pattern?
    - Centralizes all database operations in one place
    - Hides database-specific SQL differences from callers
    - Makes testing easier (can mock the repository)
    - All methods return DataFrames for consistency with existing code
"""

from datetime import date
from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects import postgresql, sqlite

from src.config import DatabaseBackend, DatabaseConfig
from src.db.engine import get_engine
from src.db.models import create_tables, games, historical_ratings, todays_games


class AnalyticsRepository:
    """
    Repository for sports betting analytics database operations.

    Automatically configures based on environment. All query methods
    return pandas DataFrames to maintain compatibility with existing code.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the repository.

        Args:
            config: Optional database configuration. If not provided,
                    loads from environment variables.
        """
        self.config = config or DatabaseConfig.from_env()
        self._engine = None

    @property
    def engine(self):
        """Lazy initialization of engine."""
        if self._engine is None:
            self._engine = get_engine(self.config)
        return self._engine

    def init_schema(self) -> None:
        """Create tables if they don't exist."""
        create_tables(self.engine)

    def health_check(self) -> bool:
        """Verify database connectivity."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def _upsert(self, table, records: list, constraint: str, index_elements: list, update_cols: list) -> int:
        """Execute upsert for a list of records. Handles SQLite vs PostgreSQL differences."""
        if not records:
            return 0
        affected = 0
        with self.engine.begin() as conn:
            for record in records:
                if self.config.is_postgresql:
                    stmt = postgresql.insert(table).values(**record)
                    stmt = stmt.on_conflict_do_update(
                        constraint=constraint,
                        set_={col: getattr(stmt.excluded, col) for col in update_cols},
                    )
                else:
                    stmt = sqlite.insert(table).values(**record)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=index_elements,
                        set_={col: getattr(stmt.excluded, col) for col in update_cols},
                    )
                result = conn.execute(stmt)
                affected += result.rowcount
        return affected

    # =========================================================================
    # Games Operations
    # =========================================================================

    def get_games(
        self,
        sport: Optional[str] = None,
        team: Optional[str] = None,
        venue: Optional[str] = None,  # 'home', 'away', or None for both
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        min_spread: Optional[float] = None,
        max_spread: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Query games with flexible filtering.

        Args:
            sport: Filter by sport (NFL, NBA, NCAAM)
            team: Filter by team name (home or away)
            venue: 'home', 'away', or None for both
            start_date: Include games on or after this date
            end_date: Include games on or before this date
            min_spread: Minimum closing spread
            max_spread: Maximum closing spread

        Returns:
            DataFrame with columns:
            - sport, game_date, home_team, away_team
            - closing_spread, home_score, away_score, spread_result
            - team, is_home, team_covered (if filtering by team)
        """
        query = select(games)

        # Build filter conditions
        if sport:
            query = query.where(games.c.sport == sport)

        if team:
            if venue == "home":
                query = query.where(games.c.home_team == team)
            elif venue == "away":
                query = query.where(games.c.away_team == team)
            else:
                query = query.where(
                    (games.c.home_team == team) | (games.c.away_team == team)
                )

        if start_date:
            query = query.where(games.c.game_date >= start_date)

        if end_date:
            query = query.where(games.c.game_date <= end_date)

        if min_spread is not None:
            query = query.where(games.c.closing_spread >= min_spread)

        if max_spread is not None:
            query = query.where(games.c.closing_spread <= max_spread)

        query = query.order_by(games.c.game_date.asc())

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # Add derived columns if filtering by team
        if team and not df.empty:
            df["team"] = team
            df["is_home"] = df["home_team"] == team
            # Team covered if: (home and spread_result >= 0) or (away and spread_result <= 0)
            df["team_covered"] = df.apply(
                lambda r: r["spread_result"] >= 0 if r["is_home"] else r["spread_result"] <= 0,
                axis=1,
            )

        return df

    def insert_games(self, df: pd.DataFrame, sport: str) -> int:
        """Insert games with upsert logic. Handles duplicates by updating existing records."""
        if df.empty:
            return 0

        records = []
        for _, row in df.iterrows():
            # Calculate spread_result if not present
            spread_result = row.get("spread_result_difference")
            if spread_result is None or pd.isna(spread_result):
                if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
                    spread_result = row["home_score"] - row["away_score"] - row["closing_spread"]
                else:
                    spread_result = None

            game_date = row["game_date"]
            if hasattr(game_date, "strftime"):
                game_date = game_date.strftime("%Y-%m-%d")

            records.append({
                "sport": sport,
                "game_date": game_date,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "closing_spread": row.get("closing_spread"),
                "home_score": int(row["home_score"]) if pd.notna(row.get("home_score")) else None,
                "away_score": int(row["away_score"]) if pd.notna(row.get("away_score")) else None,
                "spread_result": spread_result,
            })

        return self._upsert(
            games, records,
            constraint="uq_game",
            index_elements=["sport", "game_date", "home_team", "away_team"],
            update_cols=["closing_spread", "home_score", "away_score", "spread_result"],
        )

    def get_all_teams(self, sport: Optional[str] = None) -> List[str]:
        """Get sorted list of all unique teams."""
        home_query = select(games.c.home_team.distinct())
        away_query = select(games.c.away_team.distinct())

        if sport:
            home_query = home_query.where(games.c.sport == sport)
            away_query = away_query.where(games.c.sport == sport)

        with self.engine.connect() as conn:
            home_teams = {row[0] for row in conn.execute(home_query)}
            away_teams = {row[0] for row in conn.execute(away_query)}

        return sorted(home_teams | away_teams)

    def get_sports(self) -> List[str]:
        """Get list of all sports in the database."""
        query = select(games.c.sport.distinct()).order_by(games.c.sport)

        with self.engine.connect() as conn:
            return [row[0] for row in conn.execute(query)]

    def get_date_range(self, sport: Optional[str] = None) -> Tuple[date, date]:
        """Get the date range of games in the database."""
        query = select(
            func.min(games.c.game_date),
            func.max(games.c.game_date),
        )

        if sport:
            query = query.where(games.c.sport == sport)

        with self.engine.connect() as conn:
            row = conn.execute(query).fetchone()

        return row[0], row[1]

    def get_game_count(self, sport: Optional[str] = None) -> int:
        """Get total number of games."""
        query = select(func.count()).select_from(games)

        if sport:
            query = query.where(games.c.sport == sport)

        with self.engine.connect() as conn:
            return conn.execute(query).scalar()

    # =========================================================================
    # Historical Ratings Operations
    # =========================================================================

    def insert_ratings(self, df: pd.DataFrame) -> int:
        """Insert historical ratings with upsert logic."""
        if df.empty:
            return 0

        records = df.to_dict("records")
        for record in records:
            if hasattr(record.get("snapshot_date"), "strftime"):
                record["snapshot_date"] = record["snapshot_date"].strftime("%Y-%m-%d")

        return self._upsert(
            historical_ratings, records,
            constraint="uq_rating",
            index_elements=["sport", "snapshot_date", "team"],
            update_cols=["win_rating", "ats_rating", "market_gap", "games_analyzed", "win_rank", "ats_rank"],
        )

    def get_ratings(
        self,
        sport: Optional[str] = None,
        snapshot_date: Optional[date] = None,
        team: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get historical ratings with optional filters."""
        query = select(historical_ratings)

        if sport:
            query = query.where(historical_ratings.c.sport == sport)
        if snapshot_date:
            query = query.where(historical_ratings.c.snapshot_date == snapshot_date)
        if team:
            query = query.where(historical_ratings.c.team == team)

        query = query.order_by(
            historical_ratings.c.snapshot_date.desc(),
            historical_ratings.c.win_rank,
        )

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def get_latest_ratings(self, sport: str) -> pd.DataFrame:
        """Get most recent ratings snapshot for a sport."""
        # Find the most recent snapshot date
        subq = (
            select(func.max(historical_ratings.c.snapshot_date))
            .where(historical_ratings.c.sport == sport)
            .scalar_subquery()
        )

        query = (
            select(historical_ratings)
            .where(historical_ratings.c.sport == sport)
            .where(historical_ratings.c.snapshot_date == subq)
            .order_by(historical_ratings.c.win_rank)
        )

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def get_games_with_ratings(
        self,
        sport: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get games joined with pre-game ratings for backtesting.

        Returns games with each team's ratings from the day before the game.
        Used for backtesting betting strategies.

        Args:
            sport: Sport to query
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with game data plus:
            - home_win_rating, home_ats_rating, home_market_gap
            - away_win_rating, away_ats_rating, away_market_gap
        """
        # This complex query uses correlated subqueries to find
        # the most recent rating for each team before each game
        sql = text("""
            SELECT
                g.id,
                g.sport,
                g.game_date,
                g.home_team,
                g.away_team,
                g.closing_spread,
                g.home_score,
                g.away_score,
                g.spread_result,
                hr_home.win_rating as home_win_rating,
                hr_home.ats_rating as home_ats_rating,
                hr_home.market_gap as home_market_gap,
                hr_home.win_rank as home_win_rank,
                hr_home.ats_rank as home_ats_rank,
                hr_away.win_rating as away_win_rating,
                hr_away.ats_rating as away_ats_rating,
                hr_away.market_gap as away_market_gap,
                hr_away.win_rank as away_win_rank,
                hr_away.ats_rank as away_ats_rank
            FROM games g
            LEFT JOIN historical_ratings hr_home
                ON g.home_team = hr_home.team
                AND g.sport = hr_home.sport
                AND hr_home.snapshot_date = (
                    SELECT MAX(snapshot_date)
                    FROM historical_ratings
                    WHERE team = g.home_team
                    AND sport = g.sport
                    AND snapshot_date < g.game_date
                )
            LEFT JOIN historical_ratings hr_away
                ON g.away_team = hr_away.team
                AND g.sport = hr_away.sport
                AND hr_away.snapshot_date = (
                    SELECT MAX(snapshot_date)
                    FROM historical_ratings
                    WHERE team = g.away_team
                    AND sport = g.sport
                    AND snapshot_date < g.game_date
                )
            WHERE g.sport = :sport
                AND g.game_date BETWEEN :start_date AND :end_date
            ORDER BY g.game_date
        """)

        with self.engine.connect() as conn:
            return pd.read_sql(
                sql,
                conn,
                params={
                    "sport": sport,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )

    # =========================================================================
    # Today's Games Operations
    # =========================================================================

    def insert_todays_games(self, games_list: list, sport: str) -> int:
        """
        Insert today's games with upsert logic.

        Args:
            games_list: List of game dicts with keys:
                - game_date, commence_time, home_team, away_team
                - spread (optional), spread_source (optional)
            sport: Sport key (NFL, NBA, NCAAM)

        Returns:
            Number of rows affected
        """
        if not games_list:
            return 0

        from datetime import datetime

        now = datetime.utcnow()
        records = []
        for game in games_list:
            game_date = game["game_date"]
            if hasattr(game_date, "strftime"):
                game_date = game_date.strftime("%Y-%m-%d")

            commence_time = game["commence_time"]
            if hasattr(commence_time, "isoformat"):
                commence_time = commence_time.isoformat()

            records.append({
                "sport": sport,
                "game_date": game_date,
                "commence_time": commence_time,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "spread": game.get("spread"),
                "spread_source": game.get("spread_source"),
                "created_at": now,
                "updated_at": now,
            })

        return self._upsert(
            todays_games, records,
            constraint="uq_todays_game",
            index_elements=["sport", "game_date", "home_team", "away_team"],
            update_cols=["spread", "spread_source", "updated_at", "commence_time"],
        )

    def get_todays_games(self, sport: str = None, game_date: date = None) -> pd.DataFrame:
        """
        Get today's games from the database.

        Args:
            sport: Filter by sport (NFL, NBA, NCAAM), or None for all
            game_date: Filter by date, defaults to today (EST)

        Returns:
            DataFrame with columns: sport, game_date, commence_time,
            home_team, away_team, spread, spread_source, updated_at
        """
        from datetime import datetime, timedelta, timezone

        # Default to today in EST
        if game_date is None:
            est = timezone(timedelta(hours=-5))
            game_date = datetime.now(est).date()

        query = select(todays_games).where(todays_games.c.game_date == game_date)

        if sport:
            query = query.where(todays_games.c.sport == sport)

        query = query.order_by(todays_games.c.commence_time.asc())

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def clear_old_todays_games(self, before_date: date) -> int:
        """
        Delete games older than a given date (housekeeping).

        Args:
            before_date: Delete games with game_date before this date

        Returns:
            Number of rows deleted
        """
        stmt = delete(todays_games).where(todays_games.c.game_date < before_date)

        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def execute_raw(self, sql: str, params: dict = None) -> pd.DataFrame:
        """
        Execute raw SQL query and return DataFrame.

        Use sparingly - prefer the typed methods above.

        Args:
            sql: SQL query string
            params: Optional parameters for the query

        Returns:
            DataFrame with query results
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    def clear_games(self, sport: Optional[str] = None) -> int:
        """
        Delete games. USE WITH CAUTION.

        Args:
            sport: If provided, only delete games for this sport.
                   If None, deletes ALL games.

        Returns:
            Number of rows deleted
        """
        stmt = delete(games)
        if sport:
            stmt = stmt.where(games.c.sport == sport)

        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount

    def clear_ratings(self, sport: Optional[str] = None) -> int:
        """
        Delete historical ratings. USE WITH CAUTION.

        Args:
            sport: If provided, only delete ratings for this sport.
                   If None, deletes ALL ratings.

        Returns:
            Number of rows deleted
        """
        stmt = delete(historical_ratings)
        if sport:
            stmt = stmt.where(historical_ratings.c.sport == sport)

        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount
