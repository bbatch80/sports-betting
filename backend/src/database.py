"""
SQLite database module for sports betting analytics.

Provides:
- Database connection management
- Schema initialization
- Query helpers for games and metrics
"""

import sqlite3
from pathlib import Path
from datetime import date
from typing import Optional, List, Tuple
import pandas as pd


# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / 'data' / 'analytics.db'


def get_connection(db_path: Optional[Path] = None, check_same_thread: bool = True) -> sqlite3.Connection:
    """Get a database connection with row factory enabled.

    Args:
        db_path: Optional custom database path
        check_same_thread: If False, allows connection to be used across threads
                          (needed for Streamlit and other multi-threaded apps)
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    cursor = conn.cursor()

    # Core game data (one row per game)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY,
            sport TEXT NOT NULL,
            game_date DATE NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            closing_spread REAL,
            home_score INTEGER,
            away_score INTEGER,
            spread_result REAL,
            UNIQUE(sport, game_date, home_team, away_team)
        )
    ''')

    # Create indexes for common queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_games_sport ON games(sport)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team)
    ''')

    conn.commit()


def init_database(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize database and return connection."""
    conn = get_connection(db_path)
    init_schema(conn)
    return conn


# =============================================================================
# Data Loading
# =============================================================================

def insert_games(conn: sqlite3.Connection, games_df: pd.DataFrame, sport: str) -> int:
    """
    Insert games from a DataFrame into the database.

    Expected DataFrame columns:
    - game_date: datetime
    - home_team: str
    - away_team: str
    - closing_spread: float
    - home_score: int
    - away_score: int
    - spread_result_difference: float (optional, will be calculated if missing)

    Returns: Number of games inserted
    """
    cursor = conn.cursor()
    inserted = 0

    for _, row in games_df.iterrows():
        # Calculate spread_result if not present
        spread_result = row.get('spread_result_difference')
        if spread_result is None or pd.isna(spread_result):
            spread_result = row['home_score'] - row['away_score'] - row['closing_spread']

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO games
                (sport, game_date, home_team, away_team, closing_spread,
                 home_score, away_score, spread_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sport,
                row['game_date'].strftime('%Y-%m-%d') if hasattr(row['game_date'], 'strftime') else str(row['game_date']),
                row['home_team'],
                row['away_team'],
                row['closing_spread'],
                int(row['home_score']),
                int(row['away_score']),
                spread_result
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            # Game already exists, skip
            pass

    conn.commit()
    return inserted


# =============================================================================
# Query Helpers
# =============================================================================

def get_games(
    conn: sqlite3.Connection,
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

    Returns DataFrame with columns:
    - sport, game_date, home_team, away_team
    - closing_spread, home_score, away_score, spread_result
    - team (the team of interest if filtering by team)
    - is_home (True if team was home)
    - team_covered (True if team covered the spread)
    """
    conditions = []
    params = []

    if sport:
        conditions.append("sport = ?")
        params.append(sport)

    if team:
        if venue == 'home':
            conditions.append("home_team = ?")
            params.append(team)
        elif venue == 'away':
            conditions.append("away_team = ?")
            params.append(team)
        else:
            conditions.append("(home_team = ? OR away_team = ?)")
            params.extend([team, team])

    if start_date:
        conditions.append("game_date >= ?")
        params.append(start_date.strftime('%Y-%m-%d'))

    if end_date:
        conditions.append("game_date <= ?")
        params.append(end_date.strftime('%Y-%m-%d'))

    if min_spread is not None:
        conditions.append("closing_spread >= ?")
        params.append(min_spread)

    if max_spread is not None:
        conditions.append("closing_spread <= ?")
        params.append(max_spread)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f'''
        SELECT * FROM games
        WHERE {where_clause}
        ORDER BY game_date ASC
    '''

    df = pd.read_sql_query(query, conn, params=params)

    # Add derived columns if filtering by team
    if team:
        df['team'] = team
        df['is_home'] = df['home_team'] == team
        # Team covered if: (home and spread_result >= 0) or (away and spread_result <= 0)
        df['team_covered'] = df.apply(
            lambda r: r['spread_result'] >= 0 if r['is_home'] else r['spread_result'] <= 0,
            axis=1
        )

    return df


def get_all_teams(conn: sqlite3.Connection, sport: Optional[str] = None) -> List[str]:
    """Get list of all teams in the database."""
    cursor = conn.cursor()

    if sport:
        cursor.execute('''
            SELECT DISTINCT home_team FROM games WHERE sport = ?
            UNION
            SELECT DISTINCT away_team FROM games WHERE sport = ?
        ''', (sport, sport))
    else:
        cursor.execute('''
            SELECT DISTINCT home_team FROM games
            UNION
            SELECT DISTINCT away_team FROM games
        ''')

    return sorted([row[0] for row in cursor.fetchall()])


def get_sports(conn: sqlite3.Connection) -> List[str]:
    """Get list of all sports in the database."""
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT sport FROM games ORDER BY sport')
    return [row[0] for row in cursor.fetchall()]


def get_date_range(conn: sqlite3.Connection, sport: Optional[str] = None) -> Tuple[date, date]:
    """Get the date range of games in the database."""
    cursor = conn.cursor()

    if sport:
        cursor.execute('''
            SELECT MIN(game_date), MAX(game_date) FROM games WHERE sport = ?
        ''', (sport,))
    else:
        cursor.execute('SELECT MIN(game_date), MAX(game_date) FROM games')

    row = cursor.fetchone()
    return row[0], row[1]


def get_game_count(conn: sqlite3.Connection, sport: Optional[str] = None) -> int:
    """Get total number of games in the database."""
    cursor = conn.cursor()

    if sport:
        cursor.execute('SELECT COUNT(*) FROM games WHERE sport = ?', (sport,))
    else:
        cursor.execute('SELECT COUNT(*) FROM games')

    return cursor.fetchone()[0]


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Database utilities')
    parser.add_argument('--init', action='store_true', help='Initialize database')
    parser.add_argument('--stats', action='store_true', help='Show database stats')
    args = parser.parse_args()

    if args.init:
        print(f"Initializing database at {DEFAULT_DB_PATH}")
        conn = init_database()
        print("Database initialized successfully")
        conn.close()

    if args.stats:
        conn = get_connection()
        print(f"\nDatabase: {DEFAULT_DB_PATH}")
        print(f"Total games: {get_game_count(conn)}")
        for sport in get_sports(conn):
            count = get_game_count(conn, sport)
            teams = len(get_all_teams(conn, sport))
            dates = get_date_range(conn, sport)
            print(f"  {sport}: {count} games, {teams} teams, {dates[0]} to {dates[1]}")
        conn.close()
