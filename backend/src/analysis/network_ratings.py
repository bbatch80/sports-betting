"""
Network-based power rankings using iterative strength propagation.

This module computes two types of ratings:
1. Win-based: True team strength based on game outcomes
2. ATS-based: Market-beating ability based on spread coverage

The gap between them reveals market efficiency:
- Positive gap (ATS > Win): Market undervalues this team
- Negative gap (ATS < Win): Market overvalues this team
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import date
import pandas as pd
import sqlite3

from sqlalchemy import text

from ..database import get_games


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'decay': 0.92,           # Recency decay per week
    'margin_cap': 20,        # Max margin considered
    'learning_rate': 0.03,   # Rating adjustment rate
    'tolerance': 0.0005,     # Convergence threshold
    'min_games': 5,          # Minimum for "reliable" rating
}

# Sport-specific iterations (sparse vs dense graphs)
ITERATIONS_BY_SPORT = {
    'NFL': 100,
    'NBA': 100,
    'NCAAM': 150,  # More iterations for 350+ team sparse graph
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TeamRatings:
    """Network-based power ratings for a team."""
    team: str
    sport: str

    # Win-based (true strength)
    win_rating: float
    win_rank: int

    # ATS-based (market-beating)
    ats_rating: float
    ats_rank: int

    # The key insight
    market_gap: float  # ats_rating - win_rating

    # Supporting stats
    games_analyzed: int
    win_record: str      # "18-7"
    ats_record: str      # "15-10"
    is_reliable: bool    # games >= min_games

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Core Algorithm
# =============================================================================

def compute_network_ratings(
    games_df: pd.DataFrame,
    mode: str = 'ats',
    decay: float = 0.92,
    margin_cap: float = 20,
    max_iterations: int = 100,
    learning_rate: float = 0.03,
    tolerance: float = 0.0005
) -> Dict[str, float]:
    """
    Compute network-based team ratings using iterative strength propagation.

    Args:
        games_df: DataFrame with game data (must have home_team, away_team,
                  home_score, away_score, spread_result, game_date)
        mode: 'ats' for spread coverage, 'win' for game outcomes
        decay: Recency decay rate per week (0.92 = 8% decay per week)
        margin_cap: Maximum margin to consider (caps outliers)
        max_iterations: Maximum iterations before stopping
        learning_rate: How much ratings adjust per game
        tolerance: Convergence threshold for stopping

    Returns:
        Dictionary mapping team names to ratings (0-1 normalized)
    """
    teams = set(games_df['home_team']) | set(games_df['away_team'])
    ratings = {team: 0.5 for team in teams}

    if len(teams) == 0:
        return ratings

    # Filter based on mode
    if mode == 'ats':
        # Skip pushes for ATS
        games = games_df[games_df['spread_result'] != 0].copy()
    else:
        # Skip ties for win-based (rare in basketball, but handle it)
        games = games_df[games_df['home_score'] != games_df['away_score']].copy()

    if len(games) == 0:
        return ratings

    # Ensure game_date is datetime
    games['game_date'] = pd.to_datetime(games['game_date'])
    max_date = games['game_date'].max()

    for iteration in range(max_iterations):
        new_ratings = ratings.copy()

        for _, game in games.iterrows():
            # Determine winner and margin based on mode
            if mode == 'ats':
                home_won = game['spread_result'] > 0
                margin = abs(game['spread_result'])
            else:  # win mode
                home_won = game['home_score'] > game['away_score']
                margin = abs(game['home_score'] - game['away_score'])

            margin = min(margin, margin_cap)

            winner = game['home_team'] if home_won else game['away_team']
            loser = game['away_team'] if home_won else game['home_team']

            # How surprising was this result?
            winner_rating = ratings[winner]
            loser_rating = ratings[loser]
            total = winner_rating + loser_rating
            expected = winner_rating / total if total > 0 else 0.5
            surprise = 1 - expected

            # Recency weight
            days_ago = (max_date - game['game_date']).days
            recency_weight = decay ** (days_ago / 7)

            # Adjustment
            adjustment = surprise * (margin / margin_cap) * learning_rate * recency_weight

            new_ratings[winner] += adjustment
            new_ratings[loser] -= adjustment

        # Normalize using z-score + sigmoid (handles outliers better than min-max)
        import math
        # Filter out nan/inf values for mean/std calculation
        valid_values = [v for v in new_ratings.values() if math.isfinite(v)]
        if len(valid_values) > 1:
            mean_r = sum(valid_values) / len(valid_values)
            std_r = (sum((v - mean_r) ** 2 for v in valid_values) / len(valid_values)) ** 0.5
            if std_r > 0:
                new_ratings = {
                    t: 1 / (1 + math.exp(-max(-10, min(10, (r - mean_r) / std_r))))  # Clamp z-score to prevent overflow
                    if math.isfinite(r) else 0.5
                    for t, r in new_ratings.items()
                }

        # Check convergence
        max_change = max(abs(new_ratings[t] - ratings[t]) for t in teams)
        if max_change < tolerance:
            break

        ratings = new_ratings

    return ratings


# =============================================================================
# Historical Rating Functions
# =============================================================================

def compute_ratings_at_date(
    games_df: pd.DataFrame,
    as_of_date: date,
    mode: str = 'ats',
    **kwargs
) -> Dict[str, float]:
    """
    Compute network ratings using only games BEFORE as_of_date.

    This is essential for backtesting - we need to know what the ratings
    were BEFORE a game was played, not after.

    Args:
        games_df: All games for the sport (will be filtered)
        as_of_date: Compute ratings for this date (uses games < as_of_date)
        mode: 'ats' for spread coverage, 'win' for game outcomes
        **kwargs: Additional args passed to compute_network_ratings

    Returns:
        Dictionary mapping team names to ratings (0-1 normalized)
        Empty dict if no games before as_of_date
    """
    # Ensure game_date is datetime for comparison
    games_df = games_df.copy()
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    as_of_datetime = pd.to_datetime(as_of_date)

    # Filter to games BEFORE the target date
    games_before = games_df[games_df['game_date'] < as_of_datetime]

    if len(games_before) == 0:
        return {}  # No historical data yet

    # Use existing algorithm with filtered games
    return compute_network_ratings(games_before, mode=mode, **kwargs)


def generate_historical_snapshots(
    conn: sqlite3.Connection,
    sport: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_games: int = 5
) -> int:
    """
    Generate and store daily rating snapshots for a sport.

    Creates one snapshot per game date, computing ratings using only
    games played BEFORE that date. Results are stored in the
    historical_ratings table for fast retrieval.

    Args:
        conn: Database connection
        sport: 'NFL', 'NBA', or 'NCAAM'
        start_date: Optional start date for generation (defaults to season start)
        end_date: Optional end date for generation (defaults to latest game)
        min_games: Minimum games for "reliable" rating

    Returns:
        Number of snapshots created
    """
    cfg = DEFAULT_CONFIG.copy()
    max_iterations = ITERATIONS_BY_SPORT.get(sport, 100)

    # Get all games for the sport
    games_df = get_games(conn, sport=sport)
    if len(games_df) == 0:
        return 0

    # Ensure game_date is datetime
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])

    # Get unique game dates (sorted)
    game_dates = sorted(games_df['game_date'].dt.date.unique())

    cursor = conn.cursor()
    snapshots_created = 0

    for game_date in game_dates:
        # Apply date filters if provided
        if start_date and game_date < start_date:
            continue
        if end_date and game_date > end_date:
            continue

        # Compute both ratings using games BEFORE this date
        win_ratings = compute_ratings_at_date(
            games_df, game_date, mode='win',
            decay=cfg['decay'],
            margin_cap=cfg['margin_cap'],
            max_iterations=max_iterations,
            learning_rate=cfg['learning_rate'],
            tolerance=cfg['tolerance']
        )

        ats_ratings = compute_ratings_at_date(
            games_df, game_date, mode='ats',
            decay=cfg['decay'],
            margin_cap=cfg['margin_cap'],
            max_iterations=max_iterations,
            learning_rate=cfg['learning_rate'],
            tolerance=cfg['tolerance']
        )

        if not win_ratings:
            continue  # Skip if no prior games (season start)

        # Get all teams that have ratings
        teams = set(win_ratings.keys()) | set(ats_ratings.keys())

        # Compute ranks for win ratings
        win_sorted = sorted(win_ratings.items(), key=lambda x: x[1], reverse=True)
        win_ranks = {team: i + 1 for i, (team, _) in enumerate(win_sorted)}

        # Compute ranks for ATS ratings
        ats_sorted = sorted(ats_ratings.items(), key=lambda x: x[1], reverse=True)
        ats_ranks = {team: i + 1 for i, (team, _) in enumerate(ats_sorted)}

        # Count games per team (before this date)
        games_before = games_df[games_df['game_date'].dt.date < game_date]
        team_games = {}
        for team in teams:
            home = len(games_before[games_before['home_team'] == team])
            away = len(games_before[games_before['away_team'] == team])
            team_games[team] = home + away

        # Insert snapshots for each team
        for team in teams:
            win_rating = win_ratings.get(team, 0.5)
            ats_rating = ats_ratings.get(team, 0.5)
            market_gap = ats_rating - win_rating
            games_analyzed = team_games.get(team, 0)
            win_rank = win_ranks.get(team, len(teams))
            ats_rank = ats_ranks.get(team, len(teams))

            cursor.execute('''
                INSERT OR REPLACE INTO historical_ratings
                (sport, snapshot_date, team, win_rating, ats_rating, market_gap,
                 games_analyzed, win_rank, ats_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sport,
                game_date.strftime('%Y-%m-%d'),
                team,
                win_rating,
                ats_rating,
                market_gap,
                games_analyzed,
                win_rank,
                ats_rank
            ))

        snapshots_created += 1

    conn.commit()
    return snapshots_created


def get_ratings_at_date(
    conn: sqlite3.Connection,
    sport: str,
    game_date: date
) -> Dict[str, 'TeamRatings']:
    """
    Retrieve pre-computed ratings for a specific date.

    Args:
        conn: Database connection
        sport: 'NFL', 'NBA', or 'NCAAM'
        game_date: The date to get ratings for

    Returns:
        Dictionary mapping team name -> TeamRatings
    """
    query = text('''
        SELECT team, win_rating, ats_rating, market_gap,
               games_analyzed, win_rank, ats_rank
        FROM historical_ratings
        WHERE sport = :sport AND snapshot_date = :game_date
    ''')
    result = conn.execute(query, {'sport': sport, 'game_date': game_date.strftime('%Y-%m-%d')})

    results = {}
    for row in result.fetchall():
        team = row[0]
        results[team] = TeamRatings(
            team=team,
            sport=sport,
            win_rating=row[1],
            win_rank=row[5],
            ats_rating=row[2],
            ats_rank=row[6],
            market_gap=row[3],
            games_analyzed=row[4],
            win_record='',  # Not stored in historical table
            ats_record='',  # Not stored in historical table
            is_reliable=row[4] >= DEFAULT_CONFIG['min_games']
        )

    return results


def get_ratings_timeseries(
    conn: sqlite3.Connection,
    sport: str,
    team: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Get rating evolution for a team over time.

    Args:
        conn: Database connection
        sport: 'NFL', 'NBA', or 'NCAAM'
        team: Team name
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with columns:
            snapshot_date, win_rating, ats_rating, market_gap, win_rank, ats_rank
    """
    query = '''
        SELECT snapshot_date, win_rating, ats_rating, market_gap,
               games_analyzed, win_rank, ats_rank
        FROM historical_ratings
        WHERE sport = :sport AND team = :team
    '''
    params = {'sport': sport, 'team': team}

    if start_date:
        query += ' AND snapshot_date >= :start_date'
        params['start_date'] = start_date.strftime('%Y-%m-%d')

    if end_date:
        query += ' AND snapshot_date <= :end_date'
        params['end_date'] = end_date.strftime('%Y-%m-%d')

    query += ' ORDER BY snapshot_date ASC'

    df = pd.read_sql_query(text(query), conn, params=params)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    return df


# =============================================================================
# Helper Functions
# =============================================================================

def get_team_record(games_df: pd.DataFrame, team: str) -> str:
    """Get win-loss record for a team."""
    wins = 0
    losses = 0

    # Home games
    home_games = games_df[games_df['home_team'] == team]
    wins += (home_games['home_score'] > home_games['away_score']).sum()
    losses += (home_games['home_score'] < home_games['away_score']).sum()

    # Away games
    away_games = games_df[games_df['away_team'] == team]
    wins += (away_games['away_score'] > away_games['home_score']).sum()
    losses += (away_games['away_score'] < away_games['home_score']).sum()

    return f"{wins}-{losses}"


def get_team_ats_record(games_df: pd.DataFrame, team: str) -> str:
    """Get ATS record for a team (excluding pushes)."""
    wins = 0
    losses = 0

    # Home games (spread_result > 0 means home covered)
    home_games = games_df[(games_df['home_team'] == team) & (games_df['spread_result'] != 0)]
    wins += (home_games['spread_result'] > 0).sum()
    losses += (home_games['spread_result'] < 0).sum()

    # Away games (spread_result < 0 means away covered)
    away_games = games_df[(games_df['away_team'] == team) & (games_df['spread_result'] != 0)]
    wins += (away_games['spread_result'] < 0).sum()
    losses += (away_games['spread_result'] > 0).sum()

    return f"{wins}-{losses}"


def get_team_games_count(games_df: pd.DataFrame, team: str) -> int:
    """Get total games played by a team."""
    home = len(games_df[games_df['home_team'] == team])
    away = len(games_df[games_df['away_team'] == team])
    return home + away


# =============================================================================
# Main API Functions
# =============================================================================

def get_team_rankings(
    conn: sqlite3.Connection,
    sport: str,
    min_games: int = 5,
    config: Optional[dict] = None
) -> List[TeamRatings]:
    """
    Get power rankings for all teams in a sport.

    Args:
        conn: Database connection
        sport: 'NFL', 'NBA', or 'NCAAM'
        min_games: Minimum games for reliable rating
        config: Optional config overrides

    Returns:
        List of TeamRatings sorted by market_gap (highest first)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    max_iterations = ITERATIONS_BY_SPORT.get(sport, 100)

    # Get all games for the sport
    games_df = get_games(conn, sport=sport)
    if len(games_df) == 0:
        return []

    # Compute both ratings
    win_ratings = compute_network_ratings(
        games_df,
        mode='win',
        decay=cfg['decay'],
        margin_cap=cfg['margin_cap'],
        max_iterations=max_iterations,
        learning_rate=cfg['learning_rate'],
        tolerance=cfg['tolerance']
    )

    ats_ratings = compute_network_ratings(
        games_df,
        mode='ats',
        decay=cfg['decay'],
        margin_cap=cfg['margin_cap'],
        max_iterations=max_iterations,
        learning_rate=cfg['learning_rate'],
        tolerance=cfg['tolerance']
    )

    # Get all teams
    teams = set(games_df['home_team']) | set(games_df['away_team'])

    # Build rankings list
    rankings = []
    for team in teams:
        games_count = get_team_games_count(games_df, team)
        win_rating = win_ratings.get(team, 0.5)
        ats_rating = ats_ratings.get(team, 0.5)

        rankings.append({
            'team': team,
            'sport': sport,
            'win_rating': win_rating,
            'ats_rating': ats_rating,
            'market_gap': ats_rating - win_rating,
            'games_analyzed': games_count,
            'win_record': get_team_record(games_df, team),
            'ats_record': get_team_ats_record(games_df, team),
            'is_reliable': games_count >= min_games,
        })

    # Sort by win rating to assign ranks
    rankings.sort(key=lambda x: x['win_rating'], reverse=True)
    for i, r in enumerate(rankings):
        r['win_rank'] = i + 1

    # Sort by ATS rating to assign ranks
    rankings.sort(key=lambda x: x['ats_rating'], reverse=True)
    for i, r in enumerate(rankings):
        r['ats_rank'] = i + 1

    # Convert to dataclass objects
    result = [TeamRatings(**r) for r in rankings]

    # Sort by market gap (highest first = most undervalued)
    result.sort(key=lambda x: x.market_gap, reverse=True)

    return result


def get_all_rankings(
    conn: sqlite3.Connection,
    min_games: int = 5
) -> Dict[str, List[TeamRatings]]:
    """
    Get power rankings for all sports.

    Returns:
        Dictionary mapping sport to list of TeamRatings
    """
    return {
        sport: get_team_rankings(conn, sport, min_games)
        for sport in ['NFL', 'NBA', 'NCAAM']
    }


def get_rankings_dataframe(
    conn: sqlite3.Connection,
    sport: str,
    min_games: int = 5
) -> pd.DataFrame:
    """
    Get power rankings as a DataFrame for display.

    Args:
        conn: Database connection
        sport: Sport to get rankings for
        min_games: Minimum games for reliable rating

    Returns:
        DataFrame with rankings data
    """
    rankings = get_team_rankings(conn, sport, min_games)

    if not rankings:
        return pd.DataFrame()

    data = [r.to_dict() for r in rankings]
    df = pd.DataFrame(data)

    # Reorder columns for display
    columns = [
        'team', 'win_rating', 'ats_rating', 'market_gap',
        'win_record', 'ats_record', 'games_analyzed', 'is_reliable'
    ]
    df = df[columns]

    return df


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    from ..database import get_connection

    print("=" * 70)
    print("NETWORK POWER RANKINGS TEST")
    print("=" * 70)

    conn = get_connection()

    for sport in ['NFL', 'NBA', 'NCAAM']:
        print(f"\n{'='*70}")
        print(f"{sport} RANKINGS")
        print("=" * 70)

        rankings = get_team_rankings(conn, sport, min_games=5)
        reliable = [r for r in rankings if r.is_reliable]

        print(f"Total teams: {len(rankings)}")
        print(f"Reliable (5+ games): {len(reliable)}")

        if reliable:
            print(f"\nTop 10 by Market Gap (most undervalued):")
            print("-" * 70)
            print(f"{'Team':<20} {'Win':<8} {'ATS':<8} {'Gap':<8} {'Record':<10} {'ATS Rec':<10}")
            print("-" * 70)

            for r in reliable[:10]:
                print(f"{r.team:<20} {r.win_rating:.3f}    {r.ats_rating:.3f}    "
                      f"{r.market_gap:+.3f}   {r.win_record:<10} {r.ats_record:<10}")

            print(f"\nBottom 5 by Market Gap (most overvalued):")
            print("-" * 70)
            for r in reliable[-5:]:
                print(f"{r.team:<20} {r.win_rating:.3f}    {r.ats_rating:.3f}    "
                      f"{r.market_gap:+.3f}   {r.win_record:<10} {r.ats_record:<10}")

    conn.close()
