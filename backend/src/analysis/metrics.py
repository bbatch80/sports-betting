"""
Core metrics calculations for sports betting analysis.

All functions operate on DataFrames with standard columns:
- spread_result: home_score - away_score - spread (positive = home covered)
- game_date: date of game
- is_home: True if analyzing from home team perspective
- team_covered: True if the team of interest covered

For league-wide (macro) analysis, use spread_result directly.
For team-specific (micro) analysis, use team_covered column.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List


def ats_cover_rate(
    df: pd.DataFrame,
    handicap: float = 0,
    perspective: str = 'home'
) -> float:
    """
    Calculate ATS (against the spread) cover rate.

    Args:
        df: DataFrame with spread_result column
        handicap: Additional points to give (positive = more cushion)
        perspective: 'home' or 'away' - which side we're calculating for

    Returns:
        Cover rate as decimal (0.0 to 1.0)

    Example:
        # Home teams covering at 52%
        rate = ats_cover_rate(games_df, handicap=0, perspective='home')

        # Away teams with 7pt handicap covering at 58%
        rate = ats_cover_rate(games_df, handicap=7, perspective='away')
    """
    if len(df) == 0:
        return 0.0

    if perspective == 'home':
        # Home covers when spread_result + handicap >= 0
        covers = (df['spread_result'] + handicap) >= 0
    else:
        # Away covers when spread_result - handicap <= 0
        # (spread_result is from home perspective, so negative = away covered)
        covers = (df['spread_result'] - handicap) <= 0

    return covers.mean()


def ats_record(
    df: pd.DataFrame,
    handicap: float = 0,
    perspective: str = 'home'
) -> Tuple[int, int, int]:
    """
    Calculate ATS record as (wins, losses, pushes).

    Args:
        df: DataFrame with spread_result column
        handicap: Additional points to give
        perspective: 'home' or 'away'

    Returns:
        Tuple of (wins, losses, pushes)
    """
    if len(df) == 0:
        return (0, 0, 0)

    if perspective == 'home':
        adjusted = df['spread_result'] + handicap
    else:
        adjusted = -(df['spread_result'] - handicap)

    wins = (adjusted > 0).sum()
    losses = (adjusted < 0).sum()
    pushes = (adjusted == 0).sum()

    return (int(wins), int(losses), int(pushes))


def spread_margin_avg(df: pd.DataFrame, perspective: str = 'home') -> float:
    """
    Calculate average spread margin (how much team beats/misses spread by).

    Args:
        df: DataFrame with spread_result column
        perspective: 'home' or 'away'

    Returns:
        Average margin (positive = beating spread on average)
    """
    if len(df) == 0:
        return 0.0

    if perspective == 'home':
        return df['spread_result'].mean()
    else:
        return -df['spread_result'].mean()


def handicap_cover_rate(
    df: pd.DataFrame,
    handicap_range: List[int] = None,
    perspective: str = 'home'
) -> pd.DataFrame:
    """
    Calculate cover rates across multiple handicap values.

    Args:
        df: DataFrame with spread_result column
        handicap_range: List of handicap values (default: 0-15)
        perspective: 'home' or 'away'

    Returns:
        DataFrame with columns: handicap, games, covers, cover_pct
    """
    if handicap_range is None:
        handicap_range = list(range(0, 16))

    results = []
    for h in handicap_range:
        wins, losses, pushes = ats_record(df, handicap=h, perspective=perspective)
        total = wins + losses + pushes
        results.append({
            'handicap': h,
            'games': total,
            'covers': wins,
            'cover_pct': wins / total if total > 0 else 0.0
        })

    return pd.DataFrame(results)


def time_series_ats(
    df: pd.DataFrame,
    handicap: float = 0,
    perspective: str = 'home',
    cumulative: bool = True
) -> pd.DataFrame:
    """
    Calculate ATS cover rate over time.

    Args:
        df: DataFrame with spread_result and game_date columns
        handicap: Additional points to give
        perspective: 'home' or 'away'
        cumulative: If True, show cumulative rate; if False, show per-game

    Returns:
        DataFrame with columns: game_date, games, covers, cover_pct
    """
    if len(df) == 0:
        return pd.DataFrame(columns=['game_date', 'games', 'covers', 'cover_pct'])

    df = df.copy().sort_values('game_date')

    if perspective == 'home':
        df['covered'] = (df['spread_result'] + handicap) >= 0
    else:
        df['covered'] = (df['spread_result'] - handicap) <= 0

    # Group by date
    daily = df.groupby('game_date').agg(
        games=('covered', 'count'),
        covers=('covered', 'sum')
    ).reset_index()

    if cumulative:
        daily['games'] = daily['games'].cumsum()
        daily['covers'] = daily['covers'].cumsum()

    daily['cover_pct'] = daily['covers'] / daily['games']

    return daily


def team_ats_cover_rate(df: pd.DataFrame, handicap: float = 0) -> float:
    """
    Calculate ATS cover rate for a specific team.

    Expects DataFrame with 'team_covered' column (from get_games with team filter).

    Args:
        df: DataFrame with team_covered and spread_result columns
        handicap: Additional points to give the team

    Returns:
        Cover rate as decimal
    """
    if len(df) == 0 or 'is_home' not in df.columns:
        return 0.0

    # Apply handicap adjustment
    df = df.copy()
    # When team is home: spread_result + handicap >= 0 means cover
    # When team is away: spread_result - handicap <= 0 means cover
    df['covered_with_handicap'] = df.apply(
        lambda r: (r['spread_result'] + handicap >= 0) if r['is_home']
                  else (r['spread_result'] - handicap <= 0),
        axis=1
    )

    return df['covered_with_handicap'].mean()


def team_ats_record(df: pd.DataFrame, handicap: float = 0) -> Tuple[int, int, int]:
    """
    Calculate ATS record for a specific team.

    Args:
        df: DataFrame with is_home and spread_result columns
        handicap: Additional points to give

    Returns:
        Tuple of (wins, losses, pushes)
    """
    if len(df) == 0 or 'is_home' not in df.columns:
        return (0, 0, 0)

    wins = 0
    losses = 0
    pushes = 0

    for _, row in df.iterrows():
        if row['is_home']:
            adjusted = row['spread_result'] + handicap
        else:
            adjusted = -(row['spread_result'] - handicap)

        if adjusted > 0:
            wins += 1
        elif adjusted < 0:
            losses += 1
        else:
            pushes += 1

    return (wins, losses, pushes)


def streak_continuation_analysis(
    conn,
    sport: str,
    streak_length: int,
    streak_type: str,
    handicap_range: tuple = (0, 15)
) -> pd.DataFrame:
    """
    Analyze what happens after ATS winning/losing streaks across all handicaps.

    For a specific streak length and type, calculates cover rate at each
    handicap level (0-15) for the NEXT game.

    Args:
        conn: Database connection
        sport: Sport to analyze
        streak_length: The streak length to analyze (e.g., 3 for 3-game streak)
        streak_type: 'WIN' or 'LOSS'
        handicap_range: (min, max) handicap points to analyze

    Returns:
        DataFrame with columns:
        - handicap: The handicap level (0-15)
        - covers: Number of next games that covered at this handicap
        - total: Total situations
        - cover_pct: Percentage that covered
    """
    from ..database import get_games, get_all_teams

    teams = get_all_teams(conn, sport)

    # Collect all "next game" spread results after the specified streak
    next_game_spread_results = []  # List of (spread_result, is_home) tuples

    for team in teams:
        games = get_games(conn, sport=sport, team=team)
        if len(games) < streak_length + 1:
            continue

        games = games.sort_values('game_date').reset_index(drop=True)

        # Calculate cover (at 0 handicap) for each game to identify streaks
        game_data = []
        for _, row in games.iterrows():
            is_home = row['is_home']
            spread_result = row['spread_result']
            # Cover at 0 handicap
            covered_base = (spread_result > 0) if is_home else (spread_result < 0)
            game_data.append({
                'spread_result': spread_result,
                'is_home': is_home,
                'covered_base': covered_base
            })

        # Find situations matching the streak criteria
        target_type = (streak_type == 'WIN')

        for i in range(streak_length, len(game_data)):
            # Check if there's a streak of exactly streak_length ending at i-1
            streak_len = 1
            for j in range(i-2, -1, -1):
                if game_data[j]['covered_base'] == game_data[i-1]['covered_base']:
                    streak_len += 1
                else:
                    break

            if streak_len >= streak_length and game_data[i-1]['covered_base'] == target_type:
                # Record the next game's spread result for handicap analysis
                next_game_spread_results.append({
                    'spread_result': game_data[i]['spread_result'],
                    'is_home': game_data[i]['is_home']
                })

    if not next_game_spread_results:
        return pd.DataFrame()

    # Now calculate cover rate at each handicap level
    results = []
    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        covers = 0
        for ng in next_game_spread_results:
            if ng['is_home']:
                covered = (ng['spread_result'] + handicap) > 0
            else:
                covered = -(ng['spread_result'] - handicap) > 0
            if covered:
                covers += 1

        total = len(next_game_spread_results)
        results.append({
            'handicap': handicap,
            'covers': covers,
            'total': total,
            'cover_pct': covers / total if total > 0 else 0
        })

    return pd.DataFrame(results)


def baseline_handicap_coverage(
    conn,
    sport: str,
    handicap_range: tuple = (0, 15)
) -> pd.DataFrame:
    """
    Calculate league-wide baseline cover rate at each handicap level.

    This gives the "expected" cover rate if there's no streak effect.

    Args:
        conn: Database connection
        sport: Sport to analyze
        handicap_range: (min, max) handicap to analyze

    Returns:
        DataFrame with columns: handicap, baseline_cover_pct
    """
    from ..database import get_games

    games = get_games(conn, sport=sport)
    if len(games) == 0:
        return pd.DataFrame()

    results = []
    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        # Calculate cover rate at this handicap (home perspective)
        # Home covers when spread_result + handicap > 0
        covers = ((games['spread_result'] + handicap) > 0).sum()
        total = len(games)

        results.append({
            'handicap': handicap,
            'baseline_covers': int(covers),
            'baseline_total': total,
            'baseline_cover_pct': covers / total if total > 0 else 0
        })

    return pd.DataFrame(results)


def streak_summary_all_lengths(
    conn,
    sport: str,
    streak_range: tuple = (2, 10)
) -> pd.DataFrame:
    """
    Get summary of streak occurrences across all lengths.

    Returns count of situations for each streak length/type combination.
    """
    from ..database import get_games, get_all_teams

    teams = get_all_teams(conn, sport)
    counts = {}

    for team in teams:
        games = get_games(conn, sport=sport, team=team)
        if len(games) < 3:
            continue

        games = games.sort_values('game_date').reset_index(drop=True)

        # Calculate cover at 0 handicap
        covers = []
        for _, row in games.iterrows():
            covered = (row['spread_result'] > 0) if row['is_home'] else (row['spread_result'] < 0)
            covers.append(covered)

        # Count streaks
        for i in range(2, len(covers)):
            streak_len = 1
            streak_type = covers[i-1]

            for j in range(i-2, -1, -1):
                if covers[j] == streak_type:
                    streak_len += 1
                else:
                    break

            if streak_range[0] <= streak_len <= streak_range[1]:
                key = (streak_len, 'WIN' if streak_type else 'LOSS')
                counts[key] = counts.get(key, 0) + 1

    results = []
    for (length, stype), count in sorted(counts.items()):
        results.append({
            'streak_length': length,
            'streak_type': stype,
            'situations': count
        })

    return pd.DataFrame(results)


def get_streak_situations_detail(
    conn,
    sport: str,
    streak_length: int,
    streak_type: str,
    handicap: float = 0
) -> pd.DataFrame:
    """
    Get detailed list of all situations where a team had a specific streak.

    Args:
        conn: Database connection
        sport: Sport to analyze
        streak_length: The streak length to find (e.g., 3 for 3-game streak)
        streak_type: 'WIN' or 'LOSS'
        handicap: Points to add to spread

    Returns:
        DataFrame with each instance: team, date, opponent, spread, outcome
    """
    from ..database import get_games, get_all_teams

    teams = get_all_teams(conn, sport)
    situations = []

    for team in teams:
        games = get_games(conn, sport=sport, team=team)
        if len(games) < streak_length + 1:
            continue

        games = games.sort_values('game_date').reset_index(drop=True)

        # Calculate cover for each game
        game_data = []
        for _, row in games.iterrows():
            if row['is_home']:
                covered = (row['spread_result'] + handicap) > 0
                opponent = row['away_team']
            else:
                covered = (row['spread_result'] - handicap) < 0
                opponent = row['home_team']

            game_data.append({
                'date': row['game_date'],
                'opponent': opponent,
                'spread': row['closing_spread'],
                'spread_result': row['spread_result'],
                'is_home': row['is_home'],
                'covered': covered
            })

        # Find situations matching the streak criteria
        target_type = (streak_type == 'WIN')

        for i in range(streak_length, len(game_data)):
            # Check if there's a streak of exactly streak_length ending at i-1
            streak_len = 1
            for j in range(i-2, -1, -1):
                if game_data[j]['covered'] == game_data[i-1]['covered']:
                    streak_len += 1
                else:
                    break

            if streak_len == streak_length and game_data[i-1]['covered'] == target_type:
                situations.append({
                    'team': team,
                    'streak_games': f"{streak_length}-game {streak_type}",
                    'next_game_date': game_data[i]['date'],
                    'next_opponent': game_data[i]['opponent'],
                    'next_spread': game_data[i]['spread'],
                    'next_is_home': game_data[i]['is_home'],
                    'next_covered': game_data[i]['covered'],
                    'result': '✅ COVER' if game_data[i]['covered'] else '❌ LOSS'
                })

    return pd.DataFrame(situations)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

    from src.database import get_connection, get_games

    conn = get_connection()

    print("=" * 60)
    print("METRICS TEST")
    print("=" * 60)

    # Test macro metrics (league-wide)
    for sport in ['NFL', 'NBA', 'NCAAM']:
        games = get_games(conn, sport=sport)
        if len(games) == 0:
            continue

        print(f"\n{sport} ({len(games)} games):")

        # Home team stats
        home_rate = ats_cover_rate(games, handicap=0, perspective='home')
        home_record = ats_record(games, handicap=0, perspective='home')
        print(f"  Home ATS: {home_rate:.1%} ({home_record[0]}-{home_record[1]}-{home_record[2]})")

        # Away team stats
        away_rate = ats_cover_rate(games, handicap=0, perspective='away')
        away_record = ats_record(games, handicap=0, perspective='away')
        print(f"  Away ATS: {away_rate:.1%} ({away_record[0]}-{away_record[1]}-{away_record[2]})")

        # Handicap coverage
        print(f"  Home +11pt: {ats_cover_rate(games, handicap=11, perspective='home'):.1%}")
        print(f"  Away +11pt: {ats_cover_rate(games, handicap=11, perspective='away'):.1%}")

    conn.close()
