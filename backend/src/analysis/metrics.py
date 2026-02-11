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
from __future__ import annotations

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import numpy as np
except ImportError:
    np = None
from typing import Tuple, Optional, List


# =============================================================================
# Over/Under (Totals) Metrics
# =============================================================================

def ou_cover_rate(
    df: pd.DataFrame,
    handicap: float = 0,
    direction: str = 'over'
) -> float:
    """
    Calculate Over/Under cover rate at a handicap level.

    Args:
        df: DataFrame with total_result column
        handicap: Points to add (for OVER) or subtract (for UNDER)
        direction: 'over' or 'under'

    Returns:
        Cover rate as decimal (0.0 to 1.0)

    Example:
        # OVER hitting at 48%
        rate = ou_cover_rate(games_df, handicap=0, direction='over')

        # UNDER with 5pt cushion hitting at 60%
        rate = ou_cover_rate(games_df, handicap=5, direction='under')
    """
    if len(df) == 0 or 'total_result' not in df.columns:
        return 0.0

    # Filter out games without totals data
    valid = df[df['total_result'].notna()]
    if len(valid) == 0:
        return 0.0

    if direction == 'over':
        # OVER covers when total_result + handicap > 0
        covers = (valid['total_result'] + handicap) > 0
    else:  # under
        # UNDER covers when total_result - handicap < 0
        covers = (valid['total_result'] - handicap) < 0

    return covers.mean()


def ou_record(
    df: pd.DataFrame,
    handicap: float = 0,
    direction: str = 'over'
) -> Tuple[int, int, int]:
    """
    Calculate O/U record as (covers, non-covers, pushes).

    Args:
        df: DataFrame with total_result column
        handicap: Points adjustment
        direction: 'over' or 'under'

    Returns:
        Tuple of (covers, non-covers, pushes)
    """
    if len(df) == 0 or 'total_result' not in df.columns:
        return (0, 0, 0)

    # Filter out games without totals data
    valid = df[df['total_result'].notna()]
    if len(valid) == 0:
        return (0, 0, 0)

    if direction == 'over':
        adjusted = valid['total_result'] + handicap
        covers = (adjusted > 0).sum()
        non_covers = (adjusted < 0).sum()
        pushes = (adjusted == 0).sum()
    else:  # under
        adjusted = valid['total_result'] - handicap
        covers = (adjusted < 0).sum()
        non_covers = (adjusted > 0).sum()
        pushes = (adjusted == 0).sum()

    return (int(covers), int(non_covers), int(pushes))


def team_ou_cover_rate(df: pd.DataFrame, handicap: float = 0) -> float:
    """
    Calculate O/U cover rate for a specific team's games.

    For totals, there's no home/away distinction - the total is the same
    regardless of which team's perspective we're viewing from.

    Args:
        df: DataFrame with total_result column (team's games)
        handicap: Points to add (OVER covers if total_result + handicap > 0)

    Returns:
        OVER cover rate as decimal
    """
    if len(df) == 0 or 'total_result' not in df.columns:
        return 0.0

    valid = df[df['total_result'].notna()]
    if len(valid) == 0:
        return 0.0

    # OVER covers when total_result + handicap > 0
    covered = (valid['total_result'] + handicap) > 0
    return float(covered.mean())


def team_ou_record(df: pd.DataFrame, handicap: float = 0) -> Tuple[int, int, int]:
    """
    Calculate O/U record for a specific team's games.

    Args:
        df: DataFrame with total_result column
        handicap: Points adjustment

    Returns:
        Tuple of (overs, unders, pushes)
    """
    if len(df) == 0 or 'total_result' not in df.columns:
        return (0, 0, 0)

    valid = df[df['total_result'].notna()]
    if len(valid) == 0:
        return (0, 0, 0)

    adjusted = valid['total_result'] + handicap
    overs = int((adjusted > 0).sum())
    unders = int((adjusted < 0).sum())
    pushes = int((adjusted == 0).sum())

    return (overs, unders, pushes)


# =============================================================================
# ATS (Spread) Metrics
# =============================================================================

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

    Uses vectorized numpy operations for performance (10-100x faster than apply).

    Args:
        df: DataFrame with is_home and spread_result columns
        handicap: Additional points to give the team

    Returns:
        Cover rate as decimal
    """
    if len(df) == 0 or 'is_home' not in df.columns:
        return 0.0

    # Vectorized calculation using numpy
    # When team is home: spread_result + handicap >= 0 means cover
    # When team is away: spread_result - handicap <= 0 means cover
    spread_result = df['spread_result'].values
    is_home = df['is_home'].values

    covered = np.where(
        is_home,
        spread_result + handicap >= 0,
        spread_result - handicap <= 0
    )

    return float(covered.mean())


def team_ats_record(df: pd.DataFrame, handicap: float = 0) -> Tuple[int, int, int]:
    """
    Calculate ATS record for a specific team.

    Uses vectorized numpy operations for performance (10-100x faster than iterrows).

    Args:
        df: DataFrame with is_home and spread_result columns
        handicap: Additional points to give

    Returns:
        Tuple of (wins, losses, pushes)
    """
    if len(df) == 0 or 'is_home' not in df.columns:
        return (0, 0, 0)

    # Vectorized calculation using numpy
    # When home: adjusted = spread_result + handicap
    # When away: adjusted = -(spread_result - handicap)
    spread_result = df['spread_result'].values
    is_home = df['is_home'].values

    adjusted = np.where(
        is_home,
        spread_result + handicap,
        -(spread_result - handicap)
    )

    wins = int((adjusted > 0).sum())
    losses = int((adjusted < 0).sum())
    pushes = int((adjusted == 0).sum())

    return (wins, losses, pushes)


def streak_continuation_analysis(
    conn,
    sport: str,
    streak_length: int,
    streak_type: str,
    handicap_range: tuple = (0, 15),
    _all_games_cache: dict = None
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
        _all_games_cache: Optional cache dict to reuse games data across calls

    Returns:
        DataFrame with columns:
        - handicap: The handicap level (0-15)
        - covers: Number of next games that covered at this handicap
        - total: Total situations
        - cover_pct: Percentage that covered
    """
    from ..database import get_games

    # Fetch all games for this sport ONCE (not per-team)
    if _all_games_cache is not None and sport in _all_games_cache:
        all_games = _all_games_cache[sport]
    else:
        all_games = get_games(conn, sport=sport)
        if _all_games_cache is not None:
            _all_games_cache[sport] = all_games

    if len(all_games) == 0:
        return pd.DataFrame()

    # Get unique teams from the data
    teams = set(all_games['home_team']) | set(all_games['away_team'])

    # Collect all "next game" spread results after the specified streak
    next_game_spread_results = []  # List of (spread_result, is_home) tuples

    for team in teams:
        # Filter games for this team in memory (no database query)
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) < streak_length + 1:
            continue

        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Add is_home column
        team_games['is_home'] = team_games['home_team'] == team

        # Calculate cover (at 0 handicap) for each game to identify streaks
        game_data = []
        for _, row in team_games.iterrows():
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
    Calculates from BOTH home and away perspectives to match how
    streak_continuation_analysis evaluates team performance.

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
        # Calculate cover rate from BOTH perspectives (matches streak analysis logic)
        # Home covers when spread_result + handicap > 0
        home_covers = ((games['spread_result'] + handicap) > 0).sum()
        # Away covers when -(spread_result - handicap) > 0, i.e., spread_result < handicap
        away_covers = ((games['spread_result'] - handicap) < 0).sum()

        total_covers = home_covers + away_covers
        total_situations = len(games) * 2  # Each game has 2 team perspectives

        results.append({
            'handicap': handicap,
            'baseline_covers': int(total_covers),
            'baseline_total': total_situations,
            'baseline_cover_pct': total_covers / total_situations if total_situations > 0 else 0
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
    from ..database import get_games

    # Fetch all games for this sport ONCE
    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0:
        return pd.DataFrame()

    # Get unique teams from the data
    teams = set(all_games['home_team']) | set(all_games['away_team'])
    counts = {}

    for team in teams:
        # Filter games for this team in memory
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) < 3:
            continue

        team_games = team_games.sort_values('game_date').reset_index(drop=True)
        team_games['is_home'] = team_games['home_team'] == team

        # Calculate cover at 0 handicap
        covers = []
        for _, row in team_games.iterrows():
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
    from ..database import get_games

    # Fetch all games for this sport ONCE
    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0:
        return pd.DataFrame()

    # Get unique teams from the data
    teams = set(all_games['home_team']) | set(all_games['away_team'])
    situations = []

    for team in teams:
        # Filter games for this team in memory
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) < streak_length + 1:
            continue

        team_games = team_games.sort_values('game_date').reset_index(drop=True)
        team_games['is_home'] = team_games['home_team'] == team

        # Calculate cover for each game
        game_data = []
        for _, row in team_games.iterrows():
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
# Over/Under (Totals) Streak Analysis
# =============================================================================

def ou_streak_continuation_analysis(
    conn,
    sport: str,
    streak_length: int,
    streak_type: str,
    handicap_range: tuple = (0, 20),
    direction: str = 'over',
    _all_games_cache: dict = None
) -> pd.DataFrame:
    """
    Analyze what happens to O/U results after OVER/UNDER streaks.

    After X consecutive OVERs/UNDERs, what's the O/U cover rate at each handicap?

    Args:
        conn: Database connection
        sport: Sport to analyze
        streak_length: Number of consecutive games (e.g., 3)
        streak_type: 'OVER' or 'UNDER' - the streak to look for
        handicap_range: Points added to total (0-20)
        direction: 'over' or 'under' - which bet to analyze in the next game
        _all_games_cache: Optional cache dict to reuse games data across calls

    Returns:
        DataFrame with: handicap, covers, total, cover_pct
    """
    from ..database import get_games

    # Fetch all games for this sport ONCE
    if _all_games_cache is not None and sport in _all_games_cache:
        all_games = _all_games_cache[sport]
    else:
        all_games = get_games(conn, sport=sport)
        if _all_games_cache is not None:
            _all_games_cache[sport] = all_games

    if len(all_games) == 0 or 'total_result' not in all_games.columns:
        return pd.DataFrame()

    # Filter games with totals data
    all_games = all_games[all_games['total_result'].notna()]
    if len(all_games) == 0:
        return pd.DataFrame()

    # Get unique teams from the data
    teams = set(all_games['home_team']) | set(all_games['away_team'])

    # Collect all "next game" total results after the specified streak
    next_game_total_results = []

    for team in teams:
        # Filter games for this team in memory
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) < streak_length + 1:
            continue

        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Calculate O/U result for each game to identify streaks
        game_data = []
        for _, row in team_games.iterrows():
            total_result = row['total_result']
            # Skip pushes
            if total_result == 0:
                is_over = None
            else:
                is_over = total_result > 0
            game_data.append({
                'total_result': total_result,
                'is_over': is_over
            })

        # Find situations matching the streak criteria
        target_is_over = (streak_type == 'OVER')

        for i in range(streak_length, len(game_data)):
            # Skip if this game was a push
            if game_data[i]['is_over'] is None:
                continue

            # Check if there's a streak of exactly streak_length ending at i-1
            streak_len = 0
            for j in range(i-1, -1, -1):
                if game_data[j]['is_over'] is None:
                    # Push breaks the streak
                    break
                if game_data[j]['is_over'] == target_is_over:
                    streak_len += 1
                else:
                    break

            if streak_len >= streak_length:
                # Record the next game's total result for handicap analysis
                next_game_total_results.append({
                    'total_result': game_data[i]['total_result']
                })

    if not next_game_total_results:
        return pd.DataFrame()

    # Now calculate cover rate at each handicap level
    results = []
    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        covers = 0
        for ng in next_game_total_results:
            if direction == 'over':
                # OVER covers when total_result + handicap > 0
                covered = (ng['total_result'] + handicap) > 0
            else:  # under
                # UNDER covers when total_result - handicap < 0
                covered = (ng['total_result'] - handicap) < 0
            if covered:
                covers += 1

        total = len(next_game_total_results)
        results.append({
            'handicap': handicap,
            'covers': covers,
            'total': total,
            'cover_pct': covers / total if total > 0 else 0
        })

    return pd.DataFrame(results)


def baseline_ou_coverage(
    conn,
    sport: str,
    handicap_range: tuple = (0, 20),
    direction: str = 'over'
) -> pd.DataFrame:
    """
    Calculate league-wide baseline O/U cover rate at each handicap.

    Args:
        conn: Database connection
        sport: Sport to analyze
        handicap_range: (min, max) handicap to analyze
        direction: 'over' or 'under' - which bet to calculate baseline for

    Returns:
        DataFrame with: handicap, baseline_cover_pct
    """
    from ..database import get_games

    games = get_games(conn, sport=sport)
    if len(games) == 0 or 'total_result' not in games.columns:
        return pd.DataFrame()

    # Filter games with totals data
    games = games[games['total_result'].notna()]
    if len(games) == 0:
        return pd.DataFrame()

    results = []
    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        if direction == 'over':
            # OVER covers when total_result + handicap > 0
            covers = ((games['total_result'] + handicap) > 0).sum()
        else:  # under
            # UNDER covers when total_result - handicap < 0
            covers = ((games['total_result'] - handicap) < 0).sum()

        total = len(games)
        results.append({
            'handicap': handicap,
            'baseline_covers': int(covers),
            'baseline_total': total,
            'baseline_cover_pct': covers / total if total > 0 else 0
        })

    return pd.DataFrame(results)


def ou_streak_summary_all_lengths(
    conn,
    sport: str,
    streak_range: tuple = (2, 10)
) -> pd.DataFrame:
    """
    Get summary of O/U streak occurrences across all lengths.

    Returns count of situations for each streak length/type combination.

    Args:
        conn: Database connection
        sport: Sport to analyze
        streak_range: (min, max) streak lengths to count

    Returns:
        DataFrame with: streak_length, streak_type, situations
    """
    from ..database import get_games

    # Fetch all games for this sport ONCE
    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0 or 'total_result' not in all_games.columns:
        return pd.DataFrame()

    # Filter games with totals data
    all_games = all_games[all_games['total_result'].notna()]
    if len(all_games) == 0:
        return pd.DataFrame()

    # Get unique teams from the data
    teams = set(all_games['home_team']) | set(all_games['away_team'])
    counts = {}

    for team in teams:
        # Filter games for this team in memory
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) < 3:
            continue

        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Calculate O/U result for each game
        ou_results = []
        for _, row in team_games.iterrows():
            total_result = row['total_result']
            # Skip pushes
            if total_result == 0:
                ou_results.append(None)
            else:
                ou_results.append(total_result > 0)  # True = OVER, False = UNDER

        # Count streaks
        for i in range(2, len(ou_results)):
            if ou_results[i] is None:
                continue

            streak_len = 0
            streak_is_over = None

            for j in range(i-1, -1, -1):
                if ou_results[j] is None:
                    break
                if streak_is_over is None:
                    streak_is_over = ou_results[j]
                    streak_len = 1
                elif ou_results[j] == streak_is_over:
                    streak_len += 1
                else:
                    break

            if streak_range[0] <= streak_len <= streak_range[1]:
                key = (streak_len, 'OVER' if streak_is_over else 'UNDER')
                counts[key] = counts.get(key, 0) + 1

    results = []
    for (length, stype), count in sorted(counts.items()):
        results.append({
            'streak_length': length,
            'streak_type': stype,
            'situations': count
        })

    return pd.DataFrame(results)


# =============================================================================
# Team Totals (Individual O/U) Streak Analysis
# =============================================================================

def tt_streak_summary_all_lengths(
    conn,
    sport: str,
    streak_range: tuple = (1, 10)
) -> pd.DataFrame:
    """
    Get summary of team total O/U streak occurrences across all lengths.

    Uses home_team_total_result / away_team_total_result to detect per-team
    OVER/UNDER streaks (individual team score vs team total line).

    Args:
        conn: Database connection
        sport: Sport to analyze
        streak_range: (min, max) streak lengths to count

    Returns:
        DataFrame with: streak_length, streak_type, situations
    """
    from ..database import get_games

    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0:
        return pd.DataFrame()

    # Need team total columns
    for col in ['home_team_total_result', 'away_team_total_result']:
        if col not in all_games.columns:
            return pd.DataFrame()

    teams = set(all_games['home_team']) | set(all_games['away_team'])
    counts = {}

    for team in teams:
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) < 2:
            continue

        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Get TT result for each game (home or away perspective)
        tt_results = []  # True=OVER, False=UNDER, None=push/missing
        for _, row in team_games.iterrows():
            is_home = row['home_team'] == team
            margin = row['home_team_total_result'] if is_home else row['away_team_total_result']
            if pd.isna(margin) or margin == 0:
                tt_results.append(None)
            else:
                tt_results.append(margin > 0)  # True=OVER

        # Count streaks going INTO each game
        for i in range(1, len(tt_results)):
            if tt_results[i] is None:
                continue

            # Walk backwards from i-1 to count streak
            streak_len = 0
            streak_is_over = None
            for j in range(i - 1, -1, -1):
                if tt_results[j] is None:
                    break
                if streak_is_over is None:
                    streak_is_over = tt_results[j]
                    streak_len = 1
                elif tt_results[j] == streak_is_over:
                    streak_len += 1
                else:
                    break

            if streak_len >= streak_range[0] and streak_len <= streak_range[1]:
                key = (streak_len, 'OVER' if streak_is_over else 'UNDER')
                counts[key] = counts.get(key, 0) + 1

    results = []
    for (length, stype), count in sorted(counts.items()):
        results.append({
            'streak_length': length,
            'streak_type': stype,
            'situations': count
        })

    return pd.DataFrame(results)


def tt_streak_continuation_analysis(
    conn,
    sport: str,
    streak_length: int,
    streak_type: str,
    handicap_range: tuple = (0, 10),
    direction: str = 'ride'
) -> pd.DataFrame:
    """
    After a team total OVER/UNDER streak of N+, measure TT coverage in next game.

    RIDE: bet same direction as streak
      - OVER streak → OVER covers if margin > -h
      - UNDER streak → UNDER covers if margin < h
    FADE: bet opposite direction
      - OVER streak → UNDER covers if margin < h
      - UNDER streak → OVER covers if margin > -h

    Args:
        conn: Database connection
        sport: Sport to analyze
        streak_length: Min streak length to look for (N+)
        streak_type: 'OVER' or 'UNDER'
        handicap_range: (min, max) handicap
        direction: 'ride' or 'fade'

    Returns:
        DataFrame with: handicap, covers, total, cover_pct
    """
    from ..database import get_games

    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0:
        return pd.DataFrame()

    for col in ['home_team_total_result', 'away_team_total_result']:
        if col not in all_games.columns:
            return pd.DataFrame()

    teams = set(all_games['home_team']) | set(all_games['away_team'])
    next_margins = []  # TT margins for next game after streak

    for team in teams:
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) < streak_length + 1:
            continue

        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Build per-game TT results
        game_data = []
        for _, row in team_games.iterrows():
            is_home = row['home_team'] == team
            margin = row['home_team_total_result'] if is_home else row['away_team_total_result']
            if pd.isna(margin) or margin == 0:
                game_data.append({'margin': None, 'is_over': None})
            else:
                game_data.append({'margin': margin, 'is_over': margin > 0})

        target_is_over = (streak_type == 'OVER')

        for i in range(streak_length, len(game_data)):
            # Skip if next game has no TT data
            if game_data[i]['margin'] is None:
                continue

            # Count streak ending at i-1
            streak_len = 0
            for j in range(i - 1, -1, -1):
                if game_data[j]['is_over'] is None:
                    break
                if game_data[j]['is_over'] == target_is_over:
                    streak_len += 1
                else:
                    break

            if streak_len >= streak_length:
                next_margins.append(game_data[i]['margin'])

    if not next_margins:
        return pd.DataFrame()

    # Calculate coverage at each handicap
    results = []
    for h in range(handicap_range[0], handicap_range[1] + 1):
        covers = 0
        for margin in next_margins:
            if direction == 'ride':
                if streak_type == 'OVER':
                    covered = margin > -h  # OVER covers with h-pt cushion
                else:  # UNDER
                    covered = margin < h   # UNDER covers with h-pt cushion
            else:  # fade
                if streak_type == 'OVER':
                    covered = margin < h   # Fade OVER = bet UNDER
                else:  # UNDER
                    covered = margin > -h  # Fade UNDER = bet OVER

            if covered:
                covers += 1

        total = len(next_margins)
        results.append({
            'handicap': h,
            'covers': covers,
            'total': total,
            'cover_pct': covers / total if total > 0 else 0
        })

    return pd.DataFrame(results)


def tt_baseline_coverage(
    conn,
    sport: str,
    handicap_range: tuple = (0, 10),
    direction: str = 'over'
) -> pd.DataFrame:
    """
    Baseline team total coverage rate across all games (no streak filter).

    Pools all home_team_total_result + away_team_total_result values.

    Args:
        conn: Database connection
        sport: Sport to analyze
        handicap_range: (min, max) handicap
        direction: 'over' or 'under'

    Returns:
        DataFrame with: handicap, baseline_covers, baseline_total, baseline_cover_pct
    """
    from ..database import get_games

    games = get_games(conn, sport=sport)
    if len(games) == 0:
        return pd.DataFrame()

    for col in ['home_team_total_result', 'away_team_total_result']:
        if col not in games.columns:
            return pd.DataFrame()

    # Pool all TT margins (home + away), skip NaN and pushes
    home_margins = games['home_team_total_result'].dropna()
    away_margins = games['away_team_total_result'].dropna()
    all_margins = pd.concat([home_margins, away_margins], ignore_index=True)
    all_margins = all_margins[all_margins != 0]

    if len(all_margins) == 0:
        return pd.DataFrame()

    results = []
    for h in range(handicap_range[0], handicap_range[1] + 1):
        if direction == 'over':
            covers = (all_margins > -h).sum()
        else:  # under
            covers = (all_margins < h).sum()

        total = len(all_margins)
        results.append({
            'handicap': h,
            'baseline_covers': int(covers),
            'baseline_total': total,
            'baseline_cover_pct': covers / total if total > 0 else 0
        })

    return pd.DataFrame(results)


def convergent_gt_analysis(
    conn,
    sport: str,
    combo_dir: str,
    min_streak_a: int,
    min_streak_b: int = None,
    handicap_range: tuple = (0, 20),
    direction: str = 'ride'
) -> pd.DataFrame:
    """
    When both teams enter on same-direction TT streaks, measure game total coverage.

    For each game, compute both teams' TT streak going into it.
    Filter: one team has min_streak_a+ and the other has min_streak_b+
    (order doesn't matter — either team can fill either slot).

    RIDE: bet same as combo_dir on the game total
      - Both OVER → OVER covers if total_result > -h
      - Both UNDER → UNDER covers if total_result < h
    FADE: bet opposite
      - Both OVER → UNDER covers if total_result < h
      - Both UNDER → OVER covers if total_result > -h

    Args:
        conn: Database connection
        sport: Sport to analyze
        combo_dir: 'OVER' or 'UNDER' — both teams must be on this streak
        min_streak_a: Minimum streak length for Team A
        min_streak_b: Minimum streak length for Team B (defaults to min_streak_a)
        handicap_range: (min, max) handicap
        direction: 'ride' or 'fade'

    Returns:
        DataFrame with: handicap, covers, total, cover_pct, n_games
    """
    if min_streak_b is None:
        min_streak_b = min_streak_a
    from ..database import get_games

    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0:
        return pd.DataFrame()

    for col in ['home_team_total_result', 'away_team_total_result', 'total_result']:
        if col not in all_games.columns:
            return pd.DataFrame()

    # Filter to games with TT and GT data
    valid = all_games[
        all_games['home_team_total_result'].notna() &
        all_games['away_team_total_result'].notna() &
        all_games['total_result'].notna()
    ].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    teams = set(valid['home_team']) | set(valid['away_team'])
    target_is_over = (combo_dir == 'OVER')

    # Pre-compute TT history per team: list of (game_date, is_over) in order
    team_tt_history = {}
    for team in teams:
        team_games = valid[
            (valid['home_team'] == team) | (valid['away_team'] == team)
        ].sort_values('game_date')

        history = []
        for _, row in team_games.iterrows():
            is_home = row['home_team'] == team
            margin = row['home_team_total_result'] if is_home else row['away_team_total_result']
            if pd.isna(margin) or margin == 0:
                history.append((row['game_date'], row['id'], None))
            else:
                history.append((row['game_date'], row['id'], margin > 0))
        team_tt_history[team] = history

    def get_streak_into_game(team, game_id):
        """Get TT streak length for team going into a specific game."""
        history = team_tt_history.get(team, [])
        # Find the index of this game in the team's history
        idx = None
        for i, (gd, gid, is_over) in enumerate(history):
            if gid == game_id:
                idx = i
                break
        if idx is None or idx == 0:
            return 0

        streak_len = 0
        for j in range(idx - 1, -1, -1):
            if history[j][2] is None:
                break
            if history[j][2] == target_is_over:
                streak_len += 1
            else:
                break
        return streak_len

    # Find convergent games
    convergent_total_results = []
    for _, game in valid.iterrows():
        home_streak = get_streak_into_game(game['home_team'], game['id'])
        away_streak = get_streak_into_game(game['away_team'], game['id'])

        # Either team can fill either streak slot (order doesn't matter)
        lo, hi = sorted([min_streak_a, min_streak_b])
        s1, s2 = sorted([home_streak, away_streak])
        if s1 >= lo and s2 >= hi:
            convergent_total_results.append(game['total_result'])

    if not convergent_total_results:
        return pd.DataFrame()

    # Calculate GT coverage at each handicap
    results = []
    for h in range(handicap_range[0], handicap_range[1] + 1):
        covers = 0
        for tr in convergent_total_results:
            if direction == 'ride':
                if combo_dir == 'OVER':
                    covered = tr > -h  # OVER covers
                else:
                    covered = tr < h   # UNDER covers
            else:  # fade
                if combo_dir == 'OVER':
                    covered = tr < h   # Fade OVER = bet UNDER
                else:
                    covered = tr > -h  # Fade UNDER = bet OVER

            if covered:
                covers += 1

        total = len(convergent_total_results)
        results.append({
            'handicap': h,
            'covers': covers,
            'total': total,
            'cover_pct': covers / total if total > 0 else 0,
            'n_games': total
        })

    return pd.DataFrame(results)


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
