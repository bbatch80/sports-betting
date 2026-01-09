"""
Aggregation functions for macro and micro level analysis.

Macro = League-wide aggregations (all teams)
Micro = Individual team aggregations
"""

import pandas as pd
from typing import Optional, List
from datetime import date

from .metrics import (
    ats_cover_rate,
    ats_record,
    spread_margin_avg,
    handicap_cover_rate,
    time_series_ats,
    team_ats_cover_rate,
    team_ats_record,
)


def macro_ats_summary(
    df: pd.DataFrame,
    handicaps: List[int] = None
) -> pd.DataFrame:
    """
    Generate macro (league-wide) ATS summary.

    Returns DataFrame with home and away ATS rates at various handicaps.
    """
    if handicaps is None:
        handicaps = [0, 3, 5, 7, 10, 11, 13, 15]

    results = []
    for h in handicaps:
        home_w, home_l, home_p = ats_record(df, handicap=h, perspective='home')
        away_w, away_l, away_p = ats_record(df, handicap=h, perspective='away')

        results.append({
            'handicap': h,
            'home_wins': home_w,
            'home_losses': home_l,
            'home_pushes': home_p,
            'home_pct': home_w / (home_w + home_l + home_p) if (home_w + home_l + home_p) > 0 else 0,
            'away_wins': away_w,
            'away_losses': away_l,
            'away_pushes': away_p,
            'away_pct': away_w / (away_w + away_l + away_p) if (away_w + away_l + away_p) > 0 else 0,
        })

    return pd.DataFrame(results)


def macro_time_series(
    df: pd.DataFrame,
    handicap: float = 0,
    perspective: str = 'home'
) -> pd.DataFrame:
    """
    Generate time series of ATS cover rate (league-wide).

    Returns DataFrame with: game_date, games, covers, cover_pct
    """
    return time_series_ats(df, handicap=handicap, perspective=perspective, cumulative=True)


def macro_by_spread_bucket(
    df: pd.DataFrame,
    buckets: List[tuple] = None
) -> pd.DataFrame:
    """
    Analyze ATS performance by spread size buckets.

    Args:
        df: Games DataFrame
        buckets: List of (min, max) spread tuples (home perspective)
                 Negative = home favored, Positive = away favored

    Returns:
        DataFrame with ATS stats per bucket
    """
    if buckets is None:
        # Default buckets (from home team perspective)
        buckets = [
            (-100, -14, 'Home -14+'),      # Big home favorites
            (-13.5, -10, 'Home -10 to -13.5'),
            (-9.5, -7, 'Home -7 to -9.5'),
            (-6.5, -3.5, 'Home -3.5 to -6.5'),
            (-3, 0, 'Home -3 to PK'),
            (0.5, 3, 'Away -3 to PK'),
            (3.5, 6.5, 'Away -3.5 to -6.5'),
            (7, 9.5, 'Away -7 to -9.5'),
            (10, 13.5, 'Away -10 to -13.5'),
            (14, 100, 'Away -14+'),
        ]

    results = []
    for bucket in buckets:
        min_spread, max_spread, label = bucket
        subset = df[(df['closing_spread'] >= min_spread) & (df['closing_spread'] <= max_spread)]

        if len(subset) == 0:
            continue

        home_w, home_l, home_p = ats_record(subset, handicap=0, perspective='home')

        results.append({
            'bucket': label,
            'min_spread': min_spread,
            'max_spread': max_spread,
            'games': len(subset),
            'home_covers': home_w,
            'home_pct': home_w / len(subset) if len(subset) > 0 else 0,
            'away_pct': (home_l + home_p) / len(subset) if len(subset) > 0 else 0,
        })

    return pd.DataFrame(results)


def micro_team_summary(
    conn,
    sport: str,
    team: str,
    handicaps: List[int] = None
) -> dict:
    """
    Generate micro (team-specific) summary.

    Returns dict with team's ATS performance home/away/overall.
    """
    from ..database import get_games

    if handicaps is None:
        handicaps = [0, 7, 11]

    # Get all games for team
    all_games = get_games(conn, sport=sport, team=team)
    home_games = get_games(conn, sport=sport, team=team, venue='home')
    away_games = get_games(conn, sport=sport, team=team, venue='away')

    result = {
        'team': team,
        'sport': sport,
        'total_games': len(all_games),
        'home_games': len(home_games),
        'away_games': len(away_games),
    }

    for h in handicaps:
        suffix = f'_h{h}' if h > 0 else ''

        # Overall
        if len(all_games) > 0:
            result[f'overall_pct{suffix}'] = team_ats_cover_rate(all_games, handicap=h)
            w, l, p = team_ats_record(all_games, handicap=h)
            result[f'overall_record{suffix}'] = f'{w}-{l}-{p}'

        # Home
        if len(home_games) > 0:
            result[f'home_pct{suffix}'] = team_ats_cover_rate(home_games, handicap=h)
            w, l, p = team_ats_record(home_games, handicap=h)
            result[f'home_record{suffix}'] = f'{w}-{l}-{p}'

        # Away
        if len(away_games) > 0:
            result[f'away_pct{suffix}'] = team_ats_cover_rate(away_games, handicap=h)
            w, l, p = team_ats_record(away_games, handicap=h)
            result[f'away_record{suffix}'] = f'{w}-{l}-{p}'

    return result


def micro_all_teams(
    conn,
    sport: str,
    handicap: float = 0,
    min_games: int = 5
) -> pd.DataFrame:
    """
    Generate ATS summary for all teams in a sport.

    Returns DataFrame with one row per team.
    """
    from ..database import get_all_teams, get_games

    teams = get_all_teams(conn, sport=sport)
    results = []

    for team in teams:
        games = get_games(conn, sport=sport, team=team)
        if len(games) < min_games:
            continue

        home_games = games[games['is_home'] == True]
        away_games = games[games['is_home'] == False]

        w, l, p = team_ats_record(games, handicap=handicap)

        results.append({
            'team': team,
            'games': len(games),
            'home_games': len(home_games),
            'away_games': len(away_games),
            'ats_wins': w,
            'ats_losses': l,
            'ats_pushes': p,
            'ats_pct': w / (w + l + p) if (w + l + p) > 0 else 0,
            'avg_margin': spread_margin_avg(games, perspective='home') if len(home_games) > len(away_games) else -spread_margin_avg(games, perspective='home'),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('ats_pct', ascending=False)

    return df


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

    from src.database import get_connection, get_games

    conn = get_connection()

    print("=" * 60)
    print("AGGREGATIONS TEST")
    print("=" * 60)

    # Test macro summary
    games = get_games(conn, sport='NBA')
    print(f"\nNBA Macro ATS Summary ({len(games)} games):")
    summary = macro_ats_summary(games)
    print(summary.to_string(index=False))

    # Test spread buckets
    print(f"\nNBA by Spread Bucket:")
    buckets = macro_by_spread_bucket(games)
    print(buckets[['bucket', 'games', 'home_pct']].to_string(index=False))

    # Test micro - all teams
    print(f"\nNBA Top 10 Teams by ATS %:")
    teams_df = micro_all_teams(conn, 'NBA', min_games=10)
    print(teams_df.head(10)[['team', 'games', 'ats_pct', 'ats_wins', 'ats_losses']].to_string(index=False))

    conn.close()
