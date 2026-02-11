"""
Dynamic insight engine for detecting betting patterns and opportunities.

This module scans historical data for statistically significant edges
and matches current team states to identify actionable betting opportunities.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
try:
    import pandas as pd
except ImportError:
    pd = None
import sqlite3

from ..database import get_games, get_all_teams
from .metrics import (
    streak_continuation_analysis,
    baseline_handicap_coverage,
    ou_streak_continuation_analysis,
    baseline_ou_coverage,
    tt_streak_continuation_analysis,
    tt_baseline_coverage,
)


@dataclass
class InsightPattern:
    """A detected betting edge with statistical backing."""
    sport: str                    # 'NBA', 'NFL', 'NCAAM'
    pattern_type: str             # 'streak_fade', 'streak_ride'
    streak_type: str              # 'WIN'/'LOSS' (ATS) or 'OVER'/'UNDER' (O/U, TT)
    streak_length: int            # e.g., 3
    handicap: int                 # e.g., 0 (standard spread)
    cover_rate: float             # After streak, cover rate at handicap
    baseline_rate: float          # League baseline at that handicap
    edge: float                   # cover_rate - baseline_rate
    sample_size: int              # Number of observations
    confidence: str               # 'high', 'medium', 'low'
    market_type: str = 'ats'      # 'ats', 'ou', or 'tt'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TodayOpportunity:
    """A team matching a profitable pattern right now."""
    team: str
    sport: str
    current_streak: int           # Current ATS streak length
    streak_type: str              # 'WIN' or 'LOSS'
    pattern: InsightPattern       # The pattern they match
    recommendation: str           # 'FADE' or 'RIDE'
    edge_pct: float               # Expected edge vs baseline

    def to_dict(self) -> dict:
        result = asdict(self)
        result['pattern'] = self.pattern.to_dict()
        return result


def get_confidence(sample_size: int, edge: float) -> str:
    """
    Determine confidence level based on sample size and edge magnitude.

    Args:
        sample_size: Number of observations (streak situations)
        edge: Edge vs baseline (can be positive or negative)

    Returns:
        'high', 'medium', or 'low'

    Thresholds:
        - High: 50+ samples AND 8%+ edge
        - Medium: 30+ samples AND 5%+ edge
        - Low: Everything else (use with caution)

    Note: Patterns with <30 samples should be treated cautiously.
    The minimum of 5 samples is a hard floor for any pattern to be considered.
    """
    abs_edge = abs(edge)

    # High confidence: Large sample AND significant edge
    if sample_size >= 50 and abs_edge >= 0.08:
        return 'high'
    # Medium confidence: Decent sample AND meaningful edge
    elif sample_size >= 30 and abs_edge >= 0.05:
        return 'medium'
    else:
        return 'low'


def detect_patterns(
    conn: sqlite3.Connection,
    min_sample: int = 30,
    min_edge: float = 0.05,
    streak_range: tuple = (2, 7),
    handicap_range: tuple = (0, 15)
) -> List[InsightPattern]:
    """
    Scan all sports/streak combinations for statistically significant edges.

    Criteria for a valid pattern:
    - Sample size >= min_sample (default 30)
    - Absolute edge vs baseline >= min_edge (default 5%)

    Args:
        conn: Database connection
        min_sample: Minimum observations required
        min_edge: Minimum edge vs baseline (as decimal, e.g., 0.05 = 5%)
        streak_range: (min, max) streak lengths to analyze
        handicap_range: (min, max) handicap levels to analyze

    Returns:
        List of InsightPattern objects, sorted by edge magnitude (strongest first)
    """
    patterns = []

    # Cache to store games per sport (avoids N+1 queries)
    games_cache = {}

    for sport in ['NFL', 'NBA', 'NCAAM']:
        # Get baseline cover rates for this sport
        baseline = baseline_handicap_coverage(conn, sport, handicap_range)
        if len(baseline) == 0:
            continue

        for streak_length in range(streak_range[0], streak_range[1] + 1):
            for streak_type in ['WIN', 'LOSS']:
                # Get handicap coverage after this streak
                data = streak_continuation_analysis(
                    conn, sport, streak_length, streak_type, handicap_range,
                    _all_games_cache=games_cache
                )

                if len(data) == 0:
                    continue

                # Check each handicap level for edges
                for _, row in data.iterrows():
                    handicap = row['handicap']

                    # Get baseline for this handicap
                    baseline_row = baseline[baseline['handicap'] == handicap]
                    if len(baseline_row) == 0:
                        continue

                    baseline_pct = baseline_row['baseline_cover_pct'].iloc[0]
                    edge = row['cover_pct'] - baseline_pct

                    # Check if meets criteria
                    if row['total'] >= min_sample and abs(edge) >= min_edge:
                        patterns.append(InsightPattern(
                            sport=sport,
                            pattern_type='streak_fade' if edge < 0 else 'streak_ride',
                            streak_type=streak_type,
                            streak_length=streak_length,
                            handicap=int(handicap),
                            cover_rate=row['cover_pct'],
                            baseline_rate=baseline_pct,
                            edge=edge,
                            sample_size=int(row['total']),
                            confidence=get_confidence(int(row['total']), edge)
                        ))

    # Sort by absolute edge (strongest patterns first)
    return sorted(patterns, key=lambda p: abs(p.edge), reverse=True)


# =============================================================================
# Over/Under (Totals) Streaks
# =============================================================================

def detect_ou_streak(team_games: pd.DataFrame) -> Tuple[int, str]:
    """
    Detect a team's current O/U streak.

    Args:
        team_games: DataFrame of team's games with total_result column

    Returns:
        Tuple of (streak_length, 'OVER'|'UNDER')
        Returns (0, 'OVER') if no games or no totals data
    """
    if len(team_games) == 0 or 'total_result' not in team_games.columns:
        return (0, 'OVER')

    # Filter games with totals data
    valid = team_games[team_games['total_result'].notna()].copy()
    if len(valid) == 0:
        return (0, 'OVER')

    # Sort by date descending (most recent first)
    valid = valid.sort_values('game_date', ascending=False)

    streak_length = 0
    streak_type = None

    for _, row in valid.iterrows():
        total_result = row['total_result']

        # Skip pushes (total_result == 0)
        if total_result == 0:
            continue

        is_over = total_result > 0

        if streak_type is None:
            # First non-push game establishes streak type
            streak_type = 'OVER' if is_over else 'UNDER'
            streak_length = 1
        elif (streak_type == 'OVER') == is_over:
            # Streak continues
            streak_length += 1
        else:
            # Streak broken
            break

    return (streak_length, streak_type or 'OVER')


def get_current_ou_streaks(conn: sqlite3.Connection, sport: str) -> Dict[str, dict]:
    """
    Get each team's current O/U streak (live computation).

    Args:
        conn: Database connection
        sport: Sport to analyze

    Returns:
        Dictionary mapping team name to streak info:
        {team: {'streak_length': N, 'streak_type': 'OVER'|'UNDER'}}
    """
    # Fetch all games for this sport ONCE
    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0 or 'total_result' not in all_games.columns:
        return {}

    # Filter games with totals data
    all_games = all_games[all_games['total_result'].notna()]
    if len(all_games) == 0:
        return {}

    # Get unique teams from the data
    teams = set(all_games['home_team']) | set(all_games['away_team'])
    streaks = {}

    for team in teams:
        # Filter games for this team in memory
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) == 0:
            continue

        streak_length, streak_type = detect_ou_streak(team_games)

        if streak_length > 0:
            streaks[team] = {
                'streak_length': streak_length,
                'streak_type': streak_type
            }

    return streaks


def get_cached_ou_streaks(conn: sqlite3.Connection, sport: str) -> Dict[str, dict]:
    """
    Get pre-computed O/U streaks from current_ou_streaks table.

    Falls back to live computation if the table doesn't exist or is empty.

    Args:
        conn: Database connection
        sport: Sport to analyze

    Returns:
        Dictionary mapping team name to streak info:
        {team: {'streak_length': N, 'streak_type': 'OVER'|'UNDER'}}
    """
    from sqlalchemy import text

    query = text('''
        SELECT team, streak_length, streak_type
        FROM current_ou_streaks
        WHERE sport = :sport
    ''')

    try:
        result = conn.execute(query, {'sport': sport})
        rows = result.fetchall()
    except Exception:
        # Table might not exist yet - fall back to live computation
        return get_current_ou_streaks(conn, sport)

    if not rows:
        # Table is empty - fall back to live computation
        return get_current_ou_streaks(conn, sport)

    streaks = {}
    for row in rows:
        streaks[row[0]] = {
            'streak_length': row[1] or 0,
            'streak_type': row[2] or 'OVER',
        }

    return streaks


# =============================================================================
# Team Totals (Individual O/U) Streaks
# =============================================================================

def get_current_tt_streaks(conn: sqlite3.Connection, sport: str) -> Dict[str, dict]:
    """
    Get each team's current team total O/U streak (live computation).

    Uses home_team_total_result / away_team_total_result to track individual
    team scoring vs their own O/U line.

    Args:
        conn: Database connection
        sport: Sport to analyze

    Returns:
        Dictionary mapping team name to streak info:
        {team: {'streak_length': N, 'streak_type': 'OVER'|'UNDER'}}
    """
    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0:
        return {}

    for col in ['home_team_total_result', 'away_team_total_result']:
        if col not in all_games.columns:
            return {}

    teams = set(all_games['home_team']) | set(all_games['away_team'])
    streaks = {}

    for team in teams:
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) == 0:
            continue

        # Sort most recent first
        team_games = team_games.sort_values('game_date', ascending=False)

        streak_length = 0
        streak_type = None

        for _, row in team_games.iterrows():
            is_home = row['home_team'] == team
            margin = row['home_team_total_result'] if is_home else row['away_team_total_result']

            # Skip missing or pushes
            if pd.isna(margin) or margin == 0:
                continue

            is_over = margin > 0

            if streak_type is None:
                streak_type = 'OVER' if is_over else 'UNDER'
                streak_length = 1
            elif (streak_type == 'OVER') == is_over:
                streak_length += 1
            else:
                break

        if streak_length > 0:
            streaks[team] = {
                'streak_length': streak_length,
                'streak_type': streak_type or 'OVER'
            }

    return streaks


# =============================================================================
# ATS (Spread) Streaks
# =============================================================================

def get_current_streaks(conn: sqlite3.Connection, sport: str) -> Dict[str, dict]:
    """
    Get each team's current ATS streak (live computation).

    Args:
        conn: Database connection
        sport: Sport to analyze

    Returns:
        Dictionary mapping team name to streak info:
        {team: {'streak_length': N, 'streak_type': 'WIN'|'LOSS'}}
    """
    # Fetch all games for this sport ONCE
    all_games = get_games(conn, sport=sport)
    if len(all_games) == 0:
        return {}

    # Get unique teams from the data
    teams = set(all_games['home_team']) | set(all_games['away_team'])
    streaks = {}

    for team in teams:
        # Filter games for this team in memory (no database query)
        team_games = all_games[
            (all_games['home_team'] == team) | (all_games['away_team'] == team)
        ].copy()

        if len(team_games) == 0:
            continue

        # Sort by date descending (most recent first)
        team_games = team_games.sort_values('game_date', ascending=False)

        # Add is_home column
        team_games['is_home'] = team_games['home_team'] == team

        streak_length = 0
        streak_type = None

        for _, row in team_games.iterrows():
            is_home = row['is_home']
            # Cover at 0 handicap
            covered = (row['spread_result'] > 0) if is_home else (row['spread_result'] < 0)

            if streak_type is None:
                # First game establishes the streak type
                streak_type = 'WIN' if covered else 'LOSS'
                streak_length = 1
            elif (streak_type == 'WIN') == covered:
                # Streak continues
                streak_length += 1
            else:
                # Streak broken
                break

        if streak_type is not None:
            streaks[team] = {
                'streak_length': streak_length,
                'streak_type': streak_type
            }

    return streaks


def get_cached_streaks(conn: sqlite3.Connection, sport: str) -> Dict[str, dict]:
    """
    Get pre-computed streaks from current_streaks table.

    Falls back to live computation if the table is empty.
    This provides ~10x faster dashboard loads (from 300ms-1s to <50ms).

    Args:
        conn: Database connection
        sport: Sport to analyze

    Returns:
        Dictionary mapping team name to streak info:
        {team: {'streak_length': N, 'streak_type': 'WIN'|'LOSS'}}
    """
    from sqlalchemy import text

    query = text('''
        SELECT team, streak_length, streak_type
        FROM current_streaks
        WHERE sport = :sport
    ''')

    try:
        result = conn.execute(query, {'sport': sport})
        rows = result.fetchall()
    except Exception:
        # Table might not exist yet - fall back to live computation
        return get_current_streaks(conn, sport)

    if not rows:
        # Table is empty - fall back to live computation
        return get_current_streaks(conn, sport)

    streaks = {}
    for row in rows:
        streaks[row[0]] = {
            'streak_length': row[1] or 0,
            'streak_type': row[2] or 'WIN',
        }

    return streaks


def get_cached_patterns(
    conn: sqlite3.Connection,
    min_sample: int = 30,
    min_edge: float = 0.05
) -> List[InsightPattern]:
    """
    Get pre-computed patterns from detected_patterns table.

    Falls back to live computation if the table is empty.
    This provides ~100x faster dashboard loads for Today's Picks.

    Args:
        conn: Database connection
        min_sample: Minimum sample size filter
        min_edge: Minimum edge filter

    Returns:
        List of InsightPattern objects, sorted by edge magnitude
    """
    from sqlalchemy import text

    query = text('''
        SELECT sport, pattern_type, streak_type, streak_length, handicap,
               cover_rate, baseline_rate, edge, sample_size, confidence
        FROM detected_patterns
        WHERE (market_type = 'ats' OR market_type IS NULL)
          AND sample_size >= :min_sample AND ABS(edge) >= :min_edge
        ORDER BY ABS(edge) DESC
    ''')

    try:
        result = conn.execute(query, {'min_sample': min_sample, 'min_edge': min_edge})
        rows = result.fetchall()
    except Exception:
        # Table might not exist yet - fall back to live computation
        return detect_patterns(conn, min_sample=min_sample, min_edge=min_edge)

    if not rows:
        # Table is empty - fall back to live computation
        return detect_patterns(conn, min_sample=min_sample, min_edge=min_edge)

    patterns = []
    for row in rows:
        patterns.append(InsightPattern(
            sport=row[0],
            pattern_type=row[1],
            streak_type=row[2],
            streak_length=row[3],
            handicap=row[4],
            cover_rate=row[5],
            baseline_rate=row[6],
            edge=row[7],
            sample_size=row[8],
            confidence=row[9],
        ))

    return patterns


def detect_ou_patterns(
    conn: sqlite3.Connection,
    min_sample: int = 30,
    min_edge: float = 0.05,
    streak_range: tuple = (2, 7),
    handicap_range: tuple = (0, 20)
) -> List[InsightPattern]:
    """
    Scan all sports/streak combinations for significant O/U (game total) edges.

    After X consecutive OVERs or UNDERs, does the next game go OVER at a higher
    rate than the league baseline? Same structure as detect_patterns() but using
    O/U analysis functions.

    Args:
        conn: Database connection
        min_sample: Minimum observations required
        min_edge: Minimum edge vs baseline (as decimal, e.g., 0.05 = 5%)
        streak_range: (min, max) streak lengths to analyze
        handicap_range: (min, max) handicap levels to analyze

    Returns:
        List of InsightPattern objects with market_type='ou', sorted by edge magnitude
    """
    patterns = []
    games_cache = {}

    for sport in ['NFL', 'NBA', 'NCAAM']:
        # Get baseline O/U cover rates (OVER direction)
        baseline = baseline_ou_coverage(conn, sport, handicap_range, direction='over')
        if len(baseline) == 0:
            continue

        for streak_length in range(streak_range[0], streak_range[1] + 1):
            for streak_type in ['OVER', 'UNDER']:
                # Get O/U coverage after this streak (OVER direction)
                data = ou_streak_continuation_analysis(
                    conn, sport, streak_length, streak_type, handicap_range,
                    direction='over', _all_games_cache=games_cache
                )

                if len(data) == 0:
                    continue

                for _, row in data.iterrows():
                    handicap = row['handicap']

                    baseline_row = baseline[baseline['handicap'] == handicap]
                    if len(baseline_row) == 0:
                        continue

                    baseline_pct = baseline_row['baseline_cover_pct'].iloc[0]
                    edge = row['cover_pct'] - baseline_pct

                    if row['total'] >= min_sample and abs(edge) >= min_edge:
                        patterns.append(InsightPattern(
                            sport=sport,
                            pattern_type='streak_ride' if edge > 0 else 'streak_fade',
                            streak_type=streak_type,
                            streak_length=streak_length,
                            handicap=int(handicap),
                            cover_rate=row['cover_pct'],
                            baseline_rate=baseline_pct,
                            edge=edge,
                            sample_size=int(row['total']),
                            confidence=get_confidence(int(row['total']), edge),
                            market_type='ou',
                        ))

    return sorted(patterns, key=lambda p: abs(p.edge), reverse=True)


def detect_tt_patterns(
    conn: sqlite3.Connection,
    min_sample: int = 30,
    min_edge: float = 0.05,
    streak_range: tuple = (2, 7),
    handicap_range: tuple = (0, 10)
) -> List[InsightPattern]:
    """
    Scan all sports/streak combinations for significant team total edges.

    After X consecutive TT OVERs or UNDERs for a team, does the team's next
    total go OVER at a higher rate than the league baseline?

    Args:
        conn: Database connection
        min_sample: Minimum observations required
        min_edge: Minimum edge vs baseline (as decimal, e.g., 0.05 = 5%)
        streak_range: (min, max) streak lengths to analyze
        handicap_range: (min, max) handicap levels to analyze

    Returns:
        List of InsightPattern objects with market_type='tt', sorted by edge magnitude
    """
    patterns = []

    for sport in ['NFL', 'NBA', 'NCAAM']:
        # Get baseline TT cover rates (OVER direction)
        baseline = tt_baseline_coverage(conn, sport, handicap_range, direction='over')
        if len(baseline) == 0:
            continue

        for streak_length in range(streak_range[0], streak_range[1] + 1):
            for streak_type in ['OVER', 'UNDER']:
                # Get TT coverage after this streak (ride direction)
                data = tt_streak_continuation_analysis(
                    conn, sport, streak_length, streak_type, handicap_range,
                    direction='ride'
                )

                if len(data) == 0:
                    continue

                for _, row in data.iterrows():
                    handicap = row['handicap']

                    baseline_row = baseline[baseline['handicap'] == handicap]
                    if len(baseline_row) == 0:
                        continue

                    baseline_pct = baseline_row['baseline_cover_pct'].iloc[0]
                    edge = row['cover_pct'] - baseline_pct

                    if row['total'] >= min_sample and abs(edge) >= min_edge:
                        patterns.append(InsightPattern(
                            sport=sport,
                            pattern_type='streak_ride' if edge > 0 else 'streak_fade',
                            streak_type=streak_type,
                            streak_length=streak_length,
                            handicap=int(handicap),
                            cover_rate=row['cover_pct'],
                            baseline_rate=baseline_pct,
                            edge=edge,
                            sample_size=int(row['total']),
                            confidence=get_confidence(int(row['total']), edge),
                            market_type='tt',
                        ))

    return sorted(patterns, key=lambda p: abs(p.edge), reverse=True)


def get_cached_ou_patterns(
    conn: sqlite3.Connection,
    min_sample: int = 30,
    min_edge: float = 0.05
) -> List[InsightPattern]:
    """
    Get pre-computed O/U patterns from detected_patterns table.

    Falls back to live computation if no O/U patterns are cached.

    Args:
        conn: Database connection
        min_sample: Minimum sample size filter
        min_edge: Minimum edge filter

    Returns:
        List of InsightPattern objects with market_type='ou'
    """
    from sqlalchemy import text

    query = text('''
        SELECT sport, pattern_type, streak_type, streak_length, handicap,
               cover_rate, baseline_rate, edge, sample_size, confidence
        FROM detected_patterns
        WHERE market_type = 'ou'
          AND sample_size >= :min_sample AND ABS(edge) >= :min_edge
        ORDER BY ABS(edge) DESC
    ''')

    try:
        result = conn.execute(query, {'min_sample': min_sample, 'min_edge': min_edge})
        rows = result.fetchall()
    except Exception:
        return detect_ou_patterns(conn, min_sample=min_sample, min_edge=min_edge)

    if not rows:
        return detect_ou_patterns(conn, min_sample=min_sample, min_edge=min_edge)

    patterns = []
    for row in rows:
        patterns.append(InsightPattern(
            sport=row[0],
            pattern_type=row[1],
            streak_type=row[2],
            streak_length=row[3],
            handicap=row[4],
            cover_rate=row[5],
            baseline_rate=row[6],
            edge=row[7],
            sample_size=row[8],
            confidence=row[9],
            market_type='ou',
        ))

    return patterns


def get_cached_tt_patterns(
    conn: sqlite3.Connection,
    min_sample: int = 30,
    min_edge: float = 0.05
) -> List[InsightPattern]:
    """
    Get pre-computed TT patterns from detected_patterns table.

    Falls back to live computation if no TT patterns are cached.

    Args:
        conn: Database connection
        min_sample: Minimum sample size filter
        min_edge: Minimum edge filter

    Returns:
        List of InsightPattern objects with market_type='tt'
    """
    from sqlalchemy import text

    query = text('''
        SELECT sport, pattern_type, streak_type, streak_length, handicap,
               cover_rate, baseline_rate, edge, sample_size, confidence
        FROM detected_patterns
        WHERE market_type = 'tt'
          AND sample_size >= :min_sample AND ABS(edge) >= :min_edge
        ORDER BY ABS(edge) DESC
    ''')

    try:
        result = conn.execute(query, {'min_sample': min_sample, 'min_edge': min_edge})
        rows = result.fetchall()
    except Exception:
        return detect_tt_patterns(conn, min_sample=min_sample, min_edge=min_edge)

    if not rows:
        return detect_tt_patterns(conn, min_sample=min_sample, min_edge=min_edge)

    patterns = []
    for row in rows:
        patterns.append(InsightPattern(
            sport=row[0],
            pattern_type=row[1],
            streak_type=row[2],
            streak_length=row[3],
            handicap=row[4],
            cover_rate=row[5],
            baseline_rate=row[6],
            edge=row[7],
            sample_size=row[8],
            confidence=row[9],
            market_type='tt',
        ))

    return patterns


def get_cached_tt_streaks(conn: sqlite3.Connection, sport: str) -> Dict[str, dict]:
    """
    Get pre-computed TT streaks from current_tt_streaks table.

    Falls back to live computation if the table doesn't exist or is empty.

    Args:
        conn: Database connection
        sport: Sport to analyze

    Returns:
        Dictionary mapping team name to streak info:
        {team: {'streak_length': N, 'streak_type': 'OVER'|'UNDER'}}
    """
    from sqlalchemy import text

    query = text('''
        SELECT team, streak_length, streak_type
        FROM current_tt_streaks
        WHERE sport = :sport
    ''')

    try:
        result = conn.execute(query, {'sport': sport})
        rows = result.fetchall()
    except Exception:
        return get_current_tt_streaks(conn, sport)

    if not rows:
        return get_current_tt_streaks(conn, sport)

    streaks = {}
    for row in rows:
        streaks[row[0]] = {
            'streak_length': row[1] or 0,
            'streak_type': row[2] or 'OVER',
        }

    return streaks


def find_opportunities(
    conn: sqlite3.Connection,
    patterns: List[InsightPattern],
    min_confidence: str = 'low'
) -> List[TodayOpportunity]:
    """
    Match current team states to profitable patterns.

    Args:
        conn: Database connection
        patterns: List of detected patterns
        min_confidence: Minimum confidence level to include ('low', 'medium', 'high')

    Returns:
        List of TodayOpportunity objects, sorted by edge strength
    """
    confidence_order = {'low': 0, 'medium': 1, 'high': 2}
    min_conf_value = confidence_order.get(min_confidence, 0)

    opportunities = []
    seen = set()  # Avoid duplicates (team, sport, recommendation)

    for sport in ['NFL', 'NBA', 'NCAAM']:
        # Filter patterns for this sport
        sport_patterns = [
            p for p in patterns
            if p.sport == sport and confidence_order.get(p.confidence, 0) >= min_conf_value
        ]

        if not sport_patterns:
            continue

        current_streaks = get_current_streaks(conn, sport)

        for team, streak_info in current_streaks.items():
            for pattern in sport_patterns:
                # Check if team matches the pattern
                if (streak_info['streak_type'] == pattern.streak_type and
                    streak_info['streak_length'] >= pattern.streak_length):

                    # Determine recommendation
                    # FADE = bet against the team (their cover rate is BELOW baseline)
                    # RIDE = bet on the team (their cover rate is ABOVE baseline)
                    recommendation = 'FADE' if pattern.edge < 0 else 'RIDE'

                    # Avoid duplicates for same team/sport/recommendation
                    key = (team, sport, recommendation)
                    if key in seen:
                        continue
                    seen.add(key)

                    opportunities.append(TodayOpportunity(
                        team=team,
                        sport=sport,
                        current_streak=streak_info['streak_length'],
                        streak_type=streak_info['streak_type'],
                        pattern=pattern,
                        recommendation=recommendation,
                        edge_pct=abs(pattern.edge)
                    ))

    # Sort by edge strength (highest edge first)
    return sorted(opportunities, key=lambda o: o.edge_pct, reverse=True)


def get_pattern_summary(patterns: List[InsightPattern]) -> pd.DataFrame:
    """
    Create a summary DataFrame of detected patterns.

    Args:
        patterns: List of InsightPattern objects

    Returns:
        DataFrame with pattern details
    """
    if not patterns:
        return pd.DataFrame()

    data = []
    for p in patterns:
        data.append({
            'Sport': p.sport,
            'Type': p.pattern_type.replace('_', ' ').title(),
            'After': f"{p.streak_length}+ {p.streak_type.lower()} streak",
            'Handicap': f"+{p.handicap}",
            'Cover Rate': f"{p.cover_rate:.1%}",
            'Baseline': f"{p.baseline_rate:.1%}",
            'Edge': f"{p.edge:+.1%}",
            'Sample': p.sample_size,
            'Confidence': p.confidence.title()
        })

    return pd.DataFrame(data)


def get_opportunity_summary(opportunities: List[TodayOpportunity]) -> pd.DataFrame:
    """
    Create a summary DataFrame of current opportunities.

    Args:
        opportunities: List of TodayOpportunity objects

    Returns:
        DataFrame with opportunity details
    """
    if not opportunities:
        return pd.DataFrame()

    data = []
    for o in opportunities:
        data.append({
            'Team': o.team,
            'Sport': o.sport,
            'Recommendation': o.recommendation,
            'Current Streak': f"{o.current_streak} {o.streak_type.lower()}s",
            'Pattern': f"{o.pattern.streak_length}+ {o.pattern.streak_type.lower()} → {o.pattern.pattern_type.replace('_', ' ')}",
            'Edge': f"{o.edge_pct:.1%}",
            'Sample': o.pattern.sample_size,
            'Confidence': o.pattern.confidence.title()
        })

    return pd.DataFrame(data)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    from ..database import get_connection

    print("=" * 70)
    print("INSIGHT ENGINE TEST")
    print("=" * 70)

    conn = get_connection()

    # Detect patterns
    print("\nDetecting patterns...")
    patterns = detect_patterns(conn, min_sample=30, min_edge=0.05)
    print(f"Found {len(patterns)} significant patterns")

    if patterns:
        print("\nTop 10 patterns by edge:")
        summary = get_pattern_summary(patterns[:10])
        print(summary.to_string(index=False))

    # Find opportunities
    print("\n" + "-" * 70)
    print("Current opportunities...")
    opportunities = find_opportunities(conn, patterns)
    print(f"Found {len(opportunities)} opportunities")

    if opportunities:
        print("\nTop 10 opportunities:")
        opp_summary = get_opportunity_summary(opportunities[:10])
        print(opp_summary.to_string(index=False))

    conn.close()
