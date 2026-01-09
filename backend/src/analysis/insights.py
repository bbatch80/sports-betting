"""
Dynamic insight engine for detecting betting patterns and opportunities.

This module scans historical data for statistically significant edges
and matches current team states to identify actionable betting opportunities.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import pandas as pd
import sqlite3

from ..database import get_games, get_all_teams
from .metrics import streak_continuation_analysis, baseline_handicap_coverage


@dataclass
class InsightPattern:
    """A detected betting edge with statistical backing."""
    sport: str                    # 'NBA', 'NFL', 'NCAAM'
    pattern_type: str             # 'streak_fade', 'streak_ride'
    streak_type: str              # 'WIN' or 'LOSS'
    streak_length: int            # e.g., 3
    handicap: int                 # e.g., 0 (standard spread)
    cover_rate: float             # After streak, cover rate at handicap
    baseline_rate: float          # League baseline at that handicap
    edge: float                   # cover_rate - baseline_rate
    sample_size: int              # Number of observations
    confidence: str               # 'high', 'medium', 'low'

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

    for sport in ['NFL', 'NBA', 'NCAAM']:
        # Get baseline cover rates for this sport
        baseline = baseline_handicap_coverage(conn, sport, handicap_range)
        if len(baseline) == 0:
            continue

        for streak_length in range(streak_range[0], streak_range[1] + 1):
            for streak_type in ['WIN', 'LOSS']:
                # Get handicap coverage after this streak
                data = streak_continuation_analysis(
                    conn, sport, streak_length, streak_type, handicap_range
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


def get_current_streaks(conn: sqlite3.Connection, sport: str) -> Dict[str, dict]:
    """
    Get each team's current ATS streak.

    Args:
        conn: Database connection
        sport: Sport to analyze

    Returns:
        Dictionary mapping team name to streak info:
        {team: {'streak_length': N, 'streak_type': 'WIN'|'LOSS'}}
    """
    teams = get_all_teams(conn, sport)
    streaks = {}

    for team in teams:
        games = get_games(conn, sport=sport, team=team)
        if len(games) == 0:
            continue

        # Sort by date descending (most recent first)
        games = games.sort_values('game_date', ascending=False)

        streak_length = 0
        streak_type = None

        for _, row in games.iterrows():
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
