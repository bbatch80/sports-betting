"""
Tier Matchup Pattern Detection & Opportunity Engine.

This module identifies statistically significant patterns in tier matchup + handicap
combinations and surfaces them as betting opportunities/strategies.

Example patterns detected:
- NBA: Top (.7+) vs Top (.7+) with 11pt handicap → higher-rated covers 87%
- NBA: Top (.7+) vs Bottom (<.3) with 11pt handicap → lower-rated covers 85%

Tier Definitions (based on ATS Rating):
- TOP: ≥ 0.70
- HIGH: 0.55 - 0.70
- MID: 0.45 - 0.55
- LOW: 0.30 - 0.45
- BOTTOM: < 0.30
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import numpy as np
except ImportError:
    np = None
from sqlalchemy.engine import Connection

from ..database import get_games
from .backtest_ratings import get_games_with_ratings, BREAKEVEN_WIN_RATE, calculate_roi
from .network_ratings import get_team_rankings


# =============================================================================
# Constants
# =============================================================================

# Tier boundaries based on ATS rating
TIER_BOUNDARIES = {
    'TOP': (0.70, 1.0),
    'HIGH': (0.55, 0.70),
    'MID': (0.45, 0.55),
    'LOW': (0.30, 0.45),
    'BOTTOM': (0.0, 0.30),
}

# Ordered list for iteration
TIERS = ['TOP', 'HIGH', 'MID', 'LOW', 'BOTTOM']


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TierMatchupPattern:
    """A detected betting edge based on tier matchup + handicap."""
    sport: str                  # 'NBA', 'NFL', 'NCAAM'
    higher_tier: str            # 'TOP', 'HIGH', 'MID', 'LOW', 'BOTTOM'
    lower_tier: str             # 'TOP', 'HIGH', 'MID', 'LOW', 'BOTTOM'
    handicap: int               # 0-15 points
    bet_on: str                 # 'HIGHER' or 'LOWER' rated team
    cover_rate: float           # Actual cover rate for bet_on team
    baseline_rate: float        # League baseline at handicap
    edge: float                 # cover_rate - baseline_rate
    sample_size: int            # Number of games
    wins: int
    losses: int
    roi: float                  # At -110 odds
    confidence: str             # 'high', 'medium', 'low'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TierMatchupOpportunity:
    """An upcoming game matching a profitable pattern."""
    home_team: str
    away_team: str
    sport: str
    higher_rated_team: str
    lower_rated_team: str
    higher_rating: float
    lower_rating: float
    higher_tier: str
    lower_tier: str
    pattern: TierMatchupPattern
    recommendation: str         # 'BET ON <team>'
    bet_team: str               # The team to bet on
    edge_pct: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result['pattern'] = self.pattern.to_dict()
        return result


# =============================================================================
# Helper Functions
# =============================================================================

def get_tier(rating: float) -> str:
    """
    Map an ATS rating to its tier.

    Args:
        rating: ATS rating between 0 and 1

    Returns:
        Tier name: 'TOP', 'HIGH', 'MID', 'LOW', or 'BOTTOM'
    """
    if rating >= 0.70:
        return 'TOP'
    elif rating >= 0.55:
        return 'HIGH'
    elif rating >= 0.45:
        return 'MID'
    elif rating >= 0.30:
        return 'LOW'
    else:
        return 'BOTTOM'


def get_confidence(sample_size: int, edge: float, roi: float) -> str:
    """
    Determine confidence level based on sample size, edge, and ROI.

    Args:
        sample_size: Number of games in the pattern
        edge: Edge vs baseline (as decimal)
        roi: Return on investment

    Returns:
        'high', 'medium', or 'low'

    Thresholds:
        High:   ≥50 samples AND ≥8% edge AND positive ROI
        Medium: ≥30 samples AND ≥5% edge
        Low:    Everything else meeting min thresholds
    """
    abs_edge = abs(edge)

    if sample_size >= 50 and abs_edge >= 0.08 and roi > 0:
        return 'high'
    elif sample_size >= 30 and abs_edge >= 0.05:
        return 'medium'
    return 'low'


def baseline_handicap_coverage(conn: Connection, sport: str, handicap_range: tuple = (0, 15)) -> pd.DataFrame:
    """
    Calculate baseline handicap coverage rates for a sport.

    This represents the "expected" cover rate at each handicap level
    for all teams in the league, used as a baseline for comparison.

    Args:
        conn: Database connection
        sport: Sport to analyze
        handicap_range: (min, max) handicap levels

    Returns:
        DataFrame with columns: handicap, total, covers, cover_pct
    """
    games = get_games(conn, sport=sport)
    if len(games) == 0:
        return pd.DataFrame()

    # Exclude pushes
    games = games[games['spread_result'] != 0].copy()

    results = []
    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        # Home covers with +handicap: spread_result > -handicap
        # Away covers with +handicap: spread_result < handicap
        # For baseline, we look at what % of teams cover at each handicap

        # Home team covers at this handicap
        home_covers = (games['spread_result'] > -handicap).sum()
        # Away team covers at this handicap
        away_covers = (games['spread_result'] < handicap).sum()

        total_situations = len(games) * 2  # Each game has home and away perspective
        total_covers = home_covers + away_covers

        results.append({
            'handicap': handicap,
            'total': total_situations,
            'covers': total_covers,
            'cover_pct': total_covers / total_situations if total_situations > 0 else 0.5
        })

    return pd.DataFrame(results)


# =============================================================================
# Core Pattern Detection
# =============================================================================

def detect_tier_matchup_patterns(
    conn: Connection,
    sport: str,
    min_sample: int = 30,
    min_edge: float = 0.05,
    handicap_range: tuple = (0, 15)
) -> List[TierMatchupPattern]:
    """
    Scan all tier matchup + handicap combinations for significant edges.

    For each tier combination (e.g., TOP vs BOTTOM) and each handicap (0-15),
    calculate the cover rate for higher-rated and lower-rated teams,
    compare to baseline, and identify statistically significant edges.

    Args:
        conn: Database connection
        sport: Sport to analyze ('NFL', 'NBA', 'NCAAM')
        min_sample: Minimum games required for a pattern
        min_edge: Minimum edge vs baseline (as decimal, e.g., 0.05 = 5%)
        handicap_range: (min, max) handicap levels to analyze

    Returns:
        List of TierMatchupPattern sorted by |edge| descending
    """
    # Get games with historical ratings
    games_df = get_games_with_ratings(conn, sport)

    if len(games_df) == 0:
        return []

    # Add tier columns based on ATS ratings
    games_df['home_tier'] = games_df['home_ats_rating'].apply(get_tier)
    games_df['away_tier'] = games_df['away_ats_rating'].apply(get_tier)

    # Determine higher and lower rated teams
    games_df['higher_is_home'] = games_df['home_ats_rating'] > games_df['away_ats_rating']
    games_df['higher_tier'] = games_df.apply(
        lambda r: r['home_tier'] if r['higher_is_home'] else r['away_tier'],
        axis=1
    )
    games_df['lower_tier'] = games_df.apply(
        lambda r: r['away_tier'] if r['higher_is_home'] else r['home_tier'],
        axis=1
    )
    games_df['higher_rating'] = games_df.apply(
        lambda r: r['home_ats_rating'] if r['higher_is_home'] else r['away_ats_rating'],
        axis=1
    )
    games_df['lower_rating'] = games_df.apply(
        lambda r: r['away_ats_rating'] if r['higher_is_home'] else r['home_ats_rating'],
        axis=1
    )

    # Get baseline cover rates
    baseline = baseline_handicap_coverage(conn, sport, handicap_range)
    if len(baseline) == 0:
        return []

    patterns = []

    # Iterate through all tier combinations
    for higher_tier in TIERS:
        for lower_tier in TIERS:
            # Filter to this tier matchup
            matchup = games_df[
                (games_df['higher_tier'] == higher_tier) &
                (games_df['lower_tier'] == lower_tier)
            ].copy()

            if len(matchup) < min_sample:
                continue

            # Exclude pushes for analysis
            matchup = matchup[matchup['spread_result'] != 0]

            if len(matchup) < min_sample:
                continue

            # Analyze each handicap level
            for handicap in range(handicap_range[0], handicap_range[1] + 1):
                # Calculate higher-rated team covers at this handicap
                # If higher is home: home covers with +handicap → spread_result > -handicap
                # If higher is away: away covers with +handicap → spread_result < handicap
                higher_covers = (
                    (matchup['higher_is_home'] & (matchup['spread_result'] > -handicap)) |
                    (~matchup['higher_is_home'] & (matchup['spread_result'] < handicap))
                ).sum()

                total = len(matchup)
                higher_rate = higher_covers / total if total > 0 else 0.5
                lower_rate = 1 - higher_rate  # Lower-rated team cover rate

                # Get baseline for this handicap
                baseline_row = baseline[baseline['handicap'] == handicap]
                if len(baseline_row) == 0:
                    continue
                baseline_rate = baseline_row['cover_pct'].iloc[0]

                # Check edge for higher-rated team
                higher_edge = higher_rate - baseline_rate

                if abs(higher_edge) >= min_edge:
                    # Determine which team to bet on
                    if higher_edge > 0:
                        # Bet on higher-rated team
                        bet_on = 'HIGHER'
                        cover_rate = higher_rate
                        wins = higher_covers
                        losses = total - higher_covers
                    else:
                        # Bet on lower-rated team
                        bet_on = 'LOWER'
                        cover_rate = lower_rate
                        wins = total - higher_covers
                        losses = higher_covers

                    roi = calculate_roi(wins, losses)
                    edge = abs(higher_edge)

                    patterns.append(TierMatchupPattern(
                        sport=sport,
                        higher_tier=higher_tier,
                        lower_tier=lower_tier,
                        handicap=handicap,
                        bet_on=bet_on,
                        cover_rate=cover_rate,
                        baseline_rate=baseline_rate,
                        edge=edge,
                        sample_size=total,
                        wins=wins,
                        losses=losses,
                        roi=roi,
                        confidence=get_confidence(total, edge, roi)
                    ))

    # Sort by edge descending
    return sorted(patterns, key=lambda p: p.edge, reverse=True)


def detect_all_tier_matchup_patterns(
    conn: Connection,
    min_sample: int = 30,
    min_edge: float = 0.05
) -> List[TierMatchupPattern]:
    """
    Detect tier matchup patterns across all sports.

    Args:
        conn: Database connection
        min_sample: Minimum games required for a pattern
        min_edge: Minimum edge vs baseline

    Returns:
        List of all patterns sorted by edge descending
    """
    all_patterns = []

    for sport in ['NFL', 'NBA', 'NCAAM']:
        patterns = detect_tier_matchup_patterns(conn, sport, min_sample, min_edge)
        all_patterns.extend(patterns)

    return sorted(all_patterns, key=lambda p: p.edge, reverse=True)


# =============================================================================
# Opportunity Matching
# =============================================================================

def find_tier_matchup_opportunities(
    conn: Connection,
    patterns: List[TierMatchupPattern],
    min_confidence: str = 'low'
) -> List[TierMatchupOpportunity]:
    """
    Match current team ratings to profitable patterns.

    For each pattern, find teams currently in the matching tier combinations.
    When these teams play each other, we have an opportunity.

    Args:
        conn: Database connection
        patterns: List of detected patterns
        min_confidence: Minimum confidence level ('low', 'medium', 'high')

    Returns:
        List of TierMatchupOpportunity sorted by edge descending
    """
    confidence_order = {'low': 0, 'medium': 1, 'high': 2}
    min_conf_value = confidence_order.get(min_confidence, 0)

    opportunities = []
    seen = set()  # Avoid duplicates

    for sport in ['NFL', 'NBA', 'NCAAM']:
        # Get current team rankings
        rankings = get_team_rankings(conn, sport)

        if not rankings:
            continue

        # Build tier lookup: team -> (tier, ats_rating)
        team_info = {
            r.team: {
                'tier': get_tier(r.ats_rating),
                'ats_rating': r.ats_rating
            }
            for r in rankings
        }

        # Filter patterns for this sport with sufficient confidence
        sport_patterns = [
            p for p in patterns
            if p.sport == sport and confidence_order.get(p.confidence, 0) >= min_conf_value
        ]

        if not sport_patterns:
            continue

        # For each pattern, find matching team pairs
        for pattern in sport_patterns:
            # Find all teams in the higher tier
            higher_tier_teams = [
                (team, info) for team, info in team_info.items()
                if info['tier'] == pattern.higher_tier
            ]

            # Find all teams in the lower tier
            lower_tier_teams = [
                (team, info) for team, info in team_info.items()
                if info['tier'] == pattern.lower_tier
            ]

            # Create opportunities for all valid matchups
            for h_team, h_info in higher_tier_teams:
                for l_team, l_info in lower_tier_teams:
                    if h_team == l_team:
                        continue

                    # Avoid duplicate matchups
                    matchup_key = (sport, tuple(sorted([h_team, l_team])), pattern.handicap)
                    if matchup_key in seen:
                        continue
                    seen.add(matchup_key)

                    # Determine bet team
                    bet_team = h_team if pattern.bet_on == 'HIGHER' else l_team

                    opportunities.append(TierMatchupOpportunity(
                        home_team='TBD',  # We don't know who's home yet
                        away_team='TBD',
                        sport=sport,
                        higher_rated_team=h_team,
                        lower_rated_team=l_team,
                        higher_rating=h_info['ats_rating'],
                        lower_rating=l_info['ats_rating'],
                        higher_tier=pattern.higher_tier,
                        lower_tier=pattern.lower_tier,
                        pattern=pattern,
                        recommendation=f"BET ON {bet_team} +{pattern.handicap}",
                        bet_team=bet_team,
                        edge_pct=pattern.edge
                    ))

    # Sort by edge descending
    return sorted(opportunities, key=lambda o: o.edge_pct, reverse=True)


# =============================================================================
# Summary Functions for Display
# =============================================================================

def get_tier_pattern_summary(patterns: List[TierMatchupPattern]) -> pd.DataFrame:
    """
    Create a summary DataFrame of detected tier matchup patterns.

    Args:
        patterns: List of TierMatchupPattern objects

    Returns:
        DataFrame with pattern details
    """
    if not patterns:
        return pd.DataFrame()

    data = []
    for p in patterns:
        data.append({
            'Sport': p.sport,
            'Matchup': f"{p.higher_tier} vs {p.lower_tier}",
            'Handicap': f"+{p.handicap}",
            'Bet On': p.bet_on,
            'Cover Rate': f"{p.cover_rate:.1%}",
            'Baseline': f"{p.baseline_rate:.1%}",
            'Edge': f"{p.edge:+.1%}",
            'Sample': p.sample_size,
            'W-L': f"{p.wins}-{p.losses}",
            'ROI': f"{p.roi:+.1%}",
            'Confidence': p.confidence.title()
        })

    return pd.DataFrame(data)


def get_tier_opportunity_summary(opportunities: List[TierMatchupOpportunity]) -> pd.DataFrame:
    """
    Create a summary DataFrame of tier matchup opportunities.

    Args:
        opportunities: List of TierMatchupOpportunity objects

    Returns:
        DataFrame with opportunity details
    """
    if not opportunities:
        return pd.DataFrame()

    data = []
    for o in opportunities:
        data.append({
            'Sport': o.sport,
            'Higher Team': f"{o.higher_rated_team} ({o.higher_tier})",
            'Lower Team': f"{o.lower_rated_team} ({o.lower_tier})",
            'Recommendation': o.recommendation,
            'Bet On': o.bet_team,
            'Edge': f"{o.edge_pct:.1%}",
            'Handicap': f"+{o.pattern.handicap}",
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
    print("TIER MATCHUP PATTERN DETECTION TEST")
    print("=" * 70)

    conn = get_connection()

    for sport in ['NFL', 'NBA', 'NCAAM']:
        print(f"\n{'='*70}")
        print(f"{sport} TIER MATCHUP PATTERNS")
        print("=" * 70)

        patterns = detect_tier_matchup_patterns(conn, sport, min_sample=30, min_edge=0.05)
        print(f"Found {len(patterns)} patterns with ≥5% edge and ≥30 samples")

        if patterns:
            print(f"\nTop 10 patterns by edge:")
            print("-" * 70)

            summary = get_tier_pattern_summary(patterns[:10])
            print(summary.to_string(index=False))

            # High confidence only
            high_conf = [p for p in patterns if p.confidence == 'high']
            print(f"\nHigh confidence patterns: {len(high_conf)}")

    print("\n" + "=" * 70)
    print("CURRENT OPPORTUNITIES")
    print("=" * 70)

    all_patterns = detect_all_tier_matchup_patterns(conn, min_sample=30, min_edge=0.05)
    opportunities = find_tier_matchup_opportunities(conn, all_patterns, min_confidence='medium')

    print(f"Found {len(opportunities)} potential matchup opportunities (medium+ confidence)")

    if opportunities:
        print("\nTop 10 opportunities:")
        print("-" * 70)
        opp_summary = get_tier_opportunity_summary(opportunities[:10])
        print(opp_summary.to_string(index=False))

    conn.close()
