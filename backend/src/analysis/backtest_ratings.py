"""
Backtesting module for rating-based betting strategies.

Analyzes historical matchups to find profitable strategies based on
pre-game rating differentials (market gap, ATS rank, win rank).

Key insight: 52.4% win rate is breakeven at -110 odds.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import date
import pandas as pd
import sqlite3

from sqlalchemy import text

from ..database import get_games


# =============================================================================
# Constants
# =============================================================================

# Breakeven win rate at -110 odds
BREAKEVEN_WIN_RATE = 0.524

# Standard payout at -110 odds (bet $110 to win $100)
WIN_PAYOUT = 100 / 110  # 0.909...


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a backtest strategy."""
    strategy_name: str
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi: float              # Return on investment
    edge: float             # win_rate - breakeven
    is_profitable: bool     # roi > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThresholdResult:
    """Results at a specific threshold."""
    threshold: float
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    roi: float
    edge: float


# =============================================================================
# Core Data Functions
# =============================================================================

def get_games_with_ratings(conn: sqlite3.Connection, sport: str) -> pd.DataFrame:
    """
    Join each game with both teams' pre-game ratings.

    Returns DataFrame with columns:
    - game_date, home_team, away_team, closing_spread, spread_result
    - home_win_rating, home_ats_rating, home_market_gap, home_ats_rank, home_win_rank
    - away_win_rating, away_ats_rating, away_market_gap, away_ats_rank, away_win_rank
    - home_covered (bool), gap_diff (home_gap - away_gap)
    """
    # Get all games
    games_df = get_games(conn, sport=sport)
    if len(games_df) == 0:
        return pd.DataFrame()

    # Join with historical ratings for home team
    query = """
        SELECT
            g.game_date,
            g.home_team,
            g.away_team,
            g.closing_spread,
            g.home_score,
            g.away_score,
            g.spread_result,
            hr_home.win_rating AS home_win_rating,
            hr_home.ats_rating AS home_ats_rating,
            hr_home.market_gap AS home_market_gap,
            hr_home.win_rank AS home_win_rank,
            hr_home.ats_rank AS home_ats_rank,
            hr_home.games_analyzed AS home_games,
            hr_away.win_rating AS away_win_rating,
            hr_away.ats_rating AS away_ats_rating,
            hr_away.market_gap AS away_market_gap,
            hr_away.win_rank AS away_win_rank,
            hr_away.ats_rank AS away_ats_rank,
            hr_away.games_analyzed AS away_games
        FROM games g
        LEFT JOIN historical_ratings hr_home
            ON g.sport = hr_home.sport
            AND g.game_date = hr_home.snapshot_date
            AND g.home_team = hr_home.team
        LEFT JOIN historical_ratings hr_away
            ON g.sport = hr_away.sport
            AND g.game_date = hr_away.snapshot_date
            AND g.away_team = hr_away.team
        WHERE g.sport = :sport
        ORDER BY g.game_date ASC
    """

    df = pd.read_sql_query(text(query), conn, params={'sport': sport})

    # Filter to games where we have ratings for both teams
    df = df.dropna(subset=['home_win_rating', 'away_win_rating'])

    if len(df) == 0:
        return pd.DataFrame()

    # Add derived columns
    df['home_covered'] = df['spread_result'] > 0
    df['away_covered'] = df['spread_result'] < 0
    df['is_push'] = df['spread_result'] == 0

    # Gap difference: positive means home team has higher gap (more undervalued)
    df['gap_diff'] = df['home_market_gap'] - df['away_market_gap']

    # Rank difference: negative means home team has better (lower) rank
    df['ats_rank_diff'] = df['home_ats_rank'] - df['away_ats_rank']
    df['win_rank_diff'] = df['home_win_rank'] - df['away_win_rank']

    # Did the team with higher market gap cover?
    df['higher_gap_covered'] = (
        (df['gap_diff'] > 0) & df['home_covered'] |
        (df['gap_diff'] < 0) & df['away_covered']
    )

    return df


# =============================================================================
# ROI Calculation
# =============================================================================

def calculate_roi(wins: int, losses: int) -> float:
    """
    Calculate ROI assuming -110 odds on all bets.

    ROI = (profit / total_wagered) * 100

    At -110 odds:
    - Win: profit = $100 on $110 wagered = 0.909...
    - Loss: profit = -$110 on $110 wagered = -1.0
    """
    if wins + losses == 0:
        return 0.0

    profit = (wins * WIN_PAYOUT) - losses
    total_wagered = wins + losses
    return profit / total_wagered


# =============================================================================
# Gap-Based Strategies
# =============================================================================

def backtest_gap_strategy(
    games_df: pd.DataFrame,
    min_gap_diff: float = 0.1,
    bet_on: str = 'higher_gap'
) -> BacktestResult:
    """
    Backtest: Bet on team with higher/lower market gap.

    Args:
        games_df: DataFrame from get_games_with_ratings()
        min_gap_diff: Minimum gap difference to trigger a bet
        bet_on: 'higher_gap' to bet on undervalued team,
                'lower_gap' to fade (bet against) undervalued team

    Returns:
        BacktestResult with strategy performance
    """
    if len(games_df) == 0:
        return BacktestResult(
            strategy_name=f"gap_{bet_on}_{min_gap_diff}",
            total_bets=0, wins=0, losses=0, pushes=0,
            win_rate=0, roi=0, edge=0, is_profitable=False
        )

    # Filter to games meeting threshold
    # gap_diff > 0 means home team has higher gap
    if bet_on == 'higher_gap':
        # Bet on home when home gap > away gap by min_gap_diff
        home_bets = games_df[games_df['gap_diff'] >= min_gap_diff]
        # Bet on away when away gap > home gap by min_gap_diff
        away_bets = games_df[games_df['gap_diff'] <= -min_gap_diff]
    else:  # lower_gap - fade the higher gap team
        home_bets = games_df[games_df['gap_diff'] <= -min_gap_diff]
        away_bets = games_df[games_df['gap_diff'] >= min_gap_diff]

    # Count results
    home_wins = home_bets['home_covered'].sum()
    home_losses = (~home_bets['home_covered'] & ~home_bets['is_push']).sum()
    home_pushes = home_bets['is_push'].sum()

    away_wins = away_bets['away_covered'].sum()
    away_losses = (~away_bets['away_covered'] & ~away_bets['is_push']).sum()
    away_pushes = away_bets['is_push'].sum()

    total_wins = int(home_wins + away_wins)
    total_losses = int(home_losses + away_losses)
    total_pushes = int(home_pushes + away_pushes)
    total_bets = total_wins + total_losses + total_pushes

    # Calculate metrics (excluding pushes from win rate)
    decided_bets = total_wins + total_losses
    win_rate = total_wins / decided_bets if decided_bets > 0 else 0
    roi = calculate_roi(total_wins, total_losses)
    edge = win_rate - BREAKEVEN_WIN_RATE

    return BacktestResult(
        strategy_name=f"gap_{bet_on}_{min_gap_diff:.2f}",
        total_bets=total_bets,
        wins=total_wins,
        losses=total_losses,
        pushes=total_pushes,
        win_rate=win_rate,
        roi=roi,
        edge=edge,
        is_profitable=roi > 0
    )


def analyze_gap_thresholds(
    games_df: pd.DataFrame,
    thresholds: List[float] = None,
    bet_on: str = 'higher_gap'
) -> pd.DataFrame:
    """
    Analyze win rate and ROI at different gap thresholds.

    Args:
        games_df: DataFrame from get_games_with_ratings()
        thresholds: List of gap thresholds to test
        bet_on: 'higher_gap' or 'lower_gap'

    Returns:
        DataFrame with columns: threshold, bets, wins, losses, win_rate, roi, edge
    """
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    results = []
    for threshold in thresholds:
        result = backtest_gap_strategy(games_df, min_gap_diff=threshold, bet_on=bet_on)
        results.append(ThresholdResult(
            threshold=threshold,
            total_bets=result.total_bets,
            wins=result.wins,
            losses=result.losses,
            win_rate=result.win_rate,
            roi=result.roi,
            edge=result.edge
        ))

    return pd.DataFrame([asdict(r) for r in results])


# =============================================================================
# Rank-Based Strategies
# =============================================================================

def backtest_rank_strategy(
    games_df: pd.DataFrame,
    rank_type: str = 'ats',
    max_rank: int = 10,
    min_opponent_rank: int = 20
) -> BacktestResult:
    """
    Backtest: Bet on top-ranked teams vs bottom-ranked teams.

    Args:
        games_df: DataFrame from get_games_with_ratings()
        rank_type: 'ats' for ATS ranking, 'win' for win ranking
        max_rank: Bet on teams ranked this or better (lower is better)
        min_opponent_rank: Against teams ranked this or worse

    Returns:
        BacktestResult with strategy performance
    """
    if len(games_df) == 0:
        return BacktestResult(
            strategy_name=f"rank_{rank_type}_top{max_rank}_vs_bottom{min_opponent_rank}",
            total_bets=0, wins=0, losses=0, pushes=0,
            win_rate=0, roi=0, edge=0, is_profitable=False
        )

    # Select rank columns based on type
    if rank_type == 'ats':
        home_rank_col = 'home_ats_rank'
        away_rank_col = 'away_ats_rank'
    else:  # win
        home_rank_col = 'home_win_rank'
        away_rank_col = 'away_win_rank'

    # Bet on home when home is top-ranked and away is bottom-ranked
    home_bets = games_df[
        (games_df[home_rank_col] <= max_rank) &
        (games_df[away_rank_col] >= min_opponent_rank)
    ]

    # Bet on away when away is top-ranked and home is bottom-ranked
    away_bets = games_df[
        (games_df[away_rank_col] <= max_rank) &
        (games_df[home_rank_col] >= min_opponent_rank)
    ]

    # Count results
    home_wins = home_bets['home_covered'].sum()
    home_losses = (~home_bets['home_covered'] & ~home_bets['is_push']).sum()
    home_pushes = home_bets['is_push'].sum()

    away_wins = away_bets['away_covered'].sum()
    away_losses = (~away_bets['away_covered'] & ~away_bets['is_push']).sum()
    away_pushes = away_bets['is_push'].sum()

    total_wins = int(home_wins + away_wins)
    total_losses = int(home_losses + away_losses)
    total_pushes = int(home_pushes + away_pushes)
    total_bets = total_wins + total_losses + total_pushes

    # Calculate metrics
    decided_bets = total_wins + total_losses
    win_rate = total_wins / decided_bets if decided_bets > 0 else 0
    roi = calculate_roi(total_wins, total_losses)
    edge = win_rate - BREAKEVEN_WIN_RATE

    return BacktestResult(
        strategy_name=f"rank_{rank_type}_top{max_rank}_vs_bottom{min_opponent_rank}",
        total_bets=total_bets,
        wins=total_wins,
        losses=total_losses,
        pushes=total_pushes,
        win_rate=win_rate,
        roi=roi,
        edge=edge,
        is_profitable=roi > 0
    )


# =============================================================================
# Comprehensive Analysis
# =============================================================================

def generate_backtest_report(conn: sqlite3.Connection, sport: str) -> Dict:
    """
    Generate comprehensive backtest report for a sport.

    Returns:
        Dictionary with:
        - games_analyzed: number of games with ratings
        - gap_thresholds: DataFrame of gap threshold results
        - best_gap_threshold: optimal gap threshold
        - rank_strategies: list of rank strategy results
        - best_strategy: overall best performing strategy
    """
    games_df = get_games_with_ratings(conn, sport)

    if len(games_df) == 0:
        return {
            'sport': sport,
            'games_analyzed': 0,
            'error': 'No games with ratings data found'
        }

    report = {
        'sport': sport,
        'games_analyzed': len(games_df),
        'date_range': {
            'start': str(games_df['game_date'].min()),
            'end': str(games_df['game_date'].max())
        }
    }

    # Gap threshold analysis
    gap_thresholds_higher = analyze_gap_thresholds(games_df, bet_on='higher_gap')
    gap_thresholds_lower = analyze_gap_thresholds(games_df, bet_on='lower_gap')

    report['gap_analysis'] = {
        'bet_higher_gap': gap_thresholds_higher.to_dict('records'),
        'bet_lower_gap': gap_thresholds_lower.to_dict('records')
    }

    # Find best gap threshold
    profitable_gaps = gap_thresholds_higher[
        (gap_thresholds_higher['roi'] > 0) &
        (gap_thresholds_higher['total_bets'] >= 10)
    ]
    if len(profitable_gaps) > 0:
        best_gap = profitable_gaps.loc[profitable_gaps['roi'].idxmax()]
        report['best_gap_strategy'] = {
            'threshold': best_gap['threshold'],
            'bets': int(best_gap['total_bets']),
            'win_rate': best_gap['win_rate'],
            'roi': best_gap['roi'],
            'edge': best_gap['edge']
        }
    else:
        report['best_gap_strategy'] = None

    # Rank strategy analysis
    rank_results = []
    for rank_type in ['ats', 'win']:
        for max_rank in [5, 10, 15]:
            for min_opp in [15, 20, 25]:
                result = backtest_rank_strategy(
                    games_df, rank_type=rank_type,
                    max_rank=max_rank, min_opponent_rank=min_opp
                )
                if result.total_bets >= 5:
                    rank_results.append(result.to_dict())

    report['rank_strategies'] = rank_results

    # Find best rank strategy
    if rank_results:
        profitable_ranks = [r for r in rank_results if r['roi'] > 0 and r['total_bets'] >= 10]
        if profitable_ranks:
            best_rank = max(profitable_ranks, key=lambda x: x['roi'])
            report['best_rank_strategy'] = best_rank
        else:
            report['best_rank_strategy'] = None
    else:
        report['best_rank_strategy'] = None

    return report


def get_game_by_game_results(
    conn: sqlite3.Connection,
    sport: str,
    min_gap_diff: float = 0.1
) -> pd.DataFrame:
    """
    Get game-by-game betting results for analysis.

    Returns DataFrame with each game, teams, ratings, bet direction, and result.
    """
    games_df = get_games_with_ratings(conn, sport)

    if len(games_df) == 0:
        return pd.DataFrame()

    # Filter to games meeting threshold
    filtered = games_df[abs(games_df['gap_diff']) >= min_gap_diff].copy()

    # Determine bet and result
    filtered['bet_on'] = filtered.apply(
        lambda r: r['home_team'] if r['gap_diff'] > 0 else r['away_team'],
        axis=1
    )
    filtered['bet_against'] = filtered.apply(
        lambda r: r['away_team'] if r['gap_diff'] > 0 else r['home_team'],
        axis=1
    )
    filtered['bet_won'] = filtered.apply(
        lambda r: r['home_covered'] if r['gap_diff'] > 0 else r['away_covered'],
        axis=1
    )

    # Select and order columns for display
    result_cols = [
        'game_date', 'bet_on', 'bet_against',
        'home_market_gap', 'away_market_gap', 'gap_diff',
        'closing_spread', 'home_score', 'away_score', 'spread_result',
        'bet_won'
    ]

    return filtered[result_cols].sort_values('game_date', ascending=False)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__file__).replace('/src/analysis/backtest_ratings.py', ''))

    from src.database import get_connection

    print("=" * 70)
    print("BACKTEST RATINGS ANALYSIS")
    print("=" * 70)

    conn = get_connection()

    for sport in ['NFL', 'NBA', 'NCAAM']:
        print(f"\n{'='*70}")
        print(f"{sport}")
        print("=" * 70)

        report = generate_backtest_report(conn, sport)

        if 'error' in report:
            print(f"  {report['error']}")
            continue

        print(f"Games analyzed: {report['games_analyzed']}")
        print(f"Date range: {report['date_range']['start']} to {report['date_range']['end']}")

        print("\nGap Threshold Analysis (bet on higher gap team):")
        print("-" * 60)
        print(f"{'Threshold':<12} {'Bets':<8} {'Win%':<10} {'ROI':<10} {'Edge':<10}")
        print("-" * 60)

        for row in report['gap_analysis']['bet_higher_gap']:
            print(f"{row['threshold']:<12.2f} {row['total_bets']:<8} "
                  f"{row['win_rate']*100:<10.1f} {row['roi']*100:<10.1f}% "
                  f"{row['edge']*100:<10.1f}%")

        if report['best_gap_strategy']:
            bg = report['best_gap_strategy']
            print(f"\nBest Gap Strategy: threshold={bg['threshold']:.2f}, "
                  f"bets={bg['bets']}, ROI={bg['roi']*100:.1f}%")

        if report['best_rank_strategy']:
            br = report['best_rank_strategy']
            print(f"Best Rank Strategy: {br['strategy_name']}, "
                  f"bets={br['total_bets']}, ROI={br['roi']*100:.1f}%")

    conn.close()
