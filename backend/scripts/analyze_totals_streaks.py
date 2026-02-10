#!/usr/bin/env python3
"""
Totals (O/U) Streak Analysis - Validation Script

Run from backend directory:
    python scripts/analyze_totals_streaks.py

Analyzes whether O/U streaks predict future results.
Outputs RIDE vs FADE recommendations with edge calculations.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.db import AnalyticsRepository


def calculate_totals_streaks(games_df):
    """
    For each game, calculate each team's O/U streak going INTO that game.
    Returns DataFrame with streak situations and whether the streak continued.
    """
    df = games_df[games_df['total_result'].notna()].copy()
    df = df[df['total_result'] != 0]  # Exclude pushes
    df = df.sort_values('game_date').reset_index(drop=True)

    team_histories = defaultdict(list)
    results = []

    for _, game in df.iterrows():
        game_date = game['game_date']
        home_team = game['home_team']
        away_team = game['away_team']

        game_ou_result = 'OVER' if game['total_result'] > 0 else 'UNDER'

        for team in [home_team, away_team]:
            history = team_histories[team]

            streak_length = 0
            streak_type = None

            if len(history) > 0:
                streak_type = history[-1]
                for result in reversed(history):
                    if result == streak_type:
                        streak_length += 1
                    else:
                        break

            if streak_length >= 2:
                results.append({
                    'team': team,
                    'game_date': game_date,
                    'streak_length': streak_length,
                    'streak_type': streak_type,
                    'next_game_result': game_ou_result,
                    'continued': 1 if game_ou_result == streak_type else 0
                })

            team_histories[team].append(game_ou_result)

    return pd.DataFrame(results)


def get_current_streaks(games_df, min_streak=3):
    """Get teams currently on O/U streaks."""
    df = games_df[games_df['total_result'].notna()].copy()
    df = df[df['total_result'] != 0]

    teams = set(df['home_team']) | set(df['away_team'])
    current = []

    for team in teams:
        team_games = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].sort_values('game_date', ascending=False)

        if len(team_games) == 0:
            continue

        streak_length = 0
        streak_type = None

        for _, game in team_games.iterrows():
            result = 'OVER' if game['total_result'] > 0 else 'UNDER'

            if streak_type is None:
                streak_type = result
                streak_length = 1
            elif result == streak_type:
                streak_length += 1
            else:
                break

        if streak_length >= min_streak:
            current.append({'team': team, 'streak': streak_length, 'type': streak_type})

    return pd.DataFrame(current).sort_values('streak', ascending=False) if current else pd.DataFrame()


def analyze_sport(repo, sport):
    """Run full analysis for a single sport."""
    print(f"\n{'='*70}")
    print(f"  {sport} TOTALS STREAK ANALYSIS")
    print(f"{'='*70}")

    # Load data
    df = repo.get_games(sport=sport)
    if len(df) == 0:
        print(f"No games found for {sport}")
        return

    # Filter to games with totals
    df_valid = df[df['total_result'].notna()].copy()
    df_no_push = df_valid[df_valid['total_result'] != 0]

    if len(df_no_push) == 0:
        print(f"No totals data for {sport}")
        return

    # Baseline rates
    over_count = (df_no_push['total_result'] > 0).sum()
    under_count = (df_no_push['total_result'] < 0).sum()
    total_count = len(df_no_push)
    over_pct = over_count / total_count * 100
    under_pct = under_count / total_count * 100

    print(f"\nData: {total_count} games with totals")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Baseline: OVER {over_pct:.1f}% | UNDER {under_pct:.1f}%")

    # Calculate streaks
    streaks = calculate_totals_streaks(df)
    if len(streaks) == 0:
        print("No streak situations found")
        return

    print(f"Streak situations: {len(streaks)}")

    # Streak continuation analysis
    print(f"\n--- STREAK CONTINUATION RATES ---")
    print(f"{'Streak':<15} {'N':<8} {'Cont.%':<10} {'Edge':<10} {'Rec':<8}")
    print("-" * 55)

    findings = []
    for streak_type in ['OVER', 'UNDER']:
        for streak_len in range(3, 7):
            mask = (streaks['streak_type'] == streak_type) & (streaks['streak_length'] >= streak_len)
            subset = streaks[mask]

            if len(subset) < 10:
                continue

            cont_rate = subset['continued'].mean() * 100
            edge = cont_rate - 50

            if cont_rate > 52:
                rec = 'RIDE'
            elif cont_rate < 48:
                rec = 'FADE'
            else:
                rec = '-'

            sig = '*' if len(subset) >= 30 else ''
            streak_label = f"{streak_type} {streak_len}+"
            print(f"{streak_label:<15} {len(subset):<8} {cont_rate:<9.1f}% {edge:+.1f}%{sig:<5} {rec:<8}")

            if rec != '-' and len(subset) >= 20:
                findings.append({
                    'streak_type': streak_type,
                    'streak_length': streak_len,
                    'n': len(subset),
                    'rate': cont_rate,
                    'edge': edge,
                    'rec': rec
                })

    print(f"\n* = n >= 30 (statistically meaningful)")

    # Current streaks
    print(f"\n--- CURRENT STREAKS (3+ games) ---")
    current = get_current_streaks(df, min_streak=3)
    if len(current) == 0:
        print("No teams on 3+ game O/U streaks")
    else:
        for _, row in current.head(15).iterrows():
            print(f"  {row['team']}: {row['streak']}-game {row['type']} streak")
        if len(current) > 15:
            print(f"  ... and {len(current) - 15} more")

    # Summary
    if findings:
        print(f"\n--- ACTIONABLE PATTERNS ---")
        for f in findings:
            action = f"bet {f['streak_type']}" if f['rec'] == 'RIDE' else f"bet {'UNDER' if f['streak_type'] == 'OVER' else 'OVER'}"
            print(f"  {f['streak_type']} {f['streak_length']}+ streak -> {f['rec']} ({action}) | {f['edge']:+.1f}% edge, n={f['n']}")


def main():
    print("=" * 70)
    print("  TOTALS (O/U) STREAK ANALYSIS - VALIDATION")
    print("=" * 70)

    repo = AnalyticsRepository()

    # Check connection
    sports = repo.get_sports()
    if not sports:
        print("\nERROR: No sports found in database.")
        print("Check DATABASE_URL environment variable.")
        return

    print(f"\nConnected. Sports available: {', '.join(sports)}")

    # Analyze each sport
    for sport in ['NBA', 'NFL', 'NCAAM']:
        if sport in sports:
            analyze_sport(repo, sport)

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
