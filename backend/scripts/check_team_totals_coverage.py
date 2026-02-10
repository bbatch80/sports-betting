#!/usr/bin/env python3
"""
Audit: Check team totals and line data coverage in the games table.

Connects via DATABASE_URL env var and prints summary tables showing
how many games have team totals, spreads, and totals per sport.

Usage:
    DATABASE_URL=postgresql://user:pass@host:5432/dbname python -m scripts.check_team_totals_coverage
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


def get_database_url() -> str:
    url = os.environ.get('DATABASE_URL')
    if not url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    return url


def run_audit(engine):
    """Run coverage audit queries and print results."""

    # --- Team Totals Coverage ---
    team_totals_sql = text("""
        SELECT sport,
               COUNT(*) AS total_games,
               COUNT(home_team_total) AS has_home_tt,
               COUNT(away_team_total) AS has_away_tt,
               COUNT(*) - COUNT(home_team_total) AS missing_home_tt,
               MIN(CASE WHEN home_team_total IS NULL THEN game_date END) AS earliest_missing,
               MAX(CASE WHEN home_team_total IS NULL THEN game_date END) AS latest_missing
        FROM games
        WHERE home_score IS NOT NULL
        GROUP BY sport
        ORDER BY sport;
    """)

    # --- Spreads & Totals Coverage ---
    lines_sql = text("""
        SELECT sport,
               COUNT(*) AS total_games,
               COUNT(closing_spread) AS has_spread,
               COUNT(closing_total) AS has_total,
               COUNT(*) - COUNT(closing_spread) AS missing_spread,
               COUNT(*) - COUNT(closing_total) AS missing_total
        FROM games
        WHERE home_score IS NOT NULL
        GROUP BY sport
        ORDER BY sport;
    """)

    with engine.connect() as conn:
        # Team totals
        print("=" * 80)
        print("TEAM TOTALS COVERAGE (games with scores)")
        print("=" * 80)
        results = conn.execute(team_totals_sql).fetchall()

        if not results:
            print("  No completed games found.")
        else:
            header = f"{'Sport':<8} {'Total':>6} {'Home TT':>8} {'Away TT':>8} {'Miss Home':>10} {'Earliest Missing':>18} {'Latest Missing':>16}"
            print(header)
            print("-" * len(header))
            for row in results:
                sport, total, home, away, miss_home, earliest, latest = row
                print(f"{sport:<8} {total:>6} {home:>8} {away:>8} {miss_home:>10} {str(earliest or 'N/A'):>18} {str(latest or 'N/A'):>16}")

        # Spreads & totals
        print()
        print("=" * 80)
        print("SPREADS & TOTALS COVERAGE (games with scores)")
        print("=" * 80)
        results = conn.execute(lines_sql).fetchall()

        if not results:
            print("  No completed games found.")
        else:
            header = f"{'Sport':<8} {'Total':>6} {'Spread':>8} {'Total':>8} {'Miss Spread':>12} {'Miss Total':>12}"
            print(header)
            print("-" * len(header))
            for row in results:
                sport, total, spread, tot, miss_spread, miss_total = row
                print(f"{sport:<8} {total:>6} {spread:>8} {tot:>8} {miss_spread:>12} {miss_total:>12}")

    print()
    print("If gaps exist, run: python -m scripts.backfill_team_totals")


def main():
    database_url = get_database_url()
    engine = create_engine(database_url, poolclass=NullPool)
    run_audit(engine)


if __name__ == "__main__":
    main()
