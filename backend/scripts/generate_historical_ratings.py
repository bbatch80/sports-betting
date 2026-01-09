#!/usr/bin/env python3
"""
Generate historical rating snapshots for backtesting.

Computes daily network ratings snapshots for all game dates,
storing results in the historical_ratings table for fast retrieval.

Usage:
    python scripts/generate_historical_ratings.py           # All sports
    python scripts/generate_historical_ratings.py --sport nba   # Single sport
    python scripts/generate_historical_ratings.py --verify  # Check existing data
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import date
from src.database import get_connection, init_database, get_game_count
from src.analysis.network_ratings import (
    generate_historical_snapshots,
    get_ratings_at_date,
    get_ratings_timeseries
)


SPORTS = ['NFL', 'NBA', 'NCAAM']


def count_snapshots(conn, sport: str) -> int:
    """Count existing snapshots for a sport."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(DISTINCT snapshot_date) FROM historical_ratings
        WHERE sport = ?
    ''', (sport,))
    return cursor.fetchone()[0]


def count_teams_for_date(conn, sport: str, snapshot_date: str) -> int:
    """Count teams with ratings for a specific date."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) FROM historical_ratings
        WHERE sport = ? AND snapshot_date = ?
    ''', (sport, snapshot_date))
    return cursor.fetchone()[0]


def verify_snapshots(conn, sport: str) -> None:
    """Verify historical ratings data for a sport."""
    cursor = conn.cursor()

    # Get date range
    cursor.execute('''
        SELECT MIN(snapshot_date), MAX(snapshot_date), COUNT(DISTINCT snapshot_date)
        FROM historical_ratings WHERE sport = ?
    ''', (sport,))
    row = cursor.fetchone()

    if row[0] is None:
        print(f"  {sport}: No historical ratings found")
        return

    min_date, max_date, snapshot_count = row

    # Get team count
    cursor.execute('''
        SELECT COUNT(DISTINCT team) FROM historical_ratings WHERE sport = ?
    ''', (sport,))
    team_count = cursor.fetchone()[0]

    # Get total records
    cursor.execute('''
        SELECT COUNT(*) FROM historical_ratings WHERE sport = ?
    ''', (sport,))
    total_records = cursor.fetchone()[0]

    print(f"  {sport}:")
    print(f"    Date range: {min_date} to {max_date}")
    print(f"    Snapshots: {snapshot_count} days")
    print(f"    Teams: {team_count}")
    print(f"    Total records: {total_records}")


def show_sample(conn, sport: str, team: str, sample_date: date) -> None:
    """Show sample ratings for a specific team and date."""
    ratings = get_ratings_at_date(conn, sport, sample_date)

    if team in ratings:
        r = ratings[team]
        print(f"\n  Sample: {team} on {sample_date}")
        print(f"    Win Rating: {r.win_rating:.3f} (rank #{r.win_rank})")
        print(f"    ATS Rating: {r.ats_rating:.3f} (rank #{r.ats_rank})")
        print(f"    Market Gap: {r.market_gap:+.3f}")
        print(f"    Games Analyzed: {r.games_analyzed}")
    else:
        print(f"\n  No ratings found for {team} on {sample_date}")


def main():
    parser = argparse.ArgumentParser(description='Generate historical rating snapshots')
    parser.add_argument('--sport', choices=['nfl', 'nba', 'ncaam', 'all'],
                        default='all', help='Sport to process (default: all)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing snapshots without generating')
    parser.add_argument('--sample', action='store_true',
                        help='Show sample ratings after generation')
    parser.add_argument('--db-path', type=Path, default=None,
                        help='Custom database path')
    args = parser.parse_args()

    print("=" * 60)
    print("Historical Rating Snapshot Generator")
    print("=" * 60)

    # Initialize database (ensures schema is up to date)
    conn = init_database(args.db_path)

    if args.verify:
        print("\nVerifying existing snapshots:")
        for sport in SPORTS:
            if args.sport != 'all' and sport.lower() != args.sport.lower():
                continue
            verify_snapshots(conn, sport)

    else:
        total_snapshots = 0

        for sport in SPORTS:
            if args.sport != 'all' and sport.lower() != args.sport.lower():
                continue

            game_count = get_game_count(conn, sport)
            if game_count == 0:
                print(f"\n{sport}: No games in database, skipping")
                continue

            existing = count_snapshots(conn, sport)
            print(f"\n{sport}:")
            print(f"  Games in database: {game_count}")
            print(f"  Existing snapshots: {existing}")
            print(f"  Generating snapshots...")

            snapshots = generate_historical_snapshots(conn, sport)
            total_snapshots += snapshots
            print(f"  Created {snapshots} date snapshots")

            # Show sample if requested
            if args.sample:
                # Get latest date with ratings
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT MAX(snapshot_date) FROM historical_ratings
                    WHERE sport = ?
                ''', (sport,))
                latest = cursor.fetchone()[0]

                if latest:
                    # Get a sample team
                    cursor.execute('''
                        SELECT team FROM historical_ratings
                        WHERE sport = ? AND snapshot_date = ?
                        ORDER BY market_gap DESC LIMIT 1
                    ''', (sport, latest))
                    sample_team = cursor.fetchone()[0]
                    show_sample(conn, sport, sample_team, date.fromisoformat(latest))

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print(f"Total snapshots created: {total_snapshots}")
        print("=" * 60)

        # Final summary
        print("\nFinal summary:")
        for sport in SPORTS:
            verify_snapshots(conn, sport)

    conn.close()


if __name__ == '__main__':
    main()
