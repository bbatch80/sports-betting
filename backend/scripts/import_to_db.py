#!/usr/bin/env python3
"""
Import game data from Parquet files into SQLite database.

Usage:
    python scripts/import_to_db.py           # Import all sports
    python scripts/import_to_db.py --sport nfl   # Import specific sport
    python scripts/import_to_db.py --verify  # Verify import against source
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from src.database import init_database, insert_games, get_game_count, get_connection


# Data file locations
DATA_DIR = Path(__file__).parent.parent / 'data' / 'results'

SPORT_FILES = {
    'NFL': DATA_DIR / 'nfl_season_results.parquet',
    'NBA': DATA_DIR / 'nba_season_results.parquet',
    'NCAAM': DATA_DIR / 'ncaam_season_results.parquet',
}


def import_sport(conn, sport: str, filepath: Path) -> int:
    """Import a single sport's data from Parquet file."""
    if not filepath.exists():
        print(f"  Warning: File not found: {filepath}")
        return 0

    print(f"  Loading {filepath.name}...")
    df = pd.read_parquet(filepath)
    print(f"  Found {len(df)} games in source file")

    # Rename columns if needed (handle different column names)
    if 'game_date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'game_date'})

    inserted = insert_games(conn, df, sport)
    print(f"  Inserted {inserted} games into database")

    return inserted


def verify_import(conn, sport: str, filepath: Path) -> bool:
    """Verify that database matches source file."""
    if not filepath.exists():
        return True

    df = pd.read_parquet(filepath)
    source_count = len(df)
    db_count = get_game_count(conn, sport)

    if source_count == db_count:
        print(f"  {sport}: OK ({db_count} games)")
        return True
    else:
        print(f"  {sport}: MISMATCH - Source: {source_count}, DB: {db_count}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Import game data to SQLite')
    parser.add_argument('--sport', choices=['nfl', 'nba', 'ncaam', 'all'],
                        default='all', help='Sport to import (default: all)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify import against source files')
    parser.add_argument('--db-path', type=Path, default=None,
                        help='Custom database path')
    args = parser.parse_args()

    print("=" * 60)
    print("Importing game data to SQLite database")
    print("=" * 60)

    # Initialize database
    conn = init_database(args.db_path)
    print(f"Database initialized")

    if args.verify:
        print("\nVerifying database against source files:")
        all_ok = True
        for sport, filepath in SPORT_FILES.items():
            if args.sport != 'all' and sport.lower() != args.sport.lower():
                continue
            if not verify_import(conn, sport, filepath):
                all_ok = False
        print("\n" + ("All OK!" if all_ok else "Some mismatches found"))

    else:
        total_inserted = 0

        for sport, filepath in SPORT_FILES.items():
            if args.sport != 'all' and sport.lower() != args.sport.lower():
                continue

            print(f"\n{sport}:")
            inserted = import_sport(conn, sport, filepath)
            total_inserted += inserted

        print("\n" + "=" * 60)
        print(f"IMPORT COMPLETE")
        print(f"Total games inserted: {total_inserted}")
        print(f"Total games in database: {get_game_count(conn)}")
        print("=" * 60)

        # Show summary by sport
        print("\nSummary by sport:")
        for sport in SPORT_FILES.keys():
            count = get_game_count(conn, sport)
            if count > 0:
                print(f"  {sport}: {count} games")

    conn.close()


if __name__ == '__main__':
    main()
