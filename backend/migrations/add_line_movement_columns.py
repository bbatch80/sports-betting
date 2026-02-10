#!/usr/bin/env python3
"""
Migration: Add line movement columns to todays_games table.

This adds opening team totals and closing line columns for line movement analysis:
- todays_games.home_team_total (FLOAT) - Opening team total for home team (captured at 6:30 AM)
- todays_games.away_team_total (FLOAT) - Opening team total for away team (captured at 6:30 AM)
- todays_games.closing_spread (FLOAT) - Closing spread (captured 30 min before tip)
- todays_games.closing_total (FLOAT) - Closing game total (captured 30 min before tip)
- todays_games.closing_home_tt (FLOAT) - Closing home team total (captured 30 min before tip)
- todays_games.closing_away_tt (FLOAT) - Closing away team total (captured 30 min before tip)
- todays_games.closing_captured_at (TIMESTAMP) - When closing lines were captured

Usage:
    cd backend

    # Dry run
    PYTHONPATH=. python -m migrations.add_line_movement_columns \
        --pg-url postgresql://user:pass@host:5432/dbname \
        --dry-run

    # Execute migration
    PYTHONPATH=. python -m migrations.add_line_movement_columns \
        --pg-url postgresql://user:pass@host:5432/dbname
"""

import argparse
import sys
from sqlalchemy import create_engine, text


def mask_password(url: str) -> str:
    """Hide password in connection URL for safe logging."""
    if "@" in url and ":" in url:
        start = url.find("://") + 3
        at_pos = url.find("@")
        colon_pos = url.find(":", start)
        if colon_pos < at_pos:
            return url[:colon_pos + 1] + "****" + url[at_pos:]
    return url


def check_column_exists(conn, table: str, column: str) -> bool:
    """Check if a column already exists in a table."""
    result = conn.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table AND column_name = :column
    """), {"table": table, "column": column})
    return result.fetchone() is not None


def migrate(pg_url: str, dry_run: bool = False) -> bool:
    """
    Add line movement columns to todays_games table.
    """
    print("=" * 60)
    print("Migration: Add Line Movement Columns")
    print("=" * 60)
    print(f"Target:   {mask_password(pg_url)}")
    print(f"Dry run:  {dry_run}")
    print("=" * 60)

    engine = create_engine(pg_url)

    migrations = [
        # Opening team totals (captured by collect_todays_games Lambda at 6:30 AM)
        ("todays_games", "home_team_total",
         "ALTER TABLE todays_games ADD COLUMN home_team_total FLOAT"),
        ("todays_games", "away_team_total",
         "ALTER TABLE todays_games ADD COLUMN away_team_total FLOAT"),

        # Closing lines (captured 30 min before game by n8n workflow)
        ("todays_games", "closing_spread",
         "ALTER TABLE todays_games ADD COLUMN closing_spread FLOAT"),
        ("todays_games", "closing_total",
         "ALTER TABLE todays_games ADD COLUMN closing_total FLOAT"),
        ("todays_games", "closing_home_tt",
         "ALTER TABLE todays_games ADD COLUMN closing_home_tt FLOAT"),
        ("todays_games", "closing_away_tt",
         "ALTER TABLE todays_games ADD COLUMN closing_away_tt FLOAT"),
        ("todays_games", "closing_captured_at",
         "ALTER TABLE todays_games ADD COLUMN closing_captured_at TIMESTAMP"),
    ]

    success = True

    with engine.connect() as conn:
        for table, column, sql in migrations:
            exists = check_column_exists(conn, table, column)

            if exists:
                print(f"  SKIP: {table}.{column} already exists")
                continue

            if dry_run:
                print(f"  WOULD ADD: {table}.{column}")
            else:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    print(f"  ADDED: {table}.{column}")
                except Exception as e:
                    print(f"  ERROR adding {table}.{column}: {e}")
                    success = False

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE - No changes were made")
    elif success:
        print("MIGRATION COMPLETE")
    else:
        print("MIGRATION COMPLETED WITH ERRORS")
    print("=" * 60)

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Add line movement columns to todays_games table"
    )
    parser.add_argument(
        "--pg-url",
        required=True,
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    if not args.pg_url.startswith("postgresql"):
        print("ERROR: PostgreSQL URL must start with 'postgresql://'")
        sys.exit(1)

    success = migrate(args.pg_url, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
