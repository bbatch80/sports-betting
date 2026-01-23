#!/usr/bin/env python3
"""
Migration: Add over/under (totals) columns to games and todays_games tables.

This adds:
- games.closing_total (FLOAT) - The over/under line at game start
- games.total_result (FLOAT) - Computed: (home_score + away_score) - closing_total
- todays_games.total (FLOAT) - Current over/under line
- todays_games.total_source (VARCHAR) - Bookmaker name

Usage:
    cd backend

    # Dry run
    PYTHONPATH=. python -m migrations.add_totals_columns \
        --pg-url postgresql://user:pass@host:5432/dbname \
        --dry-run

    # Execute migration
    PYTHONPATH=. python -m migrations.add_totals_columns \
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
    Add totals columns to games and todays_games tables.
    """
    print("=" * 60)
    print("Migration: Add Over/Under (Totals) Columns")
    print("=" * 60)
    print(f"Target:   {mask_password(pg_url)}")
    print(f"Dry run:  {dry_run}")
    print("=" * 60)

    engine = create_engine(pg_url)

    migrations = [
        # (table, column, sql)
        ("games", "closing_total",
         "ALTER TABLE games ADD COLUMN closing_total FLOAT"),
        ("games", "total_result",
         "ALTER TABLE games ADD COLUMN total_result FLOAT"),
        ("todays_games", "total",
         "ALTER TABLE todays_games ADD COLUMN total FLOAT"),
        ("todays_games", "total_source",
         "ALTER TABLE todays_games ADD COLUMN total_source VARCHAR(50)"),
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
        description="Add totals columns to sports betting database"
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
