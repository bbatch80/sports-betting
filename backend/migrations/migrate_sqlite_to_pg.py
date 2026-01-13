#!/usr/bin/env python3
"""
One-time migration script: SQLite to PostgreSQL.

This script migrates all data from your local SQLite database to a PostgreSQL
database (typically AWS RDS).

Usage:
    cd backend

    # Dry run (shows what would be done without making changes)
    PYTHONPATH=. python -m migrations.migrate_sqlite_to_pg \\
        --sqlite-path data/analytics.db \\
        --pg-url postgresql://user:pass@host:5432/dbname \\
        --dry-run

    # Actual migration
    PYTHONPATH=. python -m migrations.migrate_sqlite_to_pg \\
        --sqlite-path data/analytics.db \\
        --pg-url postgresql://user:pass@host:5432/dbname

What it does:
    1. Connects to both SQLite and PostgreSQL databases
    2. Creates the schema in PostgreSQL (if not exists)
    3. Reads all data from SQLite tables
    4. Inserts data into PostgreSQL in batches
    5. Validates row counts match

Safety features:
    - Dry run mode to preview changes
    - Row count validation
    - Batch inserts for large tables
    - Progress bars for visibility
    - Password is hidden in output
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text, inspect

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.models import metadata, create_tables

# Batch size for inserts (tune based on row size and memory)
BATCH_SIZE = 1000


def mask_password(url: str) -> str:
    """Hide password in connection URL for safe logging."""
    if "@" in url and ":" in url:
        # Find password between : and @
        start = url.find("://") + 3
        at_pos = url.find("@")
        colon_pos = url.find(":", start)
        if colon_pos < at_pos:
            return url[:colon_pos + 1] + "****" + url[at_pos:]
    return url


def validate_sqlite(sqlite_path: str) -> bool:
    """Check if SQLite file exists and has expected tables."""
    path = Path(sqlite_path)
    if not path.exists():
        print(f"ERROR: SQLite file not found: {sqlite_path}")
        return False

    engine = create_engine(f"sqlite:///{sqlite_path}")
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    expected = {"games", "historical_ratings"}
    found = set(tables) & expected

    if found != expected:
        missing = expected - found
        print(f"WARNING: Missing tables in SQLite: {missing}")
        print(f"Found tables: {tables}")

    return len(found) > 0


def migrate(sqlite_path: str, pg_url: str, dry_run: bool = False) -> bool:
    """
    Migrate data from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to SQLite database file
        pg_url: PostgreSQL connection URL
        dry_run: If True, show what would be done without making changes

    Returns:
        True if migration succeeded, False otherwise
    """
    print("=" * 60)
    print("SQLite to PostgreSQL Migration")
    print("=" * 60)
    print(f"Source:   {sqlite_path}")
    print(f"Target:   {mask_password(pg_url)}")
    print(f"Dry run:  {dry_run}")
    print("=" * 60)

    # Validate SQLite
    if not validate_sqlite(sqlite_path):
        return False

    # Connect to both databases
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
    pg_engine = create_engine(pg_url)

    # Create schema in PostgreSQL
    if not dry_run:
        print("\nCreating schema in PostgreSQL...")
        create_tables(pg_engine)
        print("Schema created successfully")

    # Tables to migrate (in order - games first, then ratings)
    tables = ["games", "historical_ratings"]

    success = True
    for table_name in tables:
        print(f"\n{'=' * 40}")
        print(f"Migrating: {table_name}")
        print("=" * 40)

        # Check if table exists in SQLite
        inspector = inspect(sqlite_engine)
        if table_name not in inspector.get_table_names():
            print(f"  Skipping - table doesn't exist in SQLite")
            continue

        # Read from SQLite
        df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_engine)
        total_rows = len(df)
        print(f"  Source rows: {total_rows:,}")

        if total_rows == 0:
            print("  No data to migrate")
            continue

        if dry_run:
            print(f"  Would insert {total_rows:,} rows")
            # Show sample of data
            print(f"  Sample (first 3 rows):")
            print(df.head(3).to_string(index=False))
            continue

        # Remove 'id' column to let PostgreSQL auto-generate IDs
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # Insert in batches
        inserted = 0
        errors = 0

        for start_idx in range(0, total_rows, BATCH_SIZE):
            batch = df.iloc[start_idx : start_idx + BATCH_SIZE]
            try:
                batch.to_sql(
                    table_name,
                    pg_engine,
                    if_exists="append",
                    index=False,
                    method="multi",  # Batch insert for efficiency
                )
                inserted += len(batch)
                # Progress indicator
                progress = min(start_idx + BATCH_SIZE, total_rows)
                print(f"  Progress: {progress:,}/{total_rows:,} ({100*progress/total_rows:.1f}%)", end="\r")
            except Exception as e:
                errors += 1
                print(f"\n  ERROR inserting batch at row {start_idx}: {e}")
                if errors > 5:
                    print("  Too many errors, aborting table migration")
                    success = False
                    break

        print()  # New line after progress

        # Validate row count
        with pg_engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            pg_count = result.scalar()

        print(f"  Target rows: {pg_count:,}")

        if pg_count != total_rows:
            print(f"  WARNING: Row count mismatch! Expected {total_rows:,}, got {pg_count:,}")
            success = False
        else:
            print(f"  SUCCESS: All {total_rows:,} rows migrated")

    # Final summary
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE - No changes were made")
    elif success:
        print("MIGRATION COMPLETE - All data transferred successfully")
    else:
        print("MIGRATION COMPLETED WITH WARNINGS - Check output above")
    print("=" * 60)

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate sports betting data from SQLite to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from backend directory):
  Dry run (preview):
    PYTHONPATH=. python -m migrations.migrate_sqlite_to_pg \\
        --sqlite-path data/analytics.db \\
        --pg-url postgresql://user:pass@localhost:5432/analytics \\
        --dry-run

  Actual migration:
    PYTHONPATH=. python -m migrations.migrate_sqlite_to_pg \\
        --sqlite-path data/analytics.db \\
        --pg-url postgresql://user:pass@mydb.xxx.us-east-1.rds.amazonaws.com:5432/analytics
        """,
    )
    parser.add_argument(
        "--sqlite-path",
        required=True,
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--pg-url",
        required=True,
        help="PostgreSQL connection URL (postgresql://user:pass@host:port/dbname)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Validate PostgreSQL URL format
    if not args.pg_url.startswith("postgresql"):
        print("ERROR: PostgreSQL URL must start with 'postgresql://'")
        sys.exit(1)

    success = migrate(args.sqlite_path, args.pg_url, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
