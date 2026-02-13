"""
Migration: Add coverage_profile_json column to detected_patterns.

This stores the full coverage profile across all handicap levels for each
detected pattern, enabling the dashboard to show tier-based handicap breakdowns.

Usage:
    python migrations/add_coverage_profiles.py --dry-run   # Preview SQL
    python migrations/add_coverage_profiles.py             # Execute migration
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


MIGRATION_STEPS = [
    {
        'description': 'Add coverage_profile_json column to detected_patterns',
        'sql': "ALTER TABLE detected_patterns ADD COLUMN IF NOT EXISTS coverage_profile_json TEXT;",
    },
]


def get_database_url():
    """Get DATABASE_URL from environment or .env file."""
    url = os.environ.get('DATABASE_URL')
    if url:
        return url

    # Try loading from .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('DATABASE_URL='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")

    return None


def run_migration(dry_run=False):
    """Execute migration steps."""
    database_url = get_database_url()
    if not database_url:
        print("ERROR: DATABASE_URL not set. Set it in environment or .env file.")
        sys.exit(1)

    if dry_run:
        print("=" * 60)
        print("DRY RUN - No changes will be made")
        print("=" * 60)
        for i, step in enumerate(MIGRATION_STEPS, 1):
            print(f"\nStep {i}: {step['description']}")
            print(f"  SQL: {step['sql'].strip()}")
        print(f"\nTotal: {len(MIGRATION_STEPS)} steps")
        return

    engine = create_engine(database_url, poolclass=NullPool)

    print("=" * 60)
    print("Running migration: add_coverage_profiles")
    print("=" * 60)

    with engine.begin() as conn:
        for i, step in enumerate(MIGRATION_STEPS, 1):
            print(f"\nStep {i}/{len(MIGRATION_STEPS)}: {step['description']}")
            try:
                conn.execute(text(step['sql']))
                print(f"  OK")
            except Exception as e:
                print(f"  ERROR: {e}")
                raise

    print(f"\nMigration complete! {len(MIGRATION_STEPS)} steps executed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run coverage profiles migration')
    parser.add_argument('--dry-run', action='store_true', help='Preview SQL without executing')
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run)
