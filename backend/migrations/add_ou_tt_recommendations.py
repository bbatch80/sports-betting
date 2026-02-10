"""
Migration: Add O/U and TT support to detected_patterns and todays_recommendations.

Changes:
1. Add market_type column to detected_patterns (default 'ats')
2. Drop + recreate unique constraint to include market_type
3. Create current_ou_streaks table
4. Create current_tt_streaks table
5. Add total/TT columns + O/U/TT streak columns to todays_recommendations

Usage:
    python migrations/add_ou_tt_recommendations.py --dry-run   # Preview SQL
    python migrations/add_ou_tt_recommendations.py             # Execute migration
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


MIGRATION_STEPS = [
    # Step 1: Add market_type to detected_patterns
    {
        'description': 'Add market_type column to detected_patterns',
        'sql': "ALTER TABLE detected_patterns ADD COLUMN IF NOT EXISTS market_type VARCHAR(10) DEFAULT 'ats';",
    },
    # Step 2: Drop old unique constraint and create new one including market_type
    {
        'description': 'Drop old unique constraint on detected_patterns',
        'sql': "ALTER TABLE detected_patterns DROP CONSTRAINT IF EXISTS uq_detected_pattern;",
    },
    {
        'description': 'Create new unique constraint including market_type',
        'sql': """ALTER TABLE detected_patterns ADD CONSTRAINT uq_detected_pattern
                  UNIQUE (sport, market_type, pattern_type, streak_type, streak_length, handicap);""",
    },
    # Step 3: Create current_ou_streaks table
    {
        'description': 'Create current_ou_streaks table',
        'sql': """
            CREATE TABLE IF NOT EXISTS current_ou_streaks (
                id SERIAL PRIMARY KEY,
                sport VARCHAR(50) NOT NULL,
                team VARCHAR(100) NOT NULL,
                streak_length INTEGER,
                streak_type VARCHAR(10),
                computed_at TIMESTAMP,
                CONSTRAINT uq_current_ou_streak UNIQUE (sport, team)
            );
        """,
    },
    {
        'description': 'Create index on current_ou_streaks',
        'sql': "CREATE INDEX IF NOT EXISTS idx_current_ou_streaks_sport ON current_ou_streaks (sport);",
    },
    # Step 4: Create current_tt_streaks table
    {
        'description': 'Create current_tt_streaks table',
        'sql': """
            CREATE TABLE IF NOT EXISTS current_tt_streaks (
                id SERIAL PRIMARY KEY,
                sport VARCHAR(50) NOT NULL,
                team VARCHAR(100) NOT NULL,
                streak_length INTEGER,
                streak_type VARCHAR(10),
                computed_at TIMESTAMP,
                CONSTRAINT uq_current_tt_streak UNIQUE (sport, team)
            );
        """,
    },
    {
        'description': 'Create index on current_tt_streaks',
        'sql': "CREATE INDEX IF NOT EXISTS idx_current_tt_streaks_sport ON current_tt_streaks (sport);",
    },
    # Step 5: Add columns to todays_recommendations
    {
        'description': 'Add total column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS total FLOAT;",
    },
    {
        'description': 'Add total_source column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS total_source VARCHAR(100);",
    },
    {
        'description': 'Add home_team_total column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS home_team_total FLOAT;",
    },
    {
        'description': 'Add away_team_total column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS away_team_total FLOAT;",
    },
    {
        'description': 'Add home_ou_streak_length column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS home_ou_streak_length INTEGER;",
    },
    {
        'description': 'Add home_ou_streak_type column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS home_ou_streak_type VARCHAR(10);",
    },
    {
        'description': 'Add away_ou_streak_length column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS away_ou_streak_length INTEGER;",
    },
    {
        'description': 'Add away_ou_streak_type column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS away_ou_streak_type VARCHAR(10);",
    },
    {
        'description': 'Add home_tt_streak_length column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS home_tt_streak_length INTEGER;",
    },
    {
        'description': 'Add home_tt_streak_type column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS home_tt_streak_type VARCHAR(10);",
    },
    {
        'description': 'Add away_tt_streak_length column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS away_tt_streak_length INTEGER;",
    },
    {
        'description': 'Add away_tt_streak_type column to todays_recommendations',
        'sql': "ALTER TABLE todays_recommendations ADD COLUMN IF NOT EXISTS away_tt_streak_type VARCHAR(10);",
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
    print("Running migration: add_ou_tt_recommendations")
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
    parser = argparse.ArgumentParser(description='Run O/U and TT recommendations migration')
    parser.add_argument('--dry-run', action='store_true', help='Preview SQL without executing')
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run)
