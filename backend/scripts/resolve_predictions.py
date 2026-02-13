"""
Resolve prediction outcomes for a given date.

Matches unresolved predictions in the prediction_results table against
game results in the games table, determining WIN/LOSS/PUSH for each.

Usage:
    python3 scripts/resolve_predictions.py                    # resolve yesterday
    python3 scripts/resolve_predictions.py --date 2026-02-10  # resolve specific date
    python3 scripts/resolve_predictions.py --date 2026-02-10 --dry-run  # preview only
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta, timezone, date

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables (.env has DATABASE_URL for PostgreSQL)
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

from src.db.engine import get_engine
from src.analysis.prediction_tracking import resolve_prediction_outcomes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Resolve prediction outcomes')
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date to resolve (YYYY-MM-DD). Defaults to yesterday EST.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview results without writing to database'
    )
    args = parser.parse_args()

    # Determine target date
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        est = timezone(timedelta(hours=-5))
        target_date = (datetime.now(est) - timedelta(days=1)).date()

    logger.info(f"Resolving predictions for {target_date}...")
    if args.dry_run:
        logger.info("DRY RUN mode — no changes will be written")

    engine = get_engine()
    if engine is None:
        logger.error("Could not connect to database")
        sys.exit(1)

    resolved = resolve_prediction_outcomes(engine, target_date, dry_run=args.dry_run)
    logger.info(f"Done. {resolved} predictions resolved.")


if __name__ == '__main__':
    main()
