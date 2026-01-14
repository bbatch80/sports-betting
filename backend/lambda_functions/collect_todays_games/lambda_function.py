"""
Lambda Function: Collect Today's Games
Fetches today's scheduled games and spreads from the Odds API and stores in PostgreSQL.

Triggered by: EventBridge schedule (daily at 6:30 AM EST)
"""

import json
import boto3
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
secrets_client = boto3.client('secretsmanager')

# Configuration
ODDS_API_SECRET = 'odds-api-key'
DB_SECRET_NAME = 'sports-betting-db-credentials'

# Sports configuration
SPORTS = ['nfl', 'nba', 'ncaam']
SPORT_API_KEYS = {
    'nfl': 'americanfootball_nfl',
    'nba': 'basketball_nba',
    'ncaam': 'basketball_ncaab'
}

# API settings
API_RATE_LIMIT_DELAY = 1.0
DEFAULT_REGIONS = ['us']

# Database engine singleton
_db_engine = None


def get_api_key() -> str:
    """Get Odds API key from Secrets Manager."""
    response = secrets_client.get_secret_value(SecretId=ODDS_API_SECRET)
    return response['SecretString']


def get_database_url() -> Optional[str]:
    """Get DATABASE_URL from environment or Secrets Manager."""
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        return database_url

    try:
        response = secrets_client.get_secret_value(SecretId=DB_SECRET_NAME)
        secret = json.loads(response['SecretString'])
        return secret.get('url')
    except Exception as e:
        logger.error(f"Could not retrieve DATABASE_URL: {e}")
        return None


def get_db_engine():
    """Get or create database engine singleton."""
    global _db_engine
    if _db_engine is None:
        database_url = get_database_url()
        if database_url:
            _db_engine = create_engine(database_url, poolclass=NullPool)
            logger.info("✓ Database engine created")
    return _db_engine


def make_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Any:
    """Make request to Odds API."""
    url = f"https://api.the-odds-api.com/v4/{endpoint}"
    params['apiKey'] = api_key

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_spread_from_game(game: Dict[str, Any]) -> tuple:
    """Extract current spread from game data (DraftKings preferred)."""
    home_team = game.get('home_team', '')
    bookmakers = game.get('bookmakers', [])

    # Prefer DraftKings
    draftkings = None
    fallback = None

    for bookmaker in bookmakers:
        is_dk = 'draftkings' in bookmaker.get('key', '').lower()
        for market in bookmaker.get('markets', []):
            if market.get('key') == 'spreads':
                for outcome in market.get('outcomes', []):
                    if outcome.get('name') == home_team:
                        spread = outcome.get('point')
                        if is_dk:
                            return spread, 'DraftKings'
                        elif fallback is None:
                            fallback = (spread, bookmaker.get('title', 'Unknown'))

    if fallback:
        return fallback
    return None, None


def write_games_to_database(games: List[Dict[str, Any]], sport_key: str) -> int:
    """Write today's games to PostgreSQL using upsert."""
    engine = get_db_engine()
    if engine is None:
        logger.error("No database connection available")
        return 0

    if not games:
        return 0

    upsert_sql = text("""
        INSERT INTO todays_games (sport, game_date, commence_time, home_team, away_team, spread, spread_source, created_at, updated_at)
        VALUES (:sport, :game_date, :commence_time, :home_team, :away_team, :spread, :spread_source, :created_at, :updated_at)
        ON CONFLICT (sport, game_date, home_team, away_team)
        DO UPDATE SET
            spread = EXCLUDED.spread,
            spread_source = EXCLUDED.spread_source,
            commence_time = EXCLUDED.commence_time,
            updated_at = EXCLUDED.updated_at
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(upsert_sql, games)
            logger.info(f"✓ Database: upserted {result.rowcount} games for {sport_key.upper()}")
            return result.rowcount
    except Exception as e:
        logger.error(f"Database write error: {e}")
        return 0


def cleanup_old_games(days_to_keep: int = 7) -> int:
    """Delete games older than X days."""
    engine = get_db_engine()
    if engine is None:
        return 0

    est = timezone(timedelta(hours=-5))
    cutoff_date = (datetime.now(est) - timedelta(days=days_to_keep)).date()

    delete_sql = text("DELETE FROM todays_games WHERE game_date < :cutoff_date")

    try:
        with engine.begin() as conn:
            result = conn.execute(delete_sql, {'cutoff_date': cutoff_date})
            if result.rowcount > 0:
                logger.info(f"✓ Cleaned up {result.rowcount} old games")
            return result.rowcount
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return 0


def collect_sport(api_key: str, sport_key: str, target_date: datetime) -> Dict[str, Any]:
    """Collect today's games for one sport."""
    api_sport_key = SPORT_API_KEYS[sport_key]

    # Convert to EST for date comparison
    est = timezone(timedelta(hours=-5))
    target_date_est = target_date.astimezone(est) if target_date.tzinfo else target_date.replace(tzinfo=est)
    target_date_only = target_date_est.date()

    logger.info(f"Collecting {sport_key.upper()} games for {target_date_only}...")

    # Get odds (includes today's scheduled games with spreads)
    time.sleep(API_RATE_LIMIT_DELAY)
    odds_response = make_api_request(
        endpoint=f"sports/{api_sport_key}/odds",
        params={
            'regions': ','.join(DEFAULT_REGIONS),
            'markets': 'spreads',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        },
        api_key=api_key
    )

    # Filter to target date
    now = datetime.utcnow()
    games_data = []

    for game in odds_response:
        commence_time_str = game.get('commence_time')
        if not commence_time_str:
            continue

        try:
            event_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
            event_date_est = event_time.astimezone(est).date()

            if event_date_est == target_date_only:
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                spread, spread_source = extract_spread_from_game(game)

                games_data.append({
                    'sport': sport_key.upper(),
                    'game_date': target_date_only.isoformat(),
                    'commence_time': event_time.isoformat(),
                    'home_team': home_team,
                    'away_team': away_team,
                    'spread': spread,
                    'spread_source': spread_source,
                    'created_at': now,
                    'updated_at': now
                })
        except Exception as e:
            logger.warning(f"Error parsing game: {e}")
            continue

    logger.info(f"Found {len(games_data)} games for today")

    if not games_data:
        return {'sport': sport_key, 'games_found': 0, 'db_rows': 0}

    # Write to database
    db_rows = write_games_to_database(games_data, sport_key)

    return {
        'sport': sport_key,
        'games_found': len(games_data),
        'db_rows': db_rows
    }


def lambda_handler(event, context):
    """Lambda entry point - collects today's scheduled games."""
    logger.info("=" * 60)
    logger.info("Starting today's games collection")
    logger.info("=" * 60)

    try:
        api_key = get_api_key()
        logger.info("✓ Retrieved API key")

        # Calculate today (EST)
        est = timezone(timedelta(hours=-5))
        today = datetime.now(est)

        logger.info(f"Target date: {today.date()} (EST)")

        # Process each sport
        results = []
        for sport_key in SPORTS:
            try:
                result = collect_sport(api_key, sport_key, today)
                results.append(result)
                logger.info(f"✓ {sport_key.upper()}: {result['games_found']} games, {result['db_rows']} written to DB")
            except Exception as e:
                logger.error(f"✗ Error processing {sport_key}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Cleanup old games
        cleanup_old_games(days_to_keep=7)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Collection complete',
                'date': today.date().isoformat(),
                'results': results
            })
        }

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
