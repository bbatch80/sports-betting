"""
Lambda Function: Collect Yesterday's Games
Collects completed games from the Odds API and writes directly to PostgreSQL.

Triggered by: EventBridge schedule (daily at 6:00 AM EST)
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
s3_client = boto3.client('s3')

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'
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


def get_closing_odds(api_key: str, sport_key: str, event_id: str,
                     home_team: str, away_team: str, commence_time: str) -> tuple:
    """
    Get closing spread, total, and team totals from DraftKings via historical odds endpoint.

    Returns:
        tuple: (closing_spread, closing_total, home_team_total, away_team_total)
               Any value can be None if not available
    """
    api_sport_key = SPORT_API_KEYS[sport_key]

    try:
        # Parse commence time
        game_start = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        snapshot_time = game_start - timedelta(minutes=5)
        snapshot_date = snapshot_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        time.sleep(API_RATE_LIMIT_DELAY)

        historical_odds = make_api_request(
            endpoint=f"historical/sports/{api_sport_key}/events/{event_id}/odds",
            params={
                'date': snapshot_date,
                'markets': 'spreads,totals,team_totals',  # Include team_totals
                'regions': ','.join(DEFAULT_REGIONS),
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            },
            api_key=api_key
        )

        if not historical_odds or not historical_odds.get('data'):
            return None, None, None, None

        # Use the API response's own team names for matching within outcomes.
        # This avoids mismatches between our DB names (from scores endpoint)
        # and the names used in odds outcomes (e.g., "BYU Cougars" vs "Brigham Young Cougars").
        api_home = historical_odds['data'].get('home_team', '')
        api_away = historical_odds['data'].get('away_team', '')

        bookmakers = historical_odds['data'].get('bookmakers', [])
        closing_spread = None
        closing_total = None
        home_team_total = None
        away_team_total = None

        # Find DraftKings odds
        for bookmaker in bookmakers:
            if 'draftkings' in bookmaker.get('key', '').lower():
                for market in bookmaker.get('markets', []):
                    # Extract spread — match home team to get home spread
                    if market.get('key') == 'spreads' and closing_spread is None:
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            if outcome.get('name', '') == api_home:
                                closing_spread = outcome.get('point')
                                break

                    # Extract total (Over/Under have same point value)
                    elif market.get('key') == 'totals' and closing_total is None:
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            if outcome.get('point') is not None:
                                closing_total = outcome.get('point')
                                break

                    # Extract team totals — match by description against API names
                    elif market.get('key') == 'team_totals':
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            if outcome.get('name', '').lower() != 'over' or outcome.get('point') is None:
                                continue
                            desc = outcome.get('description', '')
                            if desc == api_home:
                                home_team_total = outcome['point']
                            elif desc == api_away:
                                away_team_total = outcome['point']

                break  # Found DraftKings, stop looking

        return closing_spread, closing_total, home_team_total, away_team_total
    except Exception as e:
        logger.warning(f"Error getting odds for {event_id}: {e}")
        return None, None, None, None


def extract_scores(game_data: Dict[str, Any], home_team: str, away_team: str) -> tuple:
    """Extract home and away scores from game data."""
    scores_list = game_data.get('scores') or []
    home_score = None
    away_score = None

    for score_entry in scores_list:
        if not isinstance(score_entry, dict):
            continue
        team_name = score_entry.get('name', '')
        score_value = score_entry.get('score')

        if score_value is None:
            continue

        if home_team and team_name:
            if home_team.lower() in team_name.lower() or team_name.lower() in home_team.lower():
                home_score = int(score_value)
        if away_team and team_name:
            if away_team.lower() in team_name.lower() or team_name.lower() in away_team.lower():
                away_score = int(score_value)

    return home_score, away_score


def write_games_to_database(games: List[Dict[str, Any]], sport_key: str) -> int:
    """Write games to PostgreSQL using upsert."""
    engine = get_db_engine()
    if engine is None:
        logger.error("No database connection available")
        return 0

    if not games:
        return 0

    upsert_sql = text("""
        INSERT INTO games (sport, game_date, home_team, away_team, closing_spread, closing_total,
                           home_score, away_score, spread_result, total_result,
                           home_team_total, away_team_total, home_team_total_result, away_team_total_result)
        VALUES (:sport, :game_date, :home_team, :away_team, :closing_spread, :closing_total,
                :home_score, :away_score, :spread_result, :total_result,
                :home_team_total, :away_team_total, :home_team_total_result, :away_team_total_result)
        ON CONFLICT (sport, game_date, home_team, away_team)
        DO UPDATE SET
            closing_spread = COALESCE(EXCLUDED.closing_spread, games.closing_spread),
            closing_total = COALESCE(EXCLUDED.closing_total, games.closing_total),
            home_score = COALESCE(EXCLUDED.home_score, games.home_score),
            away_score = COALESCE(EXCLUDED.away_score, games.away_score),
            spread_result = COALESCE(EXCLUDED.spread_result, games.spread_result),
            total_result = COALESCE(EXCLUDED.total_result, games.total_result),
            home_team_total = COALESCE(EXCLUDED.home_team_total, games.home_team_total),
            away_team_total = COALESCE(EXCLUDED.away_team_total, games.away_team_total),
            home_team_total_result = COALESCE(EXCLUDED.home_team_total_result, games.home_team_total_result),
            away_team_total_result = COALESCE(EXCLUDED.away_team_total_result, games.away_team_total_result)
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(upsert_sql, games)
            logger.info(f"✓ Database: upserted {result.rowcount} games for {sport_key.upper()}")
            return result.rowcount
    except Exception as e:
        logger.error(f"Database write error: {e}")
        return 0


def collect_sport(api_key: str, sport_key: str, target_date: datetime) -> Dict[str, Any]:
    """Collect games for one sport and write to database."""
    api_sport_key = SPORT_API_KEYS[sport_key]

    # Convert to EST for date comparison
    est = timezone(timedelta(hours=-5))
    target_date_est = target_date.astimezone(est) if target_date.tzinfo else target_date.replace(tzinfo=est)
    target_date_only = target_date_est.date()

    logger.info(f"Collecting {sport_key.upper()} games from {target_date_only}...")

    # Get completed games from scores endpoint
    time.sleep(API_RATE_LIMIT_DELAY)
    scores = make_api_request(
        endpoint=f"sports/{api_sport_key}/scores",
        params={'dateFormat': 'iso', 'daysFrom': '3'},
        api_key=api_key
    )

    # Filter to target date
    games_to_process = []
    for game in scores:
        if not game.get('completed'):
            continue

        commence_time = game.get('commence_time')
        if not commence_time:
            continue

        # Parse and check date
        try:
            event_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            event_date_est = event_time.astimezone(est).date()
            if event_date_est == target_date_only:
                games_to_process.append(game)
        except:
            continue

    logger.info(f"Found {len(games_to_process)} completed games")

    if not games_to_process:
        return {'sport': sport_key, 'new_games': 0, 'db_rows': 0}

    # Process each game
    games_data = []
    spreads_found = 0
    totals_found = 0
    team_totals_found = 0

    for game in games_to_process:
        event_id = game.get('id')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')

        # Extract scores
        home_score, away_score = extract_scores(game, home_team, away_team)

        # Get closing spread, total, and team totals (single API call)
        closing_spread, closing_total, home_team_total, away_team_total = get_closing_odds(
            api_key, sport_key, event_id, home_team, away_team, commence_time
        )
        if closing_spread is not None:
            spreads_found += 1
        if closing_total is not None:
            totals_found += 1
        if home_team_total is not None or away_team_total is not None:
            team_totals_found += 1

        # Calculate spread result
        spread_result = None
        if home_score is not None and away_score is not None and closing_spread is not None:
            spread_result = (home_score - away_score) + closing_spread

        # Calculate total result (positive = OVER, negative = UNDER)
        total_result = None
        if home_score is not None and away_score is not None and closing_total is not None:
            total_result = (home_score + away_score) - closing_total

        # Calculate team total results (positive = OVER, negative = UNDER)
        home_team_total_result = None
        away_team_total_result = None
        if home_score is not None and home_team_total is not None:
            home_team_total_result = home_score - home_team_total
        if away_score is not None and away_team_total is not None:
            away_team_total_result = away_score - away_team_total

        # Parse game date
        try:
            event_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            game_date = event_time.astimezone(est).date().isoformat()
        except:
            game_date = target_date_only.isoformat()

        games_data.append({
            'sport': sport_key.upper(),
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'closing_spread': closing_spread,
            'closing_total': closing_total,
            'home_score': home_score,
            'away_score': away_score,
            'spread_result': spread_result,
            'total_result': total_result,
            'home_team_total': home_team_total,
            'away_team_total': away_team_total,
            'home_team_total_result': home_team_total_result,
            'away_team_total_result': away_team_total_result
        })

    logger.info(f"Got spreads for {spreads_found}/{len(games_to_process)} games, "
                f"totals for {totals_found}/{len(games_to_process)}, "
                f"team totals for {team_totals_found}/{len(games_to_process)}")

    # Write to database
    db_rows = write_games_to_database(games_data, sport_key)

    return {
        'sport': sport_key,
        'new_games': len(games_data),
        'spreads_found': spreads_found,
        'totals_found': totals_found,
        'team_totals_found': team_totals_found,
        'db_rows': db_rows
    }


def lambda_handler(event, context):
    """Lambda entry point - collects yesterday's games."""
    logger.info("=" * 60)
    logger.info("Starting daily game collection")
    logger.info("=" * 60)

    try:
        api_key = get_api_key()
        logger.info("✓ Retrieved API key")

        # Calculate yesterday (EST)
        est = timezone(timedelta(hours=-5))
        yesterday = datetime.now(est) - timedelta(days=1)

        logger.info(f"Target date: {yesterday.date()} (EST)")

        # Process each sport
        results = []
        for sport_key in SPORTS:
            try:
                result = collect_sport(api_key, sport_key, yesterday)
                results.append(result)
                logger.info(f"✓ {sport_key.upper()}: {result['new_games']} games, {result['db_rows']} written to DB")
            except Exception as e:
                logger.error(f"✗ Error processing {sport_key}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Collection complete',
                'date': yesterday.date().isoformat(),
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
