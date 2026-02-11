"""
Lambda Function: Collect Today's Games
Fetches today's scheduled games and spreads from the Odds API and stores in PostgreSQL.

Triggered by: EventBridge schedule (daily at 6:30 AM EST)
"""

import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from sqlalchemy import text

from shared import get_db_engine, get_api_key, make_api_request, extract_odds_from_game, SPORT_API_KEYS

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Sports configuration
SPORTS = ['nfl', 'nba', 'ncaam']

# API settings
API_RATE_LIMIT_DELAY = 1.0
DEFAULT_REGIONS = ['us']


def write_games_to_database(games: List[Dict[str, Any]], sport_key: str) -> int:
    """Write today's games to PostgreSQL using upsert."""
    engine = get_db_engine()
    if engine is None:
        logger.error("No database connection available")
        return 0

    if not games:
        return 0

    upsert_sql = text("""
        INSERT INTO todays_games (sport, game_date, commence_time, home_team, away_team,
                                  spread, spread_source, total, total_source,
                                  home_team_total, away_team_total,
                                  created_at, updated_at)
        VALUES (:sport, :game_date, :commence_time, :home_team, :away_team,
                :spread, :spread_source, :total, :total_source,
                :home_team_total, :away_team_total,
                :created_at, :updated_at)
        ON CONFLICT (sport, game_date, home_team, away_team)
        DO UPDATE SET
            spread = EXCLUDED.spread,
            spread_source = EXCLUDED.spread_source,
            total = EXCLUDED.total,
            total_source = EXCLUDED.total_source,
            home_team_total = EXCLUDED.home_team_total,
            away_team_total = EXCLUDED.away_team_total,
            commence_time = EXCLUDED.commence_time,
            updated_at = EXCLUDED.updated_at
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(upsert_sql, games)
            logger.info(f"Database: upserted {result.rowcount} games for {sport_key.upper()}")
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
                logger.info(f"Cleaned up {result.rowcount} old games")
            return result.rowcount
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return 0


def collect_sport(api_key: str, sport_key: str, target_date: datetime) -> Dict[str, Any]:
    """Collect today's games for one sport."""
    api_sport_key = SPORT_API_KEYS[sport_key.upper()]

    # Convert to EST for date comparison
    est = timezone(timedelta(hours=-5))
    target_date_est = target_date.astimezone(est) if target_date.tzinfo else target_date.replace(tzinfo=est)
    target_date_only = target_date_est.date()

    logger.info(f"Collecting {sport_key.upper()} games for {target_date_only}...")

    # Get odds — bulk endpoint supports spreads + totals (not team_totals)
    time.sleep(API_RATE_LIMIT_DELAY)
    odds_response = make_api_request(
        endpoint=f"sports/{api_sport_key}/odds",
        params={
            'regions': ','.join(DEFAULT_REGIONS),
            'markets': 'spreads,totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        },
        api_key=api_key
    )

    # Filter to target date and collect event IDs for team totals lookup
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
                odds = extract_odds_from_game(game)

                games_data.append({
                    'event_id': game.get('id'),
                    'sport': sport_key.upper(),
                    'game_date': target_date_only.isoformat(),
                    'commence_time': event_time.isoformat(),
                    'home_team': game.get('home_team', ''),
                    'away_team': game.get('away_team', ''),
                    'spread': odds['spread'],
                    'spread_source': odds['spread_source'],
                    'total': odds['total'],
                    'total_source': odds['total_source'],
                    'home_team_total': None,
                    'away_team_total': None,
                    'created_at': now,
                    'updated_at': now
                })
        except Exception as e:
            logger.warning(f"Error parsing game: {e}")
            continue

    # Fetch team totals per event (requires per-event endpoint)
    tt_found = 0
    for game_data in games_data:
        event_id = game_data.pop('event_id', None)
        if not event_id:
            continue
        try:
            time.sleep(API_RATE_LIMIT_DELAY)
            tt_response = make_api_request(
                endpoint=f"sports/{api_sport_key}/events/{event_id}/odds",
                params={
                    'regions': ','.join(DEFAULT_REGIONS),
                    'markets': 'team_totals',
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                },
                api_key=api_key
            )
            tt_odds = extract_odds_from_game(tt_response)
            game_data['home_team_total'] = tt_odds['home_team_total']
            game_data['away_team_total'] = tt_odds['away_team_total']
            if tt_odds['home_team_total'] is not None:
                tt_found += 1
        except Exception as e:
            logger.warning(f"Could not get team totals for event {event_id}: {e}")

    logger.info(f"Got team totals for {tt_found}/{len(games_data)} games")

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
        logger.info("Retrieved API key")

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
                logger.info(f"{sport_key.upper()}: {result['games_found']} games, {result['db_rows']} written to DB")
            except Exception as e:
                logger.error(f"Error processing {sport_key}: {e}")
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
