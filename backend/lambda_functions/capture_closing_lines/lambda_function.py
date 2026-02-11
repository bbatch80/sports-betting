"""
Lambda Function: Capture Closing Lines
Fetches current odds ~30 minutes before tipoff and writes closing lines to todays_games.

Triggered by: EventBridge Scheduler (one-time schedule per game, created by schedule-closing-captures)
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from sqlalchemy import text

from shared import get_db_engine, get_api_key, make_api_request, extract_odds_from_game

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# API settings
API_RATE_LIMIT_DELAY = 1.0
DEFAULT_REGIONS = ['us']


def lambda_handler(event, context):
    """
    Lambda entry point - captures closing lines for a single game.

    Expected event payload:
    {
        "game_id": 1234,
        "sport": "NBA",
        "api_sport_key": "basketball_nba",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics"
    }
    """
    game_id = event.get('game_id')
    sport = event.get('sport')
    api_sport_key = event.get('api_sport_key')
    home_team = event.get('home_team')
    away_team = event.get('away_team')

    logger.info(f"Capturing closing lines for game {game_id}: {away_team} @ {home_team} ({sport})")

    if not all([game_id, sport, api_sport_key, home_team, away_team]):
        logger.error(f"Missing required fields in event: {json.dumps(event)}")
        return {'statusCode': 400, 'body': 'Missing required fields'}

    engine = get_db_engine()
    if engine is None:
        logger.error("No database connection available")
        return {'statusCode': 500, 'body': 'Database connection failed'}

    # Check if closing lines already captured (idempotent)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM todays_games WHERE id = :game_id AND closing_captured_at IS NULL"),
            {'game_id': game_id}
        ).fetchone()

    if row is None:
        logger.info(f"Game {game_id} already has closing lines or does not exist, skipping")
        return {'statusCode': 200, 'body': 'Already captured or game not found'}

    try:
        api_key = get_api_key()

        # Fetch bulk odds for spreads + totals (returns all games for the sport)
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

        # Find our game in the bulk response by matching home_team
        closing_spread = None
        closing_total = None
        closing_home_tt = None
        closing_away_tt = None
        event_id = None

        for game in odds_response:
            if game.get('home_team') == home_team and game.get('away_team') == away_team:
                event_id = game.get('id')
                odds = extract_odds_from_game(game)
                closing_spread = odds['spread']
                closing_total = odds['total']
                logger.info(f"Found game in bulk response: spread={closing_spread}, total={closing_total}")
                break

        if event_id is None:
            logger.warning(f"Game not found in bulk odds response: {away_team} @ {home_team}")
            # Still write NULLs so closing_captured_at is set (prevents re-attempts)

        # Fetch team totals per event (requires per-event endpoint)
        if event_id:
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
                closing_home_tt = tt_odds['home_team_total']
                closing_away_tt = tt_odds['away_team_total']
                logger.info(f"Team totals: home={closing_home_tt}, away={closing_away_tt}")
            except Exception as e:
                logger.warning(f"Could not get team totals for event {event_id}: {e}")

        # Write closing lines to database
        now = datetime.now(timezone.utc)
        update_sql = text("""
            UPDATE todays_games
            SET closing_spread = :closing_spread,
                closing_total = :closing_total,
                closing_home_tt = :closing_home_tt,
                closing_away_tt = :closing_away_tt,
                closing_captured_at = :closing_captured_at
            WHERE id = :game_id AND closing_captured_at IS NULL
        """)

        with engine.begin() as conn:
            result = conn.execute(update_sql, {
                'closing_spread': closing_spread,
                'closing_total': closing_total,
                'closing_home_tt': closing_home_tt,
                'closing_away_tt': closing_away_tt,
                'closing_captured_at': now,
                'game_id': game_id
            })
            logger.info(f"Updated {result.rowcount} row(s) for game {game_id}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'game_id': game_id,
                'closing_spread': closing_spread,
                'closing_total': closing_total,
                'closing_home_tt': closing_home_tt,
                'closing_away_tt': closing_away_tt
            })
        }

    except Exception as e:
        logger.error(f"Error capturing closing lines for game {game_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
