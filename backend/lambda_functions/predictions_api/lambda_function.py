"""
Lambda Function: Predictions API
Serves today's games with team data for mobile app consumption.

Simplified version - removed legacy strategy endpoints (elite-teams, strategy-performance).
Mobile app will use same data model as Streamlit dashboard.
"""

import json
import boto3
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
secrets_client = boto3.client('secretsmanager')

# Configuration
ALLOWED_SPORTS = ['nfl', 'nba', 'ncaam']
DB_SECRET_NAME = 'sports-betting-db-credentials'

# Database engine singleton
_db_engine = None


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
            logger.info("Database engine created")
    return _db_engine


def get_todays_games(sport: str) -> List[Dict[str, Any]]:
    """
    Get today's games from database with team ratings and streaks.

    Returns games in format compatible with Streamlit dashboard.
    """
    engine = get_db_engine()
    if not engine:
        return []

    est = timezone(timedelta(hours=-5))
    today = datetime.now(est).date()

    # Get today's games
    games_query = text("""
        SELECT sport, game_date, commence_time, home_team, away_team,
               spread, spread_source, updated_at
        FROM todays_games
        WHERE sport = :sport AND game_date = :game_date
        ORDER BY commence_time ASC
    """)

    # Get current rankings for team tiers/ratings
    rankings_query = text("""
        SELECT team, win_rating, ats_rating, market_gap, win_rank, ats_rank,
               win_record, ats_record, games_analyzed
        FROM current_rankings
        WHERE sport = :sport
    """)

    # Get current streaks
    streaks_query = text("""
        SELECT team, streak_length, streak_type
        FROM current_streaks
        WHERE sport = :sport
    """)

    try:
        with engine.connect() as conn:
            # Fetch games
            games_result = conn.execute(games_query, {'sport': sport.upper(), 'game_date': today})
            games_rows = games_result.fetchall()

            # Fetch rankings into lookup
            rankings_result = conn.execute(rankings_query, {'sport': sport.upper()})
            rankings = {row[0]: {
                'win_rating': row[1],
                'ats_rating': row[2],
                'market_gap': row[3],
                'win_rank': row[4],
                'ats_rank': row[5],
                'win_record': row[6],
                'ats_record': row[7],
                'games_analyzed': row[8],
                'tier': get_tier(row[2]) if row[2] else 'Unknown'
            } for row in rankings_result.fetchall()}

            # Fetch streaks into lookup
            streaks_result = conn.execute(streaks_query, {'sport': sport.upper()})
            streaks = {row[0]: {
                'streak_length': row[1],
                'streak_type': row[2]
            } for row in streaks_result.fetchall()}

            # Build games list with team data
            games = []
            for row in games_rows:
                home_team = row[3]
                away_team = row[4]

                home_data = rankings.get(home_team, {})
                away_data = rankings.get(away_team, {})
                home_streak = streaks.get(home_team, {})
                away_streak = streaks.get(away_team, {})

                game = {
                    'sport': row[0],
                    'game_date': str(row[1]),
                    'commence_time': row[2].isoformat() if hasattr(row[2], 'isoformat') else str(row[2]),
                    'game_time': format_game_time(row[2]),
                    'home_team': home_team,
                    'away_team': away_team,
                    'spread': float(row[5]) if row[5] else None,
                    'spread_source': row[6],

                    # Home team data
                    'home_tier': home_data.get('tier', 'Unknown'),
                    'home_ats_rating': home_data.get('ats_rating'),
                    'home_win_rating': home_data.get('win_rating'),
                    'home_market_gap': home_data.get('market_gap'),
                    'home_ats_rank': home_data.get('ats_rank'),
                    'home_record': home_data.get('win_record'),
                    'home_ats_record': home_data.get('ats_record'),
                    'home_streak_length': home_streak.get('streak_length', 0),
                    'home_streak_type': home_streak.get('streak_type'),

                    # Away team data
                    'away_tier': away_data.get('tier', 'Unknown'),
                    'away_ats_rating': away_data.get('ats_rating'),
                    'away_win_rating': away_data.get('win_rating'),
                    'away_market_gap': away_data.get('market_gap'),
                    'away_ats_rank': away_data.get('ats_rank'),
                    'away_record': away_data.get('win_record'),
                    'away_ats_record': away_data.get('ats_record'),
                    'away_streak_length': away_streak.get('streak_length', 0),
                    'away_streak_type': away_streak.get('streak_type'),
                }
                games.append(game)

            return games

    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def get_tier(ats_rating: float) -> str:
    """Determine team tier based on ATS rating."""
    if ats_rating is None:
        return 'Unknown'
    if ats_rating >= 0.6:
        return 'Elite'
    elif ats_rating >= 0.5:
        return 'Good'
    elif ats_rating >= 0.4:
        return 'Average'
    else:
        return 'Poor'


def format_game_time(commence_time) -> str:
    """Format commence time to readable EST string."""
    try:
        est = timezone(timedelta(hours=-5))
        if hasattr(commence_time, 'astimezone'):
            event_time_est = commence_time.astimezone(est)
        else:
            return str(commence_time)
        return event_time_est.strftime('%I:%M %p EST')
    except Exception:
        return 'TBD'


def get_historical_results(sport: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """Get historical game results from database."""
    engine = get_db_engine()
    if not engine:
        return []

    query = """
        SELECT game_date, home_team, away_team, closing_spread,
               home_score, away_score, spread_result
        FROM games
        WHERE sport = :sport
    """
    params = {'sport': sport.upper()}

    if start_date:
        query += " AND game_date >= :start_date"
        params['start_date'] = start_date
    if end_date:
        query += " AND game_date <= :end_date"
        params['end_date'] = end_date

    query += " ORDER BY game_date DESC LIMIT 100"

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

            games = []
            for row in rows:
                games.append({
                    'game_date': str(row[0]),
                    'home_team': row[1],
                    'away_team': row[2],
                    'closing_spread': float(row[3]) if row[3] else None,
                    'home_score': int(row[4]) if row[4] else None,
                    'away_score': int(row[5]) if row[5] else None,
                    'spread_result': float(row[6]) if row[6] else None,
                })
            return games
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        return []


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create API Gateway response with CORS headers."""
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,OPTIONS',
            'Content-Type': 'application/json'
        },
        'body': json.dumps(body, default=str)
    }


def lambda_handler(event, context):
    """
    Lambda handler for API Gateway requests.

    Endpoints:
        GET /api/games/{sport} - Today's games with team data
        GET /api/games/all - All sports today
        GET /api/results/{sport} - Historical results
    """
    logger.info(f"Received request: {event.get('httpMethod')} {event.get('path')}")

    try:
        # Handle OPTIONS request (CORS preflight)
        if event.get('httpMethod') == 'OPTIONS':
            return create_response(200, {'message': 'OK'})

        path = event.get('path', '')
        path_parts = [p for p in path.split('/') if p]
        query_params = event.get('queryStringParameters') or {}

        # Determine endpoint type and sport
        endpoint_type = 'games'  # default
        if 'results' in path_parts:
            endpoint_type = 'results'
        elif 'games' in path_parts:
            endpoint_type = 'games'
        elif 'predictions' in path_parts:
            # Legacy endpoint - treat as games
            endpoint_type = 'games'

        # Extract sport
        sport = None
        for part in path_parts:
            if part.lower() in ALLOWED_SPORTS:
                sport = part.lower()
                break
            elif part.lower() == 'all':
                sport = 'all'
                break

        # Handle requests
        if endpoint_type == 'games':
            if sport == 'all':
                # Return all sports
                all_games = {}
                for sp in ALLOWED_SPORTS:
                    games = get_todays_games(sp)
                    all_games[sp] = {
                        'sport': sp,
                        'sport_name': sp.upper(),
                        'games': games,
                        'game_count': len(games)
                    }
                return create_response(200, {
                    'message': "Today's games for all sports",
                    'sports': all_games
                })
            elif sport:
                games = get_todays_games(sport)
                return create_response(200, {
                    'sport': sport,
                    'sport_name': sport.upper(),
                    'games': games,
                    'game_count': len(games),
                    'date': datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d')
                })
            else:
                return create_response(400, {
                    'error': 'Sport required',
                    'message': f'Valid sports: {", ".join(ALLOWED_SPORTS)}'
                })

        elif endpoint_type == 'results':
            if not sport or sport == 'all':
                return create_response(400, {
                    'error': 'Sport required for results',
                    'message': f'Valid sports: {", ".join(ALLOWED_SPORTS)}'
                })

            start_date = query_params.get('start_date')
            end_date = query_params.get('end_date')
            results = get_historical_results(sport, start_date, end_date)

            return create_response(200, {
                'sport': sport,
                'sport_name': sport.upper(),
                'games': results,
                'game_count': len(results)
            })

        # Unknown endpoint
        return create_response(404, {
            'error': 'Not found',
            'message': 'Valid endpoints: /api/games/{sport}, /api/results/{sport}'
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return create_response(500, {
            'error': 'Internal server error',
            'message': str(e)
        })
