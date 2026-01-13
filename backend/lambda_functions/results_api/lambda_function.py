"""
Lambda Function: Results API
Serves game results from PostgreSQL via API Gateway for mobile app consumption
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
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
    """
    Get DATABASE_URL from environment variable or Secrets Manager.
    """
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


def read_results_from_database(sport_key: str, start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Read results from PostgreSQL database.

    Args:
        sport_key: Sport identifier (nfl, nba, ncaam)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Dictionary with games data
    """
    engine = get_db_engine()
    if engine is None:
        return {
            'sport': sport_key,
            'sport_name': sport_key.upper(),
            'error': 'Database connection unavailable',
            'total_games': 0,
            'games': []
        }

    # Build query with optional date filters
    query = """
        SELECT
            game_date,
            home_team,
            away_team,
            closing_spread,
            home_score,
            away_score,
            spread_result as spread_result_difference
        FROM games
        WHERE sport = :sport
    """
    params = {'sport': sport_key.upper()}

    if start_date:
        query += " AND game_date >= :start_date"
        params['start_date'] = start_date
    if end_date:
        query += " AND game_date <= :end_date"
        params['end_date'] = end_date

    query += " ORDER BY game_date DESC"

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

            games = []
            for row in rows:
                game = {
                    'game_date': row.game_date.strftime('%Y-%m-%d') if row.game_date else None,
                    'home_team': row.home_team,
                    'away_team': row.away_team,
                    'closing_spread': float(row.closing_spread) if row.closing_spread else None,
                    'home_score': int(row.home_score) if row.home_score else None,
                    'away_score': int(row.away_score) if row.away_score else None,
                    'spread_result_difference': float(row.spread_result_difference) if row.spread_result_difference else None,
                }
                games.append(game)

            logger.info(f"Retrieved {len(games)} games from database for {sport_key}")

            return {
                'sport': sport_key,
                'sport_name': sport_key.upper(),
                'total_games': len(games),
                'games': games,
                'last_updated': datetime.now().isoformat(),
                'source': 'database'
            }

    except Exception as e:
        logger.error(f"Database query error: {e}")
        return {
            'sport': sport_key,
            'sport_name': sport_key.upper(),
            'error': str(e),
            'total_games': 0,
            'games': []
        }


def create_response(status_code: int, body: Dict[str, Any], cors_headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Create API Gateway response with CORS headers"""
    if cors_headers is None:
        cors_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,OPTIONS',
            'Content-Type': 'application/json'
        }

    return {
        'statusCode': status_code,
        'headers': cors_headers,
        'body': json.dumps(body, default=str)
    }


def lambda_handler(event, context):
    """
    Lambda handler for API Gateway requests

    Event structure from API Gateway:
    {
        'path': '/api/results/nfl',
        'httpMethod': 'GET',
        'queryStringParameters': {'start_date': '2025-01-01', 'end_date': '2025-01-31'},
        ...
    }
    """
    logger.info(f"Received request: {event.get('httpMethod')} {event.get('path')}")

    try:
        # Handle OPTIONS request (CORS preflight)
        if event.get('httpMethod') == 'OPTIONS':
            return create_response(200, {'message': 'OK'})

        # Extract query parameters
        query_params = event.get('queryStringParameters') or {}
        start_date = query_params.get('start_date')
        end_date = query_params.get('end_date')

        # Extract sport from path
        path = event.get('path', '')
        path_parts = [p for p in path.split('/') if p]

        # Determine which sport(s) to return
        if 'all' in path_parts or (len(path_parts) >= 3 and path_parts[2] == 'all'):
            # Return all results
            all_results = {}
            for sport in ALLOWED_SPORTS:
                try:
                    result_data = read_results_from_database(sport, start_date, end_date)
                    all_results[sport] = result_data
                except Exception as e:
                    logger.error(f"Error getting {sport} results: {e}")
                    all_results[sport] = {
                        'sport': sport,
                        'error': str(e)
                    }

            return create_response(200, {
                'message': 'All results',
                'results': all_results
            })

        # Extract sport from path (e.g., /api/results/nfl)
        sport = None
        if len(path_parts) >= 3:
            sport = path_parts[2].lower()
        elif 'pathParameters' in event and event['pathParameters']:
            sport = event['pathParameters'].get('sport', '').lower()

        # Validate sport
        if not sport or sport not in ALLOWED_SPORTS:
            return create_response(400, {
                'error': 'Invalid sport',
                'message': f'Valid sports are: {", ".join(ALLOWED_SPORTS)}',
                'received': sport
            })

        # Get results for the requested sport
        result_data = read_results_from_database(sport, start_date, end_date)

        # Return results
        return create_response(200, result_data)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return create_response(500, {
            'error': 'Internal server error',
            'message': str(e)
        })
