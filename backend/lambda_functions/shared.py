"""
Shared utilities for all Lambda functions.

Provides database connectivity, Odds API helpers, and odds extraction logic.
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

logger = logging.getLogger()

# AWS clients
secrets_client = boto3.client('secretsmanager')

# Constants
ODDS_API_SECRET = 'odds-api-key'
DB_SECRET_NAME = 'sports-betting-db-credentials'
SPORT_API_KEYS = {
    'NFL': 'americanfootball_nfl',
    'NBA': 'basketball_nba',
    'NCAAM': 'basketball_ncaab',
}

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


def get_api_key() -> str:
    """Get Odds API key from Secrets Manager."""
    response = secrets_client.get_secret_value(SecretId=ODDS_API_SECRET)
    return response['SecretString']


def make_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Any:
    """Make request to Odds API."""
    import requests
    url = f"https://api.the-odds-api.com/v4/{endpoint}"
    params['apiKey'] = api_key

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_odds_from_game(game: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract current spread, total, and team totals from game data (DraftKings preferred).

    Uses exact name matching against the API response's own home_team/away_team fields,
    which always match the outcome name/description fields within the same response.

    Returns:
        dict with keys: spread, total, home_team_total, away_team_total
    """
    home_team = game.get('home_team', '')
    away_team = game.get('away_team', '')
    bookmakers = game.get('bookmakers', [])

    result = {
        'spread': None,
        'spread_source': None,
        'total': None,
        'total_source': None,
        'home_team_total': None,
        'away_team_total': None
    }

    # Track fallback values for non-DraftKings bookmakers
    fallback_spread = None
    fallback_spread_source = None
    fallback_total = None
    fallback_total_source = None
    fallback_home_tt = None
    fallback_away_tt = None

    for bookmaker in bookmakers:
        is_dk = 'draftkings' in bookmaker.get('key', '').lower()
        bookmaker_name = bookmaker.get('title', 'Unknown')

        for market in bookmaker.get('markets', []):
            # Extract spread
            if market.get('key') == 'spreads':
                for outcome in market.get('outcomes', []):
                    if outcome.get('name') == home_team:
                        spread = outcome.get('point')
                        if is_dk:
                            result['spread'] = spread
                            result['spread_source'] = 'DraftKings'
                        elif fallback_spread is None:
                            fallback_spread = spread
                            fallback_spread_source = bookmaker_name
                        break

            # Extract total (Over/Under have same point value)
            elif market.get('key') == 'totals':
                for outcome in market.get('outcomes', []):
                    if outcome.get('point') is not None:
                        total = outcome.get('point')
                        if is_dk:
                            result['total'] = total
                            result['total_source'] = 'DraftKings'
                        elif fallback_total is None:
                            fallback_total = total
                            fallback_total_source = bookmaker_name
                        break

            # Extract team totals -- match by exact description against API names
            elif market.get('key') == 'team_totals':
                for outcome in market.get('outcomes', []):
                    if outcome.get('name', '').lower() != 'over' or outcome.get('point') is None:
                        continue
                    desc = outcome.get('description', '')
                    if desc == home_team:
                        if is_dk:
                            result['home_team_total'] = outcome['point']
                        elif fallback_home_tt is None:
                            fallback_home_tt = outcome['point']
                    elif desc == away_team:
                        if is_dk:
                            result['away_team_total'] = outcome['point']
                        elif fallback_away_tt is None:
                            fallback_away_tt = outcome['point']

    # Use fallback values if DraftKings not found
    if result['spread'] is None and fallback_spread is not None:
        result['spread'] = fallback_spread
        result['spread_source'] = fallback_spread_source
    if result['total'] is None and fallback_total is not None:
        result['total'] = fallback_total
        result['total_source'] = fallback_total_source
    if result['home_team_total'] is None and fallback_home_tt is not None:
        result['home_team_total'] = fallback_home_tt
    if result['away_team_total'] is None and fallback_away_tt is not None:
        result['away_team_total'] = fallback_away_tt

    return result
