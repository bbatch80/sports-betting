"""
Configuration settings
"""

from datetime import datetime
from typing import Dict

# Sport definitions
SPORTS = {
    'nfl': {
        'key': 'americanfootball_nfl',
        'name': 'NFL',
    },
    'nba': {
        'key': 'basketball_nba',
        'name': 'NBA',
    },
    'nhl': {
        'key': 'icehockey_nhl',
        'name': 'NHL',
    },
    'ncaam': {
        'key': 'basketball_ncaab',
        'name': 'NCAAM',
    },
    'ncaaf': {
        'key': 'americanfootball_ncaaf',
        'name': 'NCAAF',
    },
}

# Default settings
DEFAULT_MARKETS = ['spreads']
DEFAULT_REGIONS = ['us']
DEFAULT_BOOKMAKER = ['draftkings']

# API settings
API_RATE_LIMIT_DELAY = 1.0  # Seconds between requests


def get_sport_api_key(sport_key: str) -> str:
    """Get API key for a sport"""
    if sport_key not in SPORTS:
        raise ValueError(f"Unknown sport: {sport_key}")
    return SPORTS[sport_key]['key']


def get_all_sport_keys() -> list:
    """Get list of all sport keys"""
    return list(SPORTS.keys())
