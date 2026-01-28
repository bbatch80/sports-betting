#!/usr/bin/env python3
"""
Backfill Team Totals from Historical Odds API

Fetches team-specific O/U lines (e.g., "Lakers O/U 115.5") for historical games
and updates the database with team totals and results.

API Endpoint: /historical/sports/{sport}/events/{id}/odds?markets=team_totals
Note: team_totals is only available via per-event endpoint, not batch.

Usage:
    python scripts/backfill_team_totals.py --sport nba --dry-run
    python scripts/backfill_team_totals.py --sport nba --limit 10
    python scripts/backfill_team_totals.py --sport nba
    python scripts/backfill_team_totals.py  # All sports

Cost estimate: ~10 credits per game (2 API calls: events list + event odds)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

# Configuration
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_API_KEYS = {
    'NFL': 'americanfootball_nfl',
    'NBA': 'basketball_nba',
    'NCAAM': 'basketball_ncaab'
}
API_RATE_LIMIT_DELAY = 1.1  # Seconds between API calls
DEFAULT_REGIONS = ['us']
PREFERRED_BOOKMAKERS = ['draftkings', 'fanduel', 'betmgm', 'pointsbet']


def get_api_key() -> str:
    """Get Odds API key from environment."""
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        raise ValueError("ODDS_API_KEY environment variable not set")
    return api_key


def get_database_url() -> str:
    """Get database URL - defaults to local SQLite."""
    url = os.environ.get('DATABASE_URL')
    if url:
        return url
    # Local SQLite
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'analytics.db')
    return f"sqlite:///{db_path}"


def make_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Any:
    """Make request to Odds API with rate limiting."""
    url = f"{ODDS_API_BASE}/{endpoint}"
    params['apiKey'] = api_key

    time.sleep(API_RATE_LIMIT_DELAY)

    response = requests.get(url, params=params, timeout=30)

    # Log remaining credits
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    used = response.headers.get('x-requests-used', 'unknown')

    response.raise_for_status()
    return response.json(), remaining, used


def fuzzy_match_team(team_name: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
    """Find best fuzzy match for team name among candidates."""
    team_lower = team_name.lower()
    best_match = None
    best_ratio = threshold

    for candidate in candidates:
        candidate_lower = candidate.lower()

        # Exact substring match
        if team_lower in candidate_lower or candidate_lower in team_lower:
            return candidate

        # Fuzzy match
        ratio = SequenceMatcher(None, team_lower, candidate_lower).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    return best_match


def get_historical_events(api_key: str, sport: str, date: str) -> Tuple[List[Dict], str, str]:
    """Get historical events for a specific date."""
    api_sport = SPORT_API_KEYS[sport]

    data, remaining, used = make_api_request(
        endpoint=f"historical/sports/{api_sport}/events",
        params={
            'date': date,
            'dateFormat': 'iso'
        },
        api_key=api_key
    )

    events = data.get('data', []) if isinstance(data, dict) else data
    return events, remaining, used


def get_event_team_totals(api_key: str, sport: str, event_id: str,
                          snapshot_time: str) -> Tuple[Optional[Dict], str, str]:
    """
    Get team totals for a specific event.

    Returns dict with structure:
    {
        'team_name_1': {'total': 115.5},
        'team_name_2': {'total': 108.5}
    }
    """
    api_sport = SPORT_API_KEYS[sport]

    try:
        data, remaining, used = make_api_request(
            endpoint=f"historical/sports/{api_sport}/events/{event_id}/odds",
            params={
                'date': snapshot_time,
                'markets': 'team_totals',
                'regions': ','.join(DEFAULT_REGIONS),
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            },
            api_key=api_key
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None, 'N/A', 'N/A'
        raise

    if not data or not data.get('data'):
        return None, remaining, used

    bookmakers = data['data'].get('bookmakers', [])
    team_totals = {}

    # Try preferred bookmakers first, then any available
    for preferred in PREFERRED_BOOKMAKERS + ['']:
        for bookmaker in bookmakers:
            if preferred and preferred not in bookmaker.get('key', '').lower():
                continue

            for market in bookmaker.get('markets', []):
                if market.get('key') != 'team_totals':
                    continue

                for outcome in market.get('outcomes', []):
                    team_name = outcome.get('description', '')
                    bet_type = outcome.get('name', '')  # 'Over' or 'Under'
                    point = outcome.get('point')

                    if team_name and point is not None and bet_type.lower() == 'over':
                        # Team name is in description, Over/Under in name
                        # We just need the point value (same for Over and Under)
                        team_totals[team_name] = {'total': point}

            if team_totals:
                return team_totals, remaining, used

    return None, remaining, used


def match_team_to_total(team_name: str, team_totals: Dict[str, Dict]) -> Optional[float]:
    """Match our team name to the team totals response."""
    if not team_totals:
        return None

    # Try exact match first
    for api_team, data in team_totals.items():
        if team_name.lower() == api_team.lower():
            return data['total']

    # Try fuzzy match
    matched = fuzzy_match_team(team_name, list(team_totals.keys()))
    if matched:
        return team_totals[matched]['total']

    return None


def get_games_needing_team_totals(engine, sport: Optional[str] = None,
                                   limit: Optional[int] = None) -> pd.DataFrame:
    """Get games that have scores but no team totals."""
    query = """
        SELECT id, sport, game_date, home_team, away_team,
               home_score, away_score, closing_total
        FROM games
        WHERE home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND home_team_total IS NULL
          AND away_team_total IS NULL
    """
    params = {}

    if sport:
        query += " AND sport = :sport"
        params['sport'] = sport.upper()

    query += " ORDER BY game_date DESC, sport"

    if limit:
        query += f" LIMIT {limit}"

    # Use text() for proper parameter binding with SQLAlchemy
    return pd.read_sql(text(query), engine, params=params)


def update_game_team_totals(engine, game_id: int, home_total: Optional[float],
                            away_total: Optional[float], home_score: int,
                            away_score: int) -> bool:
    """Update a game with team totals and results."""
    home_result = None
    away_result = None

    if home_total is not None:
        home_result = home_score - home_total
    if away_total is not None:
        away_result = away_score - away_total

    update_sql = text("""
        UPDATE games
        SET home_team_total = :home_total,
            away_team_total = :away_total,
            home_team_total_result = :home_result,
            away_team_total_result = :away_result
        WHERE id = :game_id
    """)

    with engine.begin() as conn:
        result = conn.execute(update_sql, {
            'game_id': game_id,
            'home_total': home_total,
            'away_total': away_total,
            'home_result': home_result,
            'away_result': away_result
        })
        return result.rowcount > 0


def backfill_sport(engine, api_key: str, sport: str, games_df: pd.DataFrame,
                   dry_run: bool = False) -> Dict[str, int]:
    """Backfill team totals for a sport."""
    print(f"\n{'=' * 60}")
    print(f"Processing {sport}: {len(games_df)} games")
    print(f"{'=' * 60}")

    stats = {
        'games_processed': 0,
        'home_totals_found': 0,
        'away_totals_found': 0,
        'both_found': 0,
        'errors': 0
    }

    # Group by date for efficient API calls
    games_by_date = games_df.groupby('game_date')

    # Cache of event IDs by date
    events_cache = {}

    for game_date, date_games in games_by_date:
        print(f"\n{game_date}: {len(date_games)} games")

        # Get events for this date if not cached
        # Use 6pm UTC (1pm EST) - late enough that day's games are posted
        date_str = f"{game_date}T18:00:00Z"

        if game_date not in events_cache:
            try:
                events, remaining, used = get_historical_events(api_key, sport, date_str)
                events_cache[game_date] = events
                print(f"  Fetched {len(events)} events (API: {remaining} remaining)")
            except Exception as e:
                print(f"  ERROR fetching events: {e}")
                stats['errors'] += len(date_games)
                continue
        else:
            events = events_cache[game_date]

        # Build lookup by team names
        event_lookup = {}
        for event in events:
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            event_lookup[(home.lower(), away.lower())] = event

        for _, game in date_games.iterrows():
            game_id = game['id']
            home_team = game['home_team']
            away_team = game['away_team']
            home_score = int(game['home_score'])
            away_score = int(game['away_score'])

            # Find matching event
            event = None
            event_key = (home_team.lower(), away_team.lower())

            if event_key in event_lookup:
                event = event_lookup[event_key]
            else:
                # Try fuzzy matching
                for (h, a), ev in event_lookup.items():
                    if (fuzzy_match_team(home_team, [h]) and
                        fuzzy_match_team(away_team, [a])):
                        event = ev
                        break

            if not event:
                # Try previous day (for games that start early UTC, stored as next day in EST)
                from datetime import date as date_type
                if isinstance(game_date, str):
                    prev_date = (datetime.strptime(game_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    prev_date = (game_date - timedelta(days=1)).isoformat() if hasattr(game_date, 'isoformat') else str(game_date)

                prev_date_str = f"{prev_date}T12:00:00Z"

                if prev_date not in events_cache:
                    try:
                        prev_events, _, _ = get_historical_events(api_key, sport, prev_date_str)
                        events_cache[prev_date] = prev_events
                    except:
                        prev_events = []
                else:
                    prev_events = events_cache[prev_date]

                # Search in previous day's events
                for ev in prev_events:
                    h = ev.get('home_team', '').lower()
                    a = ev.get('away_team', '').lower()
                    if (home_team.lower() in h or h in home_team.lower()) and \
                       (away_team.lower() in a or a in away_team.lower()):
                        event = ev
                        break

                if not event:
                    print(f"  ! No event match: {away_team} @ {home_team}")
                    stats['errors'] += 1
                    continue

            event_id = event.get('id')
            commence_time = event.get('commence_time', '')

            # Calculate snapshot time (5 min before game)
            try:
                game_start = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                snapshot = game_start - timedelta(minutes=5)
                snapshot_str = snapshot.strftime('%Y-%m-%dT%H:%M:%SZ')
            except:
                snapshot_str = date_str

            # Get team totals
            try:
                team_totals, remaining, used = get_event_team_totals(
                    api_key, sport, event_id, snapshot_str
                )
            except Exception as e:
                print(f"  ! Error getting totals for {event_id}: {e}")
                stats['errors'] += 1
                continue

            home_total = None
            away_total = None

            if team_totals:
                home_total = match_team_to_total(home_team, team_totals)
                away_total = match_team_to_total(away_team, team_totals)

            # Update stats
            stats['games_processed'] += 1
            if home_total is not None:
                stats['home_totals_found'] += 1
            if away_total is not None:
                stats['away_totals_found'] += 1
            if home_total is not None and away_total is not None:
                stats['both_found'] += 1

            # Log result
            home_result = f"{home_score - home_total:+.1f}" if home_total else "N/A"
            away_result = f"{away_score - away_total:+.1f}" if away_total else "N/A"

            print(f"  {away_team} @ {home_team}: "
                  f"Home={home_total or 'N/A'} ({home_result}), "
                  f"Away={away_total or 'N/A'} ({away_result}) "
                  f"[{remaining} credits left]")

            # Update database
            if not dry_run:
                update_game_team_totals(
                    engine, game_id, home_total, away_total, home_score, away_score
                )

    return stats


def main():
    parser = argparse.ArgumentParser(description='Backfill team totals from historical odds')
    parser.add_argument('--sport', type=str, choices=['nba', 'nfl', 'ncaam'],
                       help='Sport to process (default: all)')
    parser.add_argument('--limit', type=int, help='Limit number of games to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without updating database')
    args = parser.parse_args()

    print("=" * 60)
    print("Team Totals Backfill Script")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN - No database changes will be made")

    # Setup
    api_key = get_api_key()
    engine = create_engine(get_database_url())

    # Get games needing team totals
    sport_filter = args.sport.upper() if args.sport else None
    games = get_games_needing_team_totals(engine, sport_filter, args.limit)

    if games.empty:
        print("\nNo games need team totals backfill!")
        return

    print(f"\nFound {len(games)} games needing team totals:")
    for sport, count in games.groupby('sport').size().items():
        print(f"  {sport}: {count} games")

    # Process by sport
    all_stats = {}
    for sport in games['sport'].unique():
        sport_games = games[games['sport'] == sport]
        stats = backfill_sport(engine, api_key, sport, sport_games, args.dry_run)
        all_stats[sport] = stats

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_processed = 0
    total_both = 0

    for sport, stats in all_stats.items():
        print(f"\n{sport}:")
        print(f"  Games processed: {stats['games_processed']}")
        print(f"  Home totals found: {stats['home_totals_found']}")
        print(f"  Away totals found: {stats['away_totals_found']}")
        print(f"  Both found: {stats['both_found']}")
        print(f"  Errors: {stats['errors']}")
        total_processed += stats['games_processed']
        total_both += stats['both_found']

    success_rate = (total_both / total_processed * 100) if total_processed > 0 else 0
    print(f"\nOverall: {total_both}/{total_processed} games with complete team totals ({success_rate:.1f}%)")


if __name__ == '__main__':
    main()
