#!/usr/bin/env python3
"""
Backfill closing totals for existing games in the database.

This script fetches historical over/under lines from The Odds API
for games that have a closing_spread but no closing_total.

Usage:
    cd backend

    # Dry run - see what would be done
    python3 scripts/backfill_totals.py --dry-run

    # Run for a specific sport
    python3 scripts/backfill_totals.py --sport NBA

    # Run with a limit (for testing)
    python3 scripts/backfill_totals.py --sport NBA --limit 10

    # Run all sports
    python3 scripts/backfill_totals.py

Note: This script uses the Odds API historical endpoints which count against your API quota.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple

import boto3
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Configuration
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_API_KEYS = {
    'NFL': 'americanfootball_nfl',
    'NBA': 'basketball_nba',
    'NCAAM': 'basketball_ncaab'
}
API_RATE_LIMIT_DELAY = 1.1  # Slightly over 1 second to be safe
DEFAULT_REGIONS = ['us']


def get_api_key() -> str:
    """Get Odds API key from environment or AWS Secrets Manager."""
    api_key = os.environ.get('ODDS_API_KEY')
    if api_key:
        return api_key

    try:
        secrets_client = boto3.client('secretsmanager')
        response = secrets_client.get_secret_value(SecretId='odds-api-key')
        return response['SecretString']
    except Exception as e:
        print(f"Error getting API key: {e}")
        sys.exit(1)


def get_database_url() -> str:
    """Get DATABASE_URL from environment or AWS Secrets Manager."""
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        return database_url

    try:
        secrets_client = boto3.client('secretsmanager')
        response = secrets_client.get_secret_value(SecretId='sports-betting-db-credentials')
        secret = json.loads(response['SecretString'])
        return secret.get('url')
    except Exception as e:
        print(f"Error getting database URL: {e}")
        sys.exit(1)


def make_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Any:
    """Make request to Odds API."""
    url = f"{ODDS_API_BASE_URL}/{endpoint}"
    params['apiKey'] = api_key

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_historical_odds_for_date(api_key: str, sport: str, date: str) -> Dict[str, Dict]:
    """
    Get historical odds for games on a date using the sport-level odds endpoint.

    Uses historical/sports/{sport}/odds which has better coverage than historical/events.
    Returns a dict keyed by (home_team, away_team) with totals extracted directly.
    """
    api_sport_key = SPORT_API_KEYS[sport]
    est = timezone(timedelta(hours=-5))
    target = datetime.strptime(date, '%Y-%m-%d')
    target_date = target.date()

    # Query at evening time of target date to capture games with lines
    # Also try the day before for late-night games (UTC)
    query_times = [
        target.strftime('%Y-%m-%dT20:00:00Z'),  # 8 PM UTC on target date
        (target - timedelta(days=1)).strftime('%Y-%m-%dT20:00:00Z'),  # Day before
    ]

    events = {}

    for query_date in query_times:
        time.sleep(API_RATE_LIMIT_DELAY)

        try:
            response = make_api_request(
                endpoint=f"historical/sports/{api_sport_key}/odds",
                params={
                    'date': query_date,
                    'markets': 'totals',
                    'regions': ','.join(DEFAULT_REGIONS),
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                },
                api_key=api_key
            )

            for game in response.get('data', []):
                home = game.get('home_team', '')
                away = game.get('away_team', '')
                commence_time = game.get('commence_time', '')

                if not (commence_time and home and away):
                    continue

                # Check if game is on target date (±1 day tolerance)
                try:
                    event_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    event_date_utc = event_dt.date()
                    event_date_est = event_dt.astimezone(est).date()

                    if not (abs((event_date_utc - target_date).days) <= 1 or
                            abs((event_date_est - target_date).days) <= 1):
                        continue
                except:
                    continue

                # Skip if already found
                if (home, away) in events:
                    continue

                # Extract total from bookmakers (prefer DraftKings)
                closing_total = None
                bookmakers = game.get('bookmakers', [])

                # First try DraftKings
                for bookmaker in bookmakers:
                    if 'draftkings' in bookmaker.get('key', '').lower():
                        for market in bookmaker.get('markets', []):
                            if market.get('key') == 'totals':
                                for outcome in market.get('outcomes', []):
                                    if outcome.get('point') is not None:
                                        closing_total = outcome.get('point')
                                        break
                        break

                # Fallback to any bookmaker
                if closing_total is None:
                    for bookmaker in bookmakers:
                        for market in bookmaker.get('markets', []):
                            if market.get('key') == 'totals':
                                for outcome in market.get('outcomes', []):
                                    if outcome.get('point') is not None:
                                        closing_total = outcome.get('point')
                                        break
                        if closing_total:
                            break

                if closing_total is not None:
                    events[(home, away)] = {
                        'id': game.get('id'),
                        'commence_time': commence_time,
                        'closing_total': closing_total  # Total already extracted!
                    }

        except Exception as e:
            print(f"    Error fetching odds for {query_date}: {e}")

    return events


def get_closing_total(api_key: str, sport: str, event_id: str,
                      commence_time: str) -> Optional[float]:
    """Get closing total from DraftKings via historical odds endpoint."""
    api_sport_key = SPORT_API_KEYS[sport]

    try:
        # Parse commence time and get snapshot 5 minutes before
        game_start = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        snapshot_time = game_start - timedelta(minutes=5)
        snapshot_date = snapshot_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        time.sleep(API_RATE_LIMIT_DELAY)

        historical_odds = make_api_request(
            endpoint=f"historical/sports/{api_sport_key}/events/{event_id}/odds",
            params={
                'date': snapshot_date,
                'markets': 'totals',
                'regions': ','.join(DEFAULT_REGIONS),
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            },
            api_key=api_key
        )

        if not historical_odds or not historical_odds.get('data'):
            return None

        bookmakers = historical_odds['data'].get('bookmakers', [])

        # Find DraftKings totals
        for bookmaker in bookmakers:
            if 'draftkings' in bookmaker.get('key', '').lower():
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'totals':
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            if outcome.get('point') is not None:
                                return outcome.get('point')

        # Fallback to any bookmaker
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'totals':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        if outcome.get('point') is not None:
                            return outcome.get('point')

        return None
    except Exception as e:
        print(f"    Error getting total for {event_id}: {e}")
        return None


def update_game_total(conn, game_id: int, closing_total: float,
                      home_score: int, away_score: int) -> None:
    """Update a game with the closing total and computed total_result."""
    total_result = None
    if home_score is not None and away_score is not None and closing_total is not None:
        total_result = (home_score + away_score) - closing_total

    conn.execute(text("""
        UPDATE games
        SET closing_total = :closing_total, total_result = :total_result
        WHERE id = :game_id
    """), {
        'game_id': game_id,
        'closing_total': closing_total,
        'total_result': total_result
    })
    conn.commit()


def backfill_sport(engine, api_key: str, sport: str, limit: int = None,
                   dry_run: bool = False) -> Dict[str, int]:
    """Backfill totals for a single sport."""
    print(f"\n{'='*60}")
    print(f"Processing {sport}")
    print(f"{'='*60}")

    stats = {'processed': 0, 'found': 0, 'not_found': 0, 'errors': 0}

    with engine.connect() as conn:
        # Get games needing totals, grouped by date for efficient event lookup
        query = """
            SELECT id, game_date, home_team, away_team, home_score, away_score
            FROM games
            WHERE sport = :sport
              AND closing_spread IS NOT NULL
              AND closing_total IS NULL
            ORDER BY game_date
        """
        if limit:
            query += f" LIMIT {limit}"

        result = conn.execute(text(query), {'sport': sport})
        games = result.fetchall()

        if not games:
            print(f"  No games need backfilling for {sport}")
            return stats

        print(f"  Found {len(games)} games needing totals")

        if dry_run:
            print(f"  DRY RUN - would process {len(games)} games")
            return stats

        # Group games by date for efficient API calls
        games_by_date = {}
        for game in games:
            date_str = str(game[1])  # game_date
            if date_str not in games_by_date:
                games_by_date[date_str] = []
            games_by_date[date_str].append(game)

        print(f"  Processing {len(games_by_date)} unique dates...")

        # Process each date
        for date_str, date_games in games_by_date.items():
            print(f"\n  Date: {date_str} ({len(date_games)} games)")

            # Get all games with odds for this date (includes totals directly)
            odds_data = get_historical_odds_for_date(api_key, sport, date_str)

            if not odds_data:
                print(f"    No odds data found for {date_str}")
                stats['errors'] += len(date_games)
                continue

            # Process each game for this date
            for game in date_games:
                game_id, game_date, home_team, away_team, home_score, away_score = game
                stats['processed'] += 1

                # Find matching game in odds data
                match = odds_data.get((home_team, away_team))
                if not match:
                    # Try fuzzy match
                    for (h, a), data in odds_data.items():
                        if (home_team.lower() in h.lower() or h.lower() in home_team.lower()) and \
                           (away_team.lower() in a.lower() or a.lower() in away_team.lower()):
                            match = data
                            break

                if not match:
                    print(f"    No odds found for {away_team} @ {home_team}")
                    stats['not_found'] += 1
                    continue

                # Total is already extracted in odds_data
                closing_total = match.get('closing_total')

                if closing_total is not None:
                    update_game_total(conn, game_id, closing_total, home_score, away_score)
                    print(f"    ✓ {away_team} @ {home_team}: total={closing_total}")
                    stats['found'] += 1
                else:
                    print(f"    ✗ {away_team} @ {home_team}: no total in odds")
                    stats['not_found'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill closing totals for existing games")
    parser.add_argument('--sport', choices=['NFL', 'NBA', 'NCAAM'],
                        help="Process only this sport")
    parser.add_argument('--limit', type=int, help="Limit number of games to process")
    parser.add_argument('--dry-run', action='store_true',
                        help="Show what would be done without making changes")

    args = parser.parse_args()

    print("="*60)
    print("BACKFILL CLOSING TOTALS")
    print("="*60)

    api_key = get_api_key()
    print("✓ Retrieved API key")

    database_url = get_database_url()
    engine = create_engine(database_url, poolclass=NullPool)
    print("✓ Connected to database")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    sports = [args.sport] if args.sport else ['NFL', 'NBA', 'NCAAM']

    total_stats = {'processed': 0, 'found': 0, 'not_found': 0, 'errors': 0}

    for sport in sports:
        stats = backfill_sport(engine, api_key, sport, args.limit, args.dry_run)
        for key in total_stats:
            total_stats[key] += stats[key]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Games processed: {total_stats['processed']}")
    print(f"  Totals found:    {total_stats['found']}")
    print(f"  Totals missing:  {total_stats['not_found']}")
    print(f"  Errors:          {total_stats['errors']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
