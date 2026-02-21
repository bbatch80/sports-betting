"""
One-time backfill: re-fetch closing odds for games with scores but NULL betting data.

The collect_yesterday_games Lambda originally only extracted DraftKings odds.
When DK wasn't in the API response (~10% of NCAAM games), all odds were discarded.
This script re-fetches odds using bookmaker fallback (prefer DK, fall back to first
available US bookmaker) and updates the database.

Uses the historical/sports/{sport}/odds endpoint which returns ALL events for a date
in a single API call, making this efficient even for many games.

Usage:
    python3 scripts/backfill_closing_odds.py                # All sports, default
    python3 scripts/backfill_closing_odds.py --sport NCAAM   # Single sport
    python3 scripts/backfill_closing_odds.py --dry-run       # Preview without updating
"""

import sys
import os
import time
import logging
import argparse
import requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Optional, Tuple, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from src.db.engine import get_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

SPORT_API_KEYS = {
    'NFL': 'americanfootball_nfl',
    'NBA': 'basketball_nba',
    'NCAAM': 'basketball_ncaab',
}

API_RATE_LIMIT_DELAY = 1.0


def get_api_key() -> str:
    """Get Odds API key from environment."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(env_path)
    except ImportError:
        pass

    key = os.getenv('ODDS_API_KEY')
    if not key:
        raise RuntimeError("ODDS_API_KEY not set in environment or .env")
    return key


def make_api_request(endpoint: str, params: dict, api_key: str):
    """Make request to The Odds API."""
    url = f"https://api.the-odds-api.com/v4/{endpoint}"
    params['apiKey'] = api_key
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_odds_from_event(event: Dict, home_team: str, away_team: str) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float]
]:
    """
    Extract spread, total, and team totals from an event's bookmakers.
    Prefers DraftKings, falls back to first available.

    Returns: (closing_spread, closing_total, home_team_total, away_team_total)
    """
    api_home = event.get('home_team', home_team)
    api_away = event.get('away_team', away_team)
    bookmakers = event.get('bookmakers', [])

    dk_spread = dk_total = dk_home_tt = dk_away_tt = None
    fb_spread = fb_total = fb_home_tt = fb_away_tt = None

    for bookmaker in bookmakers:
        is_dk = 'draftkings' in bookmaker.get('key', '').lower()

        for market in bookmaker.get('markets', []):
            if market.get('key') == 'spreads':
                for outcome in market.get('outcomes', []):
                    if outcome.get('name', '') == api_home:
                        if is_dk:
                            dk_spread = outcome.get('point')
                        elif fb_spread is None:
                            fb_spread = outcome.get('point')
                        break

            elif market.get('key') == 'totals':
                for outcome in market.get('outcomes', []):
                    if outcome.get('point') is not None:
                        if is_dk:
                            dk_total = outcome.get('point')
                        elif fb_total is None:
                            fb_total = outcome.get('point')
                        break

            elif market.get('key') == 'team_totals':
                for outcome in market.get('outcomes', []):
                    if outcome.get('name', '').lower() != 'over' or outcome.get('point') is None:
                        continue
                    desc = outcome.get('description', '')
                    if desc == api_home:
                        if is_dk:
                            dk_home_tt = outcome['point']
                        elif fb_home_tt is None:
                            fb_home_tt = outcome['point']
                    elif desc == api_away:
                        if is_dk:
                            dk_away_tt = outcome['point']
                        elif fb_away_tt is None:
                            fb_away_tt = outcome['point']

    closing_spread = dk_spread if dk_spread is not None else fb_spread
    closing_total = dk_total if dk_total is not None else fb_total
    home_tt = dk_home_tt if dk_home_tt is not None else fb_home_tt
    away_tt = dk_away_tt if dk_away_tt is not None else fb_away_tt

    return closing_spread, closing_total, home_tt, away_tt


def match_event_to_game(events: List[Dict], home_team: str, away_team: str) -> Optional[Dict]:
    """Match an API event to a DB game by team names (exact then partial)."""
    home_lower = home_team.lower()
    away_lower = away_team.lower()

    # Exact match
    for event in events:
        eh = event.get('home_team', '').lower()
        ea = event.get('away_team', '').lower()
        if eh == home_lower and ea == away_lower:
            return event

    # Partial match (handles name variations like "BYU Cougars" vs "Brigham Young Cougars")
    for event in events:
        eh = event.get('home_team', '').lower()
        ea = event.get('away_team', '').lower()
        if (home_lower in eh or eh in home_lower) and (away_lower in ea or ea in away_lower):
            return event

    return None


def main():
    parser = argparse.ArgumentParser(description='Backfill NULL closing odds')
    parser.add_argument('--sport', type=str, default=None, help='Single sport to backfill (e.g. NCAAM)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without updating DB')
    args = parser.parse_args()

    engine = get_engine()
    if engine is None:
        logger.error("Could not connect to database")
        sys.exit(1)

    api_key = get_api_key()
    logger.info("Retrieved API key")

    # Find games with scores but no closing spread
    sport_filter = "AND sport = :sport" if args.sport else ""
    query = text(f"""
        SELECT id, sport, game_date, home_team, away_team,
               home_score, away_score
        FROM games
        WHERE closing_spread IS NULL
          AND home_score IS NOT NULL
          {sport_filter}
        ORDER BY game_date DESC
    """)

    params = {'sport': args.sport.upper()} if args.sport else {}
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    logger.info(f"Found {len(rows)} games with scores but NULL closing_spread")

    if not rows:
        logger.info("Nothing to backfill!")
        return

    # Group by (sport, game_date) so we can batch — one API call per sport+date
    games_by_sport_date = defaultdict(list)
    for row in rows:
        key = (row[1], row[2])  # (sport, game_date)
        games_by_sport_date[key].append({
            'id': row[0],
            'sport': row[1],
            'game_date': row[2],
            'home_team': row[3],
            'away_team': row[4],
            'home_score': row[5],
            'away_score': row[6],
        })

    updated = 0
    skipped = 0
    api_errors = 0

    for (sport, game_date), games in sorted(games_by_sport_date.items()):
        api_sport_key = SPORT_API_KEYS.get(sport)
        if not api_sport_key:
            logger.warning(f"  Unknown sport: {sport}")
            continue

        logger.info(f"\n{sport} {game_date}: {len(games)} games to backfill")

        # Fetch ALL events for this sport+date in one API call.
        # The historical all-events endpoint doesn't support team_totals,
        # so we fetch spreads+totals here. Team totals are fetched per-event below.
        snapshot_date = f"{game_date}T12:00:00Z"

        time.sleep(API_RATE_LIMIT_DELAY)
        try:
            response = make_api_request(
                endpoint=f"historical/sports/{api_sport_key}/odds",
                params={
                    'date': snapshot_date,
                    'markets': 'spreads,totals',
                    'regions': 'us',
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                },
                api_key=api_key
            )
        except Exception as e:
            logger.error(f"  API error for {sport} {game_date}: {e}")
            api_errors += len(games)
            continue

        # Response format: {'data': [event, ...], 'timestamp': '...'}
        events = response.get('data', []) if isinstance(response, dict) else response
        if isinstance(events, dict):
            events = events.get('data', [])

        logger.info(f"  API returned {len(events)} events")

        for game in games:
            event = match_event_to_game(events, game['home_team'], game['away_team'])

            if not event:
                logger.warning(f"  No API match: {game['away_team']} @ {game['home_team']}")
                skipped += 1
                continue

            # Extract spread and total from the all-events response
            closing_spread, closing_total, _, _ = extract_odds_from_event(
                event, game['home_team'], game['away_team']
            )

            if closing_spread is None:
                logger.info(f"  No spread in API: {game['away_team']} @ {game['home_team']}")
                skipped += 1
                continue

            # Fetch team totals via per-event historical endpoint (supports team_totals)
            home_tt = None
            away_tt = None
            event_id = event.get('id')
            commence_time = event.get('commence_time')
            if event_id and commence_time:
                try:
                    game_start = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    tt_snapshot = (game_start - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    time.sleep(API_RATE_LIMIT_DELAY)
                    tt_response = make_api_request(
                        endpoint=f"historical/sports/{api_sport_key}/events/{event_id}/odds",
                        params={
                            'date': tt_snapshot,
                            'markets': 'team_totals',
                            'regions': 'us',
                            'oddsFormat': 'american',
                            'dateFormat': 'iso'
                        },
                        api_key=api_key
                    )
                    if tt_response and tt_response.get('data'):
                        _, _, home_tt, away_tt = extract_odds_from_event(
                            tt_response['data'], game['home_team'], game['away_team']
                        )
                except Exception as e:
                    logger.warning(f"    Team totals fetch failed (non-fatal): {e}")

            # Calculate derived fields
            home_score = game['home_score']
            away_score = game['away_score']

            spread_result = (home_score - away_score) + closing_spread
            total_result = (home_score + away_score) - closing_total if closing_total is not None else None
            home_tt_result = home_score - home_tt if home_tt is not None else None
            away_tt_result = away_score - away_tt if away_tt is not None else None

            logger.info(f"  {game['away_team']} @ {game['home_team']}: "
                        f"spread={closing_spread}, total={closing_total}, "
                        f"spread_result={spread_result}")

            if args.dry_run:
                logger.info(f"    [DRY RUN] Would update game id={game['id']}")
                updated += 1
                continue

            update_sql = text("""
                UPDATE games SET
                    closing_spread = :closing_spread,
                    closing_total = :closing_total,
                    home_team_total = :home_tt,
                    away_team_total = :away_tt,
                    spread_result = :spread_result,
                    total_result = :total_result,
                    home_team_total_result = :home_tt_result,
                    away_team_total_result = :away_tt_result
                WHERE id = :id
            """)

            with engine.begin() as conn:
                conn.execute(update_sql, {
                    'id': game['id'],
                    'closing_spread': closing_spread,
                    'closing_total': closing_total,
                    'home_tt': home_tt,
                    'away_tt': away_tt,
                    'spread_result': spread_result,
                    'total_result': total_result,
                    'home_tt_result': home_tt_result,
                    'away_tt_result': away_tt_result,
                })

            updated += 1

    logger.info(f"\nDone! Updated: {updated}, Skipped: {skipped}, API errors: {api_errors}")


if __name__ == '__main__':
    main()
