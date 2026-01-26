"""
Today's Games Recommendations Engine.

Combines streak insights and tier matchup insights to generate
betting recommendations for today's scheduled games.

Security Note:
- API key is passed as parameter, never hardcoded
- API key is never logged or displayed
- All external API calls use HTTPS
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import sqlite3
import os

import pandas as pd
import requests

from .insights import InsightPattern, get_cached_streaks
from .tier_matchups import TierMatchupPattern, get_tier
from .network_ratings import get_cached_rankings


# =============================================================================
# Configuration
# =============================================================================

SPORTS_CONFIG = {
    'NFL': 'americanfootball_nfl',
    'NBA': 'basketball_nba',
    'NCAAM': 'basketball_ncaab'
}

DEFAULT_REGIONS = ['us']


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BetRecommendation:
    """A single betting recommendation for a game."""
    bet_on: str              # Team name to bet on
    source: str              # 'streak' or 'tier_matchup'
    edge: float              # Expected edge vs baseline
    confidence: str          # 'high', 'medium', 'low'
    rationale: str           # Human-readable explanation
    handicap: int            # Handicap level for the recommendation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GameRecommendation:
    """A game with all applicable recommendations."""
    sport: str
    game_time: str           # Formatted time string
    home_team: str
    away_team: str
    spread: Optional[float]  # Current spread (home perspective)
    spread_source: Optional[str]  # Bookmaker name
    recommendations: List[BetRecommendation]
    home_tier: Optional[str]
    away_tier: Optional[str]
    home_ats_rating: Optional[float]  # ATS rating 0-1
    away_ats_rating: Optional[float]  # ATS rating 0-1
    home_streak: Optional[Tuple[int, str]]  # (length, type)
    away_streak: Optional[Tuple[int, str]]

    def to_dict(self) -> dict:
        result = asdict(self)
        result['recommendations'] = [r.to_dict() for r in self.recommendations]
        return result


# =============================================================================
# API Key Helper
# =============================================================================

def get_api_key() -> Optional[str]:
    """
    Get Odds API key from secure sources.

    Security:
    - Never logs or displays the key
    - Returns None if unavailable (no exception with key details)
    - Uses environment variable or AWS Secrets Manager only
    """
    # Option 1: Environment variable
    api_key = os.environ.get('ODDS_API_KEY')
    if api_key:
        return api_key

    # Option 2: AWS Secrets Manager
    try:
        import boto3
        client = boto3.client('secretsmanager')
        response = client.get_secret_value(SecretId='odds-api-key')
        return response['SecretString']
    except Exception:
        # Don't expose error details
        return None


# =============================================================================
# Odds API Integration
# =============================================================================

def get_todays_games(sport: str, api_key: str) -> List[Dict]:
    """
    Fetch today's games from The Odds API.

    Args:
        sport: Sport key ('NFL', 'NBA', 'NCAAM')
        api_key: Odds API key

    Returns:
        List of game dictionaries with teams, time, and spreads

    Security:
    - Uses HTTPS only
    - API key passed as URL parameter (standard for this API)
    - Errors don't expose key details
    """
    api_sport_key = SPORTS_CONFIG.get(sport)
    if not api_sport_key:
        return []

    # Get today's date in EST
    est = timezone(timedelta(hours=-5))
    today_est = datetime.now(est).date()

    url = f"https://api.the-odds-api.com/v4/sports/{api_sport_key}/odds"
    params = {
        'apiKey': api_key,
        'regions': ','.join(DEFAULT_REGIONS),
        'markets': 'spreads',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        all_games = response.json()

        # Filter to today's games only
        today_games = []
        for game in all_games:
            commence_time = game.get('commence_time')
            if commence_time:
                try:
                    event_time = pd.to_datetime(commence_time)
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    event_time_est = event_time.astimezone(est)

                    if event_time_est.date() == today_est:
                        today_games.append(game)
                except Exception:
                    pass

        return today_games

    except Exception:
        # Generic error - don't expose API details
        return []


def extract_spread_from_game(game: Dict) -> Tuple[Optional[float], Optional[str]]:
    """Extract current spread from game data (DraftKings preferred)."""
    home_team = game.get('home_team', '')
    bookmakers = game.get('bookmakers', [])

    # Prefer DraftKings
    for bookmaker in bookmakers:
        if 'draftkings' in bookmaker.get('key', '').lower():
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'spreads':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        if outcome.get('name') == home_team:
                            return outcome.get('point'), 'DraftKings'
                    if outcomes:
                        return outcomes[0].get('point'), 'DraftKings'

    # Fallback to any bookmaker
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market.get('key') == 'spreads':
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name') == home_team:
                        return outcome.get('point'), bookmaker.get('title', 'Unknown')
                if outcomes:
                    return outcomes[0].get('point'), bookmaker.get('title', 'Unknown')

    return None, None


def format_game_time(commence_time: str) -> str:
    """Format commence time to readable EST string."""
    try:
        est = timezone(timedelta(hours=-5))
        event_time = pd.to_datetime(commence_time)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        event_time_est = event_time.astimezone(est)
        return event_time_est.strftime('%I:%M %p EST')
    except Exception:
        return 'TBD'


# =============================================================================
# Database-Backed Game Fetching
# =============================================================================

def get_todays_games_from_db(conn: sqlite3.Connection, sport: str) -> List[Dict]:
    """
    Fetch today's games from the database (pre-collected by Lambda).

    Args:
        conn: Database connection
        sport: Sport to fetch ('NFL', 'NBA', 'NCAAM')

    Returns:
        List of game dictionaries with keys:
        - home_team, away_team, commence_time, spread, spread_source
    """
    from sqlalchemy import text

    est = timezone(timedelta(hours=-5))
    today = datetime.now(est).date()

    query = text("""
        SELECT sport, game_date, commence_time, home_team, away_team, spread, spread_source, updated_at
        FROM todays_games
        WHERE sport = :sport AND game_date = :game_date
        ORDER BY commence_time ASC
    """)

    try:
        result = conn.execute(query, {'sport': sport, 'game_date': today})
        rows = result.fetchall()

        games = []
        for row in rows:
            games.append({
                'home_team': row[3],
                'away_team': row[4],
                'commence_time': row[2].isoformat() if hasattr(row[2], 'isoformat') else str(row[2]),
                'spread': row[5],
                'spread_source': row[6],
                'updated_at': row[7]
            })
        return games
    except Exception as e:
        # Rollback on error to reset connection state
        try:
            conn.rollback()
        except Exception:
            pass
        return []


def get_games_last_updated(conn: sqlite3.Connection) -> Optional[datetime]:
    """
    Get the most recent update timestamp for today's games.

    Returns:
        datetime of last update, or None if no games found
    """
    from sqlalchemy import text

    est = timezone(timedelta(hours=-5))
    today = datetime.now(est).date()

    query = text("""
        SELECT MAX(updated_at)
        FROM todays_games
        WHERE game_date = :game_date
    """)

    try:
        result = conn.execute(query, {'game_date': today})
        row = result.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        # Rollback on error to reset connection state
        try:
            conn.rollback()
        except Exception:
            pass
        return None


# =============================================================================
# Team State Functions
# =============================================================================

def get_team_tier_lookup(conn: sqlite3.Connection, sport: str) -> Dict[str, Tuple[str, float]]:
    """
    Build lookup of team -> (tier, ats_rating).

    Uses cached rankings for fast dashboard loads.

    Args:
        conn: Database connection
        sport: Sport key

    Returns:
        Dictionary mapping team name to (tier, ats_rating)
    """
    rankings = get_cached_rankings(conn, sport)
    if not rankings:
        return {}

    return {
        r.team: (get_tier(r.ats_rating), r.ats_rating)
        for r in rankings
    }


def get_team_streak_lookup(conn: sqlite3.Connection, sport: str) -> Dict[str, Tuple[int, str]]:
    """
    Build lookup of team -> (streak_length, streak_type).

    Uses cached streaks for fast dashboard loads.

    Args:
        conn: Database connection
        sport: Sport key

    Returns:
        Dictionary mapping team name to (streak_length, streak_type)
    """
    streaks = get_cached_streaks(conn, sport)
    return {
        team: (info['streak_length'], info['streak_type'])
        for team, info in streaks.items()
    }


# =============================================================================
# Recommendation Logic
# =============================================================================

def match_streak_patterns(
    team: str,
    opponent: str,
    streak: Tuple[int, str],
    patterns: List[InsightPattern],
    sport: str
) -> List[BetRecommendation]:
    """
    Check if a team's current streak matches any profitable patterns.

    Args:
        team: Team name
        opponent: Opponent team name (for FADE recommendations)
        streak: (streak_length, streak_type)
        patterns: List of detected streak patterns
        sport: Sport key

    Returns:
        List of BetRecommendation if patterns match
    """
    recommendations = []
    streak_length, streak_type = streak

    for pattern in patterns:
        # Must match sport, streak type, and meet minimum length
        if pattern.sport != sport:
            continue
        if pattern.streak_type != streak_type:
            continue
        if streak_length < pattern.streak_length:
            continue

        # Determine recommendation
        if pattern.pattern_type == 'streak_ride':
            # Bet ON this team
            rec = BetRecommendation(
                bet_on=team,
                source='streak',
                edge=abs(pattern.edge),
                confidence=pattern.confidence,
                rationale=f"{streak_length}-game ATS {streak_type.lower()} streak -> RIDE ({abs(pattern.edge):.1%} edge)",
                handicap=pattern.handicap
            )
        else:
            # streak_fade: bet AGAINST this team by betting ON their opponent
            rec = BetRecommendation(
                bet_on=opponent,
                source='streak',
                edge=abs(pattern.edge),
                confidence=pattern.confidence,
                rationale=f"Opponent on {streak_length}-game ATS {streak_type.lower()} streak -> FADE ({abs(pattern.edge):.1%} edge)",
                handicap=pattern.handicap
            )

        recommendations.append(rec)
        break  # Only one streak recommendation per team

    return recommendations


def match_tier_patterns(
    home_team: str,
    away_team: str,
    home_tier_info: Tuple[str, float],
    away_tier_info: Tuple[str, float],
    patterns: List[TierMatchupPattern],
    sport: str
) -> List[BetRecommendation]:
    """
    Check if a game's tier matchup matches any profitable patterns.

    Args:
        home_team: Home team name
        away_team: Away team name
        home_tier_info: (tier, ats_rating) for home team
        away_tier_info: (tier, ats_rating) for away team
        patterns: List of detected tier matchup patterns
        sport: Sport key

    Returns:
        List of BetRecommendation if patterns match
    """
    recommendations = []

    home_tier, home_rating = home_tier_info
    away_tier, away_rating = away_tier_info

    # Determine higher/lower rated team
    if home_rating > away_rating:
        higher_team, higher_tier = home_team, home_tier
        lower_team, lower_tier = away_team, away_tier
    else:
        higher_team, higher_tier = away_team, away_tier
        lower_team, lower_tier = home_team, home_tier

    for pattern in patterns:
        if pattern.sport != sport:
            continue
        if pattern.higher_tier != higher_tier:
            continue
        if pattern.lower_tier != lower_tier:
            continue

        # Determine which team to bet on
        bet_team = higher_team if pattern.bet_on == 'HIGHER' else lower_team
        bet_tier = higher_tier if pattern.bet_on == 'HIGHER' else lower_tier

        rec = BetRecommendation(
            bet_on=bet_team,
            source='tier_matchup',
            edge=pattern.edge,
            confidence=pattern.confidence,
            rationale=f"{higher_tier} vs {lower_tier} @ +{pattern.handicap} -> Bet {pattern.bet_on} ({pattern.edge:.1%} edge)",
            handicap=pattern.handicap
        )
        recommendations.append(rec)
        break  # Only one tier recommendation per game

    return recommendations


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_recommendations(
    conn: sqlite3.Connection,
    sport: str,
    streak_patterns: List[InsightPattern],
    tier_patterns: List[TierMatchupPattern],
) -> List[GameRecommendation]:
    """
    Generate betting recommendations for today's games.

    Combines streak insights and tier matchup insights for each game.
    Games are fetched from the database (pre-collected by Lambda).

    Args:
        conn: Database connection
        sport: Sport to analyze ('NFL', 'NBA', 'NCAAM', or 'All')
        streak_patterns: Detected streak patterns
        tier_patterns: Detected tier matchup patterns

    Returns:
        List of GameRecommendation sorted by game time
    """
    # Handle "All" sports
    sports = ['NFL', 'NBA', 'NCAAM'] if sport == 'All' else [sport]

    all_recommendations = []

    for sp in sports:
        # Get today's games from database
        games = get_todays_games_from_db(conn, sp)
        if not games:
            continue

        # Build team lookups
        tier_lookup = get_team_tier_lookup(conn, sp)
        streak_lookup = get_team_streak_lookup(conn, sp)

        for game in games:
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')

            # Get spread (already stored in DB from Lambda)
            spread = game.get('spread')
            spread_source = game.get('spread_source')

            # Get team info
            home_tier_info = tier_lookup.get(home_team)
            away_tier_info = tier_lookup.get(away_team)
            home_streak = streak_lookup.get(home_team)
            away_streak = streak_lookup.get(away_team)

            recs = []

            # Check streak patterns for both teams
            if home_streak:
                recs.extend(match_streak_patterns(
                    home_team, away_team, home_streak, streak_patterns, sp
                ))
            if away_streak:
                recs.extend(match_streak_patterns(
                    away_team, home_team, away_streak, streak_patterns, sp
                ))

            # Check tier matchup patterns
            if home_tier_info and away_tier_info:
                recs.extend(match_tier_patterns(
                    home_team, away_team,
                    home_tier_info, away_tier_info,
                    tier_patterns, sp
                ))

            # Create game recommendation
            game_rec = GameRecommendation(
                sport=sp,
                game_time=format_game_time(game.get('commence_time', '')),
                home_team=home_team,
                away_team=away_team,
                spread=spread,
                spread_source=spread_source,
                recommendations=recs,
                home_tier=home_tier_info[0] if home_tier_info else None,
                away_tier=away_tier_info[0] if away_tier_info else None,
                home_ats_rating=home_tier_info[1] if home_tier_info else None,
                away_ats_rating=away_tier_info[1] if away_tier_info else None,
                home_streak=home_streak,
                away_streak=away_streak
            )

            all_recommendations.append(game_rec)

    # Sort by game time (simple string sort works for 12-hour format)
    # Games with recommendations first, then by time
    all_recommendations.sort(key=lambda g: (
        0 if g.recommendations else 1,
        g.game_time
    ))

    return all_recommendations


def get_combined_confidence(recommendations: List[BetRecommendation]) -> str:
    """
    Determine combined confidence when multiple recommendations agree.

    Args:
        recommendations: List of recommendations

    Returns:
        'HIGH', 'MEDIUM', or 'LOW'
    """
    if not recommendations:
        return 'LOW'

    # Check if recommendations agree on the same team
    teams = set(r.bet_on for r in recommendations)
    if len(teams) > 1:
        # Conflicting recommendations
        return 'LOW'

    # All agree - combine confidence
    conf_scores = {'high': 3, 'medium': 2, 'low': 1}
    total_score = sum(conf_scores.get(r.confidence.lower(), 1) for r in recommendations)

    if total_score >= 5:  # e.g., high + medium, or medium + medium + low
        return 'HIGH'
    elif total_score >= 3:  # e.g., medium + low, or high alone
        return 'MEDIUM'
    return 'LOW'


# =============================================================================
# Cached Recommendations (Pre-Computed by Lambda)
# =============================================================================

def get_cached_recommendations(conn, sport: str = None) -> List[GameRecommendation]:
    """
    Read pre-computed recommendations from the database cache table.

    This is the fast path - reads from todays_recommendations table
    that was populated by the generate_current_rankings Lambda at 6:20 AM.

    Falls back to live computation if cache is empty (slow but functional).

    Args:
        conn: Database connection
        sport: Sport to filter ('NFL', 'NBA', 'NCAAM', or 'All' for all sports)

    Returns:
        List of GameRecommendation objects sorted by game time
    """
    import json
    from sqlalchemy import text

    est = timezone(timedelta(hours=-5))
    today = datetime.now(est).date()

    # Build query
    if sport and sport != 'All':
        query = text("""
            SELECT sport, game_date, home_team, away_team, game_time,
                   spread, spread_source, home_tier, away_tier,
                   home_ats_rating, away_ats_rating,
                   home_streak_length, home_streak_type,
                   away_streak_length, away_streak_type,
                   recommendations_json
            FROM todays_recommendations
            WHERE game_date = :game_date AND sport = :sport
            ORDER BY game_time
        """)
        params = {'game_date': today, 'sport': sport}
    else:
        query = text("""
            SELECT sport, game_date, home_team, away_team, game_time,
                   spread, spread_source, home_tier, away_tier,
                   home_ats_rating, away_ats_rating,
                   home_streak_length, home_streak_type,
                   away_streak_length, away_streak_type,
                   recommendations_json
            FROM todays_recommendations
            WHERE game_date = :game_date
            ORDER BY game_time
        """)
        params = {'game_date': today}

    try:
        result = conn.execute(query, params)
        rows = result.fetchall()
    except Exception:
        rows = []

    if not rows:
        # Fallback to live computation (expensive, but still works)
        # This happens if Lambda hasn't run yet or no games today
        from .insights import get_cached_patterns
        patterns = get_cached_patterns(conn, min_sample=30, min_edge=0.05)
        return generate_recommendations(conn, sport or 'All', patterns, [])

    # Convert rows to GameRecommendation objects
    recommendations = []
    for row in rows:
        # Parse recommendations JSON
        recs_json = row[15]  # recommendations_json column
        bet_recs = []
        if recs_json:
            try:
                recs_data = json.loads(recs_json)
                for r in recs_data:
                    bet_recs.append(BetRecommendation(
                        bet_on=r['bet_on'],
                        source=r['source'],
                        edge=r['edge'],
                        confidence=r['confidence'],
                        rationale=r['rationale'],
                        handicap=r['handicap']
                    ))
            except (json.JSONDecodeError, KeyError):
                pass

        # Build streak tuples
        home_streak = None
        if row[11] is not None and row[12]:  # home_streak_length, home_streak_type
            home_streak = (row[11], row[12])

        away_streak = None
        if row[13] is not None and row[14]:  # away_streak_length, away_streak_type
            away_streak = (row[13], row[14])

        game_rec = GameRecommendation(
            sport=row[0],
            game_time=row[4] or 'TBD',
            home_team=row[2],
            away_team=row[3],
            spread=row[5],
            spread_source=row[6],
            recommendations=bet_recs,
            home_tier=row[7],
            away_tier=row[8],
            home_ats_rating=row[9],
            away_ats_rating=row[10],
            home_streak=home_streak,
            away_streak=away_streak
        )
        recommendations.append(game_rec)

    # Sort: games with recommendations first, then by game time
    recommendations.sort(key=lambda g: (
        0 if g.recommendations else 1,
        g.game_time
    ))

    return recommendations


def write_todays_recommendations(
    engine,
    recommendations: List[GameRecommendation],
    game_date,
    computed_at: datetime
) -> int:
    """
    Write today's recommendations to the database cache table.

    Called by the generate_current_rankings Lambda function.

    Args:
        engine: SQLAlchemy engine
        recommendations: List of GameRecommendation objects
        game_date: Date of the games
        computed_at: Timestamp when computed

    Returns:
        Number of rows written
    """
    import json
    from sqlalchemy import text

    if not recommendations:
        return 0

    # Clear existing recommendations for this date first
    delete_sql = text("DELETE FROM todays_recommendations WHERE game_date = :game_date")

    upsert_sql = text("""
        INSERT INTO todays_recommendations
        (sport, game_date, home_team, away_team, game_time,
         spread, spread_source, home_tier, away_tier,
         home_ats_rating, away_ats_rating,
         home_streak_length, home_streak_type,
         away_streak_length, away_streak_type,
         recommendations_json, computed_at)
        VALUES (:sport, :game_date, :home_team, :away_team, :game_time,
                :spread, :spread_source, :home_tier, :away_tier,
                :home_ats_rating, :away_ats_rating,
                :home_streak_length, :home_streak_type,
                :away_streak_length, :away_streak_type,
                :recommendations_json, :computed_at)
        ON CONFLICT (sport, game_date, home_team, away_team)
        DO UPDATE SET
            game_time = EXCLUDED.game_time,
            spread = EXCLUDED.spread,
            spread_source = EXCLUDED.spread_source,
            home_tier = EXCLUDED.home_tier,
            away_tier = EXCLUDED.away_tier,
            home_ats_rating = EXCLUDED.home_ats_rating,
            away_ats_rating = EXCLUDED.away_ats_rating,
            home_streak_length = EXCLUDED.home_streak_length,
            home_streak_type = EXCLUDED.home_streak_type,
            away_streak_length = EXCLUDED.away_streak_length,
            away_streak_type = EXCLUDED.away_streak_type,
            recommendations_json = EXCLUDED.recommendations_json,
            computed_at = EXCLUDED.computed_at
    """)

    rows = []
    for rec in recommendations:
        # Serialize recommendations to JSON
        recs_json = json.dumps([r.to_dict() for r in rec.recommendations]) if rec.recommendations else None

        rows.append({
            'sport': rec.sport,
            'game_date': game_date,
            'home_team': rec.home_team,
            'away_team': rec.away_team,
            'game_time': rec.game_time,
            'spread': rec.spread,
            'spread_source': rec.spread_source,
            'home_tier': rec.home_tier,
            'away_tier': rec.away_tier,
            'home_ats_rating': rec.home_ats_rating,
            'away_ats_rating': rec.away_ats_rating,
            'home_streak_length': rec.home_streak[0] if rec.home_streak else None,
            'home_streak_type': rec.home_streak[1] if rec.home_streak else None,
            'away_streak_length': rec.away_streak[0] if rec.away_streak else None,
            'away_streak_type': rec.away_streak[1] if rec.away_streak else None,
            'recommendations_json': recs_json,
            'computed_at': computed_at,
        })

    with engine.begin() as conn:
        conn.execute(delete_sql, {'game_date': game_date})
        if rows:
            conn.execute(upsert_sql, rows)

    return len(rows)
