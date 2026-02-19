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
from sqlalchemy.engine import Connection
import os

try:
    import pandas as pd
except ImportError:
    pd = None  # pandas not required in Lambda — stdlib datetime used instead

from .insights import (
    InsightPattern,
    get_cached_streaks,
    get_cached_ou_streaks,
    get_cached_tt_streaks,
)
from .tier_matchups import get_tier
from .network_ratings import get_cached_rankings


# =============================================================================
# Configuration
# =============================================================================

HANDICAP_TIERS = {
    'ats': [
        {'name': 'h0',  'label': 'H=0',  'handicap': 0},
        {'name': 'h2',  'label': 'H=2',  'handicap': 2},
        {'name': 'h5',  'label': 'H=5',  'handicap': 5},
        {'name': 'h8',  'label': 'H=8',  'handicap': 8},
        {'name': 'h10', 'label': 'H=10', 'handicap': 10},
    ],
    'ou': [
        {'name': 'h0',  'label': 'H=0',  'handicap': 0},
        {'name': 'h3',  'label': 'H=3',  'handicap': 3},
        {'name': 'h10', 'label': 'H=10', 'handicap': 10},
        {'name': 'h14', 'label': 'H=14', 'handicap': 14},
    ],
    'tt': [
        {'name': 'h0',  'label': 'H=0',  'handicap': 0},
        {'name': 'h2',  'label': 'H=2',  'handicap': 2},
        {'name': 'h5',  'label': 'H=5',  'handicap': 5},
        {'name': 'h8',  'label': 'H=8',  'handicap': 8},
        {'name': 'h10', 'label': 'H=10', 'handicap': 10},
    ],
}

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
    source: str              # 'streak', 'ou_streak', or 'tt_streak'
    edge: float              # Expected edge vs baseline
    confidence: str          # 'high', 'medium', 'low'
    rationale: str           # Human-readable explanation
    handicap: int            # Handicap level for the recommendation
    tier_data: Optional[List[Dict]] = None  # Coverage stats at each handicap tier

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
    home_streak: Optional[Tuple[int, str]]  # (length, type) - ATS streak
    away_streak: Optional[Tuple[int, str]]
    # Totals info
    total: Optional[float] = None            # Game total line
    total_source: Optional[str] = None       # Bookmaker name for total
    home_team_total: Optional[float] = None  # Home team total line
    away_team_total: Optional[float] = None  # Away team total line
    # O/U and TT streaks
    home_ou_streak: Optional[Tuple[int, str]] = None
    away_ou_streak: Optional[Tuple[int, str]] = None
    home_tt_streak: Optional[Tuple[int, str]] = None
    away_tt_streak: Optional[Tuple[int, str]] = None

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
        import requests
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        all_games = response.json()

        # Filter to today's games only
        today_games = []
        for game in all_games:
            commence_time = game.get('commence_time')
            if commence_time:
                try:
                    event_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
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
        event_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        event_time_est = event_time.astimezone(est)
        return event_time_est.strftime('%I:%M %p EST')
    except Exception:
        return 'TBD'


# =============================================================================
# Database-Backed Game Fetching
# =============================================================================

def get_todays_games_from_db(conn: Connection, sport: str) -> List[Dict]:
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
        SELECT sport, game_date, commence_time, home_team, away_team,
               spread, spread_source, updated_at,
               total, total_source, home_team_total, away_team_total
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
                'updated_at': row[7],
                'total': row[8],
                'total_source': row[9],
                'home_team_total': row[10],
                'away_team_total': row[11],
            })
        return games
    except Exception as e:
        # Rollback on error to reset connection state
        try:
            conn.rollback()
        except Exception:
            pass
        return []


def get_games_last_updated(conn: Connection) -> Optional[datetime]:
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


def get_closing_lines(conn, sport: str = None) -> Dict[Tuple[str, str, str], Dict]:
    """
    Fetch closing lines from todays_games for games where closing data has been captured.

    Returns dict keyed by (sport, home_team, away_team) for O(1) lookup per game card.
    No Streamlit caching — we want the freshest data on every page load.
    """
    from sqlalchemy import text

    est = timezone(timedelta(hours=-5))
    today = datetime.now(est).date()

    sport_filter = "AND sport = :sport" if sport and sport != "All" else ""
    query = text(f"""
        SELECT sport, home_team, away_team,
               closing_spread, closing_total, closing_home_tt, closing_away_tt
        FROM todays_games
        WHERE game_date = :game_date
          AND closing_captured_at IS NOT NULL
          {sport_filter}
    """)

    params = {'game_date': today}
    if sport and sport != "All":
        params['sport'] = sport

    try:
        result = conn.execute(query, params)
        rows = result.fetchall()
        return {
            (row[0], row[1], row[2]): {
                'closing_spread': row[3],
                'closing_total': row[4],
                'closing_home_tt': row[5],
                'closing_away_tt': row[6],
            }
            for row in rows
        }
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return {}


# =============================================================================
# Team State Functions
# =============================================================================

def get_team_tier_lookup(conn: Connection, sport: str) -> Dict[str, Tuple[str, float]]:
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


def get_team_streak_lookup(conn: Connection, sport: str) -> Dict[str, Tuple[int, str]]:
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

def _build_tier_data(pattern: InsightPattern) -> Optional[List[Dict]]:
    """Build tier-level coverage stats from a pattern's coverage profile.

    All markets store both directions in the profile:
    - ATS: team_cover_rate / opp_cover_rate
    - O/U: over_cover_rate / under_cover_rate
    - TT:  over_cover_rate / under_cover_rate

    We pick the correct bet direction directly — no flipping needed.
    At H>0, "team covers" and "opponent covers" (or OVER/UNDER) are NOT
    complementary due to overlap zones, so 1-rate would be wrong.
    """
    if not pattern.coverage_profile:
        return None
    market = pattern.market_type or 'ats'
    is_fade = pattern.pattern_type == 'streak_fade'
    tiers = HANDICAP_TIERS.get(market, HANDICAP_TIERS['ats'])
    result = []
    for tier in tiers:
        h = tier['handicap']
        entry = pattern.coverage_profile.get(h)
        if not entry:
            continue

        if market in ('ou', 'tt') and 'over_cover_rate' in entry:
            # O/U and TT: pick over or under based on bet direction
            if is_fade:
                bet_dir = 'under' if pattern.streak_type == 'OVER' else 'over'
            else:
                bet_dir = 'over' if pattern.streak_type == 'OVER' else 'under'

            cover_rate = entry[f'{bet_dir}_cover_rate']
            baseline_rate = entry[f'{bet_dir}_baseline']
            result.append({
                **tier,
                'cover_rate': cover_rate,
                'baseline_rate': baseline_rate,
                'edge': round(cover_rate - baseline_rate, 4),
                'sample_size': entry['sample_size'],
            })
        elif market == 'ats' and 'team_cover_rate' in entry:
            # ATS: pick team or opponent based on ride/fade
            if is_fade:
                cover_rate = entry['opp_cover_rate']
                baseline_rate = entry['opp_baseline']
            else:
                cover_rate = entry['team_cover_rate']
                baseline_rate = entry['team_baseline']

            result.append({
                **tier,
                'cover_rate': cover_rate,
                'baseline_rate': baseline_rate,
                'edge': round(cover_rate - baseline_rate, 4),
                'sample_size': entry['sample_size'],
            })
        else:
            # Legacy fallback (old profiles without directional fields)
            if is_fade:
                result.append({
                    **tier,
                    'cover_rate': round(1 - entry['cover_rate'], 4),
                    'baseline_rate': round(1 - entry['baseline_rate'], 4),
                    'edge': round(entry['baseline_rate'] - entry['cover_rate'], 4),
                    'sample_size': entry['sample_size'],
                })
            else:
                result.append({**tier, **entry})
    return result if result else None


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
        tier_data = _build_tier_data(pattern)
        if pattern.pattern_type == 'streak_ride':
            # Bet ON this team
            rec = BetRecommendation(
                bet_on=team,
                source='streak',
                edge=abs(pattern.edge),
                confidence=pattern.confidence,
                rationale=f"{streak_length}-game ATS {streak_type.lower()} streak -> RIDE ({abs(pattern.edge):.1%} edge)",
                handicap=pattern.handicap,
                tier_data=tier_data,
            )
        else:
            # streak_fade: bet AGAINST this team by betting ON their opponent
            rec = BetRecommendation(
                bet_on=opponent,
                source='streak',
                edge=abs(pattern.edge),
                confidence=pattern.confidence,
                rationale=f"Opponent on {streak_length}-game ATS {streak_type.lower()} streak -> FADE ({abs(pattern.edge):.1%} edge)",
                handicap=pattern.handicap,
                tier_data=tier_data,
            )

        recommendations.append(rec)
        break  # Only one streak recommendation per team

    return recommendations



def match_ou_streak_patterns(
    team: str,
    opponent: str,
    ou_streak: Tuple[int, str],
    patterns: List[InsightPattern],
    sport: str
) -> List[BetRecommendation]:
    """
    Check if a team's O/U streak matches any profitable O/U patterns.

    Args:
        team: Team name (whose streak we're checking)
        opponent: Opponent team name
        ou_streak: (streak_length, streak_type) where type is 'OVER'|'UNDER'
        patterns: List of detected O/U patterns (market_type='ou')
        sport: Sport key

    Returns:
        List of BetRecommendation for O/U bets
    """
    recommendations = []
    streak_length, streak_type = ou_streak

    for pattern in patterns:
        if pattern.sport != sport:
            continue
        if pattern.market_type != 'ou':
            continue
        if pattern.streak_type != streak_type:
            continue
        if streak_length < pattern.streak_length:
            continue

        # Determine OVER or UNDER recommendation
        if pattern.pattern_type == 'streak_ride':
            # Ride = bet same direction as streak
            bet_direction = streak_type  # OVER or UNDER
        else:
            # Fade = bet opposite direction
            bet_direction = 'UNDER' if streak_type == 'OVER' else 'OVER'

        rec = BetRecommendation(
            bet_on=f"Game Total {bet_direction}",
            source='ou_streak',
            edge=abs(pattern.edge),
            confidence=pattern.confidence,
            rationale=f"{team} on {streak_length}-game O/U {streak_type.lower()} streak -> {bet_direction} ({abs(pattern.edge):.1%} edge)",
            handicap=pattern.handicap,
            tier_data=_build_tier_data(pattern),
        )

        recommendations.append(rec)
        break  # Only one O/U recommendation per team's streak

    return recommendations


def match_tt_streak_patterns(
    team: str,
    tt_streak: Tuple[int, str],
    patterns: List[InsightPattern],
    sport: str
) -> List[BetRecommendation]:
    """
    Check if a team's TT streak matches any profitable team total patterns.

    Args:
        team: Team name (whose streak we're checking)
        tt_streak: (streak_length, streak_type) where type is 'OVER'|'UNDER'
        patterns: List of detected TT patterns (market_type='tt')
        sport: Sport key

    Returns:
        List of BetRecommendation for team total bets
    """
    recommendations = []
    streak_length, streak_type = tt_streak

    for pattern in patterns:
        if pattern.sport != sport:
            continue
        if pattern.market_type != 'tt':
            continue
        if pattern.streak_type != streak_type:
            continue
        if streak_length < pattern.streak_length:
            continue

        # Determine OVER or UNDER recommendation
        if pattern.pattern_type == 'streak_ride':
            bet_direction = streak_type
        else:
            bet_direction = 'UNDER' if streak_type == 'OVER' else 'OVER'

        rec = BetRecommendation(
            bet_on=f"{team} TT {bet_direction}",
            source='tt_streak',
            edge=abs(pattern.edge),
            confidence=pattern.confidence,
            rationale=f"{team} on {streak_length}-game TT {streak_type.lower()} streak -> TT {bet_direction} ({abs(pattern.edge):.1%} edge)",
            handicap=pattern.handicap,
            tier_data=_build_tier_data(pattern),
        )

        recommendations.append(rec)
        break  # Only one TT recommendation per team's streak

    return recommendations


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_recommendations(
    conn: Connection,
    sport: str,
    streak_patterns: List[InsightPattern],
    ou_patterns: List[InsightPattern] = None,
    tt_patterns: List[InsightPattern] = None,
) -> List[GameRecommendation]:
    """
    Generate betting recommendations for today's games.

    Combines ATS streak, O/U streak, and TT streak insights.
    Games are fetched from the database (pre-collected by Lambda).

    Args:
        conn: Database connection
        sport: Sport to analyze ('NFL', 'NBA', 'NCAAM', or 'All')
        streak_patterns: Detected ATS streak patterns
        ou_patterns: Detected O/U streak patterns (optional)
        tt_patterns: Detected TT streak patterns (optional)

    Returns:
        List of GameRecommendation sorted by game time
    """
    if ou_patterns is None:
        ou_patterns = []
    if tt_patterns is None:
        tt_patterns = []

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

        # Build O/U and TT streak lookups
        ou_streak_lookup = {}
        tt_streak_lookup = {}
        if ou_patterns:
            ou_streaks = get_cached_ou_streaks(conn, sp)
            ou_streak_lookup = {
                team: (info['streak_length'], info['streak_type'])
                for team, info in ou_streaks.items()
            }
        if tt_patterns:
            tt_streaks = get_cached_tt_streaks(conn, sp)
            tt_streak_lookup = {
                team: (info['streak_length'], info['streak_type'])
                for team, info in tt_streaks.items()
            }

        for game in games:
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')

            # Get spread (already stored in DB from Lambda)
            spread = game.get('spread')
            spread_source = game.get('spread_source')

            # Get totals info
            total = game.get('total')
            total_source = game.get('total_source')
            home_team_total = game.get('home_team_total')
            away_team_total = game.get('away_team_total')

            # Get team info
            home_tier_info = tier_lookup.get(home_team)
            away_tier_info = tier_lookup.get(away_team)
            home_streak = streak_lookup.get(home_team)
            away_streak = streak_lookup.get(away_team)

            # Get O/U and TT streaks
            home_ou_streak = ou_streak_lookup.get(home_team)
            away_ou_streak = ou_streak_lookup.get(away_team)
            home_tt_streak = tt_streak_lookup.get(home_team)
            away_tt_streak = tt_streak_lookup.get(away_team)

            recs = []

            # Check ATS streak patterns for both teams
            if home_streak:
                recs.extend(match_streak_patterns(
                    home_team, away_team, home_streak, streak_patterns, sp
                ))
            if away_streak:
                recs.extend(match_streak_patterns(
                    away_team, home_team, away_streak, streak_patterns, sp
                ))

            # Check O/U streak patterns for both teams
            if home_ou_streak and ou_patterns:
                recs.extend(match_ou_streak_patterns(
                    home_team, away_team, home_ou_streak, ou_patterns, sp
                ))
            if away_ou_streak and ou_patterns:
                recs.extend(match_ou_streak_patterns(
                    away_team, home_team, away_ou_streak, ou_patterns, sp
                ))

            # Check TT streak patterns for both teams
            if home_tt_streak and tt_patterns:
                recs.extend(match_tt_streak_patterns(
                    home_team, home_tt_streak, tt_patterns, sp
                ))
            if away_tt_streak and tt_patterns:
                recs.extend(match_tt_streak_patterns(
                    away_team, away_tt_streak, tt_patterns, sp
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
                away_streak=away_streak,
                total=total,
                total_source=total_source,
                home_team_total=home_team_total,
                away_team_total=away_team_total,
                home_ou_streak=home_ou_streak,
                away_ou_streak=away_ou_streak,
                home_tt_streak=home_tt_streak,
                away_tt_streak=away_tt_streak,
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
    that was populated by the daily-precompute Lambda at 6:20 AM.

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

    # Build query - use column names for clarity
    select_cols = """
            SELECT sport, game_date, home_team, away_team, game_time,
                   spread, spread_source, home_tier, away_tier,
                   home_ats_rating, away_ats_rating,
                   home_streak_length, home_streak_type,
                   away_streak_length, away_streak_type,
                   recommendations_json,
                   total, total_source, home_team_total, away_team_total,
                   home_ou_streak_length, home_ou_streak_type,
                   away_ou_streak_length, away_ou_streak_type,
                   home_tt_streak_length, home_tt_streak_type,
                   away_tt_streak_length, away_tt_streak_type
    """

    if sport and sport != 'All':
        query = text(f"""
            {select_cols}
            FROM todays_recommendations
            WHERE game_date = :game_date AND sport = :sport
            ORDER BY game_time
        """)
        params = {'game_date': today, 'sport': sport}
    else:
        query = text(f"""
            {select_cols}
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
        from .insights import get_cached_patterns, get_cached_ou_patterns, get_cached_tt_patterns
        patterns = get_cached_patterns(conn, min_sample=30, min_edge=0.05)
        ou_pats = get_cached_ou_patterns(conn, min_sample=30, min_edge=0.05)
        tt_pats = get_cached_tt_patterns(conn, min_sample=30, min_edge=0.05)
        return generate_recommendations(conn, sport or 'All', patterns,
                                        ou_patterns=ou_pats, tt_patterns=tt_pats)

    # Convert rows to GameRecommendation objects
    # Column indices:
    #  0=sport, 1=game_date, 2=home_team, 3=away_team, 4=game_time,
    #  5=spread, 6=spread_source, 7=home_tier, 8=away_tier,
    #  9=home_ats_rating, 10=away_ats_rating,
    # 11=home_streak_length, 12=home_streak_type,
    # 13=away_streak_length, 14=away_streak_type,
    # 15=recommendations_json,
    # 16=total, 17=total_source, 18=home_team_total, 19=away_team_total,
    # 20=home_ou_streak_length, 21=home_ou_streak_type,
    # 22=away_ou_streak_length, 23=away_ou_streak_type,
    # 24=home_tt_streak_length, 25=home_tt_streak_type,
    # 26=away_tt_streak_length, 27=away_tt_streak_type
    recommendations = []
    for row in rows:
        # Parse recommendations JSON
        recs_json = row[15]
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
                        handicap=r['handicap'],
                        tier_data=r.get('tier_data'),
                    ))
            except (json.JSONDecodeError, KeyError):
                pass

        # Build streak tuples
        home_streak = None
        if row[11] is not None and row[12]:
            home_streak = (row[11], row[12])

        away_streak = None
        if row[13] is not None and row[14]:
            away_streak = (row[13], row[14])

        # Build O/U streak tuples
        home_ou_streak = None
        if row[20] is not None and row[21]:
            home_ou_streak = (row[20], row[21])

        away_ou_streak = None
        if row[22] is not None and row[23]:
            away_ou_streak = (row[22], row[23])

        # Build TT streak tuples
        home_tt_streak = None
        if row[24] is not None and row[25]:
            home_tt_streak = (row[24], row[25])

        away_tt_streak = None
        if row[26] is not None and row[27]:
            away_tt_streak = (row[26], row[27])

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
            away_streak=away_streak,
            total=row[16],
            total_source=row[17],
            home_team_total=row[18],
            away_team_total=row[19],
            home_ou_streak=home_ou_streak,
            away_ou_streak=away_ou_streak,
            home_tt_streak=home_tt_streak,
            away_tt_streak=away_tt_streak,
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

    Called by the daily-precompute Lambda function.

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
         spread, spread_source, total, total_source,
         home_team_total, away_team_total,
         home_tier, away_tier,
         home_ats_rating, away_ats_rating,
         home_streak_length, home_streak_type,
         away_streak_length, away_streak_type,
         home_ou_streak_length, home_ou_streak_type,
         away_ou_streak_length, away_ou_streak_type,
         home_tt_streak_length, home_tt_streak_type,
         away_tt_streak_length, away_tt_streak_type,
         recommendations_json, computed_at)
        VALUES (:sport, :game_date, :home_team, :away_team, :game_time,
                :spread, :spread_source, :total, :total_source,
                :home_team_total, :away_team_total,
                :home_tier, :away_tier,
                :home_ats_rating, :away_ats_rating,
                :home_streak_length, :home_streak_type,
                :away_streak_length, :away_streak_type,
                :home_ou_streak_length, :home_ou_streak_type,
                :away_ou_streak_length, :away_ou_streak_type,
                :home_tt_streak_length, :home_tt_streak_type,
                :away_tt_streak_length, :away_tt_streak_type,
                :recommendations_json, :computed_at)
        ON CONFLICT (sport, game_date, home_team, away_team)
        DO UPDATE SET
            game_time = EXCLUDED.game_time,
            spread = EXCLUDED.spread,
            spread_source = EXCLUDED.spread_source,
            total = EXCLUDED.total,
            total_source = EXCLUDED.total_source,
            home_team_total = EXCLUDED.home_team_total,
            away_team_total = EXCLUDED.away_team_total,
            home_tier = EXCLUDED.home_tier,
            away_tier = EXCLUDED.away_tier,
            home_ats_rating = EXCLUDED.home_ats_rating,
            away_ats_rating = EXCLUDED.away_ats_rating,
            home_streak_length = EXCLUDED.home_streak_length,
            home_streak_type = EXCLUDED.home_streak_type,
            away_streak_length = EXCLUDED.away_streak_length,
            away_streak_type = EXCLUDED.away_streak_type,
            home_ou_streak_length = EXCLUDED.home_ou_streak_length,
            home_ou_streak_type = EXCLUDED.home_ou_streak_type,
            away_ou_streak_length = EXCLUDED.away_ou_streak_length,
            away_ou_streak_type = EXCLUDED.away_ou_streak_type,
            home_tt_streak_length = EXCLUDED.home_tt_streak_length,
            home_tt_streak_type = EXCLUDED.home_tt_streak_type,
            away_tt_streak_length = EXCLUDED.away_tt_streak_length,
            away_tt_streak_type = EXCLUDED.away_tt_streak_type,
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
            'total': rec.total,
            'total_source': rec.total_source,
            'home_team_total': rec.home_team_total,
            'away_team_total': rec.away_team_total,
            'home_tier': rec.home_tier,
            'away_tier': rec.away_tier,
            'home_ats_rating': rec.home_ats_rating,
            'away_ats_rating': rec.away_ats_rating,
            'home_streak_length': rec.home_streak[0] if rec.home_streak else None,
            'home_streak_type': rec.home_streak[1] if rec.home_streak else None,
            'away_streak_length': rec.away_streak[0] if rec.away_streak else None,
            'away_streak_type': rec.away_streak[1] if rec.away_streak else None,
            'home_ou_streak_length': rec.home_ou_streak[0] if rec.home_ou_streak else None,
            'home_ou_streak_type': rec.home_ou_streak[1] if rec.home_ou_streak else None,
            'away_ou_streak_length': rec.away_ou_streak[0] if rec.away_ou_streak else None,
            'away_ou_streak_type': rec.away_ou_streak[1] if rec.away_ou_streak else None,
            'home_tt_streak_length': rec.home_tt_streak[0] if rec.home_tt_streak else None,
            'home_tt_streak_type': rec.home_tt_streak[1] if rec.home_tt_streak else None,
            'away_tt_streak_length': rec.away_tt_streak[0] if rec.away_tt_streak else None,
            'away_tt_streak_type': rec.away_tt_streak[1] if rec.away_tt_streak else None,
            'recommendations_json': recs_json,
            'computed_at': computed_at,
        })

    with engine.begin() as conn:
        conn.execute(delete_sql, {'game_date': game_date})
        if rows:
            conn.execute(upsert_sql, rows)

    return len(rows)
