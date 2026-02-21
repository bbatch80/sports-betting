"""
Backfill predictions for past dates by reconstructing exact streaks.

For each target date:
1. Query the games table for games played ON that date (these are the "today's games")
2. Query all games BEFORE that date (to compute exact streaks as of that morning)
3. Compute ATS, O/U, and TT streaks for every team involved
4. Match against current detected_patterns to generate recommendations
5. Write to prediction_results via write_prediction_tracking()
6. Immediately resolve outcomes since the games have already been played

Usage:
    python3 scripts/backfill_predictions.py                    # defaults: Feb 14-16
    python3 scripts/backfill_predictions.py 2025-02-14         # single date
    python3 scripts/backfill_predictions.py 2025-02-14 2025-02-16  # date range
"""

import sys
import os
import logging
from datetime import date, datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from src.db.engine import get_engine
from src.analysis.insights import (
    InsightPattern,
    get_cached_patterns,
    get_cached_ou_patterns,
    get_cached_tt_patterns,
)
from src.analysis.todays_recommendations import (
    BetRecommendation,
    GameRecommendation,
    match_streak_patterns,
    match_ou_streak_patterns,
    match_tt_streak_patterns,
    get_team_tier_lookup,
    format_game_time,
)
from src.analysis.prediction_tracking import (
    write_prediction_tracking,
    resolve_prediction_outcomes,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

SPORTS = ['NFL', 'NBA', 'NCAAM']


# =============================================================================
# Streak Computation from Historical Games
# =============================================================================

def compute_ats_streak(games: List[Dict], team: str) -> Tuple[int, str]:
    """Compute ATS streak for a team from historical game dicts."""
    team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]
    if not team_games:
        return (0, 'WIN')

    team_games.sort(key=lambda g: g['game_date'], reverse=True)

    streak_length = 0
    streak_type = None
    for game in team_games:
        is_home = game['home_team'] == team
        spread_result = game.get('spread_result')
        if spread_result == 0:
            continue  # Push — skip, doesn't break streak
        if spread_result is None:
            break  # Missing data — streak cannot continue through gap
        covered = (spread_result > 0) if is_home else (spread_result < 0)
        if streak_type is None:
            streak_type = 'WIN' if covered else 'LOSS'
            streak_length = 1
        elif (streak_type == 'WIN') == covered:
            streak_length += 1
        else:
            break

    return (streak_length, streak_type or 'WIN')


def compute_ou_streak(games: List[Dict], team: str) -> Tuple[int, str]:
    """Compute O/U streak for a team from historical game dicts."""
    team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]
    if not team_games:
        return (0, 'OVER')

    team_games.sort(key=lambda g: g['game_date'], reverse=True)

    streak_length = 0
    streak_type = None
    for game in team_games:
        total_result = game.get('total_result')
        if total_result == 0:
            continue  # Push — skip, doesn't break streak
        if total_result is None:
            break  # Missing data — streak cannot continue through gap
        is_over = total_result > 0
        if streak_type is None:
            streak_type = 'OVER' if is_over else 'UNDER'
            streak_length = 1
        elif (streak_type == 'OVER') == is_over:
            streak_length += 1
        else:
            break

    return (streak_length, streak_type or 'OVER')


def compute_tt_streak(games: List[Dict], team: str) -> Tuple[int, str]:
    """Compute team total O/U streak for a team from historical game dicts."""
    team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]
    if not team_games:
        return (0, 'OVER')

    team_games.sort(key=lambda g: g['game_date'], reverse=True)

    streak_length = 0
    streak_type = None
    for game in team_games:
        is_home = game['home_team'] == team
        margin = game.get('home_team_total_result') if is_home else game.get('away_team_total_result')
        if margin == 0:
            continue  # Push — skip, doesn't break streak
        if margin is None:
            break  # Missing data — streak cannot continue through gap
        is_over = margin > 0
        if streak_type is None:
            streak_type = 'OVER' if is_over else 'UNDER'
            streak_length = 1
        elif (streak_type == 'OVER') == is_over:
            streak_length += 1
        else:
            break

    return (streak_length, streak_type or 'OVER')


# =============================================================================
# Main Backfill Logic
# =============================================================================

def backfill_date(engine, target_date: date, patterns, ou_patterns, tt_patterns) -> int:
    """
    Reconstruct predictions for a single past date and resolve outcomes.

    Returns number of predictions written.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Backfilling predictions for {target_date}")
    logger.info(f"{'='*60}")

    # Step 1: Get games that were played on the target date
    games_on_date_sql = text("""
        SELECT sport, game_date, home_team, away_team,
               closing_spread, closing_total, home_team_total, away_team_total,
               home_score, away_score,
               spread_result, total_result,
               home_team_total_result, away_team_total_result
        FROM games
        WHERE game_date = :target_date
        ORDER BY sport, home_team
    """)

    # Step 2: Get all games BEFORE the target date (for streak computation)
    history_sql = text("""
        SELECT sport, game_date, home_team, away_team,
               closing_spread, closing_total, home_team_total, away_team_total,
               home_score, away_score,
               spread_result, total_result,
               home_team_total_result, away_team_total_result
        FROM games
        WHERE game_date < :target_date
        ORDER BY game_date DESC
    """)

    with engine.connect() as conn:
        target_games = conn.execute(games_on_date_sql, {'target_date': target_date}).fetchall()
        history_rows = conn.execute(history_sql, {'target_date': target_date}).fetchall()

    if not target_games:
        logger.info(f"  No games found for {target_date}")
        return 0

    # Convert to dicts
    columns = ['sport', 'game_date', 'home_team', 'away_team',
               'closing_spread', 'closing_total', 'home_team_total', 'away_team_total',
               'home_score', 'away_score',
               'spread_result', 'total_result',
               'home_team_total_result', 'away_team_total_result']

    target_game_dicts = [dict(zip(columns, row)) for row in target_games]
    history_dicts = [dict(zip(columns, row)) for row in history_rows]

    logger.info(f"  {len(target_game_dicts)} games on {target_date}, {len(history_dicts)} historical games for streaks")

    # Group target games and history by sport
    games_by_sport = {}
    history_by_sport = {}
    for g in target_game_dicts:
        games_by_sport.setdefault(g['sport'], []).append(g)
    for g in history_dicts:
        history_by_sport.setdefault(g['sport'], []).append(g)

    # Step 3: For each sport, compute streaks and match patterns
    all_recommendations = []

    for sport in SPORTS:
        sport_games = games_by_sport.get(sport, [])
        if not sport_games:
            continue

        sport_history = history_by_sport.get(sport, [])
        logger.info(f"  {sport}: {len(sport_games)} games, {len(sport_history)} history games")

        # Get all unique teams playing on this date
        teams = set()
        for g in sport_games:
            teams.add(g['home_team'])
            teams.add(g['away_team'])

        # Compute streaks for each team as of the target date morning
        ats_streaks = {}
        ou_streaks = {}
        tt_streaks = {}
        for team in teams:
            ats_streaks[team] = compute_ats_streak(sport_history, team)
            ou_streaks[team] = compute_ou_streak(sport_history, team)
            tt_streaks[team] = compute_tt_streak(sport_history, team)

        # Get tier lookups (using current ratings as approximation)
        with engine.connect() as conn:
            tier_lookup = get_team_tier_lookup(conn, sport)

        # Match each game against patterns
        for game in sport_games:
            home_team = game['home_team']
            away_team = game['away_team']

            home_ats = ats_streaks.get(home_team, (0, 'WIN'))
            away_ats = ats_streaks.get(away_team, (0, 'WIN'))
            home_ou = ou_streaks.get(home_team, (0, 'OVER'))
            away_ou = ou_streaks.get(away_team, (0, 'OVER'))
            home_tt = tt_streaks.get(home_team, (0, 'OVER'))
            away_tt = tt_streaks.get(away_team, (0, 'OVER'))

            recs = []

            # ATS streak matching
            if home_ats[0] > 0:
                recs.extend(match_streak_patterns(home_team, away_team, home_ats, patterns, sport))
            if away_ats[0] > 0:
                recs.extend(match_streak_patterns(away_team, home_team, away_ats, patterns, sport))

            # O/U streak matching
            if home_ou[0] > 0 and ou_patterns:
                recs.extend(match_ou_streak_patterns(home_team, away_team, home_ou, ou_patterns, sport))
            if away_ou[0] > 0 and ou_patterns:
                recs.extend(match_ou_streak_patterns(away_team, home_team, away_ou, ou_patterns, sport))

            # TT streak matching
            if home_tt[0] > 0 and tt_patterns:
                recs.extend(match_tt_streak_patterns(home_team, home_tt, tt_patterns, sport))
            if away_tt[0] > 0 and tt_patterns:
                recs.extend(match_tt_streak_patterns(away_team, away_tt, tt_patterns, sport))

            if not recs:
                continue

            # Build GameRecommendation
            home_tier_info = tier_lookup.get(home_team)
            away_tier_info = tier_lookup.get(away_team)

            game_rec = GameRecommendation(
                sport=sport,
                game_time='TBD',  # We don't have commence_time for historical games
                home_team=home_team,
                away_team=away_team,
                spread=game.get('closing_spread'),
                spread_source='Closing',
                recommendations=recs,
                home_tier=home_tier_info[0] if home_tier_info else None,
                away_tier=away_tier_info[0] if away_tier_info else None,
                home_ats_rating=home_tier_info[1] if home_tier_info else None,
                away_ats_rating=away_tier_info[1] if away_tier_info else None,
                home_streak=home_ats,
                away_streak=away_ats,
                total=game.get('closing_total'),
                total_source='Closing',
                home_team_total=game.get('home_team_total'),
                away_team_total=game.get('away_team_total'),
                home_ou_streak=home_ou,
                away_ou_streak=away_ou,
                home_tt_streak=home_tt,
                away_tt_streak=away_tt,
            )
            all_recommendations.append(game_rec)

            for rec in recs:
                logger.info(f"    {away_team} @ {home_team}: {rec.source} -> {rec.bet_on} ({rec.rationale})")

    if not all_recommendations:
        logger.info(f"  No pattern matches found for {target_date}")
        return 0

    # Step 4: Write predictions
    computed_at = datetime(target_date.year, target_date.month, target_date.day,
                          11, 30, 0, tzinfo=timezone.utc)  # Simulate 6:30 AM EST
    tracked = write_prediction_tracking(engine, all_recommendations, target_date, computed_at)
    logger.info(f"  Wrote {tracked} predictions for {target_date}")

    # Step 5: Immediately resolve outcomes
    resolved = resolve_prediction_outcomes(engine, target_date)
    logger.info(f"  Resolved {resolved} predictions for {target_date}")

    return tracked


def main():
    # Parse arguments
    if len(sys.argv) == 1:
        # Default: Feb 14-16, 2026
        target_dates = [date(2026, 2, 14), date(2026, 2, 15), date(2026, 2, 16)]
    elif len(sys.argv) == 2:
        target_dates = [date.fromisoformat(sys.argv[1])]
    elif len(sys.argv) == 3:
        start = date.fromisoformat(sys.argv[1])
        end = date.fromisoformat(sys.argv[2])
        target_dates = []
        d = start
        while d <= end:
            target_dates.append(d)
            d += timedelta(days=1)
    else:
        print("Usage: python3 backfill_predictions.py [start_date] [end_date]")
        sys.exit(1)

    logger.info(f"Backfilling predictions for {len(target_dates)} dates: {target_dates}")

    engine = get_engine()
    if engine is None:
        logger.error("Could not connect to database")
        sys.exit(1)

    # Load current detected patterns (best approximation for past dates)
    with engine.connect() as conn:
        patterns = get_cached_patterns(conn, min_sample=30, min_edge=0.05)
        ou_patterns = get_cached_ou_patterns(conn, min_sample=30, min_edge=0.05)
        tt_patterns = get_cached_tt_patterns(conn, min_sample=30, min_edge=0.05)

    logger.info(f"Loaded {len(patterns)} ATS, {len(ou_patterns)} O/U, {len(tt_patterns)} TT patterns")

    total_predictions = 0
    for target_date in target_dates:
        count = backfill_date(engine, target_date, patterns, ou_patterns, tt_patterns)
        total_predictions += count

    logger.info(f"\nDone! Total predictions backfilled: {total_predictions}")


if __name__ == '__main__':
    main()
