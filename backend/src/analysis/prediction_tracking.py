"""
Prediction Result Tracking — capture and resolution.

Captures each individual BetRecommendation at prediction time with full context,
then resolves outcomes after game results are available.

Functions:
    write_prediction_tracking() — called by daily_precompute Lambda after recs are generated
    resolve_prediction_outcomes() — called by collect_yesterday_games Lambda + CLI script
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


# =============================================================================
# Derived Column Helpers
# =============================================================================

def _spread_bucket(spread: Optional[float]) -> Optional[str]:
    """Categorize absolute spread into buckets for analysis."""
    if spread is None:
        return None
    abs_spread = abs(spread)
    if abs_spread <= 1:
        return "PK-1"
    elif abs_spread <= 3.5:
        return "1.5-3.5"
    elif abs_spread <= 7:
        return "4-7"
    elif abs_spread <= 10:
        return "7.5-10"
    else:
        return "10+"


def _tier_matchup(home_tier: Optional[str], away_tier: Optional[str]) -> Optional[str]:
    """Create tier matchup label like 'Elite vs Weak'."""
    if home_tier and away_tier:
        return f"{home_tier} vs {away_tier}"
    return None


def _market_type_from_source(source: str) -> str:
    """Map source field to market_type."""
    mapping = {'streak': 'ats', 'ou_streak': 'ou', 'tt_streak': 'tt'}
    return mapping.get(source, source)


def _pattern_type_from_rationale(rationale: Optional[str]) -> Optional[str]:
    """Extract pattern type (ride/fade) from rationale text."""
    if not rationale:
        return None
    rationale_upper = rationale.upper()
    if "RIDE" in rationale_upper:
        return "streak_ride"
    elif "FADE" in rationale_upper:
        return "streak_fade"
    return None


def _bet_is_home(bet_on: str, source: str, home_team: str) -> Optional[int]:
    """Determine if bet is on home team. NULL for totals bets."""
    if source != 'streak':
        return None  # O/U and TT bets aren't home/away
    return 1 if bet_on == home_team else 0


# =============================================================================
# Capture Function
# =============================================================================

def write_prediction_tracking(engine, recommendations, game_date, computed_at: datetime) -> int:
    """
    Write individual bet recommendations to prediction_results for tracking.

    Called by daily_precompute Lambda after write_todays_recommendations().
    Each BetRecommendation becomes one row (a game with 3 recs = 3 rows).

    Args:
        engine: SQLAlchemy engine
        recommendations: List of GameRecommendation objects
        game_date: Date of the games
        computed_at: Timestamp when recommendations were computed

    Returns:
        Number of rows written
    """
    if not recommendations:
        return 0

    insert_sql = text("""
        INSERT INTO prediction_results
        (sport, game_date, home_team, away_team, game_time,
         spread, total, home_team_total, away_team_total,
         home_tier, away_tier, home_ats_rating, away_ats_rating,
         home_streak_length, home_streak_type,
         away_streak_length, away_streak_type,
         home_ou_streak_length, home_ou_streak_type,
         away_ou_streak_length, away_ou_streak_type,
         home_tt_streak_length, home_tt_streak_type,
         away_tt_streak_length, away_tt_streak_type,
         bet_on, source, edge, confidence, rationale, handicap,
         market_type, pattern_type, cover_rate, baseline_rate, sample_size,
         bet_is_home, tier_matchup, spread_bucket, rating_diff,
         captured_at)
        VALUES (:sport, :game_date, :home_team, :away_team, :game_time,
                :spread, :total, :home_team_total, :away_team_total,
                :home_tier, :away_tier, :home_ats_rating, :away_ats_rating,
                :home_streak_length, :home_streak_type,
                :away_streak_length, :away_streak_type,
                :home_ou_streak_length, :home_ou_streak_type,
                :away_ou_streak_length, :away_ou_streak_type,
                :home_tt_streak_length, :home_tt_streak_type,
                :away_tt_streak_length, :away_tt_streak_type,
                :bet_on, :source, :edge, :confidence, :rationale, :handicap,
                :market_type, :pattern_type, :cover_rate, :baseline_rate, :sample_size,
                :bet_is_home, :tier_matchup, :spread_bucket, :rating_diff,
                :captured_at)
        ON CONFLICT (sport, game_date, home_team, away_team, bet_on, source)
        DO NOTHING
    """)

    # Lookup pattern stats from detected_patterns table
    pattern_lookup = _load_pattern_lookup(engine)

    rows = []
    for game_rec in recommendations:
        for bet_rec in game_rec.recommendations:
            market = _market_type_from_source(bet_rec.source)
            pat_type = _pattern_type_from_rationale(bet_rec.rationale)

            # Look up pattern stats
            cover_rate = None
            baseline_rate = None
            sample_size = None
            if pat_type:
                key = (game_rec.sport, market, pat_type, bet_rec.handicap)
                pat = pattern_lookup.get(key)
                if pat:
                    cover_rate = pat['cover_rate']
                    baseline_rate = pat['baseline_rate']
                    sample_size = pat['sample_size']

            rating_diff = None
            if game_rec.home_ats_rating is not None and game_rec.away_ats_rating is not None:
                rating_diff = round(game_rec.home_ats_rating - game_rec.away_ats_rating, 3)

            rows.append({
                'sport': game_rec.sport,
                'game_date': game_date,
                'home_team': game_rec.home_team,
                'away_team': game_rec.away_team,
                'game_time': game_rec.game_time,
                'spread': game_rec.spread,
                'total': game_rec.total,
                'home_team_total': game_rec.home_team_total,
                'away_team_total': game_rec.away_team_total,
                'home_tier': game_rec.home_tier,
                'away_tier': game_rec.away_tier,
                'home_ats_rating': game_rec.home_ats_rating,
                'away_ats_rating': game_rec.away_ats_rating,
                'home_streak_length': game_rec.home_streak[0] if game_rec.home_streak else None,
                'home_streak_type': game_rec.home_streak[1] if game_rec.home_streak else None,
                'away_streak_length': game_rec.away_streak[0] if game_rec.away_streak else None,
                'away_streak_type': game_rec.away_streak[1] if game_rec.away_streak else None,
                'home_ou_streak_length': game_rec.home_ou_streak[0] if game_rec.home_ou_streak else None,
                'home_ou_streak_type': game_rec.home_ou_streak[1] if game_rec.home_ou_streak else None,
                'away_ou_streak_length': game_rec.away_ou_streak[0] if game_rec.away_ou_streak else None,
                'away_ou_streak_type': game_rec.away_ou_streak[1] if game_rec.away_ou_streak else None,
                'home_tt_streak_length': game_rec.home_tt_streak[0] if game_rec.home_tt_streak else None,
                'home_tt_streak_type': game_rec.home_tt_streak[1] if game_rec.home_tt_streak else None,
                'away_tt_streak_length': game_rec.away_tt_streak[0] if game_rec.away_tt_streak else None,
                'away_tt_streak_type': game_rec.away_tt_streak[1] if game_rec.away_tt_streak else None,
                'bet_on': bet_rec.bet_on,
                'source': bet_rec.source,
                'edge': bet_rec.edge,
                'confidence': bet_rec.confidence,
                'rationale': bet_rec.rationale,
                'handicap': bet_rec.handicap,
                'market_type': market,
                'pattern_type': pat_type,
                'cover_rate': cover_rate,
                'baseline_rate': baseline_rate,
                'sample_size': sample_size,
                'bet_is_home': _bet_is_home(bet_rec.bet_on, bet_rec.source, game_rec.home_team),
                'tier_matchup': _tier_matchup(game_rec.home_tier, game_rec.away_tier),
                'spread_bucket': _spread_bucket(game_rec.spread),
                'rating_diff': rating_diff,
                'captured_at': computed_at,
            })

    if rows:
        with engine.begin() as conn:
            conn.execute(insert_sql, rows)

    return len(rows)


def _load_pattern_lookup(engine) -> dict:
    """Load detected_patterns into a lookup dict keyed by (sport, market_type, pattern_type, handicap)."""
    sql = text("""
        SELECT sport, market_type, pattern_type, handicap,
               cover_rate, baseline_rate, sample_size
        FROM detected_patterns
    """)
    lookup = {}
    with engine.connect() as conn:
        result = conn.execute(sql)
        for row in result:
            key = (row.sport, row.market_type, row.pattern_type, row.handicap)
            lookup[key] = {
                'cover_rate': row.cover_rate,
                'baseline_rate': row.baseline_rate,
                'sample_size': row.sample_size,
            }
    return lookup


# =============================================================================
# Outcome Resolution
# =============================================================================

def determine_outcome(bet_on: str, source: str, handicap: int,
                      home_team: str, game_row: dict) -> dict:
    """
    Determine WIN/LOSS/PUSH for a single prediction against game results.

    The games table stores pre-calculated results against closing lines:
      - spread_result = home_score - away_score - closing_spread
      - total_result = (home_score + away_score) - closing_total
      - home_team_total_result = home_score - home_team_total
      - away_team_total_result = away_score - away_team_total

    Handicap provides additional cushion (higher = easier to cover).

    Args:
        bet_on: The bet identifier (team name, "Game Total OVER", etc.)
        source: 'streak', 'ou_streak', or 'tt_streak'
        handicap: Points of cushion
        home_team: Home team name (to determine bet direction)
        game_row: Dict with game results from the games table

    Returns:
        Dict with 'outcome' ('WIN'/'LOSS'/'PUSH') and 'margin' (float)
    """
    handicap = handicap or 0

    if source == 'streak':
        # ATS bet: bet_on is a plain team name
        spread_result = game_row.get('spread_result')
        if spread_result is None:
            return {'outcome': None, 'margin': None}

        is_home_bet = (bet_on == home_team)
        if is_home_bet:
            # Home bet: spread_result already = home perspective
            margin = spread_result + handicap
        else:
            # Away bet: flip perspective
            margin = -spread_result + handicap

    elif source == 'ou_streak':
        # Game Total O/U: bet_on = "Game Total OVER" or "Game Total UNDER"
        total_result = game_row.get('total_result')
        if total_result is None:
            return {'outcome': None, 'margin': None}

        is_over = "OVER" in bet_on.upper()
        if is_over:
            margin = total_result + handicap
        else:
            margin = -total_result + handicap

    elif source == 'tt_streak':
        # Team Total: bet_on = "Team Name TT OVER" or "Team Name TT UNDER"
        is_over = "OVER" in bet_on.upper()

        # Determine which team's TT result to use
        # bet_on format: "Team Name TT OVER" — extract team by removing " TT OVER"/" TT UNDER"
        bet_team = bet_on.replace(" TT OVER", "").replace(" TT UNDER", "")
        is_home_team = (bet_team == home_team)

        if is_home_team:
            tt_result = game_row.get('home_team_total_result')
        else:
            tt_result = game_row.get('away_team_total_result')

        if tt_result is None:
            return {'outcome': None, 'margin': None}

        if is_over:
            margin = tt_result + handicap
        else:
            margin = -tt_result + handicap
    else:
        return {'outcome': None, 'margin': None}

    margin = round(margin, 1)
    if margin > 0:
        outcome = 'WIN'
    elif margin < 0:
        outcome = 'LOSS'
    else:
        outcome = 'PUSH'

    return {'outcome': outcome, 'margin': margin}


def resolve_prediction_outcomes(engine, target_date, dry_run: bool = False) -> int:
    """
    Resolve unresolved predictions for a given date by matching against game results.

    Idempotent: only processes predictions with outcome IS NULL.
    Safe to re-run — already-resolved predictions are skipped.

    Args:
        engine: SQLAlchemy engine
        target_date: Date to resolve (date object)
        dry_run: If True, print results without writing to database

    Returns:
        Number of predictions resolved
    """
    # Step A: Query unresolved predictions
    pred_sql = text("""
        SELECT id, sport, game_date, home_team, away_team,
               bet_on, source, handicap
        FROM prediction_results
        WHERE outcome IS NULL AND game_date = :target_date
    """)

    # Step B: Query game results
    games_sql = text("""
        SELECT sport, game_date, home_team, away_team,
               closing_spread, closing_total, home_team_total, away_team_total,
               home_score, away_score,
               spread_result, total_result,
               home_team_total_result, away_team_total_result
        FROM games WHERE game_date = :target_date
    """)

    with engine.connect() as conn:
        predictions = conn.execute(pred_sql, {'target_date': target_date}).fetchall()
        game_rows = conn.execute(games_sql, {'target_date': target_date}).fetchall()

    if not predictions:
        logger.info(f"No unresolved predictions for {target_date}")
        return 0

    # Build game lookup
    game_lookup = {}
    for g in game_rows:
        key = (g.sport, str(g.game_date), g.home_team, g.away_team)
        game_lookup[key] = {
            'closing_spread': g.closing_spread,
            'closing_total': g.closing_total,
            'home_team_total': g.home_team_total,
            'away_team_total': g.away_team_total,
            'home_score': g.home_score,
            'away_score': g.away_score,
            'spread_result': g.spread_result,
            'total_result': g.total_result,
            'home_team_total_result': g.home_team_total_result,
            'away_team_total_result': g.away_team_total_result,
        }

    logger.info(f"Found {len(predictions)} unresolved predictions, {len(game_lookup)} games for {target_date}")

    # Step C & D: Match and determine outcomes
    update_sql = text("""
        UPDATE prediction_results SET
            outcome = :outcome,
            margin = :margin,
            closing_spread = :closing_spread,
            closing_total = :closing_total,
            closing_home_tt = :closing_home_tt,
            closing_away_tt = :closing_away_tt,
            home_score = :home_score,
            away_score = :away_score,
            resolved_at = :resolved_at
        WHERE id = :pred_id
    """)

    updates = []
    results_summary = {'WIN': 0, 'LOSS': 0, 'PUSH': 0, 'unmatched': 0}
    now = datetime.now(timezone.utc)

    for pred in predictions:
        key = (pred.sport, str(pred.game_date), pred.home_team, pred.away_team)
        game = game_lookup.get(key)

        if game is None:
            results_summary['unmatched'] += 1
            logger.debug(f"  No game found for {pred.away_team} @ {pred.home_team}")
            continue

        result = determine_outcome(
            bet_on=pred.bet_on,
            source=pred.source,
            handicap=pred.handicap,
            home_team=pred.home_team,
            game_row=game,
        )

        if result['outcome'] is None:
            results_summary['unmatched'] += 1
            continue

        results_summary[result['outcome']] += 1
        logger.info(f"  {pred.away_team} @ {pred.home_team} - {pred.source} {pred.bet_on} -> "
                     f"{result['outcome']} ({result['margin']:+.1f} margin)")

        updates.append({
            'pred_id': pred.id,
            'outcome': result['outcome'],
            'margin': result['margin'],
            'closing_spread': game['closing_spread'],
            'closing_total': game['closing_total'],
            'closing_home_tt': game['home_team_total'],
            'closing_away_tt': game['away_team_total'],
            'home_score': game['home_score'],
            'away_score': game['away_score'],
            'resolved_at': now,
        })

    # Step E: Write updates
    if updates and not dry_run:
        with engine.begin() as conn:
            conn.execute(update_sql, updates)

    total_resolved = results_summary['WIN'] + results_summary['LOSS'] + results_summary['PUSH']
    win_rate = results_summary['WIN'] / total_resolved * 100 if total_resolved > 0 else 0

    logger.info(f"Summary: WIN: {results_summary['WIN']} ({win_rate:.1f}%) | "
                f"LOSS: {results_summary['LOSS']} | PUSH: {results_summary['PUSH']} | "
                f"Unmatched: {results_summary['unmatched']}")

    if dry_run:
        logger.info("DRY RUN — no changes written to database")

    return total_resolved
