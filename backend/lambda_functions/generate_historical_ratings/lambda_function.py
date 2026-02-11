"""
Lambda Function: Generate Historical Ratings
Computes daily team rating snapshots for backtesting and analysis.

Triggered by: EventBridge schedule (daily at 6:15 AM EST, after collect_yesterday_games)

This function:
1. Reads all games from the database
2. Identifies game dates that don't have rating snapshots yet
3. Computes network-based ratings (win-based and ATS-based) for each missing date
4. Writes snapshots to the historical_ratings table
"""

import json
import os
import math
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, List, Set
from sqlalchemy import text

from shared import get_db_engine

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
SPORTS = ['NFL', 'NBA', 'NCAAM']

# Rating algorithm config
RATING_CONFIG = {
    'decay': 0.92,           # Recency decay per week
    'margin_cap': 20,        # Max margin considered
    'learning_rate': 0.03,   # Rating adjustment rate
    'tolerance': 0.0005,     # Convergence threshold
}

# Sport-specific iterations (sparse vs dense graphs)
ITERATIONS_BY_SPORT = {
    'NFL': 100,
    'NBA': 100,
    'NCAAM': 150,  # More iterations for 350+ team sparse graph
}


def get_games_for_sport(engine, sport: str) -> List[Dict[str, Any]]:
    """Fetch all games for a sport from the database."""
    query = text("""
        SELECT game_date, home_team, away_team, home_score, away_score,
               closing_spread, spread_result
        FROM games
        WHERE sport = :sport
        ORDER BY game_date
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {'sport': sport})
        games = [dict(row._mapping) for row in result]

    return games


def get_existing_snapshot_dates(engine, sport: str) -> Set[date]:
    """Get dates that already have rating snapshots."""
    query = text("""
        SELECT DISTINCT snapshot_date
        FROM historical_ratings
        WHERE sport = :sport
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {'sport': sport})
        dates = {row[0] for row in result}

    return dates


def compute_network_ratings(
    games: List[Dict[str, Any]],
    mode: str = 'ats',
    decay: float = 0.92,
    margin_cap: float = 20,
    max_iterations: int = 100,
    learning_rate: float = 0.03,
    tolerance: float = 0.0005
) -> Dict[str, float]:
    """
    Compute network-based team ratings using iterative strength propagation.

    Args:
        games: List of game dictionaries
        mode: 'ats' for spread coverage, 'win' for game outcomes
        Other args: algorithm parameters

    Returns:
        Dictionary mapping team names to ratings (0-1 normalized)
    """
    if not games:
        return {}

    # Get all teams
    teams = set()
    for g in games:
        teams.add(g['home_team'])
        teams.add(g['away_team'])

    ratings = {team: 0.5 for team in teams}

    if len(teams) == 0:
        return ratings

    # Filter games based on mode
    if mode == 'ats':
        # Skip pushes for ATS
        valid_games = [g for g in games if g.get('spread_result') and g['spread_result'] != 0]
    else:
        # Skip ties for win-based
        valid_games = [g for g in games if g.get('home_score') is not None
                      and g.get('away_score') is not None
                      and g['home_score'] != g['away_score']]

    if not valid_games:
        return ratings

    # Find max date for recency weighting
    max_date = max(g['game_date'] for g in valid_games)
    if isinstance(max_date, str):
        max_date = datetime.strptime(max_date, '%Y-%m-%d').date()

    for iteration in range(max_iterations):
        new_ratings = ratings.copy()

        for game in valid_games:
            # Determine winner and margin based on mode
            if mode == 'ats':
                home_won = game['spread_result'] > 0
                margin = abs(game['spread_result'])
            else:  # win mode
                home_won = game['home_score'] > game['away_score']
                margin = abs(game['home_score'] - game['away_score'])

            margin = min(margin, margin_cap)

            winner = game['home_team'] if home_won else game['away_team']
            loser = game['away_team'] if home_won else game['home_team']

            # How surprising was this result?
            winner_rating = ratings[winner]
            loser_rating = ratings[loser]
            total = winner_rating + loser_rating
            expected = winner_rating / total if total > 0 else 0.5
            surprise = 1 - expected

            # Recency weight
            game_date = game['game_date']
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
            days_ago = (max_date - game_date).days
            recency_weight = decay ** (days_ago / 7)

            # Adjustment
            adjustment = surprise * (margin / margin_cap) * learning_rate * recency_weight

            new_ratings[winner] += adjustment
            new_ratings[loser] -= adjustment

        # Normalize using z-score + sigmoid
        valid_values = [v for v in new_ratings.values() if math.isfinite(v)]
        if len(valid_values) > 1:
            mean_r = sum(valid_values) / len(valid_values)
            std_r = (sum((v - mean_r) ** 2 for v in valid_values) / len(valid_values)) ** 0.5
            if std_r > 0:
                new_ratings = {
                    t: 1 / (1 + math.exp(-max(-10, min(10, (r - mean_r) / std_r))))
                    if math.isfinite(r) else 0.5
                    for t, r in new_ratings.items()
                }

        # Check convergence
        max_change = max(abs(new_ratings[t] - ratings[t]) for t in teams)
        if max_change < tolerance:
            break

        ratings = new_ratings

    return ratings


def compute_ratings_at_date(
    games: List[Dict[str, Any]],
    as_of_date: date,
    mode: str = 'ats',
    **kwargs
) -> Dict[str, float]:
    """
    Compute network ratings using only games BEFORE as_of_date.
    """
    # Filter to games before the target date
    games_before = []
    for g in games:
        game_date = g['game_date']
        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
        if game_date < as_of_date:
            games_before.append(g)

    if not games_before:
        return {}

    return compute_network_ratings(games_before, mode=mode, **kwargs)


def write_ratings_snapshot(
    engine,
    sport: str,
    snapshot_date: date,
    win_ratings: Dict[str, float],
    ats_ratings: Dict[str, float],
    games: List[Dict[str, Any]]
) -> int:
    """Write rating snapshots for all teams to the database."""
    if not win_ratings:
        return 0

    # Get all teams that have ratings
    teams = set(win_ratings.keys()) | set(ats_ratings.keys())

    # Compute ranks for win ratings
    win_sorted = sorted(win_ratings.items(), key=lambda x: x[1], reverse=True)
    win_ranks = {team: i + 1 for i, (team, _) in enumerate(win_sorted)}

    # Compute ranks for ATS ratings
    ats_sorted = sorted(ats_ratings.items(), key=lambda x: x[1], reverse=True)
    ats_ranks = {team: i + 1 for i, (team, _) in enumerate(ats_sorted)}

    # Count games per team (before this date)
    team_games = {}
    for team in teams:
        count = 0
        for g in games:
            game_date = g['game_date']
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
            if game_date < snapshot_date:
                if g['home_team'] == team or g['away_team'] == team:
                    count += 1
        team_games[team] = count

    # Build insert data
    rows = []
    for team in teams:
        win_rating = win_ratings.get(team, 0.5)
        ats_rating = ats_ratings.get(team, 0.5)
        rows.append({
            'sport': sport,
            'snapshot_date': snapshot_date,
            'team': team,
            'win_rating': win_rating,
            'ats_rating': ats_rating,
            'market_gap': ats_rating - win_rating,
            'games_analyzed': team_games.get(team, 0),
            'win_rank': win_ranks.get(team, len(teams)),
            'ats_rank': ats_ranks.get(team, len(teams))
        })

    # Upsert to database
    upsert_sql = text("""
        INSERT INTO historical_ratings
        (sport, snapshot_date, team, win_rating, ats_rating, market_gap, games_analyzed, win_rank, ats_rank)
        VALUES (:sport, :snapshot_date, :team, :win_rating, :ats_rating, :market_gap, :games_analyzed, :win_rank, :ats_rank)
        ON CONFLICT (sport, snapshot_date, team)
        DO UPDATE SET
            win_rating = EXCLUDED.win_rating,
            ats_rating = EXCLUDED.ats_rating,
            market_gap = EXCLUDED.market_gap,
            games_analyzed = EXCLUDED.games_analyzed,
            win_rank = EXCLUDED.win_rank,
            ats_rank = EXCLUDED.ats_rank
    """)

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    return len(rows)


def process_sport(engine, sport: str) -> Dict[str, Any]:
    """Generate missing rating snapshots for a sport."""
    logger.info(f"Processing {sport}...")

    # Get all games
    games = get_games_for_sport(engine, sport)
    if not games:
        logger.info(f"  No games found for {sport}")
        return {'sport': sport, 'snapshots_created': 0, 'teams_processed': 0}

    logger.info(f"  Found {len(games)} games")

    # Get unique game dates
    game_dates = set()
    for g in games:
        game_date = g['game_date']
        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
        game_dates.add(game_date)

    # Get existing snapshot dates
    existing_dates = get_existing_snapshot_dates(engine, sport)
    logger.info(f"  Existing snapshots: {len(existing_dates)} dates")

    # Find missing dates
    missing_dates = sorted(game_dates - existing_dates)
    logger.info(f"  Missing snapshots: {len(missing_dates)} dates")

    if not missing_dates:
        return {'sport': sport, 'snapshots_created': 0, 'teams_processed': 0}

    # Get algorithm config
    cfg = RATING_CONFIG.copy()
    max_iterations = ITERATIONS_BY_SPORT.get(sport, 100)

    snapshots_created = 0
    total_teams = 0

    for snapshot_date in missing_dates:
        # Compute both ratings using games BEFORE this date
        win_ratings = compute_ratings_at_date(
            games, snapshot_date, mode='win',
            decay=cfg['decay'],
            margin_cap=cfg['margin_cap'],
            max_iterations=max_iterations,
            learning_rate=cfg['learning_rate'],
            tolerance=cfg['tolerance']
        )

        ats_ratings = compute_ratings_at_date(
            games, snapshot_date, mode='ats',
            decay=cfg['decay'],
            margin_cap=cfg['margin_cap'],
            max_iterations=max_iterations,
            learning_rate=cfg['learning_rate'],
            tolerance=cfg['tolerance']
        )

        if win_ratings:
            teams_written = write_ratings_snapshot(
                engine, sport, snapshot_date, win_ratings, ats_ratings, games
            )
            snapshots_created += 1
            total_teams += teams_written

    logger.info(f"  Created {snapshots_created} snapshots with {total_teams} team ratings")

    return {
        'sport': sport,
        'snapshots_created': snapshots_created,
        'teams_processed': total_teams,
        'missing_dates': len(missing_dates)
    }


def lambda_handler(event, context):
    """Lambda entry point - generates missing historical rating snapshots."""
    logger.info("=" * 60)
    logger.info("Starting historical ratings generation")
    logger.info("=" * 60)

    try:
        engine = get_db_engine()
        if engine is None:
            raise Exception("Could not connect to database")

        logger.info("✓ Connected to database")

        # Process each sport
        results = []
        total_snapshots = 0

        for sport in SPORTS:
            try:
                result = process_sport(engine, sport)
                results.append(result)
                total_snapshots += result['snapshots_created']
                logger.info(f"✓ {sport}: {result['snapshots_created']} new snapshots")
            except Exception as e:
                logger.error(f"✗ Error processing {sport}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info("=" * 60)
        logger.info(f"COMPLETE: Created {total_snapshots} total snapshots")
        logger.info("=" * 60)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Historical ratings generation complete',
                'total_snapshots': total_snapshots,
                'results': results
            })
        }

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


# For local testing
if __name__ == '__main__':
    # Test locally with environment variable
    import os
    if not os.environ.get('DATABASE_URL'):
        print("Set DATABASE_URL environment variable for local testing")
        print("Example: export DATABASE_URL='postgresql://user:pass@host:5432/dbname'")
    else:
        result = lambda_handler({}, None)
        print(json.dumps(json.loads(result['body']), indent=2))
