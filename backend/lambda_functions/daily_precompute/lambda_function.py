"""
Lambda Function: Generate Current Rankings, Streaks, and Patterns
Pre-computes team rankings, streaks, and betting patterns daily for fast dashboard loads.

Triggered by: EventBridge schedule (daily at 6:20 AM EST, after generate_historical_ratings)

This function:
1. For each sport (NFL, NBA, NCAAM):
   - Computes network-based ratings (win-based and ATS-based)
   - Calculates win/ATS records for each team
   - Computes current ATS streak for each team
   - Detects profitable streak patterns across handicap levels
2. Upserts results to current_rankings, current_streaks, and detected_patterns tables

Performance benefit: Dashboard pages load in <100ms instead of 3-7 seconds.
"""

import json
import os
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple
from sqlalchemy import text

from shared import get_db_engine

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
SPORTS = ['NFL', 'NBA', 'NCAAM']
MIN_GAMES = 5  # Minimum games for "reliable" rating

# Rating algorithm config
RATING_CONFIG = {
    'decay': 0.92,
    'margin_cap': 20,
    'learning_rate': 0.03,
    'tolerance': 0.0005,
}

ITERATIONS_BY_SPORT = {
    'NFL': 100,
    'NBA': 100,
    'NCAAM': 150,
}


def get_games_for_sport(engine, sport: str) -> List[Dict[str, Any]]:
    """Fetch all games for a sport from the database."""
    query = text("""
        SELECT game_date, home_team, away_team, home_score, away_score,
               closing_spread, spread_result,
               total_result, home_team_total, away_team_total,
               home_team_total_result, away_team_total_result, closing_total
        FROM games
        WHERE sport = :sport
        ORDER BY game_date
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {'sport': sport})
        games = [dict(row._mapping) for row in result]

    return games


def compute_network_ratings(
    games: List[Dict[str, Any]],
    mode: str = 'ats',
    decay: float = 0.92,
    margin_cap: float = 20,
    max_iterations: int = 100,
    learning_rate: float = 0.03,
    tolerance: float = 0.0005
) -> Dict[str, float]:
    """Compute network-based team ratings using iterative strength propagation."""
    if not games:
        return {}

    teams = set()
    for g in games:
        teams.add(g['home_team'])
        teams.add(g['away_team'])

    ratings = {team: 0.5 for team in teams}

    if len(teams) == 0:
        return ratings

    if mode == 'ats':
        valid_games = [g for g in games if g.get('spread_result') and g['spread_result'] != 0]
    else:
        valid_games = [g for g in games if g.get('home_score') is not None
                      and g.get('away_score') is not None
                      and g['home_score'] != g['away_score']]

    if not valid_games:
        return ratings

    max_date = max(g['game_date'] for g in valid_games)
    if isinstance(max_date, str):
        from datetime import datetime as dt
        max_date = dt.strptime(max_date, '%Y-%m-%d').date()

    for iteration in range(max_iterations):
        new_ratings = ratings.copy()

        for game in valid_games:
            if mode == 'ats':
                home_won = game['spread_result'] > 0
                margin = abs(game['spread_result'])
            else:
                home_won = game['home_score'] > game['away_score']
                margin = abs(game['home_score'] - game['away_score'])

            margin = min(margin, margin_cap)

            winner = game['home_team'] if home_won else game['away_team']
            loser = game['away_team'] if home_won else game['home_team']

            winner_rating = ratings[winner]
            loser_rating = ratings[loser]
            total = winner_rating + loser_rating
            expected = winner_rating / total if total > 0 else 0.5
            surprise = 1 - expected

            game_date = game['game_date']
            if isinstance(game_date, str):
                from datetime import datetime as dt
                game_date = dt.strptime(game_date, '%Y-%m-%d').date()
            days_ago = (max_date - game_date).days
            recency_weight = decay ** (days_ago / 7)

            adjustment = surprise * (margin / margin_cap) * learning_rate * recency_weight

            new_ratings[winner] += adjustment
            new_ratings[loser] -= adjustment

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

        max_change = max(abs(new_ratings[t] - ratings[t]) for t in teams)
        if max_change < tolerance:
            break

        ratings = new_ratings

    return ratings


def get_team_record(games: List[Dict], team: str) -> str:
    """Get win-loss record for a team."""
    wins = 0
    losses = 0

    for g in games:
        if g['home_team'] == team:
            if g.get('home_score') is not None and g.get('away_score') is not None:
                if g['home_score'] > g['away_score']:
                    wins += 1
                elif g['home_score'] < g['away_score']:
                    losses += 1
        elif g['away_team'] == team:
            if g.get('home_score') is not None and g.get('away_score') is not None:
                if g['away_score'] > g['home_score']:
                    wins += 1
                elif g['away_score'] < g['home_score']:
                    losses += 1

    return f"{wins}-{losses}"


def get_team_ats_record(games: List[Dict], team: str) -> str:
    """Get ATS record for a team (excluding pushes)."""
    wins = 0
    losses = 0

    for g in games:
        spread_result = g.get('spread_result')
        if spread_result is None or spread_result == 0:
            continue

        if g['home_team'] == team:
            if spread_result > 0:
                wins += 1
            else:
                losses += 1
        elif g['away_team'] == team:
            if spread_result < 0:
                wins += 1
            else:
                losses += 1

    return f"{wins}-{losses}"


def get_team_games_count(games: List[Dict], team: str) -> int:
    """Get total games played by a team."""
    count = 0
    for g in games:
        if g['home_team'] == team or g['away_team'] == team:
            count += 1
    return count


def compute_current_streak(games: List[Dict], team: str) -> Tuple[int, str]:
    """Compute current ATS streak for a team."""
    team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]

    if not team_games:
        return (0, 'WIN')

    # Sort by date descending
    from datetime import datetime as dt
    def get_date(g):
        d = g['game_date']
        if isinstance(d, str):
            return dt.strptime(d, '%Y-%m-%d').date()
        return d

    team_games.sort(key=get_date, reverse=True)

    streak_length = 0
    streak_type = None

    for game in team_games:
        is_home = game['home_team'] == team
        spread_result = game.get('spread_result')

        if spread_result is None or spread_result == 0:
            continue

        covered = (spread_result > 0) if is_home else (spread_result < 0)

        if streak_type is None:
            streak_type = 'WIN' if covered else 'LOSS'
            streak_length = 1
        elif (streak_type == 'WIN') == covered:
            streak_length += 1
        else:
            break

    return (streak_length, streak_type or 'WIN')


def compute_current_ou_streak(games: List[Dict], team: str) -> Tuple[int, str]:
    """Compute current O/U streak for a team (game total over/under)."""
    team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]

    if not team_games:
        return (0, 'OVER')

    from datetime import datetime as dt
    def get_date(g):
        d = g['game_date']
        if isinstance(d, str):
            return dt.strptime(d, '%Y-%m-%d').date()
        return d

    team_games.sort(key=get_date, reverse=True)

    streak_length = 0
    streak_type = None

    for game in team_games:
        total_result = game.get('total_result')

        if total_result is None or total_result == 0:
            continue

        is_over = total_result > 0

        if streak_type is None:
            streak_type = 'OVER' if is_over else 'UNDER'
            streak_length = 1
        elif (streak_type == 'OVER') == is_over:
            streak_length += 1
        else:
            break

    return (streak_length, streak_type or 'OVER')


def compute_current_tt_streak(games: List[Dict], team: str) -> Tuple[int, str]:
    """Compute current team total O/U streak for a team."""
    team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]

    if not team_games:
        return (0, 'OVER')

    from datetime import datetime as dt
    def get_date(g):
        d = g['game_date']
        if isinstance(d, str):
            return dt.strptime(d, '%Y-%m-%d').date()
        return d

    team_games.sort(key=get_date, reverse=True)

    streak_length = 0
    streak_type = None

    for game in team_games:
        is_home = game['home_team'] == team
        margin = game.get('home_team_total_result') if is_home else game.get('away_team_total_result')

        if margin is None or margin == 0:
            continue

        is_over = margin > 0

        if streak_type is None:
            streak_type = 'OVER' if is_over else 'UNDER'
            streak_length = 1
        elif (streak_type == 'OVER') == is_over:
            streak_length += 1
        else:
            break

    return (streak_length, streak_type or 'OVER')


def write_current_ou_streaks(engine, sport: str, streaks_data: Dict[str, Tuple[int, str]], computed_at: datetime) -> int:
    """Write current O/U streaks to database."""
    if not streaks_data:
        return 0

    upsert_sql = text("""
        INSERT INTO current_ou_streaks
        (sport, team, streak_length, streak_type, computed_at)
        VALUES (:sport, :team, :streak_length, :streak_type, :computed_at)
        ON CONFLICT (sport, team)
        DO UPDATE SET
            streak_length = EXCLUDED.streak_length,
            streak_type = EXCLUDED.streak_type,
            computed_at = EXCLUDED.computed_at
    """)

    rows = []
    for team, (length, stype) in streaks_data.items():
        rows.append({
            'sport': sport,
            'team': team,
            'streak_length': length,
            'streak_type': stype,
            'computed_at': computed_at,
        })

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    return len(rows)


def write_current_tt_streaks(engine, sport: str, streaks_data: Dict[str, Tuple[int, str]], computed_at: datetime) -> int:
    """Write current TT streaks to database."""
    if not streaks_data:
        return 0

    upsert_sql = text("""
        INSERT INTO current_tt_streaks
        (sport, team, streak_length, streak_type, computed_at)
        VALUES (:sport, :team, :streak_length, :streak_type, :computed_at)
        ON CONFLICT (sport, team)
        DO UPDATE SET
            streak_length = EXCLUDED.streak_length,
            streak_type = EXCLUDED.streak_type,
            computed_at = EXCLUDED.computed_at
    """)

    rows = []
    for team, (length, stype) in streaks_data.items():
        rows.append({
            'sport': sport,
            'team': team,
            'streak_length': length,
            'streak_type': stype,
            'computed_at': computed_at,
        })

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    return len(rows)


def write_current_rankings(engine, sport: str, rankings_data: List[Dict], computed_at: datetime) -> int:
    """Write current rankings to database."""
    if not rankings_data:
        return 0

    upsert_sql = text("""
        INSERT INTO current_rankings
        (sport, team, win_rating, ats_rating, market_gap, win_rank, ats_rank,
         win_record, ats_record, games_analyzed, is_reliable, computed_at)
        VALUES (:sport, :team, :win_rating, :ats_rating, :market_gap, :win_rank, :ats_rank,
                :win_record, :ats_record, :games_analyzed, :is_reliable, :computed_at)
        ON CONFLICT (sport, team)
        DO UPDATE SET
            win_rating = EXCLUDED.win_rating,
            ats_rating = EXCLUDED.ats_rating,
            market_gap = EXCLUDED.market_gap,
            win_rank = EXCLUDED.win_rank,
            ats_rank = EXCLUDED.ats_rank,
            win_record = EXCLUDED.win_record,
            ats_record = EXCLUDED.ats_record,
            games_analyzed = EXCLUDED.games_analyzed,
            is_reliable = EXCLUDED.is_reliable,
            computed_at = EXCLUDED.computed_at
    """)

    rows = []
    for r in rankings_data:
        rows.append({
            'sport': sport,
            'team': r['team'],
            'win_rating': r['win_rating'],
            'ats_rating': r['ats_rating'],
            'market_gap': r['market_gap'],
            'win_rank': r['win_rank'],
            'ats_rank': r['ats_rank'],
            'win_record': r['win_record'],
            'ats_record': r['ats_record'],
            'games_analyzed': r['games_analyzed'],
            'is_reliable': 1 if r['is_reliable'] else 0,
            'computed_at': computed_at,
        })

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    return len(rows)


# =============================================================================
# Pattern Detection
# =============================================================================

# Pattern detection config
PATTERN_CONFIG = {
    'min_sample': 30,
    'min_edge': 0.05,
    'streak_range': (2, 7),
    'handicap_range': (0, 15),
}


def get_confidence(sample_size: int, edge: float) -> str:
    """Determine confidence based on sample size and edge magnitude."""
    abs_edge = abs(edge)
    if sample_size >= 50 and abs_edge >= 0.08:
        return 'high'
    elif sample_size >= 30 and abs_edge >= 0.05:
        return 'medium'
    return 'low'


def compute_baseline_coverage(games: List[Dict], handicap_range: Tuple[int, int] = (0, 15)) -> Dict[int, float]:
    """Compute baseline cover rates at each handicap level."""
    baseline = {}

    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        covers = 0
        total = 0

        for g in games:
            spread_result = g.get('spread_result')
            if spread_result is None:
                continue

            # Home team cover at this handicap
            home_covered = (spread_result + handicap) > 0
            total += 1
            if home_covered:
                covers += 1

        baseline[handicap] = covers / total if total > 0 else 0.5

    return baseline


def compute_streak_continuation(
    games: List[Dict],
    teams: Set[str],
    streak_length: int,
    streak_type: str,
    handicap_range: Tuple[int, int] = (0, 15)
) -> Dict[int, Tuple[int, int]]:
    """
    Compute cover counts at each handicap for games following a streak.

    Returns: {handicap: (covers, total)}
    """
    results = {h: [0, 0] for h in range(handicap_range[0], handicap_range[1] + 1)}

    for team in teams:
        team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]
        if len(team_games) < streak_length + 1:
            continue

        # Sort by date
        from datetime import datetime as dt
        def get_date(g):
            d = g['game_date']
            if isinstance(d, str):
                return dt.strptime(d, '%Y-%m-%d').date()
            return d

        team_games.sort(key=get_date)

        # Build game data
        game_data = []
        for g in team_games:
            is_home = g['home_team'] == team
            spread_result = g.get('spread_result')
            if spread_result is None:
                continue
            covered_base = (spread_result > 0) if is_home else (spread_result < 0)
            game_data.append({
                'spread_result': spread_result,
                'is_home': is_home,
                'covered_base': covered_base
            })

        # Find situations matching streak criteria
        target_covered = (streak_type == 'WIN')

        for i in range(streak_length, len(game_data)):
            # Check if there's a streak of exactly streak_length ending at i-1
            streak_count = 0
            for j in range(i - 1, -1, -1):
                if game_data[j]['covered_base'] == target_covered:
                    streak_count += 1
                else:
                    break
                if streak_count >= streak_length:
                    break

            if streak_count >= streak_length:
                # Game i is AFTER the streak - check coverage at each handicap
                next_game = game_data[i]
                is_home = next_game['is_home']
                spread_result = next_game['spread_result']

                for handicap in range(handicap_range[0], handicap_range[1] + 1):
                    if is_home:
                        covered = (spread_result + handicap) > 0
                    else:
                        covered = (spread_result - handicap) < 0

                    results[handicap][1] += 1  # total
                    if covered:
                        results[handicap][0] += 1  # covers

    return {h: tuple(v) for h, v in results.items()}


def detect_patterns_for_sport(games: List[Dict], sport: str) -> List[Dict]:
    """Detect all significant streak patterns for a sport."""
    cfg = PATTERN_CONFIG
    patterns = []

    teams = set()
    for g in games:
        teams.add(g['home_team'])
        teams.add(g['away_team'])

    # Compute baseline
    baseline = compute_baseline_coverage(games, cfg['handicap_range'])

    # Check each streak type and length
    for streak_length in range(cfg['streak_range'][0], cfg['streak_range'][1] + 1):
        for streak_type in ['WIN', 'LOSS']:
            continuation = compute_streak_continuation(
                games, teams, streak_length, streak_type, cfg['handicap_range']
            )

            # Build full coverage profile for this (streak_type, streak_length)
            coverage_profile = {}
            for h in range(cfg['handicap_range'][0], cfg['handicap_range'][1] + 1):
                covers, total = continuation[h]
                if total >= 5:
                    coverage_profile[h] = {
                        'cover_rate': round(covers / total, 4),
                        'baseline_rate': round(baseline[h], 4),
                        'edge': round((covers / total) - baseline[h], 4),
                        'sample_size': total,
                    }
            profile_json = json.dumps(coverage_profile) if coverage_profile else None

            for handicap in range(cfg['handicap_range'][0], cfg['handicap_range'][1] + 1):
                covers, total = continuation[handicap]
                if total < cfg['min_sample']:
                    continue

                cover_rate = covers / total
                baseline_rate = baseline[handicap]
                edge = cover_rate - baseline_rate

                if abs(edge) < cfg['min_edge']:
                    continue

                pattern_type = 'streak_ride' if edge > 0 else 'streak_fade'

                patterns.append({
                    'sport': sport,
                    'pattern_type': pattern_type,
                    'streak_type': streak_type,
                    'streak_length': streak_length,
                    'handicap': handicap,
                    'cover_rate': cover_rate,
                    'baseline_rate': baseline_rate,
                    'edge': edge,
                    'sample_size': total,
                    'confidence': get_confidence(total, edge),
                    'coverage_profile_json': profile_json,
                })

    return patterns


def compute_ou_baseline_coverage(games: List[Dict], handicap_range: Tuple[int, int] = (0, 20)) -> Dict[int, float]:
    """Compute baseline O/U OVER cover rates at each handicap level."""
    baseline = {}

    # Filter games with total_result
    valid_games = [g for g in games if g.get('total_result') is not None]
    if not valid_games:
        return baseline

    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        covers = 0
        total = len(valid_games)

        for g in valid_games:
            # OVER covers when total_result + handicap > 0
            if (g['total_result'] + handicap) > 0:
                covers += 1

        baseline[handicap] = covers / total if total > 0 else 0.5

    return baseline


def compute_ou_streak_continuation(
    games: List[Dict],
    teams: Set[str],
    streak_length: int,
    streak_type: str,
    handicap_range: Tuple[int, int] = (0, 20)
) -> Dict[int, Tuple[int, int]]:
    """
    Compute OVER cover counts at each handicap for games following an O/U streak.

    Returns: {handicap: (covers, total)}
    """
    results = {h: [0, 0] for h in range(handicap_range[0], handicap_range[1] + 1)}

    target_is_over = (streak_type == 'OVER')

    for team in teams:
        team_games = [g for g in games
                      if (g['home_team'] == team or g['away_team'] == team)
                      and g.get('total_result') is not None]
        if len(team_games) < streak_length + 1:
            continue

        from datetime import datetime as dt
        def get_date(g):
            d = g['game_date']
            if isinstance(d, str):
                return dt.strptime(d, '%Y-%m-%d').date()
            return d

        team_games.sort(key=get_date)

        # Build game data
        game_data = []
        for g in team_games:
            total_result = g['total_result']
            if total_result == 0:
                game_data.append({'total_result': total_result, 'is_over': None})
            else:
                game_data.append({'total_result': total_result, 'is_over': total_result > 0})

        for i in range(streak_length, len(game_data)):
            if game_data[i]['is_over'] is None:
                continue

            streak_len = 0
            for j in range(i - 1, -1, -1):
                if game_data[j]['is_over'] is None:
                    break
                if game_data[j]['is_over'] == target_is_over:
                    streak_len += 1
                else:
                    break

            if streak_len >= streak_length:
                total_result = game_data[i]['total_result']
                for handicap in range(handicap_range[0], handicap_range[1] + 1):
                    covered = (total_result + handicap) > 0  # OVER at this handicap
                    results[handicap][1] += 1
                    if covered:
                        results[handicap][0] += 1

    return {h: tuple(v) for h, v in results.items()}


def detect_ou_patterns_for_sport(games: List[Dict], sport: str) -> List[Dict]:
    """Detect all significant O/U streak patterns for a sport."""
    cfg = PATTERN_CONFIG
    ou_handicap_range = (0, 20)
    patterns = []

    teams = set()
    for g in games:
        teams.add(g['home_team'])
        teams.add(g['away_team'])

    baseline = compute_ou_baseline_coverage(games, ou_handicap_range)
    if not baseline:
        return patterns

    for streak_length in range(cfg['streak_range'][0], cfg['streak_range'][1] + 1):
        for streak_type in ['OVER', 'UNDER']:
            continuation = compute_ou_streak_continuation(
                games, teams, streak_length, streak_type, ou_handicap_range
            )

            # Build full coverage profile for this (streak_type, streak_length)
            coverage_profile = {}
            for h in range(ou_handicap_range[0], ou_handicap_range[1] + 1):
                covers, total = continuation[h]
                if total >= 5:
                    coverage_profile[h] = {
                        'cover_rate': round(covers / total, 4),
                        'baseline_rate': round(baseline.get(h, 0.5), 4),
                        'edge': round((covers / total) - baseline.get(h, 0.5), 4),
                        'sample_size': total,
                    }
            profile_json = json.dumps(coverage_profile) if coverage_profile else None

            for handicap in range(ou_handicap_range[0], ou_handicap_range[1] + 1):
                covers, total = continuation[handicap]
                if total < cfg['min_sample']:
                    continue

                cover_rate = covers / total
                baseline_rate = baseline.get(handicap, 0.5)
                edge = cover_rate - baseline_rate

                if abs(edge) < cfg['min_edge']:
                    continue

                pattern_type = 'streak_ride' if edge > 0 else 'streak_fade'

                patterns.append({
                    'sport': sport,
                    'market_type': 'ou',
                    'pattern_type': pattern_type,
                    'streak_type': streak_type,
                    'streak_length': streak_length,
                    'handicap': handicap,
                    'cover_rate': cover_rate,
                    'baseline_rate': baseline_rate,
                    'edge': edge,
                    'sample_size': total,
                    'confidence': get_confidence(total, edge),
                    'coverage_profile_json': profile_json,
                })

    return patterns


def compute_tt_baseline_coverage(games: List[Dict], handicap_range: Tuple[int, int] = (0, 10)) -> Dict[int, float]:
    """Compute baseline TT OVER cover rates at each handicap level."""
    baseline = {}

    # Pool all TT margins (home + away)
    all_margins = []
    for g in games:
        for col in ['home_team_total_result', 'away_team_total_result']:
            m = g.get(col)
            if m is not None and m != 0:
                all_margins.append(m)

    if not all_margins:
        return baseline

    for handicap in range(handicap_range[0], handicap_range[1] + 1):
        covers = sum(1 for m in all_margins if m > -handicap)
        total = len(all_margins)
        baseline[handicap] = covers / total if total > 0 else 0.5

    return baseline


def compute_tt_streak_continuation(
    games: List[Dict],
    teams: Set[str],
    streak_length: int,
    streak_type: str,
    handicap_range: Tuple[int, int] = (0, 10)
) -> Dict[int, Tuple[int, int]]:
    """
    Compute TT OVER cover counts at each handicap for games following a TT streak.

    Uses ride direction: after OVER streak, check if next game TT margin > -h (OVER covers).

    Returns: {handicap: (covers, total)}
    """
    results = {h: [0, 0] for h in range(handicap_range[0], handicap_range[1] + 1)}

    target_is_over = (streak_type == 'OVER')

    for team in teams:
        team_games = [g for g in games if g['home_team'] == team or g['away_team'] == team]
        if len(team_games) < streak_length + 1:
            continue

        from datetime import datetime as dt
        def get_date(g):
            d = g['game_date']
            if isinstance(d, str):
                return dt.strptime(d, '%Y-%m-%d').date()
            return d

        team_games.sort(key=get_date)

        # Build per-game TT results
        game_data = []
        for g in team_games:
            is_home = g['home_team'] == team
            margin = g.get('home_team_total_result') if is_home else g.get('away_team_total_result')
            if margin is None or margin == 0:
                game_data.append({'margin': None, 'is_over': None})
            else:
                game_data.append({'margin': margin, 'is_over': margin > 0})

        for i in range(streak_length, len(game_data)):
            if game_data[i]['margin'] is None:
                continue

            streak_len = 0
            for j in range(i - 1, -1, -1):
                if game_data[j]['is_over'] is None:
                    break
                if game_data[j]['is_over'] == target_is_over:
                    streak_len += 1
                else:
                    break

            if streak_len >= streak_length:
                margin = game_data[i]['margin']
                for handicap in range(handicap_range[0], handicap_range[1] + 1):
                    # Ride direction: OVER streak -> OVER covers if margin > -h
                    if streak_type == 'OVER':
                        covered = margin > -handicap
                    else:  # UNDER streak -> UNDER covers if margin < h
                        covered = margin < handicap

                    results[handicap][1] += 1
                    if covered:
                        results[handicap][0] += 1

    return {h: tuple(v) for h, v in results.items()}


def detect_tt_patterns_for_sport(games: List[Dict], sport: str) -> List[Dict]:
    """Detect all significant TT streak patterns for a sport."""
    cfg = PATTERN_CONFIG
    tt_handicap_range = (0, 10)
    patterns = []

    teams = set()
    for g in games:
        teams.add(g['home_team'])
        teams.add(g['away_team'])

    baseline = compute_tt_baseline_coverage(games, tt_handicap_range)
    if not baseline:
        return patterns

    for streak_length in range(cfg['streak_range'][0], cfg['streak_range'][1] + 1):
        for streak_type in ['OVER', 'UNDER']:
            continuation = compute_tt_streak_continuation(
                games, teams, streak_length, streak_type, tt_handicap_range
            )

            # Build full coverage profile for this (streak_type, streak_length)
            coverage_profile = {}
            for h in range(tt_handicap_range[0], tt_handicap_range[1] + 1):
                covers, total = continuation[h]
                if total >= 5:
                    coverage_profile[h] = {
                        'cover_rate': round(covers / total, 4),
                        'baseline_rate': round(baseline.get(h, 0.5), 4),
                        'edge': round((covers / total) - baseline.get(h, 0.5), 4),
                        'sample_size': total,
                    }
            profile_json = json.dumps(coverage_profile) if coverage_profile else None

            for handicap in range(tt_handicap_range[0], tt_handicap_range[1] + 1):
                covers, total = continuation[handicap]
                if total < cfg['min_sample']:
                    continue

                cover_rate = covers / total
                baseline_rate = baseline.get(handicap, 0.5)
                edge = cover_rate - baseline_rate

                if abs(edge) < cfg['min_edge']:
                    continue

                pattern_type = 'streak_ride' if edge > 0 else 'streak_fade'

                patterns.append({
                    'sport': sport,
                    'market_type': 'tt',
                    'pattern_type': pattern_type,
                    'streak_type': streak_type,
                    'streak_length': streak_length,
                    'handicap': handicap,
                    'cover_rate': cover_rate,
                    'baseline_rate': baseline_rate,
                    'edge': edge,
                    'sample_size': total,
                    'confidence': get_confidence(total, edge),
                    'coverage_profile_json': profile_json,
                })

    return patterns


def write_detected_patterns(engine, patterns: List[Dict], computed_at: datetime) -> int:
    """Write detected patterns to database."""
    if not patterns:
        return 0

    # Clear old patterns for these sport+market_type combos
    sport_markets = set((p['sport'], p.get('market_type', 'ats')) for p in patterns)
    delete_sql = text("DELETE FROM detected_patterns WHERE sport = :sport AND market_type = :market_type")

    upsert_sql = text("""
        INSERT INTO detected_patterns
        (sport, market_type, pattern_type, streak_type, streak_length, handicap,
         cover_rate, baseline_rate, edge, sample_size, confidence, coverage_profile_json, computed_at)
        VALUES (:sport, :market_type, :pattern_type, :streak_type, :streak_length, :handicap,
                :cover_rate, :baseline_rate, :edge, :sample_size, :confidence, :coverage_profile_json, :computed_at)
        ON CONFLICT (sport, market_type, pattern_type, streak_type, streak_length, handicap)
        DO UPDATE SET
            cover_rate = EXCLUDED.cover_rate,
            baseline_rate = EXCLUDED.baseline_rate,
            edge = EXCLUDED.edge,
            sample_size = EXCLUDED.sample_size,
            confidence = EXCLUDED.confidence,
            coverage_profile_json = EXCLUDED.coverage_profile_json,
            computed_at = EXCLUDED.computed_at
    """)

    rows = []
    for p in patterns:
        rows.append({
            'sport': p['sport'],
            'market_type': p.get('market_type', 'ats'),
            'pattern_type': p['pattern_type'],
            'streak_type': p['streak_type'],
            'streak_length': p['streak_length'],
            'handicap': p['handicap'],
            'cover_rate': p['cover_rate'],
            'baseline_rate': p['baseline_rate'],
            'edge': p['edge'],
            'sample_size': p['sample_size'],
            'confidence': p['confidence'],
            'coverage_profile_json': p.get('coverage_profile_json'),
            'computed_at': computed_at,
        })

    with engine.begin() as conn:
        for sport, market_type in sport_markets:
            conn.execute(delete_sql, {'sport': sport, 'market_type': market_type})
        if rows:
            conn.execute(upsert_sql, rows)

    return len(rows)


def write_current_streaks(engine, sport: str, streaks_data: Dict[str, Tuple[int, str]], computed_at: datetime) -> int:
    """Write current streaks to database."""
    if not streaks_data:
        return 0

    upsert_sql = text("""
        INSERT INTO current_streaks
        (sport, team, streak_length, streak_type, computed_at)
        VALUES (:sport, :team, :streak_length, :streak_type, :computed_at)
        ON CONFLICT (sport, team)
        DO UPDATE SET
            streak_length = EXCLUDED.streak_length,
            streak_type = EXCLUDED.streak_type,
            computed_at = EXCLUDED.computed_at
    """)

    rows = []
    for team, (length, stype) in streaks_data.items():
        rows.append({
            'sport': sport,
            'team': team,
            'streak_length': length,
            'streak_type': stype,
            'computed_at': computed_at,
        })

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    return len(rows)


def process_sport(engine, sport: str, computed_at: datetime) -> Dict[str, Any]:
    """Process rankings, streaks, and patterns for a single sport."""
    logger.info(f"Processing {sport}...")

    games = get_games_for_sport(engine, sport)
    if not games:
        logger.info(f"  No games found for {sport}")
        return {'sport': sport, 'teams': 0, 'rankings': 0, 'streaks': 0, 'patterns': 0}

    logger.info(f"  Found {len(games)} games")

    # Get algorithm config
    cfg = RATING_CONFIG.copy()
    max_iterations = ITERATIONS_BY_SPORT.get(sport, 100)

    # Compute ratings
    win_ratings = compute_network_ratings(
        games, mode='win',
        decay=cfg['decay'],
        margin_cap=cfg['margin_cap'],
        max_iterations=max_iterations,
        learning_rate=cfg['learning_rate'],
        tolerance=cfg['tolerance']
    )

    ats_ratings = compute_network_ratings(
        games, mode='ats',
        decay=cfg['decay'],
        margin_cap=cfg['margin_cap'],
        max_iterations=max_iterations,
        learning_rate=cfg['learning_rate'],
        tolerance=cfg['tolerance']
    )

    # Get all teams
    teams = set(win_ratings.keys()) | set(ats_ratings.keys())
    logger.info(f"  Found {len(teams)} teams")

    # Build rankings list
    rankings = []
    for team in teams:
        games_count = get_team_games_count(games, team)
        win_rating = win_ratings.get(team, 0.5)
        ats_rating = ats_ratings.get(team, 0.5)

        rankings.append({
            'team': team,
            'win_rating': win_rating,
            'ats_rating': ats_rating,
            'market_gap': ats_rating - win_rating,
            'games_analyzed': games_count,
            'win_record': get_team_record(games, team),
            'ats_record': get_team_ats_record(games, team),
            'is_reliable': games_count >= MIN_GAMES,
        })

    # Sort and assign win ranks
    rankings.sort(key=lambda x: x['win_rating'], reverse=True)
    for i, r in enumerate(rankings):
        r['win_rank'] = i + 1

    # Sort and assign ATS ranks
    rankings.sort(key=lambda x: x['ats_rating'], reverse=True)
    for i, r in enumerate(rankings):
        r['ats_rank'] = i + 1

    # Compute ATS streaks
    streaks = {}
    for team in teams:
        length, stype = compute_current_streak(games, team)
        streaks[team] = (length, stype)

    # Compute O/U streaks
    ou_streaks = {}
    for team in teams:
        length, stype = compute_current_ou_streak(games, team)
        if length > 0:
            ou_streaks[team] = (length, stype)

    # Compute TT streaks
    tt_streaks = {}
    for team in teams:
        length, stype = compute_current_tt_streak(games, team)
        if length > 0:
            tt_streaks[team] = (length, stype)

    # Detect ATS patterns
    logger.info(f"  Detecting ATS patterns...")
    ats_patterns = detect_patterns_for_sport(games, sport)
    # Tag ATS patterns with market_type
    for p in ats_patterns:
        p['market_type'] = 'ats'
    logger.info(f"  Found {len(ats_patterns)} ATS patterns")

    # Detect O/U patterns
    logger.info(f"  Detecting O/U patterns...")
    ou_patterns = detect_ou_patterns_for_sport(games, sport)
    logger.info(f"  Found {len(ou_patterns)} O/U patterns")

    # Detect TT patterns
    logger.info(f"  Detecting TT patterns...")
    tt_patterns = detect_tt_patterns_for_sport(games, sport)
    logger.info(f"  Found {len(tt_patterns)} TT patterns")

    all_patterns = ats_patterns + ou_patterns + tt_patterns

    # Write to database
    rankings_written = write_current_rankings(engine, sport, rankings, computed_at)
    streaks_written = write_current_streaks(engine, sport, streaks, computed_at)
    ou_streaks_written = write_current_ou_streaks(engine, sport, ou_streaks, computed_at)
    tt_streaks_written = write_current_tt_streaks(engine, sport, tt_streaks, computed_at)
    patterns_written = write_detected_patterns(engine, all_patterns, computed_at)

    logger.info(f"  Wrote {rankings_written} rankings, {streaks_written} ATS streaks, "
                f"{ou_streaks_written} O/U streaks, {tt_streaks_written} TT streaks, "
                f"{patterns_written} patterns")

    return {
        'sport': sport,
        'teams': len(teams),
        'rankings': rankings_written,
        'streaks': streaks_written,
        'ou_streaks': ou_streaks_written,
        'tt_streaks': tt_streaks_written,
        'patterns': patterns_written,
    }


def generate_todays_recommendations(engine, computed_at: datetime) -> int:
    """
    Generate and store today's betting recommendations.

    This combines pattern detection with today's scheduled games to create
    pre-computed recommendations that the dashboard can read instantly.
    """
    from datetime import timedelta

    logger.info("Generating today's recommendations...")

    # Get today's date in EST
    est = timezone(timedelta(hours=-5))
    today_est = datetime.now(est).date()

    # Import recommendation functions (these are in the src package)
    # We need to add src to path since Lambda packages differently
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    try:
        from src.analysis.todays_recommendations import generate_recommendations, write_todays_recommendations
        from src.analysis.insights import get_cached_patterns, get_cached_ou_patterns, get_cached_tt_patterns
    except ImportError as e:
        logger.warning(f"Could not import recommendation modules: {e}")
        return 0

    # Get cached patterns (already computed by process_sport)
    with engine.connect() as conn:
        patterns = get_cached_patterns(conn, min_sample=30, min_edge=0.05)
        ou_patterns = get_cached_ou_patterns(conn, min_sample=30, min_edge=0.05)
        tt_patterns = get_cached_tt_patterns(conn, min_sample=30, min_edge=0.05)

    logger.info(f"  Found {len(patterns)} ATS, {len(ou_patterns)} O/U, {len(tt_patterns)} TT cached patterns")

    # Generate recommendations for all sports
    all_recommendations = []
    with engine.connect() as conn:
        for sport in SPORTS:
            try:
                sport_recs = generate_recommendations(
                    conn, sport, patterns,
                    ou_patterns=ou_patterns, tt_patterns=tt_patterns
                )
                all_recommendations.extend(sport_recs)
                logger.info(f"  {sport}: {len(sport_recs)} games analyzed")
            except Exception as e:
                logger.warning(f"  Error generating recommendations for {sport}: {e}")

    # Write to database
    if all_recommendations:
        written = write_todays_recommendations(engine, all_recommendations, today_est, computed_at)
        logger.info(f"  Wrote {written} recommendations to cache")

        # Persist individual predictions for outcome tracking
        try:
            from src.analysis.prediction_tracking import write_prediction_tracking
            tracked = write_prediction_tracking(engine, all_recommendations, today_est, computed_at)
            logger.info(f"  Tracked {tracked} individual predictions")
        except Exception as e:
            logger.warning(f"  Error tracking predictions (non-fatal): {e}")

        return written
    else:
        logger.info("  No games scheduled today")
        return 0


def lambda_handler(event, context):
    """Lambda entry point - generates current rankings, streaks, patterns, and recommendations."""
    logger.info("=" * 60)
    logger.info("Starting current rankings, streaks, patterns, and recommendations generation")
    logger.info("=" * 60)

    try:
        engine = get_db_engine()
        if engine is None:
            raise Exception("Could not connect to database")

        logger.info("Connected to database")

        computed_at = datetime.now(timezone.utc)
        results = []
        total_rankings = 0
        total_streaks = 0
        total_patterns = 0

        for sport in SPORTS:
            try:
                result = process_sport(engine, sport, computed_at)
                results.append(result)
                total_rankings += result['rankings']
                total_streaks += result['streaks']
                total_patterns += result['patterns']
                logger.info(f"{sport}: {result['rankings']} rankings, {result['streaks']} streaks, {result['patterns']} patterns")
            except Exception as e:
                logger.error(f"Error processing {sport}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Generate today's recommendations (uses patterns we just computed)
        total_recommendations = 0
        try:
            total_recommendations = generate_todays_recommendations(engine, computed_at)
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.info("=" * 60)
        logger.info(f"COMPLETE: {total_rankings} rankings, {total_streaks} streaks, {total_patterns} patterns, {total_recommendations} recommendations")
        logger.info("=" * 60)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Current rankings, streaks, patterns, and recommendations generation complete',
                'total_rankings': total_rankings,
                'total_streaks': total_streaks,
                'total_patterns': total_patterns,
                'total_recommendations': total_recommendations,
                'results': results,
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
    if not os.environ.get('DATABASE_URL'):
        print("Set DATABASE_URL environment variable for local testing")
        print("Example: export DATABASE_URL='postgresql://user:pass@host:5432/dbname'")
    else:
        result = lambda_handler({}, None)
        print(json.dumps(json.loads(result['body']), indent=2))
