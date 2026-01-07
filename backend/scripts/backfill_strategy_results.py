"""
Backfill strategy results from season start to yesterday.

This script processes all historical predictions and matches them to actual
game outcomes to build the cumulative strategy performance data.

Usage:
    python backfill_strategy_results.py --sport nba --start-date 2024-10-22
    python backfill_strategy_results.py --sport nfl --start-date 2024-09-05
    python backfill_strategy_results.py --sport ncaam --start-date 2024-11-04
    python backfill_strategy_results.py --all-sports
"""

import os
import sys
import argparse
import json
import boto3
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from collections import defaultdict
import io

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'

# Season start dates for each sport (approximate)
SEASON_START_DATES = {
    'nfl': '2024-09-05',
    'nba': '2024-10-22',
    'ncaam': '2024-11-04'
}

# Sport-specific handicaps for home_focus and away_focus strategies
SPORT_HANDICAP = {
    'nfl': 5,
    'nba': 9,
    'ncaam': 10
}

# Strategy configurations
STRATEGY_CONFIG = {
    'home_focus': {'handicap': 'variable', 'bet_on_field': None},
    'away_focus': {'handicap': 'variable', 'bet_on_field': None},
    'coverage_based': {'handicap': 0, 'bet_on_field': 'better_team'},
    'elite_team': {'handicap': 0, 'bet_on_field': 'elite_team'},
    'hot_vs_cold': {'handicap': 11, 'bet_on_field': 'bet_on'},
    'opponent_perfect_form': {'handicap': 11, 'bet_on_field': 'bet_on'},
    'common_opponent': {'handicap': 0, 'bet_on_field': None}
}


def get_s3_client():
    """Get S3 client"""
    return boto3.client('s3')


def read_json_from_s3(s3_key: str) -> Optional[Dict]:
    """Read JSON file from S3"""
    try:
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Error reading {s3_key}: {e}")
        return None


def write_json_to_s3(s3_key: str, data: Dict):
    """Write JSON file to S3"""
    s3_client = get_s3_client()
    json_data = json.dumps(data, indent=2, default=str)
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json_data.encode('utf-8'),
        ContentType='application/json'
    )
    print(f"  Saved: s3://{BUCKET_NAME}/{s3_key}")


def read_excel_from_s3(s3_key: str) -> Optional[pd.DataFrame]:
    """Read Excel file from S3"""
    try:
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        excel_data = response['Body'].read()
        return pd.read_excel(io.BytesIO(excel_data))
    except Exception as e:
        print(f"Error reading Excel from {s3_key}: {e}")
        return None


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching"""
    if not name:
        return ""
    return name.lower().strip().replace(" ", "").replace("-", "").replace(".", "")


def calculate_bet_result(
    home_team: str,
    away_team: str,
    bet_on: str,
    home_score: float,
    away_score: float,
    closing_spread: float,
    handicap: float
) -> Dict[str, Any]:
    """
    Calculate whether a bet won or lost.

    Args:
        home_team: Home team name
        away_team: Away team name
        bet_on: Which team was bet on ('home' or 'away')
        home_score: Final home score
        away_score: Final away score
        closing_spread: The closing spread (negative means home favored)
        handicap: Additional handicap points for the strategy

    Returns:
        Dict with result ('win', 'loss', 'push'), margin, etc.
    """
    # Calculate spread result difference
    # spread_result_diff = (home_score - away_score) + closing_spread
    # Positive means home covered, negative means away covered
    actual_margin = home_score - away_score
    spread_result_diff = actual_margin + closing_spread

    # Determine bet outcome based on which team was bet on
    is_home_bet = normalize_team_name(bet_on) == normalize_team_name(home_team)

    if is_home_bet:
        # Betting on home: win if spread_result_diff > handicap
        if spread_result_diff > handicap:
            result = 'win'
        elif spread_result_diff < handicap:
            result = 'loss'
        else:
            result = 'push'
    else:
        # Betting on away: win if spread_result_diff < -handicap
        if spread_result_diff < -handicap:
            result = 'win'
        elif spread_result_diff > -handicap:
            result = 'loss'
        else:
            result = 'push'

    return {
        'result': result,
        'actual_margin': actual_margin,
        'spread_result_diff': spread_result_diff,
        'handicap_applied': handicap
    }


def find_game_result(
    prediction: Dict,
    results_df: pd.DataFrame,
    game_date: str
) -> Optional[Dict]:
    """Find the matching game result for a prediction"""
    home_team_pred = normalize_team_name(prediction.get('home_team', ''))
    away_team_pred = normalize_team_name(prediction.get('away_team', ''))

    for _, row in results_df.iterrows():
        home_team_result = normalize_team_name(str(row.get('home_team', '')))
        away_team_result = normalize_team_name(str(row.get('away_team', '')))

        # Check if teams match
        if home_team_pred == home_team_result and away_team_pred == away_team_result:
            # Check if scores are available
            if pd.notna(row.get('home_score')) and pd.notna(row.get('away_score')):
                return {
                    'home_team': row.get('home_team'),
                    'away_team': row.get('away_team'),
                    'home_score': float(row.get('home_score')),
                    'away_score': float(row.get('away_score')),
                    'closing_spread': float(row.get('closing_spread', 0))
                }

    return None


def get_strategy_predictions_from_opportunity(
    opp: Dict,
    sport: str
) -> List[Dict]:
    """
    Extract strategy predictions from an opportunity.
    Returns list of (strategy_name, bet_on, handicap) tuples.
    """
    predictions = []
    home_team = opp.get('home_team')
    away_team = opp.get('away_team')
    sport_handicap = SPORT_HANDICAP.get(sport, 9)

    # Check each strategy
    strategies = opp.get('strategies', {})

    for strategy_name, strategy_data in strategies.items():
        if not strategy_data or not strategy_data.get('recommended'):
            continue

        bet_on = strategy_data.get('bet_on')
        if not bet_on:
            continue

        # Get handicap for this strategy
        config = STRATEGY_CONFIG.get(strategy_name, {})
        if config.get('handicap') == 'variable':
            handicap = sport_handicap
        else:
            handicap = config.get('handicap', 0)

        predictions.append({
            'strategy': strategy_name,
            'bet_on': bet_on,
            'handicap': handicap,
            'home_team': home_team,
            'away_team': away_team,
            'spread': opp.get('current_spread'),
            'confidence': strategy_data.get('confidence'),
            'reason': strategy_data.get('reason')
        })

    # Also check for direct strategy fields in opportunities
    # home_focus / away_focus
    if opp.get('home_cover_pct_handicap') and opp.get('away_cover_pct_handicap'):
        home_pct = opp.get('home_cover_pct_handicap', 0)
        away_pct = opp.get('away_cover_pct_handicap', 0)

        if home_pct > away_pct:
            predictions.append({
                'strategy': 'home_focus',
                'bet_on': home_team,
                'handicap': sport_handicap,
                'home_team': home_team,
                'away_team': away_team,
                'spread': opp.get('current_spread'),
                'confidence': home_pct
            })
        elif away_pct > home_pct:
            predictions.append({
                'strategy': 'away_focus',
                'bet_on': away_team,
                'handicap': sport_handicap,
                'home_team': home_team,
                'away_team': away_team,
                'spread': opp.get('current_spread'),
                'confidence': away_pct
            })

    # coverage_based
    if 'handicap_pct_difference' in opp:
        pct_diff = opp.get('handicap_pct_difference', 0)
        if abs(pct_diff) >= 10:
            bet_on = home_team if pct_diff > 0 else away_team
            predictions.append({
                'strategy': 'coverage_based',
                'bet_on': bet_on,
                'handicap': 0,
                'home_team': home_team,
                'away_team': away_team,
                'spread': opp.get('current_spread'),
                'confidence': abs(pct_diff)
            })

    # hot_vs_cold
    if opp.get('hot_vs_cold'):
        hvc = opp.get('hot_vs_cold')
        if hvc.get('bet_on'):
            predictions.append({
                'strategy': 'hot_vs_cold',
                'bet_on': hvc.get('bet_on'),
                'handicap': 11,
                'home_team': home_team,
                'away_team': away_team,
                'spread': opp.get('current_spread'),
                'confidence': hvc.get('confidence')
            })

    # opponent_perfect_form
    if opp.get('opponent_perfect_form'):
        opf = opp.get('opponent_perfect_form')
        if opf.get('bet_on'):
            predictions.append({
                'strategy': 'opponent_perfect_form',
                'bet_on': opf.get('bet_on'),
                'handicap': 11,
                'home_team': home_team,
                'away_team': away_team,
                'spread': opp.get('current_spread'),
                'confidence': opf.get('confidence')
            })

    # elite_team
    if opp.get('elite_team'):
        et = opp.get('elite_team')
        if et.get('bet_on') or et.get('elite_team'):
            predictions.append({
                'strategy': 'elite_team',
                'bet_on': et.get('bet_on') or et.get('elite_team'),
                'handicap': 0,
                'home_team': home_team,
                'away_team': away_team,
                'spread': opp.get('current_spread'),
                'confidence': et.get('confidence')
            })

    # common_opponent (NCAAM only)
    if sport == 'ncaam' and opp.get('common_opponent'):
        co = opp.get('common_opponent')
        if co.get('bet_on') or co.get('better_team'):
            predictions.append({
                'strategy': 'common_opponent',
                'bet_on': co.get('bet_on') or co.get('better_team'),
                'handicap': 0,
                'home_team': home_team,
                'away_team': away_team,
                'spread': opp.get('current_spread'),
                'confidence': co.get('confidence')
            })

    # Deduplicate by strategy name
    seen = set()
    unique_predictions = []
    for p in predictions:
        if p['strategy'] not in seen:
            seen.add(p['strategy'])
            unique_predictions.append(p)

    return unique_predictions


def process_date(
    sport: str,
    date_str: str,
    results_df: pd.DataFrame
) -> Optional[Dict]:
    """Process predictions for a single date"""
    # Load predictions for this date
    predictions_key = f"predictions/predictions_{sport}_{date_str}.json"
    predictions_data = read_json_from_s3(predictions_key)

    if not predictions_data:
        return None

    opportunities = predictions_data.get('opportunities', [])
    if not opportunities:
        return None

    # Strategy results for this date
    strategy_results = defaultdict(lambda: {
        'predictions': 0,
        'wins': 0,
        'losses': 0,
        'pushes': 0,
        'games': []
    })

    # Process each opportunity
    for opp in opportunities:
        # Get strategy predictions from opportunity
        strategy_preds = get_strategy_predictions_from_opportunity(opp, sport)

        for pred in strategy_preds:
            # Find matching game result
            game_result = find_game_result(pred, results_df, date_str)

            if not game_result:
                continue

            # Calculate bet result
            bet_result = calculate_bet_result(
                home_team=pred['home_team'],
                away_team=pred['away_team'],
                bet_on=pred['bet_on'],
                home_score=game_result['home_score'],
                away_score=game_result['away_score'],
                closing_spread=game_result['closing_spread'],
                handicap=pred['handicap']
            )

            strategy_name = pred['strategy']
            strategy_results[strategy_name]['predictions'] += 1

            if bet_result['result'] == 'win':
                strategy_results[strategy_name]['wins'] += 1
            elif bet_result['result'] == 'loss':
                strategy_results[strategy_name]['losses'] += 1
            else:
                strategy_results[strategy_name]['pushes'] += 1

            strategy_results[strategy_name]['games'].append({
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'bet_on': pred['bet_on'],
                'spread': game_result['closing_spread'],
                'handicap': pred['handicap'],
                'home_score': game_result['home_score'],
                'away_score': game_result['away_score'],
                'actual_margin': bet_result['actual_margin'],
                'result': bet_result['result']
            })

    if not strategy_results:
        return None

    return {
        'date': date_str,
        'sport': sport,
        'strategy_results': dict(strategy_results)
    }


def build_cumulative_performance(
    sport: str,
    daily_results: List[Dict],
    season_start: str
) -> Dict:
    """Build cumulative performance data from daily results"""
    # Initialize cumulative tracking per strategy
    cumulative = defaultdict(lambda: {
        'total_predictions': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_pushes': 0,
        'current_streak': 0,
        'streak_type': None,
        'daily_cumulative': []
    })

    # Process daily results in chronological order
    for daily in sorted(daily_results, key=lambda x: x['date']):
        date_str = daily['date']

        for strategy_name, results in daily['strategy_results'].items():
            strat = cumulative[strategy_name]

            # Update totals
            strat['total_predictions'] += results['predictions']
            strat['total_wins'] += results['wins']
            strat['total_losses'] += results['losses']
            strat['total_pushes'] += results.get('pushes', 0)

            # Update streak
            for game in results.get('games', []):
                if game['result'] == 'win':
                    if strat['streak_type'] == 'win':
                        strat['current_streak'] += 1
                    else:
                        strat['streak_type'] = 'win'
                        strat['current_streak'] = 1
                elif game['result'] == 'loss':
                    if strat['streak_type'] == 'loss':
                        strat['current_streak'] += 1
                    else:
                        strat['streak_type'] = 'loss'
                        strat['current_streak'] = 1

            # Calculate cumulative rate
            total = strat['total_wins'] + strat['total_losses']
            rate = strat['total_wins'] / total if total > 0 else 0

            # Add daily cumulative data point
            strat['daily_cumulative'].append({
                'date': date_str,
                'cumulative_wins': strat['total_wins'],
                'cumulative_losses': strat['total_losses'],
                'cumulative_total': total,
                'cumulative_rate': round(rate, 4)
            })

    # Calculate final metrics
    for strategy_name, strat in cumulative.items():
        total = strat['total_wins'] + strat['total_losses']
        strat['current_win_rate'] = round(strat['total_wins'] / total, 4) if total > 0 else 0

    return {
        'sport': sport,
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'season_start': season_start,
        'strategies': dict(cumulative)
    }


def backfill_sport(sport: str, start_date: str, end_date: Optional[str] = None):
    """Backfill strategy results for a single sport"""
    print(f"\n{'='*80}")
    print(f"Backfilling {sport.upper()} strategy results")
    print(f"Start date: {start_date}")
    print(f"{'='*80}\n")

    # Load season results
    results_key = f"data/results/{sport}_season_results.xlsx"
    results_df = read_excel_from_s3(results_key)

    if results_df is None or len(results_df) == 0:
        print(f"ERROR: No results data found for {sport}")
        return

    # Filter to completed games
    results_df = results_df[
        (results_df['home_score'].notna()) &
        (results_df['away_score'].notna()) &
        (results_df['closing_spread'].notna())
    ].copy()

    print(f"Loaded {len(results_df)} completed games from results")

    # Convert game_date if needed
    if 'game_date' in results_df.columns:
        results_df['game_date'] = pd.to_datetime(results_df['game_date'])

    # Determine date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        # Default to yesterday (use naive datetime for comparison)
        est = timezone(timedelta(hours=-5))
        end_aware = datetime.now(est) - timedelta(days=1)
        end = datetime(end_aware.year, end_aware.month, end_aware.day)

    # Process each date
    daily_results = []
    current = start
    dates_processed = 0
    dates_with_data = 0

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        # Filter results to this date
        if 'game_date' in results_df.columns:
            date_results = results_df[
                results_df['game_date'].dt.date == current.date()
            ]
        else:
            date_results = results_df

        result = process_date(sport, date_str, date_results)

        if result:
            daily_results.append(result)
            dates_with_data += 1

            # Save daily results
            daily_key = f"strategy_tracking/results_{sport}_{date_str}.json"
            write_json_to_s3(daily_key, result)

        dates_processed += 1
        current += timedelta(days=1)

        # Progress indicator
        if dates_processed % 10 == 0:
            print(f"  Processed {dates_processed} dates, {dates_with_data} with data...")

    print(f"\nProcessed {dates_processed} dates, {dates_with_data} had prediction data")

    # Build and save cumulative performance
    if daily_results:
        performance = build_cumulative_performance(sport, daily_results, start_date)
        perf_key = f"strategy_tracking/performance/{sport}_strategy_performance.json"
        write_json_to_s3(perf_key, performance)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Strategy Performance Summary for {sport.upper()}")
        print(f"{'='*60}")

        for strategy_name, strat in performance['strategies'].items():
            total = strat['total_wins'] + strat['total_losses']
            if total > 0:
                print(f"\n{strategy_name}:")
                print(f"  Record: {strat['total_wins']}-{strat['total_losses']}")
                print(f"  Win Rate: {strat['current_win_rate']*100:.1f}%")
                print(f"  Streak: {strat['current_streak']} {strat['streak_type']}s")
    else:
        print("No prediction data found for any dates")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill strategy results from season start'
    )
    parser.add_argument(
        '--sport',
        choices=['nfl', 'nba', 'ncaam'],
        help='Sport to backfill'
    )
    parser.add_argument(
        '--start-date',
        help='Start date (YYYY-MM-DD). Defaults to season start.'
    )
    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD). Defaults to yesterday.'
    )
    parser.add_argument(
        '--all-sports',
        action='store_true',
        help='Process all sports from their season starts'
    )

    args = parser.parse_args()

    if args.all_sports:
        # Process all sports
        for sport, default_start in SEASON_START_DATES.items():
            start = args.start_date or default_start
            backfill_sport(sport, start, args.end_date)
    elif args.sport:
        # Process single sport
        start = args.start_date or SEASON_START_DATES.get(args.sport)
        if not start:
            print(f"ERROR: No start date specified for {args.sport}")
            sys.exit(1)
        backfill_sport(args.sport, start, args.end_date)
    else:
        parser.print_help()
        sys.exit(1)

    print("\n" + "="*80)
    print("Backfill complete!")
    print("="*80)


if __name__ == "__main__":
    main()
