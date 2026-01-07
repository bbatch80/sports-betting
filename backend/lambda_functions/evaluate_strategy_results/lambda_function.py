"""
Lambda Function: Evaluate Strategy Results
Matches yesterday's predictions to actual game outcomes and tracks strategy performance.

Triggered by: EventBridge schedule (daily at 3:00 AM EST, after all games complete)
"""

import json
import boto3
import pandas as pd
import io
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import logging

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'
REGION = 'us-east-1'

# Sports configuration
SPORTS = ['nfl', 'nba', 'ncaam']

# Strategy configurations with their handicap points
STRATEGY_CONFIG = {
    'home_focus': {'handicap': 'variable', 'bet_on_field': None},
    'away_focus': {'handicap': 'variable', 'bet_on_field': None},
    'coverage_based': {'handicap': 0, 'bet_on_field': 'better_team'},
    'elite_team': {'handicap': 0, 'bet_on_field': 'elite_team'},
    'hot_vs_cold': {'handicap': 11, 'bet_on_field': 'bet_on'},
    'opponent_perfect_form': {'handicap': 11, 'bet_on_field': 'bet_on'},
    'common_opponent': {'handicap': 0, 'bet_on_field': None}  # NCAAM only
}

# Sport-specific handicap values (used for home_focus, away_focus)
SPORT_HANDICAP = {
    'nfl': 5,
    'nba': 9,
    'ncaam': 10
}


def read_predictions_from_s3(sport: str, date: str) -> Optional[Dict[str, Any]]:
    """Read predictions JSON from S3 for a specific date"""
    s3_key = f"predictions/predictions_{sport}_{date}.json"

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        predictions = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Read predictions from S3: {s3_key}")
        return predictions
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"No predictions file found: {s3_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading predictions from S3: {e}")
        return None


def read_results_from_s3(sport: str) -> Optional[pd.DataFrame]:
    """Read season results Excel from S3"""
    s3_key = f"data/results/{sport}_season_results.xlsx"

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        excel_data = response['Body'].read()
        df = pd.read_excel(io.BytesIO(excel_data))
        logger.info(f"Read {len(df)} results from S3: {s3_key}")
        return df
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"No results file found: {s3_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading results from S3: {e}")
        return None


def read_performance_from_s3(sport: str) -> Dict[str, Any]:
    """Read existing cumulative performance data from S3"""
    s3_key = f"strategy_tracking/performance/{sport}_strategy_performance.json"

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        performance = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Read existing performance from S3: {s3_key}")
        return performance
    except s3_client.exceptions.NoSuchKey:
        logger.info(f"No existing performance file, will create new: {s3_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading performance from S3: {e}")
        return None


def write_daily_results_to_s3(results: Dict[str, Any], sport: str, date: str):
    """Write daily strategy results to S3"""
    s3_key = f"strategy_tracking/results_{sport}_{date}.json"

    try:
        json_data = json.dumps(results, indent=2, default=str)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Wrote daily results to S3: {s3_key}")
    except Exception as e:
        logger.error(f"Error writing daily results to S3: {e}")
        raise


def write_performance_to_s3(performance: Dict[str, Any], sport: str):
    """Write cumulative performance data to S3"""
    s3_key = f"strategy_tracking/performance/{sport}_strategy_performance.json"

    try:
        json_data = json.dumps(performance, indent=2, default=str)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Wrote performance to S3: {s3_key}")
    except Exception as e:
        logger.error(f"Error writing performance to S3: {e}")
        raise


def find_game_result(
    home_team: str,
    away_team: str,
    game_date: str,
    results_df: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Find the actual game result for a prediction

    Args:
        home_team: Home team name from prediction
        away_team: Away team name from prediction
        game_date: Date of the game (YYYY-MM-DD)
        results_df: DataFrame with season results

    Returns:
        Dict with game result or None if not found
    """
    # Convert game_date to match results format
    target_date = pd.to_datetime(game_date).date()

    # Try to match the game
    for _, row in results_df.iterrows():
        row_date = pd.to_datetime(row['game_date']).date() if pd.notna(row.get('game_date')) else None

        if row_date != target_date:
            continue

        # Check if teams match (handle different naming conventions)
        row_home = str(row.get('home_team', '')).lower()
        row_away = str(row.get('away_team', '')).lower()
        pred_home = home_team.lower()
        pred_away = away_team.lower()

        # Match if both teams are found (either direction handles potential mismatches)
        if (pred_home in row_home or row_home in pred_home) and \
           (pred_away in row_away or row_away in pred_away):
            return {
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_score': row.get('home_score'),
                'away_score': row.get('away_score'),
                'closing_spread': row.get('closing_spread'),
                'spread_result_difference': row.get('spread_result_difference'),
                'game_date': str(row_date)
            }

    return None


def calculate_bet_result(
    bet_on_team: str,
    bet_on_position: str,  # 'home' or 'away'
    handicap_points: float,
    game_result: Dict[str, Any]
) -> str:
    """
    Calculate if a bet won or lost

    Args:
        bet_on_team: Team name we bet on
        bet_on_position: 'home' or 'away' - position of the team we bet on
        handicap_points: Handicap points applied to the bet
        game_result: Dict with actual game results

    Returns:
        'win', 'loss', or 'push'
    """
    spread_result_diff = game_result.get('spread_result_difference')

    if spread_result_diff is None:
        return 'unknown'

    # spread_result_difference = (home_score - away_score) + closing_spread
    # Positive means home team covered the spread
    # Negative means away team covered the spread

    if bet_on_position == 'home':
        # For home team bet: need spread_result_diff > handicap
        adjusted_result = spread_result_diff - handicap_points
        if adjusted_result > 0:
            return 'win'
        elif adjusted_result < 0:
            return 'loss'
        else:
            return 'push'
    else:  # away
        # For away team bet: need spread_result_diff < -handicap
        adjusted_result = spread_result_diff + handicap_points
        if adjusted_result < 0:
            return 'win'
        elif adjusted_result > 0:
            return 'loss'
        else:
            return 'push'


def evaluate_strategy_predictions(
    strategy_name: str,
    opportunities: List[Dict[str, Any]],
    results_df: pd.DataFrame,
    sport: str,
    prediction_date: str
) -> Dict[str, Any]:
    """
    Evaluate predictions for a single strategy

    Args:
        strategy_name: Name of the strategy
        opportunities: List of opportunity predictions
        results_df: DataFrame with actual game results
        sport: Sport key
        prediction_date: Date of predictions (YYYY-MM-DD)

    Returns:
        Dict with strategy evaluation results
    """
    strategy_config = STRATEGY_CONFIG.get(strategy_name, {})
    handicap = strategy_config.get('handicap', 0)
    bet_on_field = strategy_config.get('bet_on_field')

    # Get sport-specific handicap for variable strategies
    if handicap == 'variable':
        handicap = SPORT_HANDICAP.get(sport, 0)

    evaluated_games = []
    wins = 0
    losses = 0
    pushes = 0

    for opp in opportunities:
        home_team = opp.get('home_team', '')
        away_team = opp.get('away_team', '')

        # Find the actual game result
        game_result = find_game_result(home_team, away_team, prediction_date, results_df)

        if not game_result:
            logger.warning(f"Could not find result for {away_team} @ {home_team} on {prediction_date}")
            continue

        # Determine which team we bet on
        if bet_on_field:
            bet_on_team = opp.get(bet_on_field, '')
        elif strategy_name == 'home_focus':
            bet_on_team = home_team
        elif strategy_name == 'away_focus':
            bet_on_team = away_team
        else:
            # For common_opponent, use projected winner
            projections = opp.get('projections', {})
            actual_game = projections.get('actual_game', {})
            bet_on_team = actual_game.get('winner', home_team)

        # Determine bet position
        if bet_on_team.lower() in home_team.lower() or home_team.lower() in bet_on_team.lower():
            bet_on_position = 'home'
        else:
            bet_on_position = 'away'

        # Calculate result
        result = calculate_bet_result(bet_on_team, bet_on_position, handicap, game_result)

        if result == 'win':
            wins += 1
        elif result == 'loss':
            losses += 1
        elif result == 'push':
            pushes += 1

        evaluated_games.append({
            'home_team': home_team,
            'away_team': away_team,
            'bet_on': bet_on_team,
            'bet_position': bet_on_position,
            'handicap': handicap,
            'spread': game_result.get('closing_spread'),
            'spread_result_diff': game_result.get('spread_result_difference'),
            'home_score': game_result.get('home_score'),
            'away_score': game_result.get('away_score'),
            'result': result
        })

    return {
        'predictions': len(evaluated_games),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'games': evaluated_games
    }


def update_cumulative_performance(
    existing_performance: Optional[Dict[str, Any]],
    daily_results: Dict[str, Any],
    sport: str,
    date: str
) -> Dict[str, Any]:
    """
    Update cumulative performance with daily results

    Args:
        existing_performance: Existing cumulative data or None
        daily_results: Today's evaluation results
        sport: Sport key
        date: Date string (YYYY-MM-DD)

    Returns:
        Updated cumulative performance dict
    """
    if existing_performance is None:
        # Initialize new performance structure
        performance = {
            'sport': sport,
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'strategies': {}
        }
    else:
        performance = existing_performance.copy()
        performance['last_updated'] = datetime.now(timezone.utc).isoformat()

    strategy_results = daily_results.get('strategy_results', {})

    for strategy_name, results in strategy_results.items():
        if strategy_name not in performance['strategies']:
            # Initialize strategy tracking
            performance['strategies'][strategy_name] = {
                'total_predictions': 0,
                'total_wins': 0,
                'total_losses': 0,
                'total_pushes': 0,
                'current_win_rate': 0.0,
                'current_streak': 0,
                'streak_type': None,
                'daily_cumulative': []
            }

        strategy_perf = performance['strategies'][strategy_name]

        # Update totals
        strategy_perf['total_predictions'] += results.get('predictions', 0)
        strategy_perf['total_wins'] += results.get('wins', 0)
        strategy_perf['total_losses'] += results.get('losses', 0)
        strategy_perf['total_pushes'] += results.get('pushes', 0)

        # Calculate win rate (excluding pushes)
        decided = strategy_perf['total_wins'] + strategy_perf['total_losses']
        if decided > 0:
            strategy_perf['current_win_rate'] = round(strategy_perf['total_wins'] / decided, 4)

        # Update streak
        day_wins = results.get('wins', 0)
        day_losses = results.get('losses', 0)

        if day_wins > day_losses:
            if strategy_perf['streak_type'] == 'win':
                strategy_perf['current_streak'] += 1
            else:
                strategy_perf['streak_type'] = 'win'
                strategy_perf['current_streak'] = 1
        elif day_losses > day_wins:
            if strategy_perf['streak_type'] == 'loss':
                strategy_perf['current_streak'] += 1
            else:
                strategy_perf['streak_type'] = 'loss'
                strategy_perf['current_streak'] = 1
        # If equal, maintain current streak

        # Add daily cumulative entry
        strategy_perf['daily_cumulative'].append({
            'date': date,
            'day_predictions': results.get('predictions', 0),
            'day_wins': results.get('wins', 0),
            'day_losses': results.get('losses', 0),
            'cumulative_wins': strategy_perf['total_wins'],
            'cumulative_total': decided,
            'cumulative_rate': strategy_perf['current_win_rate']
        })

    return performance


def evaluate_sport(sport: str, date: str) -> Dict[str, Any]:
    """
    Evaluate all strategies for a sport on a given date

    Args:
        sport: Sport key (nfl, nba, ncaam)
        date: Date to evaluate (YYYY-MM-DD)

    Returns:
        Dict with evaluation results
    """
    logger.info(f"Evaluating {sport.upper()} for {date}")

    # Read predictions for that date
    predictions = read_predictions_from_s3(sport, date)
    if not predictions:
        logger.info(f"No predictions found for {sport} on {date}")
        return {'sport': sport, 'date': date, 'strategy_results': {}, 'error': 'No predictions'}

    # Read current results
    results_df = read_results_from_s3(sport)
    if results_df is None or len(results_df) == 0:
        logger.warning(f"No results data for {sport}")
        return {'sport': sport, 'date': date, 'strategy_results': {}, 'error': 'No results data'}

    # Filter to completed games only
    results_df = results_df[
        (results_df['home_score'].notna()) &
        (results_df['away_score'].notna()) &
        (results_df['spread_result_difference'].notna())
    ].copy()

    # Get prediction date from predictions
    prediction_date = predictions.get('prediction_date', date)

    # Evaluate each strategy
    strategies = predictions.get('strategies', {})
    strategy_results = {}

    for strategy_name, strategy_data in strategies.items():
        opportunities = strategy_data.get('opportunities', [])

        if not opportunities:
            logger.info(f"No opportunities for {strategy_name}")
            strategy_results[strategy_name] = {
                'predictions': 0,
                'wins': 0,
                'losses': 0,
                'games': []
            }
            continue

        results = evaluate_strategy_predictions(
            strategy_name=strategy_name,
            opportunities=opportunities,
            results_df=results_df,
            sport=sport,
            prediction_date=prediction_date
        )

        strategy_results[strategy_name] = results
        logger.info(f"{strategy_name}: {results['wins']}W-{results['losses']}L from {results['predictions']} predictions")

    return {
        'sport': sport,
        'date': date,
        'prediction_date': prediction_date,
        'strategy_results': strategy_results
    }


def lambda_handler(event, context):
    """
    Lambda handler - evaluates yesterday's predictions against actual results

    Triggered daily at 3 AM EST to ensure all games have completed
    """
    logger.info("="*80)
    logger.info("Starting strategy results evaluation")
    logger.info("="*80)

    try:
        # Determine yesterday's date (the predictions we're evaluating)
        est = timezone(timedelta(hours=-5))
        now_est = datetime.now(est)
        yesterday = (now_est - timedelta(days=1)).date()
        yesterday_str = yesterday.isoformat()

        logger.info(f"Evaluating predictions from: {yesterday_str}")

        results_summary = []

        for sport in SPORTS:
            try:
                # Evaluate the sport
                daily_results = evaluate_sport(sport, yesterday_str)

                if 'error' not in daily_results:
                    # Save daily results
                    write_daily_results_to_s3(daily_results, sport, yesterday_str)

                    # Update cumulative performance
                    existing_perf = read_performance_from_s3(sport)
                    updated_perf = update_cumulative_performance(
                        existing_perf, daily_results, sport, yesterday_str
                    )
                    write_performance_to_s3(updated_perf, sport)

                    # Summarize
                    total_predictions = sum(
                        r.get('predictions', 0)
                        for r in daily_results.get('strategy_results', {}).values()
                    )
                    total_wins = sum(
                        r.get('wins', 0)
                        for r in daily_results.get('strategy_results', {}).values()
                    )
                    total_losses = sum(
                        r.get('losses', 0)
                        for r in daily_results.get('strategy_results', {}).values()
                    )

                    results_summary.append({
                        'sport': sport,
                        'date': yesterday_str,
                        'predictions': total_predictions,
                        'wins': total_wins,
                        'losses': total_losses,
                        'status': 'success'
                    })

                    logger.info(f"✓ {sport.upper()}: {total_wins}W-{total_losses}L from {total_predictions} predictions")
                else:
                    results_summary.append({
                        'sport': sport,
                        'date': yesterday_str,
                        'status': 'skipped',
                        'reason': daily_results.get('error')
                    })
                    logger.info(f"⚠ {sport.upper()}: Skipped - {daily_results.get('error')}")

            except Exception as e:
                logger.error(f"✗ Error processing {sport}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results_summary.append({
                    'sport': sport,
                    'date': yesterday_str,
                    'status': 'error',
                    'error': str(e)
                })

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Strategy results evaluated',
                'evaluation_date': yesterday_str,
                'results': results_summary,
                'evaluated_at': datetime.now(timezone.utc).isoformat()
            })
        }

    except Exception as e:
        logger.error(f"Fatal error in Lambda function: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
