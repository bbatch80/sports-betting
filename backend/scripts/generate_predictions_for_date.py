"""
Generate predictions for a specific date
This script can be used to generate predictions for yesterday or any past date
for testing purposes.

Usage:
    python generate_predictions_for_date.py --sport nfl --date 2025-01-27
    python generate_predictions_for_date.py --sport nba --date yesterday
"""

import os
import sys
import argparse
import json
import boto3
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from odds_api_client import OddsAPIClient, OddsAPIError
import config
import score_storage

# Import functions from generate_predictions lambda
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda_functions', 'generate_predictions'))

# We'll need to copy the key functions from the lambda
# For now, let's create a simplified version

SPORTS_CONFIG = {
    'nfl': {
        'api_key': 'americanfootball_nfl',
        'handicap_points': 5,
        'name': 'NFL'
    },
    'nba': {
        'api_key': 'basketball_nba',
        'handicap_points': 9,
        'name': 'NBA'
    },
    'ncaam': {
        'api_key': 'basketball_ncaab',
        'handicap_points': 9,  # Note: Lambda uses 10, but you said 9
        'name': 'NCAAM'
    }
}

DEFAULT_REGIONS = ['us']
API_RATE_LIMIT_DELAY = 1.0
BUCKET_NAME = 'sports-betting-analytics-data'


def parse_date(date_str: str) -> datetime:
    """Parse date string (YYYY-MM-DD or 'yesterday')"""
    if date_str.lower() == 'yesterday':
        est = timezone(timedelta(hours=-5))
        yesterday = datetime.now(est) - timedelta(days=1)
        return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            est = timezone(timedelta(hours=-5))
            return date.replace(tzinfo=est)
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or 'yesterday'")


def fetch_games_for_date(api_key: str, sport_key: str, target_date: datetime) -> List[Dict[str, Any]]:
    """Fetch games for a specific date from Odds API"""
    api_sport_key = SPORTS_CONFIG[sport_key]['api_key']
    target_date_est = target_date.date()
    
    print(f"Fetching {sport_key.upper()} games for {target_date_est}...")
    
    try:
        client = OddsAPIClient()
        time.sleep(API_RATE_LIMIT_DELAY)
        
        # Get historical odds for the date
        # We need to use the historical endpoint or scores endpoint
        scores = client.get_scores(sport=api_sport_key, days_from=3)
        
        # Filter to target date
        target_games = []
        for game in scores:
            if not isinstance(game, dict):
                continue
            
            commence_time = game.get('commence_time')
            if commence_time:
                try:
                    event_time = pd.to_datetime(commence_time)
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    
                    est = timezone(timedelta(hours=-5))
                    event_time_est = event_time.astimezone(est)
                    
                    if event_time_est.date() == target_date_est:
                        target_games.append(game)
                except:
                    pass
        
        print(f"Found {len(target_games)} games for {target_date_est}")
        return target_games
        
    except Exception as e:
        print(f"Error fetching games: {e}")
        return []


def load_historical_data(sport_key: str) -> pd.DataFrame:
    """Load historical results from S3"""
    import io
    s3_client = boto3.client('s3')
    s3_key = f"data/results/{sport_key}_season_results.xlsx"
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        excel_data = response['Body'].read()
        df = pd.read_excel(io.BytesIO(excel_data))
        print(f"Loaded {len(df)} historical games from S3")
        return df
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return pd.DataFrame()


def generate_predictions_for_date(sport_key: str, target_date: datetime):
    """
    Generate predictions for a specific date
    
    This is a simplified version - you may need to adapt it based on your
    actual prediction generation logic from the lambda function.
    """
    print(f"\n{'='*80}")
    print(f"Generating predictions for {sport_key.upper()} on {target_date.date()}")
    print(f"{'='*80}\n")
    
    # Load historical data
    df_historical = load_historical_data(sport_key)
    
    if len(df_historical) == 0:
        print("No historical data available")
        return None
    
    # Filter to completed games
    df_completed = df_historical[
        (df_historical['home_score'].notna()) & 
        (df_historical['away_score'].notna()) & 
        (df_historical['closing_spread'].notna())
    ].copy()
    
    print(f"Using {len(df_completed)} completed games for statistics")
    
    # Get games for target date
    # Note: For past dates, we need to get games from results data instead
    # since the Odds API won't have upcoming games for past dates
    target_date_str = target_date.date().isoformat()
    
    # Convert game_date to datetime if it's not already
    if 'game_date' in df_historical.columns:
        df_historical['game_date'] = pd.to_datetime(df_historical['game_date'])
    
    # Get games from results that match the target date
    target_date_only = target_date.date()
    games_from_results = df_historical[
        df_historical['game_date'].dt.date == target_date_only
    ].copy()
    
    if len(games_from_results) == 0:
        print(f"No games found in results for {target_date_str}")
        print(f"Available dates in results: {df_historical['game_date'].dt.date.unique()[-5:] if len(df_historical) > 0 else 'None'}")
        return None
    
    print(f"Found {len(games_from_results)} games from results for {target_date_str}")
    
    # For now, create a basic predictions structure
    # You'll need to implement the full prediction logic here
    # This is a placeholder that creates predictions based on historical data
    
    opportunities = []
    for _, game in games_from_results.iterrows():
        # Create a basic opportunity structure
        # You'll need to calculate the actual percentages based on team stats
        opportunities.append({
            'game_time_est': f"{target_date_str} 12:00:00 EST",  # Placeholder
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'current_spread': game.get('closing_spread'),
            'home_cover_pct_handicap': 50.0,  # Placeholder - calculate from stats
            'away_cover_pct_handicap': 50.0,  # Placeholder
            'handicap_pct_difference': 0.0,  # Placeholder
        })
    
    predictions_data = {
        'sport': sport_key,
        'sport_name': SPORTS_CONFIG[sport_key]['name'],
        'handicap_points': SPORTS_CONFIG[sport_key]['handicap_points'],
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'prediction_date': target_date_str,
        'games': [],
        'opportunities': opportunities,
        'summary': {
            'total_games': len(games_from_results),
            'opportunities': len(opportunities)
        }
    }
    
    # Save to S3 with date suffix
    s3_client = boto3.client('s3')
    s3_key = f"predictions/predictions_{sport_key}_{target_date_str}.json"
    
    json_data = json.dumps(predictions_data, indent=2, default=str)
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json_data.encode('utf-8'),
        ContentType='application/json'
    )
    
    print(f"\n✓ Saved predictions to S3: {s3_key}")
    print(f"  Opportunities: {len(opportunities)}")
    
    return predictions_data


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for a specific date')
    parser.add_argument('--sport', required=True, choices=['nfl', 'nba', 'ncaam'],
                       help='Sport to generate predictions for')
    parser.add_argument('--date', required=True,
                       help="Date in YYYY-MM-DD format or 'yesterday'")
    
    args = parser.parse_args()
    
    try:
        target_date = parse_date(args.date)
        predictions = generate_predictions_for_date(args.sport, target_date)
        
        if predictions:
            print(f"\n✓ Successfully generated predictions for {args.sport} on {target_date.date()}")
        else:
            print(f"\n✗ Failed to generate predictions")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import io
    main()

