"""
Lambda Function: Collect Yesterday's Games
This function collects completed games from yesterday and updates the data in S3.

Triggered by: EventBridge schedule (daily at 6:00 AM EST)
"""

import json
import boto3
import time
import pandas as pd
import io
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients (initialized once, reused across invocations)
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Configuration - these match what we set up in Phase 1
BUCKET_NAME = 'sports-betting-analytics-data'
SECRET_NAME = 'odds-api-key'
REGION = 'us-east-1'

# Sports we collect data for
SPORTS = ['nfl', 'nba', 'ncaam']

# Sport API key mappings (from config.py)
SPORT_API_KEYS = {
    'nfl': 'americanfootball_nfl',
    'nba': 'basketball_nba',
    'ncaam': 'basketball_ncaab'
}

# Default settings
DEFAULT_REGIONS = ['us']
API_RATE_LIMIT_DELAY = 1.0

# Database configuration
DB_SECRET_NAME = 'sports-betting-db-credentials'
_db_engine = None


def get_database_url() -> Optional[str]:
    """
    Get DATABASE_URL from environment variable or Secrets Manager.
    Environment variable takes precedence (for testing).
    """
    # Check environment variable first
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        return database_url

    # Fall back to Secrets Manager
    try:
        response = secrets_client.get_secret_value(SecretId=DB_SECRET_NAME)
        secret = json.loads(response['SecretString'])
        return secret.get('url')
    except Exception as e:
        logger.warning(f"Could not retrieve DATABASE_URL from Secrets Manager: {e}")
        return None


def get_db_engine():
    """
    Get or create database engine singleton.
    Uses NullPool for Lambda (let RDS Proxy handle pooling).
    """
    global _db_engine
    if _db_engine is None:
        database_url = get_database_url()
        if database_url:
            _db_engine = create_engine(database_url, poolclass=NullPool)
            logger.info("✓ Database engine created")
    return _db_engine


def write_games_to_database(df: pd.DataFrame, sport_key: str) -> int:
    """
    Write games to PostgreSQL database using upsert logic.

    Returns number of rows affected.
    """
    engine = get_db_engine()
    if engine is None:
        logger.warning("No database connection available, skipping database write")
        return 0

    if len(df) == 0:
        return 0

    # Prepare data for insertion
    games_data = []
    for _, row in df.iterrows():
        game_date = row.get('game_date')
        if hasattr(game_date, 'date'):
            game_date = game_date.date()
        elif isinstance(game_date, str):
            game_date = pd.to_datetime(game_date).date()

        # Calculate spread_result if not present
        spread_result = row.get('spread_result_difference')
        if pd.isna(spread_result):
            home_score = row.get('home_score')
            away_score = row.get('away_score')
            closing_spread = row.get('closing_spread')
            if pd.notna(home_score) and pd.notna(away_score) and pd.notna(closing_spread):
                spread_result = float(home_score) - float(away_score) + float(closing_spread)

        games_data.append({
            'sport': sport_key.upper(),
            'game_date': game_date.isoformat() if game_date else None,
            'home_team': row.get('home_team'),
            'away_team': row.get('away_team'),
            'closing_spread': float(row['closing_spread']) if pd.notna(row.get('closing_spread')) else None,
            'home_score': int(row['home_score']) if pd.notna(row.get('home_score')) else None,
            'away_score': int(row['away_score']) if pd.notna(row.get('away_score')) else None,
            'spread_result': float(spread_result) if pd.notna(spread_result) else None,
        })

    # Upsert using PostgreSQL ON CONFLICT
    upsert_sql = text("""
        INSERT INTO games (sport, game_date, home_team, away_team, closing_spread, home_score, away_score, spread_result)
        VALUES (:sport, :game_date, :home_team, :away_team, :closing_spread, :home_score, :away_score, :spread_result)
        ON CONFLICT (sport, game_date, home_team, away_team)
        DO UPDATE SET
            closing_spread = COALESCE(EXCLUDED.closing_spread, games.closing_spread),
            home_score = COALESCE(EXCLUDED.home_score, games.home_score),
            away_score = COALESCE(EXCLUDED.away_score, games.away_score),
            spread_result = COALESCE(EXCLUDED.spread_result, games.spread_result)
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(upsert_sql, games_data)
            affected = result.rowcount
            logger.info(f"✓ Database: upserted {affected} games for {sport_key.upper()}")
            return affected
    except Exception as e:
        logger.error(f"✗ Database write error: {e}")
        return 0


def get_api_key() -> str:
    """
    Get Odds API key from AWS Secrets Manager
    
    This replaces reading from environment variables.
    The key is stored securely in Secrets Manager.
    """
    try:
        response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        return response['SecretString']
    except Exception as e:
        logger.error(f"Error getting API key from Secrets Manager: {e}")
        raise


def read_excel_from_s3(sport_key: str) -> pd.DataFrame:
    """
    Read Excel file from S3
    
    Original script: pd.read_excel('data/results/nfl_season_results.xlsx')
    Lambda version: Read from S3 bucket
    """
    s3_key = f"data/results/{sport_key}_season_results.xlsx"
    
    try:
        # Download file from S3 to memory
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        excel_data = response['Body'].read()
        
        # Read Excel from bytes (using io.BytesIO)
        df = pd.read_excel(io.BytesIO(excel_data))
        logger.info(f"Read {len(df)} rows from S3: {s3_key}")
        return df
    except s3_client.exceptions.NoSuchKey:
        # File doesn't exist yet (first run) - return empty DataFrame
        logger.info(f"No existing file found in S3: {s3_key}, starting fresh")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading Excel from S3: {e}")
        # Return empty DataFrame on error
        return pd.DataFrame()


def write_excel_to_s3(df: pd.DataFrame, sport_key: str):
    """
    Write Excel file to S3
    
    Original script: df.to_excel('data/results/nfl_season_results.xlsx')
    Lambda version: Write to S3 bucket
    """
    s3_key = f"data/results/{sport_key}_season_results.xlsx"
    
    try:
        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'{sport_key.upper()} Season', index=False)
        
        excel_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=excel_buffer.read(),
            ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        logger.info(f"Wrote {len(df)} rows to S3: {s3_key}")
    except Exception as e:
        logger.error(f"Error writing Excel to S3: {e}")
        raise


def write_parquet_to_s3(df: pd.DataFrame, sport_key: str):
    """
    Write Parquet file to S3 (backup format)
    
    Original script: df.to_parquet('data/results/nfl_season_results.parquet')
    Lambda version: Write to S3 bucket
    """
    s3_key = f"data/results/{sport_key}_season_results.parquet"
    
    try:
        # Prepare DataFrame for Parquet (ensure game_date is datetime, not date)
        parquet_df = df.copy()
        if 'game_date' in parquet_df.columns:
            # Convert date to datetime for Parquet compatibility
            parquet_df['game_date'] = pd.to_datetime(parquet_df['game_date'])
        
        # Create Parquet file in memory
        parquet_buffer = io.BytesIO()
        parquet_df.to_parquet(parquet_buffer, compression='snappy', index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=parquet_buffer.read(),
            ContentType='application/octet-stream'
        )
        logger.info(f"Wrote {len(df)} rows to S3 (Parquet): {s3_key}")
    except Exception as e:
        logger.error(f"Error writing Parquet to S3: {e}")
        # Don't raise - Parquet is backup, Excel is primary


def load_scores_from_s3(sport_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Load stored scores from S3 JSON file
    
    Original script: score_storage.load_stored_scores(sport_key)
    Lambda version: Read JSON from S3
    """
    s3_key = f"scores/{sport_key}_scores.json"
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        scores_data = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Loaded {len(scores_data)} stored scores from S3")
        return scores_data
    except s3_client.exceptions.NoSuchKey:
        # File doesn't exist yet - return empty dict
        return {}
    except Exception as e:
        logger.error(f"Error loading scores from S3: {e}")
        return {}


def save_scores_to_s3(sport_key: str, scores: Dict[str, Dict[str, Any]]):
    """
    Save scores to S3 JSON file
    
    Original script: score_storage.save_score(...)
    Lambda version: Write JSON to S3
    """
    s3_key = f"scores/{sport_key}_scores.json"
    
    try:
        scores_json = json.dumps(scores, indent=2)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=scores_json.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Saved {len(scores)} scores to S3: {s3_key}")
    except Exception as e:
        logger.error(f"Error saving scores to S3: {e}")


def make_odds_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Any:
    """
    Make request to Odds API
    
    This replaces the OddsAPIClient class for Lambda.
    We use requests directly since we have the API key.
    """
    import requests
    
    url = f"https://api.the-odds-api.com/v4/{endpoint}"
    params['apiKey'] = api_key
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error calling Odds API: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error calling Odds API: {e}")
        raise


def get_closing_spread(api_key: str, sport_key: str, event_id: str, 
                       home_team: str, away_team: str, commence_time: str) -> Optional[Dict[str, Any]]:
    """
    Get closing spread from DraftKings using historical event odds endpoint
    
    This is the same logic as the original script, but uses the API key parameter.
    """
    api_sport_key = SPORT_API_KEYS[sport_key]
    
    try:
        # Get snapshot 5 minutes before game start
        game_start = pd.to_datetime(commence_time)
        if game_start.tzinfo is None:
            game_start = game_start.replace(tzinfo=timezone.utc)
        snapshot_time = game_start - timedelta(minutes=5)
        snapshot_date = snapshot_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        time.sleep(API_RATE_LIMIT_DELAY)
        
        # Get historical odds
        historical_odds = make_odds_api_request(
            endpoint=f"historical/sports/{api_sport_key}/events/{event_id}/odds",
            params={
                'date': snapshot_date,
                'markets': 'spreads',
                'regions': ','.join(DEFAULT_REGIONS),
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            },
            api_key=api_key
        )
        
        if not historical_odds or not historical_odds.get('data'):
            return None
        
        event_odds = historical_odds['data']
        bookmakers = event_odds.get('bookmakers', [])
        
        # Find DraftKings
        for bookmaker in bookmakers:
            if 'draftkings' in bookmaker.get('key', '').lower():
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'spreads':
                        outcomes = market.get('outcomes', [])
                        if len(outcomes) < 2:
                            continue
                        
                        # Find home team outcome
                        for outcome in outcomes:
                            outcome_name = outcome.get('name', '')
                            if home_team.lower() in outcome_name.lower() or outcome_name.lower() in home_team.lower():
                                return {
                                    'spread_point': outcome.get('point'),
                                    'odds': outcome.get('price')
                                }
                        
                        # If can't match, use first outcome
                        return {
                            'spread_point': outcomes[0].get('point'),
                            'odds': outcomes[0].get('price')
                        }
        
        return None
    except Exception as e:
        logger.error(f"Error getting closing spread: {e}")
        return None


def extract_scores(game_data: Dict[str, Any], home_team: str, away_team: str) -> tuple:
    """
    Extract home and away scores from game data
    
    This is the same as the original script.
    """
    scores_list = game_data.get('scores') or []
    home_score = None
    away_score = None
    
    for score_entry in scores_list:
        if not isinstance(score_entry, dict):
            continue
        team_name = score_entry.get('name', '')
        score_value = score_entry.get('score')
        
        if score_value is None:
            continue
        
        if home_team and team_name:
            if home_team.lower() in team_name.lower() or team_name.lower() in home_team.lower():
                home_score = score_value
        if away_team and team_name:
            if away_team.lower() in team_name.lower() or team_name.lower() in away_team.lower():
                away_score = score_value
    
    return home_score, away_score


def collect_games_for_date(api_key: str, sport_key: str, target_date: datetime) -> pd.DataFrame:
    """
    Collect completed games for a specific date with closing spreads and scores
    
    This is adapted from the original script to work with Lambda:
    - Uses api_key parameter instead of creating OddsAPIClient
    - Returns DataFrame (doesn't save to files - that's done separately)
    """
    api_sport_key = SPORT_API_KEYS[sport_key]
    
    # Convert target date to EST for comparison (EST is UTC-5)
    est = timezone(timedelta(hours=-5))
    target_date_est = target_date.astimezone(est)
    target_date_only = target_date_est.date()
    
    date_str = target_date_only.strftime('%Y-%m-%d')
    logger.info(f"Collecting {sport_key.upper()} games from {date_str} (EST)...")
    
    all_games = []
    event_ids_seen = set()
    
    # Get games from scores endpoint (last 3 days covers yesterday)
    try:
        time.sleep(API_RATE_LIMIT_DELAY)
        scores = make_odds_api_request(
            endpoint=f"sports/{api_sport_key}/scores",
            params={
                'dateFormat': 'iso',
                'daysFrom': '3'
            },
            api_key=api_key
        )
        
        for game in scores:
            if game.get('completed') == True:
                event_id = game.get('id')
                commence_time = game.get('commence_time')
                
                if event_id and commence_time:
                    try:
                        event_time_utc = pd.to_datetime(commence_time)
                        if event_time_utc.tzinfo is None:
                            event_time_utc = event_time_utc.replace(tzinfo=timezone.utc)
                        elif isinstance(event_time_utc, pd.Timestamp):
                            if event_time_utc.tz is None:
                                event_time_utc = event_time_utc.tz_localize('UTC').to_pydatetime()
                            else:
                                event_time_utc = event_time_utc.to_pydatetime()
                        
                        # Convert to EST for date comparison
                        event_time_est = event_time_utc.astimezone(est)
                        event_date_est = event_time_est.date()
                        
                        # Only include games from target date (using EST date)
                        if event_date_est == target_date_only:
                            if event_id not in event_ids_seen:
                                event_ids_seen.add(event_id)
                                all_games.append(game)
                    except:
                        pass
        
        logger.info(f"Found {len(all_games)} completed games from {date_str}")
    except Exception as e:
        logger.error(f"Error getting scores: {e}")
        return pd.DataFrame()
    
    if len(all_games) == 0:
        logger.info(f"No games found for {date_str}")
        return pd.DataFrame()
    
    # Load stored scores from S3
    stored_scores = load_scores_from_s3(sport_key)
    
    # Process games to get closing spreads
    logger.info(f"Getting closing spreads for {len(all_games)} games...")
    
    results = []
    successful = 0
    no_spread = 0
    
    for i, game in enumerate(all_games, 1):
        event_id = game.get('id')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')
        
        # Get scores
        home_score, away_score = extract_scores(game, home_team, away_team)
        
        # Try stored scores if not found
        if home_score is None or away_score is None:
            stored = stored_scores.get(event_id)
            if stored:
                home_score = stored.get('home_score')
                away_score = stored.get('away_score')
        
        # Convert to numeric
        try:
            home_score = int(home_score) if home_score is not None else None
            away_score = int(away_score) if away_score is not None else None
        except:
            home_score = None
            away_score = None
        
        # Get closing spread
        spread_data = get_closing_spread(api_key, sport_key, event_id, home_team, away_team, commence_time)
        
        if spread_data and spread_data.get('spread_point') is not None:
            closing_spread = float(spread_data['spread_point'])
            successful += 1
        else:
            closing_spread = None
            no_spread += 1
        
        # Parse game date (use EST date)
        try:
            est = timezone(timedelta(hours=-5))
            event_time_utc = pd.to_datetime(commence_time)
            if event_time_utc.tzinfo is None:
                event_time_utc = event_time_utc.replace(tzinfo=timezone.utc)
            elif isinstance(event_time_utc, pd.Timestamp):
                if event_time_utc.tz is None:
                    event_time_utc = event_time_utc.tz_localize('UTC').to_pydatetime()
                else:
                    event_time_utc = event_time_utc.to_pydatetime()
            # Convert to EST for date
            event_time_est = event_time_utc.astimezone(est)
            game_date = event_time_est.date()
        except:
            game_date = None
        
        # Calculate spread_result_difference = (home_score - away_score) + closing_spread
        if home_score is not None and away_score is not None and closing_spread is not None:
            spread_result_difference = (home_score - away_score) + closing_spread
        else:
            spread_result_difference = None
        
        # Store score in S3 if we have it
        if home_score is not None and away_score is not None:
            stored_scores[event_id] = {
                "home_score": home_score,
                "away_score": away_score,
                "home_team": home_team,
                "away_team": away_team,
                "game_date": game_date.isoformat() if game_date else None,
                "stored_at": datetime.now(timezone.utc).isoformat()
            }
        
        results.append({
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'closing_spread': closing_spread,
            'home_score': home_score,
            'away_score': away_score,
            'spread_result_difference': spread_result_difference
        })
    
    logger.info(f"Summary: {successful} with spreads, {no_spread} without spreads")
    
    # Save updated scores to S3
    if stored_scores:
        save_scores_to_s3(sport_key, stored_scores)
    
    df = pd.DataFrame(results)
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)
    
    return df


def merge_with_existing_data(new_games_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new games with existing data, avoiding duplicates
    
    This preserves manually filled scores in existing data.
    """
    if len(existing_df) == 0:
        return new_games_df
    
    if len(new_games_df) == 0:
        return existing_df
    
    # Create merge key
    existing_df['merge_key'] = (
        existing_df['game_date'].astype(str) + 
        existing_df['home_team'] + 
        existing_df['away_team']
    )
    new_games_df['merge_key'] = (
        new_games_df['game_date'].astype(str) + 
        new_games_df['home_team'] + 
        new_games_df['away_team']
    )
    
    # Merge, preserving existing scores if new ones are blank
    updated_rows = []
    preserved_count = 0
    
    for _, new_row in new_games_df.iterrows():
        matching = existing_df[existing_df['merge_key'] == new_row['merge_key']]
        if not matching.empty:
            existing_row = matching.iloc[0].copy()
            
            # Preserve existing scores if new scores are blank
            if (pd.isna(new_row['home_score']) and pd.isna(new_row['away_score']) and
                pd.notna(existing_row['home_score']) and pd.notna(existing_row['away_score'])):
                new_row['home_score'] = existing_row['home_score']
                new_row['away_score'] = existing_row['away_score']
                preserved_count += 1
            
            # Recalculate spread_result_difference if needed
            if (pd.notna(new_row['home_score']) and pd.notna(new_row['away_score']) and 
                pd.notna(new_row['closing_spread'])):
                new_row['spread_result_difference'] = (
                    (new_row['home_score'] - new_row['away_score']) + new_row['closing_spread']
                )
            
            updated_rows.append(new_row)
        else:
            updated_rows.append(new_row)
    
    # Remove old versions of updated games
    existing_df_filtered = existing_df[~existing_df['merge_key'].isin(new_games_df['merge_key'])]
    
    # Combine
    combined_df = pd.concat([existing_df_filtered, pd.DataFrame(updated_rows)], ignore_index=True)
    combined_df = combined_df.drop(columns=['merge_key'])
    
    if preserved_count > 0:
        logger.info(f"Preserved {preserved_count} manually filled scores")
    
    return combined_df


def process_sport(api_key: str, sport_key: str, target_date: datetime) -> Dict[str, Any]:
    """
    Process one sport: collect games and update S3
    
    Returns summary statistics
    """
    logger.info(f"Processing {sport_key.upper()}...")
    
    # Read existing data from S3
    existing_df = read_excel_from_s3(sport_key)
    logger.info(f"Loaded {len(existing_df)} existing games from S3")
    
    # Collect new games
    new_games_df = collect_games_for_date(api_key, sport_key, target_date)
    
    if len(new_games_df) == 0:
        logger.info(f"No new games for {sport_key.upper()}")
        return {
            'sport': sport_key,
            'existing_games': len(existing_df),
            'new_games': 0,
            'total_games': len(existing_df)
        }
    
    # Merge with existing data
    combined_df = merge_with_existing_data(new_games_df, existing_df)
    
    # Prepare for Excel (convert game_date to date)
    excel_df = combined_df.copy()
    if 'game_date' in excel_df.columns:
        excel_df['game_date'] = pd.to_datetime(excel_df['game_date']).dt.date
    
    # Format numeric columns
    if 'closing_spread' in excel_df.columns:
        excel_df['closing_spread'] = excel_df['closing_spread'].round(1)
    if 'spread_result_difference' in excel_df.columns:
        excel_df['spread_result_difference'] = excel_df['spread_result_difference'].round(1)
    
    # Write to S3 (Excel and Parquet) - backup storage
    write_excel_to_s3(excel_df, sport_key)
    write_parquet_to_s3(combined_df, sport_key)

    # Write to PostgreSQL database - primary storage
    db_rows = write_games_to_database(new_games_df, sport_key)

    return {
        'sport': sport_key,
        'existing_games': len(existing_df),
        'new_games': len(new_games_df),
        'total_games': len(combined_df),
        'db_rows_affected': db_rows
    }


def lambda_handler(event, context):
    """
    Lambda handler function - this is the entry point AWS calls
    
    Event: Triggered by EventBridge schedule
    Context: Lambda context (runtime info, timeout, etc.)
    """
    logger.info("="*80)
    logger.info("Starting daily game collection")
    logger.info("="*80)
    
    try:
        # Get API key from Secrets Manager
        api_key = get_api_key()
        logger.info("✓ Retrieved API key from Secrets Manager")
        
        # Calculate yesterday's date (EST)
        est = timezone(timedelta(hours=-5))
        now_est = datetime.now(est)
        yesterday_est = now_est - timedelta(days=1)
        yesterday_date = yesterday_est.date()
        
        logger.info(f"Collecting games from: {yesterday_date} (EST)")
        
        # Process each sport
        results = []
        for sport_key in SPORTS:
            try:
                result = process_sport(api_key, sport_key, yesterday_est)
                results.append(result)
                logger.info(f"✓ {sport_key.upper()}: {result['new_games']} new games, {result['total_games']} total")
            except Exception as e:
                logger.error(f"✗ Error processing {sport_key.upper()}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Return summary
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Collection complete',
                'date': yesterday_date.isoformat(),
                'results': results
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

