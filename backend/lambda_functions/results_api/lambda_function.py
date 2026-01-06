"""
Lambda Function: Results API
Serves game results from S3 via API Gateway for mobile app consumption
"""

import json
import boto3
import pandas as pd
import io
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'
ALLOWED_SPORTS = ['nfl', 'nba', 'ncaam']


def read_results_from_s3(sport_key: str) -> Dict[str, Any]:
    """
    Read results from S3 (prefer Parquet, fallback to Excel)
    
    Returns results data with games and their outcomes
    """
    # Try Parquet first (more efficient)
    parquet_key = f"data/results/{sport_key}_season_results.parquet"
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=parquet_key)
        parquet_data = response['Body'].read()
        df = pd.read_parquet(io.BytesIO(parquet_data))
        logger.info(f"Successfully read {len(df)} results from Parquet for {sport_key}")
        
        # Convert DataFrame to list of dicts
        # Convert game_date to string for JSON serialization
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
        
        # Replace NaN with None for JSON serialization
        df = df.where(pd.notna(df), None)
        
        games = df.to_dict('records')
        
        return {
            'sport': sport_key,
            'sport_name': sport_key.upper(),
            'total_games': len(games),
            'games': games,
            'last_updated': datetime.now().isoformat()
        }
        
    except s3_client.exceptions.NoSuchKey:
        # Try Excel as fallback
        excel_key = f"data/results/{sport_key}_season_results.xlsx"
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=excel_key)
            excel_data = response['Body'].read()
            df = pd.read_excel(io.BytesIO(excel_data))
            logger.info(f"Successfully read {len(df)} results from Excel for {sport_key}")
            
            # Convert DataFrame to list of dicts
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
            
            # Replace NaN with None for JSON serialization
            df = df.where(pd.notna(df), None)
            
            games = df.to_dict('records')
            
            return {
                'sport': sport_key,
                'sport_name': sport_key.upper(),
                'total_games': len(games),
                'games': games,
                'last_updated': datetime.now().isoformat()
            }
            
        except s3_client.exceptions.NoSuchKey:
            logger.warning(f"No results found for {sport_key}")
            return {
                'sport': sport_key,
                'sport_name': sport_key.upper(),
                'error': 'No results available',
                'message': f'Results for {sport_key.upper()} are not yet available',
                'total_games': 0,
                'games': []
            }
        except Exception as e:
            logger.error(f"Error reading Excel from S3: {e}")
            raise
    except Exception as e:
        logger.error(f"Error reading Parquet from S3: {e}")
        raise


def filter_results_by_date(games: List[Dict], start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> List[Dict]:
    """
    Filter games by date range
    
    Args:
        games: List of game dictionaries
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
    
    Returns:
        Filtered list of games
    """
    if not start_date and not end_date:
        return games
    
    filtered = []
    for game in games:
        game_date = game.get('game_date')
        if not game_date:
            continue
        
        # Compare dates (game_date is already a string in YYYY-MM-DD format)
        if start_date and game_date < start_date:
            continue
        if end_date and game_date > end_date:
            continue
        
        filtered.append(game)
    
    return filtered


def create_response(status_code: int, body: Dict[str, Any], cors_headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Create API Gateway response with CORS headers"""
    if cors_headers is None:
        cors_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,OPTIONS',
            'Content-Type': 'application/json'
        }
    
    return {
        'statusCode': status_code,
        'headers': cors_headers,
        'body': json.dumps(body, default=str)
    }


def lambda_handler(event, context):
    """
    Lambda handler for API Gateway requests
    
    Event structure from API Gateway:
    {
        'path': '/api/results/nfl',
        'httpMethod': 'GET',
        'queryStringParameters': {'start_date': '2025-01-01', 'end_date': '2025-01-31'},
        ...
    }
    """
    logger.info(f"Received request: {event.get('httpMethod')} {event.get('path')}")
    
    try:
        # Handle OPTIONS request (CORS preflight)
        if event.get('httpMethod') == 'OPTIONS':
            return create_response(200, {'message': 'OK'})
        
        # Extract sport from path
        path = event.get('path', '')
        path_parts = [p for p in path.split('/') if p]
        
        # Determine which sport(s) to return
        if 'all' in path_parts or (len(path_parts) >= 3 and path_parts[2] == 'all'):
            # Return all results
            all_results = {}
            for sport in ALLOWED_SPORTS:
                try:
                    result_data = read_results_from_s3(sport)
                    # Apply date filters if provided
                    query_params = event.get('queryStringParameters') or {}
                    start_date = query_params.get('start_date')
                    end_date = query_params.get('end_date')
                    
                    if start_date or end_date:
                        games = result_data.get('games', [])
                        filtered_games = filter_results_by_date(games, start_date, end_date)
                        result_data['games'] = filtered_games
                        result_data['total_games'] = len(filtered_games)
                    
                    all_results[sport] = result_data
                except Exception as e:
                    logger.error(f"Error getting {sport} results: {e}")
                    all_results[sport] = {
                        'sport': sport,
                        'error': str(e)
                    }
            
            return create_response(200, {
                'message': 'All results',
                'results': all_results
            })
        
        # Extract sport from path (e.g., /api/results/nfl)
        sport = None
        if len(path_parts) >= 3:
            sport = path_parts[2].lower()
        elif 'pathParameters' in event and event['pathParameters']:
            # Try to get from path parameters
            sport = event['pathParameters'].get('sport', '').lower()
        
        # Validate sport
        if not sport or sport not in ALLOWED_SPORTS:
            return create_response(400, {
                'error': 'Invalid sport',
                'message': f'Valid sports are: {", ".join(ALLOWED_SPORTS)}',
                'received': sport
            })
        
        # Get results for the requested sport
        result_data = read_results_from_s3(sport)
        
        # Apply date filters if provided
        query_params = event.get('queryStringParameters') or {}
        start_date = query_params.get('start_date')
        end_date = query_params.get('end_date')
        
        if start_date or end_date:
            games = result_data.get('games', [])
            filtered_games = filter_results_by_date(games, start_date, end_date)
            result_data['games'] = filtered_games
            result_data['total_games'] = len(filtered_games)
        
        # Return results
        return create_response(200, result_data)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return create_response(500, {
            'error': 'Internal server error',
            'message': str(e)
        })

