"""
Lambda Function: Predictions API
Serves predictions from S3 via API Gateway for mobile app consumption
"""

import json
import boto3
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'
ALLOWED_SPORTS = ['nfl', 'nba', 'ncaam']


def get_elite_teams_from_s3(sport_key: str = None) -> Dict[str, Any]:
    """
    Get elite team opportunities from predictions

    Args:
        sport_key: Optional sport key. If None, returns elite teams for all sports
    """
    sports_to_check = [sport_key] if sport_key else ALLOWED_SPORTS

    elite_teams_data = {
        'elite_teams': [],
        'sports': {},
        'summary': {
            'total_opportunities': 0,
            'sports_with_opportunities': 0
        }
    }

    for sport in sports_to_check:
        s3_key = f"predictions/predictions_{sport}.json"

        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            predictions_data = json.loads(response['Body'].read().decode('utf-8'))

            # Extract elite_team strategy
            strategies = predictions_data.get('strategies', {})
            elite_team_strategy = strategies.get('elite_team', {})
            opportunities = elite_team_strategy.get('opportunities', [])

            if opportunities:
                # Add sport name to each opportunity
                for opp in opportunities:
                    opp['sport'] = sport
                    opp['sport_name'] = predictions_data.get('sport_name', sport.upper())

                elite_teams_data['elite_teams'].extend(opportunities)
                elite_teams_data['sports'][sport] = {
                    'sport_name': predictions_data.get('sport_name', sport.upper()),
                    'opportunities': opportunities,
                    'prediction_date': predictions_data.get('prediction_date'),
                    'summary': elite_team_strategy.get('summary', {})
                }
                elite_teams_data['summary']['sports_with_opportunities'] += 1
                elite_teams_data['summary']['total_opportunities'] += len(opportunities)
            else:
                elite_teams_data['sports'][sport] = {
                    'sport_name': predictions_data.get('sport_name', sport.upper()),
                    'opportunities': [],
                    'prediction_date': predictions_data.get('prediction_date'),
                    'summary': elite_team_strategy.get('summary', {})
                }

        except s3_client.exceptions.NoSuchKey:
            logger.info(f"No predictions found for {sport}")
            elite_teams_data['sports'][sport] = {
                'opportunities': [],
                'error': 'No predictions available'
            }
        except Exception as e:
            logger.error(f"Error reading elite teams for {sport}: {e}")
            elite_teams_data['sports'][sport] = {
                'opportunities': [],
                'error': str(e)
            }

    # Sort all elite teams by last 5 spread performance (best first)
    elite_teams_data['elite_teams'].sort(
        key=lambda x: x.get('elite_team_last_5_spread') or 0,
        reverse=True
    )

    return elite_teams_data


def get_predictions_from_s3(sport_key: str, date: str = None) -> Dict[str, Any]:
    """
    Read predictions JSON from S3
    
    Args:
        sport_key: Sport key (nfl, nba, ncaam)
        date: Optional date in YYYY-MM-DD format. If provided, looks for predictions_{sport}_{date}.json
              If not provided, looks for predictions_{sport}.json (today's predictions)
    """
    if date:
        # Try date-specific predictions first
        s3_key = f"predictions/predictions_{sport_key}_{date}.json"
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            predictions_data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Successfully retrieved predictions for {sport_key} on {date}")
            return predictions_data
        except s3_client.exceptions.NoSuchKey:
            logger.info(f"No date-specific predictions found for {sport_key} on {date}, trying default")
            # Fall through to default
    
    # Default: today's predictions
    s3_key = f"predictions/predictions_{sport_key}.json"
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        predictions_data = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Successfully retrieved predictions for {sport_key}")
        return predictions_data
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"No predictions found for {sport_key}")
        return {
            'sport': sport_key,
            'error': 'No predictions available',
            'message': f'Predictions for {sport_key.upper()} are not yet available'
        }
    except Exception as e:
        logger.error(f"Error reading predictions from S3: {e}")
        raise


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
        'path': '/api/predictions/nfl',
        'httpMethod': 'GET',
        'pathParameters': {...},
        'queryStringParameters': {...},
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
        
        # Check for elite-teams endpoint: /api/elite-teams or /api/elite-teams/{sport}
        if 'elite-teams' in path_parts:
            elite_idx = path_parts.index('elite-teams')
            # Check if a specific sport is requested
            sport = None
            if len(path_parts) > elite_idx + 1:
                sport = path_parts[elite_idx + 1].lower()
                if sport not in ALLOWED_SPORTS:
                    return create_response(400, {
                        'error': 'Invalid sport',
                        'message': f'Valid sports are: {", ".join(ALLOWED_SPORTS)}',
                        'received': sport
                    })

            elite_teams = get_elite_teams_from_s3(sport)
            return create_response(200, elite_teams)

        # Determine which sport(s) to return
        if 'all' in path_parts or (len(path_parts) >= 3 and path_parts[2] == 'all'):
            # Return all predictions
            all_predictions = {}
            for sport in ALLOWED_SPORTS:
                try:
                    all_predictions[sport] = get_predictions_from_s3(sport)
                except Exception as e:
                    logger.error(f"Error getting {sport} predictions: {e}")
                    all_predictions[sport] = {
                        'sport': sport,
                        'error': str(e)
                    }

            return create_response(200, {
                'message': 'All predictions',
                'predictions': all_predictions
            })

        # Extract sport from path (e.g., /api/predictions/nfl)
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
        
        # Get date from query parameters if provided
        query_params = event.get('queryStringParameters') or {}
        date = query_params.get('date')  # Format: YYYY-MM-DD
        
        # Validate date format if provided
        if date:
            try:
                from datetime import datetime
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                return create_response(400, {
                    'error': 'Invalid date format',
                    'message': 'Date must be in YYYY-MM-DD format',
                    'received': date
                })
        
        # Get predictions for the requested sport and date
        predictions = get_predictions_from_s3(sport, date)
        
        # Return predictions
        return create_response(200, predictions)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return create_response(500, {
            'error': 'Internal server error',
            'message': str(e)
        })



