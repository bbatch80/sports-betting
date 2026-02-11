"""
Lambda Function: Schedule Closing Captures
Queries today's games and creates one-time EventBridge Scheduler schedules
to capture closing lines 30 minutes before each game's tipoff.

Triggered by: EventBridge cron — daily at 7:00 AM EST (12:00 UTC)
"""

import json
import boto3
import os
from datetime import datetime, timedelta, timezone
import logging
from sqlalchemy import text

from shared import get_db_engine, SPORT_API_KEYS

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
scheduler_client = boto3.client('scheduler')
sts_client = boto3.client('sts')

# Configuration
CAPTURE_LAMBDA_NAME = 'capture-closing-lines'
SCHEDULE_GROUP = 'default'
MINUTES_BEFORE_TIPOFF = 30


def get_capture_lambda_arn() -> str:
    """Get the ARN of the capture-closing-lines Lambda function."""
    lambda_client = boto3.client('lambda')
    response = lambda_client.get_function(FunctionName=CAPTURE_LAMBDA_NAME)
    return response['Configuration']['FunctionArn']


def get_scheduler_role_arn() -> str:
    """Get or construct the ARN for the EventBridge Scheduler execution role."""
    # Use environment variable if set, otherwise construct from account ID
    role_arn = os.environ.get('SCHEDULER_ROLE_ARN')
    if role_arn:
        return role_arn

    account_id = sts_client.get_caller_identity()['Account']
    return f"arn:aws:iam::{account_id}:role/EventBridgeSchedulerClosingLinesRole"


def create_game_schedule(game: dict, capture_lambda_arn: str, scheduler_role_arn: str) -> bool:
    """Create a one-time EventBridge Scheduler schedule for a single game."""
    game_id = game['id']
    sport = game['sport']
    home_team = game['home_team']
    away_team = game['away_team']
    commence_time = game['commence_time']

    # Calculate capture time (30 min before tipoff)
    capture_time = commence_time - timedelta(minutes=MINUTES_BEFORE_TIPOFF)
    now = datetime.now(timezone.utc)

    if capture_time <= now:
        logger.info(f"Skipping game {game_id} ({away_team} @ {home_team}) - capture time already passed")
        return False

    # Schedule name: closing-{sport}-{game_id}
    schedule_name = f"closing-{sport}-{game_id}"

    # EventBridge Scheduler "at" expression: at(yyyy-mm-ddThh:mm:ss)
    schedule_expression = f"at({capture_time.strftime('%Y-%m-%dT%H:%M:%S')})"

    # Payload for the capture Lambda
    payload = json.dumps({
        'game_id': game_id,
        'sport': sport,
        'api_sport_key': SPORT_API_KEYS.get(sport, ''),
        'home_team': home_team,
        'away_team': away_team
    })

    try:
        scheduler_client.create_schedule(
            Name=schedule_name,
            GroupName=SCHEDULE_GROUP,
            ScheduleExpression=schedule_expression,
            ScheduleExpressionTimezone='UTC',
            FlexibleTimeWindow={'Mode': 'OFF'},
            Target={
                'Arn': capture_lambda_arn,
                'RoleArn': scheduler_role_arn,
                'Input': payload
            },
            ActionAfterCompletion='DELETE'
        )
        logger.info(f"Created schedule {schedule_name} at {capture_time.strftime('%H:%M UTC')} "
                     f"for {away_team} @ {home_team}")
        return True

    except scheduler_client.exceptions.ConflictException:
        # Schedule already exists (idempotent)
        logger.info(f"Schedule {schedule_name} already exists, skipping")
        return True

    except Exception as e:
        logger.error(f"Failed to create schedule {schedule_name}: {e}")
        return False


def cleanup_old_schedules() -> int:
    """Delete any leftover closing-* schedules from previous days."""
    deleted = 0
    try:
        paginator = scheduler_client.get_paginator('list_schedules')
        for page in paginator.paginate(GroupName=SCHEDULE_GROUP, NamePrefix='closing-'):
            for schedule in page.get('Schedules', []):
                name = schedule['Name']
                try:
                    scheduler_client.delete_schedule(
                        Name=name,
                        GroupName=SCHEDULE_GROUP
                    )
                    deleted += 1
                except scheduler_client.exceptions.ResourceNotFoundException:
                    pass
                except Exception as e:
                    logger.warning(f"Could not delete schedule {name}: {e}")
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")

    if deleted > 0:
        logger.info(f"Cleaned up {deleted} old schedule(s)")
    return deleted


def lambda_handler(event, context):
    """Lambda entry point - queries today's games and creates per-game schedules."""
    logger.info("=" * 60)
    logger.info("Starting closing line schedule creation")
    logger.info("=" * 60)

    try:
        engine = get_db_engine()
        if engine is None:
            logger.error("No database connection available")
            return {'statusCode': 500, 'body': 'Database connection failed'}

        # Clean up any leftover schedules from previous runs
        cleanup_old_schedules()

        # Get the capture Lambda ARN and scheduler role ARN
        capture_lambda_arn = get_capture_lambda_arn()
        scheduler_role_arn = get_scheduler_role_arn()
        logger.info(f"Capture Lambda ARN: {capture_lambda_arn}")
        logger.info(f"Scheduler Role ARN: {scheduler_role_arn}")

        # Query today's games that have recommendations and don't have closing lines yet
        est = timezone(timedelta(hours=-5))
        today = datetime.now(est).date()

        query = text("""
            SELECT tg.id, tg.sport, tg.home_team, tg.away_team, tg.commence_time
            FROM todays_games tg
            INNER JOIN todays_recommendations tr
                ON tg.sport = tr.sport
                AND tg.game_date = tr.game_date
                AND tg.home_team = tr.home_team
                AND tg.away_team = tr.away_team
            WHERE tg.game_date = :today AND tg.closing_captured_at IS NULL
            ORDER BY tg.commence_time
        """)

        with engine.connect() as conn:
            rows = conn.execute(query, {'today': today.isoformat()}).fetchall()

        logger.info(f"Found {len(rows)} games needing closing line capture for {today}")

        if not rows:
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No games to schedule', 'date': today.isoformat()})
            }

        # Create a schedule for each game
        created = 0
        skipped = 0

        for row in rows:
            game = {
                'id': row[0],
                'sport': row[1],
                'home_team': row[2],
                'away_team': row[3],
                'commence_time': row[4]
            }

            # Ensure commence_time is timezone-aware (UTC)
            ct = game['commence_time']
            if ct.tzinfo is None:
                game['commence_time'] = ct.replace(tzinfo=timezone.utc)

            if create_game_schedule(game, capture_lambda_arn, scheduler_role_arn):
                created += 1
            else:
                skipped += 1

        logger.info(f"Created {created} schedule(s), skipped {skipped}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Schedules created',
                'date': today.isoformat(),
                'games_found': len(rows),
                'schedules_created': created,
                'skipped': skipped
            })
        }

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
