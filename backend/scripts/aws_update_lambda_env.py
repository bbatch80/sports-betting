#!/usr/bin/env python3
"""
Update Lambda Environment Variables for PostgreSQL

This script retrieves the DATABASE_URL from Secrets Manager and
updates all Lambda functions with the necessary environment variables.

Usage:
    python3 scripts/aws_update_lambda_env.py
"""

import boto3
import json
from botocore.exceptions import ClientError

# Configuration
REGION = 'us-east-1'
RDS_SECRET_NAME = 'sports-betting-db-credentials'

# Lambda functions to update
LAMBDA_FUNCTIONS = [
    'collect-yesterday-games',
    'generate-predictions',
    'evaluate-strategy-results',
    'predictions-api',
    'results-api',
]


def get_database_url(secrets_client, secret_name):
    """Retrieve DATABASE_URL from Secrets Manager."""
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret.get('url')
    except ClientError as e:
        if 'ResourceNotFoundException' in str(e):
            print(f"✗ Secret '{secret_name}' not found")
            print(f"  Run aws_rds_setup.py first to create the RDS instance")
        else:
            print(f"✗ Error retrieving secret: {e}")
        return None


def update_lambda_environment(lambda_client, function_name, database_url):
    """Update Lambda function environment variables."""
    try:
        # Get current configuration
        try:
            response = lambda_client.get_function_configuration(
                FunctionName=function_name
            )
            current_env = response.get('Environment', {}).get('Variables', {})
        except ClientError as e:
            if 'ResourceNotFoundException' in str(e):
                print(f"  ⚠ Function '{function_name}' not found, skipping")
                return None
            raise

        # Add/update DATABASE_URL
        new_env = current_env.copy()
        new_env['DATABASE_URL'] = database_url
        new_env['USE_NULL_POOL'] = 'false'  # Set to 'true' if using RDS Proxy

        # Update function
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment={'Variables': new_env}
        )
        print(f"  ✓ Updated {function_name}")
        return True

    except Exception as e:
        print(f"  ✗ Error updating {function_name}: {e}")
        return False


def main():
    """Main function."""
    print("=" * 80)
    print("UPDATE LAMBDA ENVIRONMENT VARIABLES")
    print("=" * 80)
    print()

    # Initialize clients
    secrets_client = boto3.client('secretsmanager', region_name=REGION)
    lambda_client = boto3.client('lambda', region_name=REGION)

    # Get DATABASE_URL
    print("Retrieving DATABASE_URL from Secrets Manager...")
    database_url = get_database_url(secrets_client, RDS_SECRET_NAME)

    if not database_url:
        return 1

    # Mask password in output
    masked_url = database_url.split('@')[0].rsplit(':', 1)[0] + ':****@' + database_url.split('@')[1]
    print(f"✓ Retrieved DATABASE_URL: {masked_url}")
    print()

    # Update Lambda functions
    print("Updating Lambda functions...")
    success_count = 0
    skip_count = 0

    for function_name in LAMBDA_FUNCTIONS:
        result = update_lambda_environment(lambda_client, function_name, database_url)
        if result is True:
            success_count += 1
        elif result is None:
            skip_count += 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Updated: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {len(LAMBDA_FUNCTIONS) - success_count - skip_count}")
    print()
    print("Lambda functions will now use PostgreSQL via DATABASE_URL.")
    print("Test by invoking a function and checking CloudWatch logs.")

    return 0


if __name__ == "__main__":
    exit(main())
