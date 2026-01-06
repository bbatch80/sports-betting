#!/usr/bin/env python3
"""
AWS Setup Script - Phase 1
Creates S3 bucket, uploads data, sets up Secrets Manager, and creates IAM roles
"""

import boto3
import json
import os
from pathlib import Path
from botocore.exceptions import ClientError, BotoCoreError

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'  # Must be globally unique - change if needed
REGION = 'us-east-1'  # Change to your preferred region
SECRET_NAME = 'odds-api-key'
ROLE_NAME = 'SportsBettingLambdaRole'

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'results'


def check_aws_credentials():
    """Verify AWS credentials are configured"""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials configured")
        print(f"  Account ID: {identity['Account']}")
        print(f"  User ARN: {identity['Arn']}")
        return True
    except (ClientError, BotoCoreError) as e:
        print(f"✗ AWS credentials not configured: {e}")
        print("\nPlease run: aws configure")
        return False


def create_s3_bucket(s3_client, bucket_name, region):
    """Create S3 bucket if it doesn't exist"""
    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ S3 bucket '{bucket_name}' already exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                if region == 'us-east-1':
                    # us-east-1 doesn't need LocationConstraint
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                
                # Enable versioning
                s3_client.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                
                # Block public access
                s3_client.put_public_access_block(
                    Bucket=bucket_name,
                    PublicAccessBlockConfiguration={
                        'BlockPublicAcls': True,
                        'IgnorePublicAcls': True,
                        'BlockPublicPolicy': True,
                        'RestrictPublicBuckets': True
                    }
                )
                
                print(f"✓ Created S3 bucket '{bucket_name}'")
                print(f"  Region: {region}")
                print(f"  Versioning: Enabled")
                print(f"  Public access: Blocked")
                return True
            else:
                print(f"✗ Error checking bucket: {e}")
                return False
    except ClientError as e:
        print(f"✗ Error creating S3 bucket: {e}")
        if 'BucketAlreadyExists' in str(e) or 'BucketAlreadyOwnedByYou' in str(e):
            print(f"  Bucket '{bucket_name}' already exists (owned by you)")
            return True
        return False


def upload_data_files(s3_client, bucket_name):
    """Upload existing data files to S3"""
    if not DATA_DIR.exists():
        print(f"✗ Data directory not found: {DATA_DIR}")
        return False
    
    files_uploaded = 0
    files_skipped = 0
    
    # Files to upload
    files_to_upload = [
        'nfl_season_results.xlsx',
        'nba_season_results.xlsx',
        'ncaam_season_results.xlsx',
        'nfl_season_results.parquet',
        'nba_season_results.parquet',
        'ncaam_season_results.parquet',
    ]
    
    print(f"\nUploading files to S3...")
    
    for filename in files_to_upload:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"  ⚠ Skipping {filename} (not found)")
            files_skipped += 1
            continue
        
        s3_key = f"data/results/{filename}"
        
        try:
            # Check if file already exists
            try:
                s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                print(f"  ⚠ {filename} already exists in S3, skipping")
                files_skipped += 1
                continue
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Upload file
            s3_client.upload_file(str(file_path), bucket_name, s3_key)
            print(f"  ✓ Uploaded {filename}")
            files_uploaded += 1
        except ClientError as e:
            print(f"  ✗ Error uploading {filename}: {e}")
    
    print(f"\n  Total uploaded: {files_uploaded}")
    print(f"  Total skipped: {files_skipped}")
    return files_uploaded > 0


def setup_secrets_manager(secrets_client, secret_name):
    """Set up Secrets Manager with Odds API key"""
    # Get API key from environment or prompt
    api_key = os.getenv('ODDS_API_KEY')
    
    if not api_key:
        print(f"\n⚠ ODDS_API_KEY not found in environment")
        api_key = input("Enter your Odds API key: ").strip()
        if not api_key:
            print("✗ API key required")
            return False
    
    try:
        # Check if secret already exists
        try:
            secrets_client.describe_secret(SecretId=secret_name)
            print(f"✓ Secret '{secret_name}' already exists")
            
            # Ask if user wants to update
            update = input(f"  Update existing secret? (y/n): ").strip().lower()
            if update == 'y':
                secrets_client.put_secret_value(
                    SecretId=secret_name,
                    SecretString=api_key
                )
                print(f"  ✓ Updated secret")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                # Secret doesn't exist, create it
                secrets_client.create_secret(
                    Name=secret_name,
                    SecretString=api_key,
                    Description='Odds API key for Sports Betting Analytics'
                )
                print(f"✓ Created secret '{secret_name}' in Secrets Manager")
                return True
            else:
                raise
    except ClientError as e:
        print(f"✗ Error setting up Secrets Manager: {e}")
        return False


def create_iam_role(iam_client, role_name):
    """Create IAM role for Lambda functions"""
    try:
        # Check if role exists
        try:
            iam_client.get_role(RoleName=role_name)
            print(f"✓ IAM role '{role_name}' already exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchEntity':
                raise
        
        # Trust policy for Lambda
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Create role
        iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for Sports Betting Analytics Lambda functions'
        )
        
        # Attach policies
        policies = [
            'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/SecretsManagerReadWrite'
        ]
        
        for policy_arn in policies:
            try:
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
            except ClientError as e:
                print(f"  ⚠ Warning: Could not attach policy {policy_arn}: {e}")
        
        print(f"✓ Created IAM role '{role_name}'")
        print(f"  Attached policies: Lambda execution, S3 access, Secrets Manager access")
        return True
        
    except ClientError as e:
        print(f"✗ Error creating IAM role: {e}")
        return False


def main():
    """Main setup function"""
    print("="*80)
    print("AWS SETUP - Phase 1")
    print("="*80)
    print()
    
    # Check AWS credentials
    if not check_aws_credentials():
        return
    
    print()
    
    # Initialize AWS clients
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        secrets_client = boto3.client('secretsmanager', region_name=REGION)
        iam_client = boto3.client('iam')
    except Exception as e:
        print(f"✗ Error initializing AWS clients: {e}")
        return
    
    success = True
    
    # Step 1: Create S3 bucket
    print("\n" + "="*80)
    print("Step 1: Creating S3 bucket")
    print("="*80)
    if not create_s3_bucket(s3_client, BUCKET_NAME, REGION):
        print("⚠ Continuing with existing bucket...")
    
    # Step 2: Upload data files
    print("\n" + "="*80)
    print("Step 2: Uploading data files to S3")
    print("="*80)
    upload_data_files(s3_client, BUCKET_NAME)
    
    # Step 3: Set up Secrets Manager
    print("\n" + "="*80)
    print("Step 3: Setting up Secrets Manager")
    print("="*80)
    if not setup_secrets_manager(secrets_client, SECRET_NAME):
        success = False
    
    # Step 4: Create IAM role
    print("\n" + "="*80)
    print("Step 4: Creating IAM role for Lambda")
    print("="*80)
    if not create_iam_role(iam_client, ROLE_NAME):
        success = False
    
    # Summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    if success:
        print("✓ Phase 1 setup complete!")
        print(f"\nResources created:")
        print(f"  - S3 Bucket: {BUCKET_NAME}")
        print(f"  - Secret: {SECRET_NAME}")
        print(f"  - IAM Role: {ROLE_NAME}")
        print(f"\nNext steps:")
        print(f"  1. Verify setup: python3 scripts/aws_verify_setup.py")
        print(f"  2. Proceed to Phase 2: Convert scripts to Lambda functions")
    else:
        print("⚠ Setup completed with some errors. Please review above.")
    
    print("="*80)


if __name__ == "__main__":
    main()



