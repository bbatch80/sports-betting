#!/usr/bin/env python3
"""
AWS Setup Verification Script
Verifies that Phase 1 setup was completed successfully
"""

import boto3
from botocore.exceptions import ClientError, BotoCoreError

# Configuration (must match aws_setup.py)
BUCKET_NAME = 'sports-betting-analytics-data'
REGION = 'us-east-1'
SECRET_NAME = 'odds-api-key'
ROLE_NAME = 'SportsBettingLambdaRole'


def verify_s3_bucket(s3_client, bucket_name):
    """Verify S3 bucket exists and is configured correctly"""
    print("Checking S3 bucket...")
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"  ✓ Bucket '{bucket_name}' exists")
        
        # Check versioning
        versioning = s3_client.get_bucket_versioning(Bucket=bucket_name)
        if versioning.get('Status') == 'Enabled':
            print(f"  ✓ Versioning is enabled")
        else:
            print(f"  ⚠ Versioning is not enabled")
        
        # Check public access block
        try:
            public_access = s3_client.get_public_access_block(Bucket=bucket_name)
            config = public_access['PublicAccessBlockConfiguration']
            if all([config.get('BlockPublicAcls'), config.get('BlockPublicPolicy'),
                   config.get('RestrictPublicBuckets'), config.get('IgnorePublicAcls')]):
                print(f"  ✓ Public access is blocked")
            else:
                print(f"  ⚠ Public access may not be fully blocked")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchPublicAccessBlockConfiguration':
                print(f"  ⚠ Public access block not configured")
        
        # List files in bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='data/results/')
        if 'Contents' in response:
            file_count = len(response['Contents'])
            print(f"  ✓ Found {file_count} files in data/results/")
            
            # List file names
            files = [obj['Key'] for obj in response['Contents']]
            print(f"    Files:")
            for file_key in files[:10]:  # Show first 10
                print(f"      - {file_key}")
            if len(files) > 10:
                print(f"      ... and {len(files) - 10} more")
        else:
            print(f"  ⚠ No files found in data/results/")
        
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"  ✗ Bucket '{bucket_name}' does not exist")
        else:
            print(f"  ✗ Error checking bucket: {e}")
        return False


def verify_secrets_manager(secrets_client, secret_name):
    """Verify Secrets Manager secret exists"""
    print("\nChecking Secrets Manager...")
    try:
        response = secrets_client.describe_secret(SecretId=secret_name)
        print(f"  ✓ Secret '{secret_name}' exists")
        print(f"    Created: {response.get('CreatedDate')}")
        print(f"    Description: {response.get('Description', 'N/A')}")
        
        # Try to get secret value (just verify it exists, don't print it)
        try:
            secrets_client.get_secret_value(SecretId=secret_name)
            print(f"  ✓ Secret value is accessible")
        except ClientError as e:
            print(f"  ⚠ Warning: Cannot access secret value: {e}")
        
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"  ✗ Secret '{secret_name}' does not exist")
        else:
            print(f"  ✗ Error checking secret: {e}")
        return False


def verify_iam_role(iam_client, role_name):
    """Verify IAM role exists and has correct policies"""
    print("\nChecking IAM role...")
    try:
        response = iam_client.get_role(RoleName=role_name)
        print(f"  ✓ Role '{role_name}' exists")
        print(f"    ARN: {response['Role']['Arn']}")
        
        # Check attached policies
        policies = iam_client.list_attached_role_policies(RoleName=role_name)
        if policies['AttachedPolicies']:
            print(f"  ✓ Attached policies:")
            for policy in policies['AttachedPolicies']:
                print(f"      - {policy['PolicyName']}")
        else:
            print(f"  ⚠ No policies attached")
        
        # Check trust policy
        trust_policy = response['Role']['AssumeRolePolicyDocument']
        if 'lambda.amazonaws.com' in str(trust_policy):
            print(f"  ✓ Trust policy allows Lambda service")
        else:
            print(f"  ⚠ Trust policy may not allow Lambda")
        
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print(f"  ✗ Role '{role_name}' does not exist")
        else:
            print(f"  ✗ Error checking role: {e}")
        return False


def main():
    """Main verification function"""
    print("="*80)
    print("AWS SETUP VERIFICATION")
    print("="*80)
    print()
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"Region: {REGION}")
        print()
    except Exception as e:
        print(f"✗ AWS credentials not configured: {e}")
        return
    
    # Initialize AWS clients
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        secrets_client = boto3.client('secretsmanager', region_name=REGION)
        iam_client = boto3.client('iam')
    except Exception as e:
        print(f"✗ Error initializing AWS clients: {e}")
        return
    
    results = []
    
    # Verify each component
    results.append(verify_s3_bucket(s3_client, BUCKET_NAME))
    results.append(verify_secrets_manager(secrets_client, SECRET_NAME))
    results.append(verify_iam_role(iam_client, ROLE_NAME))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if all(results):
        print("✓ All components verified successfully!")
        print("\nPhase 1 setup is complete and ready for Phase 2.")
    else:
        print("⚠ Some components failed verification.")
        print("\nPlease run: python3 scripts/aws_setup.py")
        print("Or fix the issues manually in AWS Console.")
    
    print("="*80)


if __name__ == "__main__":
    main()



