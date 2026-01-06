"""
Test AWS Connections
Tests that we can connect to S3 and Secrets Manager
"""

import boto3
import sys
from datetime import datetime

BUCKET_NAME = 'sports-betting-analytics-data'
SECRET_NAME = 'odds-api-key'
REGION = 'us-east-1'

def test_s3_connection():
    """Test reading and writing to S3"""
    print("\n" + "="*80)
    print("TEST 1: S3 Connection")
    print("="*80)
    
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        
        # Test 1: List bucket contents
        print("\n1. Testing bucket access...")
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=5)
        print(f"   ✓ Can list bucket contents")
        if 'Contents' in response:
            print(f"   ✓ Found {len(response['Contents'])} objects (showing first 5)")
            for obj in response['Contents'][:3]:
                print(f"     - {obj['Key']}")
        else:
            print(f"   ✓ Bucket is empty (expected for first test)")
        
        # Test 2: Read a file (if exists)
        print("\n2. Testing file read...")
        test_key = "data/results/nfl_season_results.xlsx"
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=test_key)
            print(f"   ✓ Can read file: {test_key}")
            print(f"   ✓ File size: {response['ContentLength']} bytes")
        except s3_client.exceptions.NoSuchKey:
            print(f"   ✓ File doesn't exist yet (this is OK)")
        
        # Test 3: Write a test file
        print("\n3. Testing file write...")
        test_key = "test/test_file.txt"
        test_content = f"Test file created at {datetime.now().isoformat()}"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=test_key,
            Body=test_content.encode('utf-8'),
            ContentType='text/plain'
        )
        print(f"   ✓ Can write file: {test_key}")
        
        # Test 4: Delete test file
        print("\n4. Testing file delete...")
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=test_key)
        print(f"   ✓ Can delete file: {test_key}")
        
        print("\n✓ S3 connection test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ S3 connection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_secrets_manager():
    """Test reading from Secrets Manager"""
    print("\n" + "="*80)
    print("TEST 2: Secrets Manager Connection")
    print("="*80)
    
    try:
        secrets_client = boto3.client('secretsmanager', region_name=REGION)
        
        # Test: Get API key
        print("\n1. Testing secret retrieval...")
        response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        api_key = response['SecretString']
        
        if api_key:
            print(f"   ✓ Can retrieve secret: {SECRET_NAME}")
            print(f"   ✓ API key length: {len(api_key)} characters")
            print(f"   ✓ API key starts with: {api_key[:8]}...")
        else:
            print(f"   ✗ Secret is empty")
            return False
        
        print("\n✓ Secrets Manager connection test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Secrets Manager connection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_iam_permissions():
    """Test that IAM role has correct permissions"""
    print("\n" + "="*80)
    print("TEST 3: IAM Permissions Check")
    print("="*80)
    
    try:
        sts_client = boto3.client('sts')
        
        # Get current identity
        identity = sts_client.get_caller_identity()
        print(f"\n1. Current AWS Identity:")
        print(f"   Account: {identity.get('Account')}")
        print(f"   User/Role: {identity.get('Arn')}")
        
        # Test S3 permissions
        print("\n2. Testing S3 permissions...")
        s3_client = boto3.client('s3', region_name=REGION)
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"   ✓ Can access bucket: {BUCKET_NAME}")
        
        # Test Secrets Manager permissions
        print("\n3. Testing Secrets Manager permissions...")
        secrets_client = boto3.client('secretsmanager', region_name=REGION)
        secrets_client.describe_secret(SecretId=SECRET_NAME)
        print(f"   ✓ Can access secret: {SECRET_NAME}")
        
        print("\n✓ IAM permissions test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ IAM permissions test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("AWS CONNECTION TESTS")
    print("="*80)
    print("\nTesting AWS service connections before deploying Lambda functions...")
    
    results = []
    
    # Test S3
    results.append(("S3 Connection", test_s3_connection()))
    
    # Test Secrets Manager
    results.append(("Secrets Manager", test_secrets_manager()))
    
    # Test IAM Permissions
    results.append(("IAM Permissions", test_iam_permissions()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All AWS connection tests PASSED!")
        print("  Ready to proceed with Lambda function testing.")
        return 0
    else:
        print("\n✗ Some tests FAILED!")
        print("  Please fix AWS configuration before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())



