"""
Deploy Lambda Functions to AWS
Packages and deploys both Lambda functions with their dependencies
"""

import os
import sys
import zipfile
import shutil
import subprocess
import boto3
import json
from pathlib import Path

# Configuration
REGION = 'us-east-1'
ROLE_NAME = 'SportsBettingLambdaRole'
BUCKET_NAME = 'sports-betting-analytics-data'
SECRET_NAME = 'odds-api-key'

# Lambda function configurations
LAMBDA_FUNCTIONS = {
    'collect-yesterday-games': {
        'description': 'Collects yesterday\'s games and updates S3 data files',
        'timeout': 900,  # 15 minutes
        'memory': 512,   # MB
        'handler': 'lambda_function.lambda_handler',
        'runtime': 'python3.12',
        'schedule': 'cron(0 11 * * ? *)'  # 6:00 AM EST (11:00 UTC)
    },
    'generate-predictions': {
        'description': 'Generates betting opportunity predictions for today\'s games',
        'timeout': 600,  # 10 minutes
        'memory': 1024,  # MB (more for pandas operations)
        'handler': 'lambda_function.lambda_handler',
        'runtime': 'python3.12',
        'schedule': 'cron(30 11 * * ? *)'  # 6:30 AM EST (11:30 UTC)
    }
}


def get_role_arn():
    """Get the IAM role ARN"""
    iam = boto3.client('iam')
    try:
        response = iam.get_role(RoleName=ROLE_NAME)
        return response['Role']['Arn']
    except Exception as e:
        print(f"✗ Error getting IAM role: {e}")
        raise


def install_dependencies(function_dir, requirements_file):
    """Install dependencies into a package directory"""
    print(f"  Installing dependencies from {requirements_file}...")
    
    package_dir = os.path.join(function_dir, 'package')
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    
    # Install dependencies into package directory
    # Use --platform manylinux2014_x86_64 for Lambda compatibility
    # Use --only-binary :all: to avoid building from source
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 
         '-r', requirements_file,
         '-t', package_dir,
         '--platform', 'manylinux2014_x86_64',
         '--only-binary', ':all:',
         '--upgrade'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        # Try without platform specification (for local builds)
        print(f"  ⚠ Platform-specific install failed, trying standard install...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 
             '-r', requirements_file,
             '-t', package_dir,
             '--upgrade'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  ✗ Error installing dependencies: {result.stderr}")
            return False
    
    # Remove any __pycache__ directories and .pyc files
    for root, dirs, files in os.walk(package_dir):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))
            dirs.remove('__pycache__')
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))
    
    # Remove any .dist-info directories that might cause issues
    for item in os.listdir(package_dir):
        item_path = os.path.join(package_dir, item)
        if os.path.isdir(item_path) and item.endswith('.dist-info'):
            # Keep dist-info but clean it up
            pass
    
    print(f"  ✓ Dependencies installed")
    return True


def create_deployment_package(function_dir, function_name):
    """Create a zip file for Lambda deployment"""
    print(f"  Creating deployment package...")
    
    package_dir = os.path.join(function_dir, 'package')
    zip_path = os.path.join(function_dir, f'{function_name}.zip')
    
    # Remove old zip if exists
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    # Create zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add Lambda function code FIRST (so it's at the root of the zip)
        lambda_file = os.path.join(function_dir, 'lambda_function.py')
        if os.path.exists(lambda_file):
            zipf.write(lambda_file, 'lambda_function.py')
        
        # Add all files from package directory (dependencies)
        if os.path.exists(package_dir):
            for root, dirs, files in os.walk(package_dir):
                # Skip __pycache__ directories and .pyc files
                dirs[:] = [d for d in dirs if d != '__pycache__' and not d.endswith('.dist-info')]
                for file in files:
                    if file.endswith('.pyc') or file.endswith('.pyo'):
                        continue
                    file_path = os.path.join(root, file)
                    # Keep the directory structure from package_dir
                    arcname = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arcname)
    
    file_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    print(f"  ✓ Created {zip_path} ({file_size:.2f} MB)")
    
    return zip_path


def deploy_lambda_function(lambda_client, s3_client, function_name, config, zip_path, role_arn, bucket_name):
    """Deploy or update a Lambda function"""
    print(f"\nDeploying {function_name}...")
    
    try:
        # Check zip file size
        zip_size = os.path.getsize(zip_path)
        zip_size_mb = zip_size / (1024 * 1024)
        use_s3 = zip_size > 50 * 1024 * 1024  # 50 MB limit for direct upload
        
        if use_s3:
            print(f"  Package size ({zip_size_mb:.2f} MB) exceeds 50 MB, uploading to S3 first...")
            # Upload to S3
            s3_key = f"lambda-deployments/{function_name}.zip"
            with open(zip_path, 'rb') as f:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=f.read()
                )
            print(f"  ✓ Uploaded to S3: s3://{bucket_name}/{s3_key}")
            
            # Get S3 object version (if versioning enabled) or use latest
            code_dict = {
                'S3Bucket': bucket_name,
                'S3Key': s3_key
            }
        else:
            # Read zip file for direct upload
            with open(zip_path, 'rb') as f:
                zip_content = f.read()
            code_dict = {'ZipFile': zip_content}
            print(f"  Package size ({zip_size_mb:.2f} MB) is under 50 MB, uploading directly...")
        
        # Check if function exists
        try:
            lambda_client.get_function(FunctionName=function_name)
            function_exists = True
            print(f"  Function exists, updating...")
        except lambda_client.exceptions.ResourceNotFoundException:
            function_exists = False
            print(f"  Function doesn't exist, creating...")
        
        # Wait for any in-progress updates
        if function_exists:
            max_wait = 30  # Wait up to 30 seconds
            wait_time = 0
            while wait_time < max_wait:
                try:
                    response = lambda_client.get_function(FunctionName=function_name)
                    state = response['Configuration']['State']
                    if state == 'Active':
                        break
                    print(f"  Waiting for function to be ready (state: {state})...")
                    import time
                    time.sleep(2)
                    wait_time += 2
                except Exception:
                    break
        
        if function_exists:
            # Update function code
            lambda_client.update_function_code(
                FunctionName=function_name,
                **code_dict
            )
            print(f"  ✓ Updated function code")
            
            # Wait a moment for code update to complete
            import time
            time.sleep(2)
            
            # Update configuration
            try:
                lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Description=config['description'],
                    Timeout=config['timeout'],
                    MemorySize=config['memory'],
                    Runtime=config['runtime'],
                    Handler=config['handler']
                )
                print(f"  ✓ Updated function configuration")
            except Exception as e:
                if 'ResourceConflictException' in str(e):
                    print(f"  ⚠ Configuration update in progress, will complete automatically")
                else:
                    raise
        else:
            # Create new function
            lambda_client.create_function(
                FunctionName=function_name,
                Runtime=config['runtime'],
                Role=role_arn,
                Handler=config['handler'],
                Code=code_dict,
                Description=config['description'],
                Timeout=config['timeout'],
                MemorySize=config['memory'],
                Environment={
                    'Variables': {}
                }
            )
            print(f"  ✓ Created function")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error deploying function: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_eventbridge_rule(events_client, function_name, schedule, lambda_arn):
    """Create or update EventBridge rule to trigger Lambda"""
    rule_name = f"{function_name}-schedule"
    
    print(f"\nSetting up EventBridge schedule for {function_name}...")
    
    try:
        # Check if rule exists
        try:
            events_client.describe_rule(Name=rule_name)
            rule_exists = True
            print(f"  Rule exists, updating...")
        except events_client.exceptions.ResourceNotFoundException:
            rule_exists = False
            print(f"  Rule doesn't exist, creating...")
        
        if rule_exists:
            # Update rule
            events_client.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule,
                Description=f"Daily schedule for {function_name}",
                State='ENABLED'
            )
        else:
            # Create rule
            events_client.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule,
                Description=f"Daily schedule for {function_name}",
                State='ENABLED'
            )
        
        print(f"  ✓ Rule created/updated: {rule_name}")
        
        # Add Lambda as target
        try:
            events_client.put_targets(
                Rule=rule_name,
                Targets=[{
                    'Id': '1',
                    'Arn': lambda_arn
                }]
            )
            print(f"  ✓ Lambda added as target")
        except Exception as e:
            # Target might already exist
            if 'already exists' not in str(e).lower():
                print(f"  ⚠ Warning adding target: {e}")
            else:
                print(f"  ✓ Target already exists")
        
        # Add permission for EventBridge to invoke Lambda
        lambda_client = boto3.client('lambda', region_name=REGION)
        try:
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'{rule_name}-invoke',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=f'arn:aws:events:{REGION}:{boto3.client("sts").get_caller_identity()["Account"]}:rule/{rule_name}'
            )
            print(f"  ✓ Added EventBridge invoke permission")
        except Exception as e:
            if 'already exists' not in str(e).lower():
                print(f"  ⚠ Warning adding permission: {e}")
            else:
                print(f"  ✓ Permission already exists")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error setting up EventBridge: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("DEPLOY LAMBDA FUNCTIONS TO AWS")
    print("="*80)
    
    # Verify AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"\n✓ AWS credentials verified")
        print(f"  Account: {identity.get('Account')}")
        print(f"  User/Role: {identity.get('Arn')}")
    except Exception as e:
        print(f"\n✗ AWS credentials not configured: {e}")
        return 1
    
    # Get IAM role ARN
    try:
        role_arn = get_role_arn()
        print(f"\n✓ IAM role found: {role_arn}")
    except Exception as e:
        print(f"\n✗ IAM role not found. Please run aws_setup.py first.")
        return 1
    
    # Initialize AWS clients
    lambda_client = boto3.client('lambda', region_name=REGION)
    events_client = boto3.client('events', region_name=REGION)
    s3_client = boto3.client('s3', region_name=REGION)
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    lambda_functions_dir = base_dir / 'lambda_functions'
    
    # Process each Lambda function
    results = []
    
    for function_name, config in LAMBDA_FUNCTIONS.items():
        print("\n" + "="*80)
        print(f"PROCESSING: {function_name}")
        print("="*80)
        
        function_dir = lambda_functions_dir / function_name.replace('-', '_')
        requirements_file = function_dir / 'requirements.txt'
        
        if not function_dir.exists():
            print(f"✗ Function directory not found: {function_dir}")
            results.append((function_name, False))
            continue
        
        if not requirements_file.exists():
            print(f"✗ Requirements file not found: {requirements_file}")
            results.append((function_name, False))
            continue
        
        # Step 1: Install dependencies
        print(f"\nStep 1: Installing dependencies...")
        if not install_dependencies(str(function_dir), str(requirements_file)):
            print(f"✗ Failed to install dependencies")
            results.append((function_name, False))
            continue
        
        # Step 2: Create deployment package
        print(f"\nStep 2: Creating deployment package...")
        zip_path = create_deployment_package(str(function_dir), function_name)
        if not zip_path or not os.path.exists(zip_path):
            print(f"✗ Failed to create deployment package")
            results.append((function_name, False))
            continue
        
        # Step 3: Deploy to AWS Lambda
        print(f"\nStep 3: Deploying to AWS Lambda...")
        if not deploy_lambda_function(lambda_client, s3_client, function_name, config, zip_path, role_arn, BUCKET_NAME):
            print(f"✗ Failed to deploy function")
            results.append((function_name, False))
            continue
        
        # Step 4: Get Lambda ARN for EventBridge
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            lambda_arn = response['Configuration']['FunctionArn']
        except Exception as e:
            print(f"✗ Error getting Lambda ARN: {e}")
            results.append((function_name, False))
            continue
        
        # Step 5: Set up EventBridge schedule
        print(f"\nStep 4: Setting up EventBridge schedule...")
        if not create_eventbridge_rule(events_client, function_name, config['schedule'], lambda_arn):
            print(f"✗ Failed to set up EventBridge")
            results.append((function_name, False))
            continue
        
        results.append((function_name, True))
        print(f"\n✓ {function_name} deployed successfully!")
    
    # Summary
    print("\n" + "="*80)
    print("DEPLOYMENT SUMMARY")
    print("="*80)
    
    for function_name, success in results:
        status = "✓ DEPLOYED" if success else "✗ FAILED"
        print(f"{function_name}: {status}")
    
    all_success = all(result[1] for result in results)
    
    if all_success:
        print("\n✓ All Lambda functions deployed successfully!")
        print("\nNext steps:")
        print("  1. Test functions manually in AWS Console")
        print("  2. Check CloudWatch logs")
        print("  3. Verify S3 files are updated")
        return 0
    else:
        print("\n✗ Some deployments failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

