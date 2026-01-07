"""
Deploy API Gateway and API Lambda Functions
Creates REST API endpoints for mobile app to fetch predictions and results
"""

import os
import sys
import zipfile
import shutil
import subprocess
import boto3
import json
from pathlib import Path
import time

# Configuration
REGION = 'us-east-1'
ROLE_NAME = 'SportsBettingLambdaRole'
BUCKET_NAME = 'sports-betting-analytics-data'
API_NAME = 'sports-betting-predictions-api'
STAGE_NAME = 'prod'

# Lambda function configurations
PREDICTIONS_API_FUNCTION_NAME = 'predictions-api'
PREDICTIONS_API_CONFIG = {
    'description': 'Serves predictions from S3 via API Gateway',
    'timeout': 30,
    'memory': 256,
    'handler': 'lambda_function.lambda_handler',
    'runtime': 'python3.12',
    'has_dependencies': False
}

RESULTS_API_FUNCTION_NAME = 'results-api'
RESULTS_API_CONFIG = {
    'description': 'Serves game results from S3 via API Gateway',
    'timeout': 30,
    'memory': 512,  # More memory for pandas operations
    'handler': 'lambda_function.lambda_handler',
    'runtime': 'python3.12',
    'has_dependencies': True
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
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))
            dirs.remove('__pycache__')
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))
    
    print(f"  ✓ Dependencies installed")
    return True


def create_deployment_package(function_dir, function_name, has_dependencies=False):
    """Create a zip file for Lambda deployment"""
    print(f"  Creating deployment package...")
    
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
        
        # Add dependencies if needed
        if has_dependencies:
            package_dir = os.path.join(function_dir, 'package')
            if os.path.exists(package_dir):
                for root, dirs, files in os.walk(package_dir):
                    # Skip __pycache__ directories
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


def deploy_api_lambda_function(lambda_client, s3_client, function_name, config, zip_path, role_arn, bucket_name):
    """Deploy or update the API Lambda function"""
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
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                try:
                    response = lambda_client.get_function(FunctionName=function_name)
                    state = response['Configuration']['State']
                    if state == 'Active':
                        break
                    time.sleep(2)
                    wait_time += 2
                except Exception:
                    break
        
        # Wait for any in-progress updates
        if function_exists:
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                try:
                    response = lambda_client.get_function(FunctionName=function_name)
                    state = response['Configuration']['State']
                    if state == 'Active':
                        break
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
                if 'ResourceConflictException' not in str(e):
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
                MemorySize=config['memory']
            )
            print(f"  ✓ Created function")
        
        # Get function ARN
        response = lambda_client.get_function(FunctionName=function_name)
        function_arn = response['Configuration']['FunctionArn']
        
        return function_arn
        
    except Exception as e:
        print(f"  ✗ Error deploying function: {e}")
        import traceback
        traceback.print_exc()
        return None


def setup_api_resource(apigw_client, api_id, api_resource_id, resource_path, resource_name, 
                       lambda_arn, function_name, api_name):
    """Set up an API Gateway resource (predictions or results)"""
    # Refresh resources to get latest
    resources = apigw_client.get_resources(restApiId=api_id)
    
    # Find or create the resource
    target_resource_id = None
    for resource in resources['items']:
        if resource['path'] == resource_path:
            target_resource_id = resource['id']
            break
    
    if not target_resource_id:
        # Build the resource path step by step
        resource_parts = [p for p in resource_path.split('/') if p]  # ['api', 'predictions', '{sport}']
        parent_id = api_resource_id
        
        # Skip 'api' since we already have api_resource_id
        for i, part in enumerate(resource_parts[1:], 1):  # Start from index 1
            current_path = '/' + '/'.join(resource_parts[:i+1])
            
            # Check if this path already exists
            found = False
            for resource in resources['items']:
                if resource['path'] == current_path:
                    parent_id = resource['id']
                    found = True
                    break
            
            if not found:
                # Create the resource
                new_resource = apigw_client.create_resource(
                    restApiId=api_id,
                    parentId=parent_id,
                    pathPart=part
                )
                parent_id = new_resource['id']
                # Refresh resources list
                resources = apigw_client.get_resources(restApiId=api_id)
        
        target_resource_id = parent_id
        print(f"  ✓ Created {resource_path} resource")
    else:
        print(f"  {resource_path} resource already exists")
    
    # Create GET method
    try:
        apigw_client.get_method(restApiId=api_id, resourceId=target_resource_id, httpMethod='GET')
        print(f"  GET method already exists for {resource_path}")
    except apigw_client.exceptions.NotFoundException:
        apigw_client.put_method(
            restApiId=api_id,
            resourceId=target_resource_id,
            httpMethod='GET',
            authorizationType='NONE'
        )
        print(f"  ✓ Created GET method for {resource_path}")
    
    # Create OPTIONS method for CORS
    try:
        apigw_client.get_method(restApiId=api_id, resourceId=target_resource_id, httpMethod='OPTIONS')
        print(f"  OPTIONS method already exists for {resource_path}")
    except apigw_client.exceptions.NotFoundException:
        apigw_client.put_method(
            restApiId=api_id,
            resourceId=target_resource_id,
            httpMethod='OPTIONS',
            authorizationType='NONE'
        )
        print(f"  ✓ Created OPTIONS method for {resource_path}")
    
    # Set up Lambda integration for GET
    lambda_uri = f"arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
    try:
        apigw_client.put_integration(
            restApiId=api_id,
            resourceId=target_resource_id,
            httpMethod='GET',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=lambda_uri
        )
        print(f"  ✓ Set up Lambda integration for GET on {resource_path}")
    except Exception as e:
        if 'already exists' not in str(e).lower():
            print(f"  ⚠ Integration may already exist: {e}")
    
    # Set up Lambda integration for OPTIONS
    try:
        apigw_client.put_integration(
            restApiId=api_id,
            resourceId=target_resource_id,
            httpMethod='OPTIONS',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=lambda_uri
        )
        print(f"  ✓ Set up Lambda integration for OPTIONS on {resource_path}")
    except Exception as e:
        if 'already exists' not in str(e).lower():
            print(f"  ⚠ Integration may already exist: {e}")
    
    # Add permission for API Gateway to invoke Lambda
    lambda_client = boto3.client('lambda', region_name=REGION)
    try:
        account_id = boto3.client('sts').get_caller_identity()['Account']
        source_arn = f"arn:aws:execute-api:{REGION}:{account_id}:{api_id}/*/*"
        
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f'{api_name}-{resource_name}-invoke',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=source_arn
        )
        print(f"  ✓ Added API Gateway invoke permission for {function_name}")
    except Exception as e:
        if 'already exists' not in str(e).lower():
            print(f"  ⚠ Permission may already exist: {e}")


def create_api_gateway(apigw_client, api_name, predictions_lambda_arn, predictions_function_name,
                       results_lambda_arn, results_function_name):
    """Create API Gateway REST API with both predictions and results endpoints"""
    print(f"\nCreating API Gateway...")
    
    try:
        # Check if API already exists
        apis = apigw_client.get_rest_apis()
        api_id = None
        for api in apis['items']:
            if api['name'] == api_name:
                api_id = api['id']
                print(f"  API exists, using existing: {api_id}")
                break
        
        if not api_id:
            # Create new REST API
            response = apigw_client.create_rest_api(
                name=api_name,
                description='Sports Betting Predictions and Results API',
                endpointConfiguration={
                    'types': ['REGIONAL']
                }
            )
            api_id = response['id']
            print(f"  ✓ Created API: {api_id}")
        
        # Get root resource ID
        resources = apigw_client.get_resources(restApiId=api_id)
        root_resource_id = None
        for resource in resources['items']:
            if resource['path'] == '/':
                root_resource_id = resource['id']
                break
        
        if not root_resource_id:
            raise Exception("Could not find root resource")
        
        # Create /api resource
        api_resource_id = None
        for resource in resources['items']:
            if resource['path'] == '/api':
                api_resource_id = resource['id']
                break
        
        if not api_resource_id:
            api_resource = apigw_client.create_resource(
                restApiId=api_id,
                parentId=root_resource_id,
                pathPart='api'
            )
            api_resource_id = api_resource['id']
            print(f"  ✓ Created /api resource")
        
        # Set up predictions endpoints
        print(f"\n  Setting up predictions endpoints...")
        setup_api_resource(
            apigw_client, api_id, api_resource_id,
            '/api/predictions/{sport}', 'predictions',
            predictions_lambda_arn, predictions_function_name, api_name
        )

        # Set up elite-teams endpoints (uses predictions Lambda)
        print(f"\n  Setting up elite-teams endpoints...")
        setup_api_resource(
            apigw_client, api_id, api_resource_id,
            '/api/elite-teams', 'elite-teams',
            predictions_lambda_arn, predictions_function_name, api_name
        )
        setup_api_resource(
            apigw_client, api_id, api_resource_id,
            '/api/elite-teams/{sport}', 'elite-teams-sport',
            predictions_lambda_arn, predictions_function_name, api_name
        )

        # Set up strategy-performance endpoint (uses predictions Lambda)
        print(f"\n  Setting up strategy-performance endpoints...")
        setup_api_resource(
            apigw_client, api_id, api_resource_id,
            '/api/strategy-performance/{sport}', 'strategy-performance',
            predictions_lambda_arn, predictions_function_name, api_name
        )

        # Set up results endpoints
        print(f"\n  Setting up results endpoints...")
        setup_api_resource(
            apigw_client, api_id, api_resource_id,
            '/api/results/{sport}', 'results',
            results_lambda_arn, results_function_name, api_name
        )
        
        # Deploy API to stage
        try:
            apigw_client.create_deployment(
                restApiId=api_id,
                stageName=STAGE_NAME,
                description='Production deployment'
            )
            print(f"  ✓ Deployed API to {STAGE_NAME} stage")
        except Exception as e:
            # Deployment might already exist, update it
            try:
                apigw_client.create_deployment(
                    restApiId=api_id,
                    stageName=STAGE_NAME
                )
                print(f"  ✓ Updated deployment")
            except:
                print(f"  ⚠ Deployment may already exist")
        
        # Get API endpoint URL
        api_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/{STAGE_NAME}"
        print(f"\n  ✓ API Gateway URL: {api_url}")
        
        return api_id, api_url
        
    except Exception as e:
        print(f"  ✗ Error creating API Gateway: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    print("="*80)
    print("DEPLOY API GATEWAY AND API LAMBDA FUNCTIONS")
    print("="*80)
    
    # Verify AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"\n✓ AWS credentials verified")
        print(f"  Account: {identity.get('Account')}")
    except Exception as e:
        print(f"\n✗ AWS credentials not configured: {e}")
        return 1
    
    # Get IAM role ARN
    try:
        role_arn = get_role_arn()
        print(f"✓ IAM role found: {role_arn}")
    except Exception as e:
        print(f"\n✗ IAM role not found. Please run aws_setup.py first.")
        return 1
    
    # Initialize AWS clients
    lambda_client = boto3.client('lambda', region_name=REGION)
    s3_client = boto3.client('s3', region_name=REGION)
    apigw_client = boto3.client('apigateway', region_name=REGION)
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    lambda_functions_dir = base_dir / 'lambda_functions'
    
    # ===== DEPLOY PREDICTIONS API =====
    print("\n" + "="*80)
    print("DEPLOYING PREDICTIONS API")
    print("="*80)
    
    predictions_function_dir = lambda_functions_dir / 'predictions_api'
    if not predictions_function_dir.exists():
        print(f"✗ Function directory not found: {predictions_function_dir}")
        return 1
    
    # Step 1: Package predictions API
    print("\nStep 1: Packaging predictions API...")
    predictions_zip_path = create_deployment_package(
        str(predictions_function_dir), 
        PREDICTIONS_API_FUNCTION_NAME,
        has_dependencies=PREDICTIONS_API_CONFIG['has_dependencies']
    )
    if not predictions_zip_path or not os.path.exists(predictions_zip_path):
        print(f"✗ Failed to create deployment package")
        return 1
    
    # Step 2: Deploy predictions Lambda function
    print("\nStep 2: Deploying predictions Lambda function...")
    predictions_lambda_arn = deploy_api_lambda_function(
        lambda_client, s3_client,
        PREDICTIONS_API_FUNCTION_NAME, 
        PREDICTIONS_API_CONFIG, 
        predictions_zip_path, 
        role_arn,
        BUCKET_NAME
    )
    
    if not predictions_lambda_arn:
        print(f"✗ Failed to deploy predictions Lambda function")
        return 1
    
    # ===== DEPLOY RESULTS API =====
    print("\n" + "="*80)
    print("DEPLOYING RESULTS API")
    print("="*80)
    
    results_function_dir = lambda_functions_dir / 'results_api'
    if not results_function_dir.exists():
        print(f"✗ Function directory not found: {results_function_dir}")
        return 1
    
    # Step 1: Install dependencies for results API
    print("\nStep 1: Installing dependencies for results API...")
    requirements_file = results_function_dir / 'requirements.txt'
    if not requirements_file.exists():
        print(f"✗ Requirements file not found: {requirements_file}")
        return 1
    
    if not install_dependencies(str(results_function_dir), str(requirements_file)):
        print(f"✗ Failed to install dependencies")
        return 1
    
    # Step 2: Package results API
    print("\nStep 2: Packaging results API...")
    results_zip_path = create_deployment_package(
        str(results_function_dir), 
        RESULTS_API_FUNCTION_NAME,
        has_dependencies=RESULTS_API_CONFIG['has_dependencies']
    )
    if not results_zip_path or not os.path.exists(results_zip_path):
        print(f"✗ Failed to create deployment package")
        return 1
    
    # Step 3: Deploy results Lambda function
    print("\nStep 3: Deploying results Lambda function...")
    results_lambda_arn = deploy_api_lambda_function(
        lambda_client, s3_client,
        RESULTS_API_FUNCTION_NAME, 
        RESULTS_API_CONFIG, 
        results_zip_path, 
        role_arn,
        BUCKET_NAME
    )
    
    if not results_lambda_arn:
        print(f"✗ Failed to deploy results Lambda function")
        return 1
    
    # ===== CREATE API GATEWAY =====
    print("\n" + "="*80)
    print("CREATING API GATEWAY")
    print("="*80)
    api_id, api_url = create_api_gateway(
        apigw_client, API_NAME, 
        predictions_lambda_arn, PREDICTIONS_API_FUNCTION_NAME,
        results_lambda_arn, RESULTS_API_FUNCTION_NAME
    )
    
    if not api_id:
        print(f"✗ Failed to create API Gateway")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("DEPLOYMENT COMPLETE!")
    print("="*80)
    print(f"\n✓ Predictions API Lambda function deployed: {PREDICTIONS_API_FUNCTION_NAME}")
    print(f"✓ Results API Lambda function deployed: {RESULTS_API_FUNCTION_NAME}")
    print(f"✓ API Gateway created: {API_NAME}")
    print(f"\nAPI Endpoints:")
    print(f"  Predictions:")
    print(f"    {api_url}/api/predictions/nfl")
    print(f"    {api_url}/api/predictions/nba")
    print(f"    {api_url}/api/predictions/ncaam")
    print(f"    {api_url}/api/predictions/all")
    print(f"  Elite Teams:")
    print(f"    {api_url}/api/elite-teams")
    print(f"    {api_url}/api/elite-teams/nfl")
    print(f"    {api_url}/api/elite-teams/nba")
    print(f"    {api_url}/api/elite-teams/ncaam")
    print(f"  Results:")
    print(f"    {api_url}/api/results/nfl")
    print(f"    {api_url}/api/results/nba")
    print(f"    {api_url}/api/results/ncaam")
    print(f"    {api_url}/api/results/all")
    print(f"\nTest the APIs:")
    print(f"  curl {api_url}/api/predictions/nba")
    print(f"  curl {api_url}/api/elite-teams")
    print(f"  curl {api_url}/api/results/nba")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



