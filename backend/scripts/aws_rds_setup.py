#!/usr/bin/env python3
"""
AWS RDS Setup Script
Creates PostgreSQL RDS instance for Sports Betting Analytics.

This script:
1. Creates security group for RDS
2. Creates RDS PostgreSQL instance (Free Tier: db.t3.micro)
3. Stores database credentials in Secrets Manager
4. Updates Lambda IAM role with RDS permissions

Usage:
    python3 scripts/aws_rds_setup.py

Prerequisites:
    - AWS credentials configured
    - aws_setup.py already run (IAM role exists)
"""

import boto3
import json
import os
import secrets
import string
import time
from botocore.exceptions import ClientError

# Configuration - match existing aws_setup.py
REGION = 'us-east-1'
ROLE_NAME = 'SportsBettingLambdaRole'

# RDS Configuration
RDS_INSTANCE_ID = 'sports-betting-analytics'
RDS_DB_NAME = 'analytics'
RDS_USERNAME = 'sbadmin'
RDS_INSTANCE_CLASS = 'db.t3.micro'  # Free Tier eligible
RDS_ENGINE = 'postgres'
RDS_ENGINE_VERSION = '15.10'  # PostgreSQL 15
RDS_ALLOCATED_STORAGE = 20  # GB (Free Tier: up to 20 GB)
RDS_SECURITY_GROUP_NAME = 'sports-betting-rds-sg'
RDS_SECRET_NAME = 'sports-betting-db-credentials'

# VPC Configuration (using default VPC for simplicity)
USE_DEFAULT_VPC = True


def check_aws_credentials():
    """Verify AWS credentials are configured."""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials configured")
        print(f"  Account ID: {identity['Account']}")
        print(f"  User ARN: {identity['Arn']}")
        return identity['Account']
    except Exception as e:
        print(f"✗ AWS credentials not configured: {e}")
        return None


def generate_password(length=24):
    """Generate a secure random password."""
    # AWS RDS password requirements: letters, digits, and specific symbols
    alphabet = string.ascii_letters + string.digits + "!#$%&*+=-_"
    # Ensure at least one of each type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("!#$%&*+=-_"),
    ]
    # Fill the rest
    password += [secrets.choice(alphabet) for _ in range(length - 4)]
    # Shuffle
    secrets.SystemRandom().shuffle(password)
    return ''.join(password)


def get_default_vpc(ec2_client):
    """Get the default VPC ID."""
    try:
        response = ec2_client.describe_vpcs(
            Filters=[{'Name': 'is-default', 'Values': ['true']}]
        )
        if response['Vpcs']:
            vpc_id = response['Vpcs'][0]['VpcId']
            print(f"✓ Found default VPC: {vpc_id}")
            return vpc_id
        else:
            print("✗ No default VPC found")
            return None
    except Exception as e:
        print(f"✗ Error getting default VPC: {e}")
        return None


def get_default_subnets(ec2_client, vpc_id):
    """Get subnet IDs in the default VPC (need at least 2 for RDS subnet group)."""
    try:
        response = ec2_client.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
        )
        subnet_ids = [subnet['SubnetId'] for subnet in response['Subnets']]
        azs = [subnet['AvailabilityZone'] for subnet in response['Subnets']]
        print(f"✓ Found {len(subnet_ids)} subnets in VPC")
        for subnet_id, az in zip(subnet_ids, azs):
            print(f"    {subnet_id} ({az})")
        return subnet_ids
    except Exception as e:
        print(f"✗ Error getting subnets: {e}")
        return []


def create_security_group(ec2_client, vpc_id, sg_name):
    """Create security group for RDS."""
    try:
        # Check if security group already exists
        response = ec2_client.describe_security_groups(
            Filters=[
                {'Name': 'group-name', 'Values': [sg_name]},
                {'Name': 'vpc-id', 'Values': [vpc_id]}
            ]
        )

        if response['SecurityGroups']:
            sg_id = response['SecurityGroups'][0]['GroupId']
            print(f"✓ Security group '{sg_name}' already exists: {sg_id}")
            return sg_id

        # Create new security group
        response = ec2_client.create_security_group(
            GroupName=sg_name,
            Description='Security group for Sports Betting RDS instance',
            VpcId=vpc_id
        )
        sg_id = response['GroupId']

        # Add inbound rule for PostgreSQL (port 5432)
        # WARNING: 0.0.0.0/0 allows access from anywhere - restrict in production!
        ec2_client.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 5432,
                    'ToPort': 5432,
                    'IpRanges': [
                        {
                            'CidrIp': '0.0.0.0/0',
                            'Description': 'PostgreSQL access - RESTRICT IN PRODUCTION'
                        }
                    ]
                }
            ]
        )

        print(f"✓ Created security group: {sg_id}")
        print(f"  ⚠ WARNING: Allows inbound from 0.0.0.0/0 - restrict for production!")
        return sg_id

    except ClientError as e:
        if 'InvalidGroup.Duplicate' in str(e):
            # Get existing group
            response = ec2_client.describe_security_groups(
                Filters=[
                    {'Name': 'group-name', 'Values': [sg_name]},
                    {'Name': 'vpc-id', 'Values': [vpc_id]}
                ]
            )
            sg_id = response['SecurityGroups'][0]['GroupId']
            print(f"✓ Security group already exists: {sg_id}")
            return sg_id
        print(f"✗ Error creating security group: {e}")
        return None


def create_db_subnet_group(rds_client, subnet_ids):
    """Create DB subnet group for RDS."""
    subnet_group_name = 'sports-betting-db-subnet-group'

    try:
        # Check if subnet group exists
        try:
            rds_client.describe_db_subnet_groups(
                DBSubnetGroupName=subnet_group_name
            )
            print(f"✓ DB subnet group '{subnet_group_name}' already exists")
            return subnet_group_name
        except ClientError as e:
            if 'DBSubnetGroupNotFoundFault' not in str(e):
                raise

        # Create subnet group
        rds_client.create_db_subnet_group(
            DBSubnetGroupName=subnet_group_name,
            DBSubnetGroupDescription='Subnet group for Sports Betting RDS',
            SubnetIds=subnet_ids
        )
        print(f"✓ Created DB subnet group: {subnet_group_name}")
        return subnet_group_name

    except Exception as e:
        print(f"✗ Error creating DB subnet group: {e}")
        return None


def store_credentials_in_secrets_manager(secrets_client, secret_name, username, password, host, port, db_name):
    """Store database credentials in AWS Secrets Manager."""
    secret_value = {
        'username': username,
        'password': password,
        'host': host,
        'port': port,
        'dbname': db_name,
        'engine': 'postgres',
        # Convenience: full connection URL
        'url': f'postgresql://{username}:{password}@{host}:{port}/{db_name}'
    }

    try:
        # Check if secret exists
        try:
            secrets_client.describe_secret(SecretId=secret_name)
            # Update existing secret
            secrets_client.put_secret_value(
                SecretId=secret_name,
                SecretString=json.dumps(secret_value)
            )
            print(f"✓ Updated existing secret: {secret_name}")
            return True
        except ClientError as e:
            if 'ResourceNotFoundException' not in str(e):
                raise

        # Create new secret
        secrets_client.create_secret(
            Name=secret_name,
            Description='PostgreSQL credentials for Sports Betting Analytics',
            SecretString=json.dumps(secret_value)
        )
        print(f"✓ Created secret: {secret_name}")
        return True

    except Exception as e:
        print(f"✗ Error storing credentials: {e}")
        return False


def create_rds_instance(rds_client, instance_id, db_name, username, password,
                        security_group_id, subnet_group_name):
    """Create RDS PostgreSQL instance."""
    try:
        # Check if instance already exists
        try:
            response = rds_client.describe_db_instances(
                DBInstanceIdentifier=instance_id
            )
            instance = response['DBInstances'][0]
            status = instance['DBInstanceStatus']
            print(f"✓ RDS instance '{instance_id}' already exists (status: {status})")

            if status == 'available':
                endpoint = instance['Endpoint']
                return endpoint['Address'], endpoint['Port']
            else:
                print(f"  Waiting for instance to become available...")
                return wait_for_rds_available(rds_client, instance_id)

        except ClientError as e:
            if 'DBInstanceNotFound' not in str(e):
                raise

        print(f"Creating RDS instance '{instance_id}'...")
        print(f"  Instance class: {RDS_INSTANCE_CLASS}")
        print(f"  Engine: PostgreSQL {RDS_ENGINE_VERSION}")
        print(f"  Storage: {RDS_ALLOCATED_STORAGE} GB")
        print(f"  This will take 5-10 minutes...")

        rds_client.create_db_instance(
            DBInstanceIdentifier=instance_id,
            DBInstanceClass=RDS_INSTANCE_CLASS,
            Engine=RDS_ENGINE,
            EngineVersion=RDS_ENGINE_VERSION,
            MasterUsername=username,
            MasterUserPassword=password,
            DBName=db_name,
            AllocatedStorage=RDS_ALLOCATED_STORAGE,
            VpcSecurityGroupIds=[security_group_id],
            DBSubnetGroupName=subnet_group_name,
            PubliclyAccessible=True,  # Required for Lambda access without VPC config
            BackupRetentionPeriod=0,  # Disabled for Free Tier (set to 1-7 after upgrading)
            MultiAZ=False,  # Free Tier
            StorageType='gp2',
            StorageEncrypted=True,
            DeletionProtection=False,  # Set True for production
            Tags=[
                {'Key': 'Project', 'Value': 'sports-betting-analytics'},
                {'Key': 'Environment', 'Value': 'development'}
            ]
        )

        print(f"✓ RDS instance creation initiated")
        return wait_for_rds_available(rds_client, instance_id)

    except Exception as e:
        print(f"✗ Error creating RDS instance: {e}")
        return None, None


def wait_for_rds_available(rds_client, instance_id, max_wait_minutes=15):
    """Wait for RDS instance to become available."""
    print(f"  Waiting for RDS instance to be available (max {max_wait_minutes} min)...")

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print(f"✗ Timeout waiting for RDS instance")
            return None, None

        try:
            response = rds_client.describe_db_instances(
                DBInstanceIdentifier=instance_id
            )
            instance = response['DBInstances'][0]
            status = instance['DBInstanceStatus']

            if status == 'available':
                endpoint = instance['Endpoint']
                print(f"✓ RDS instance is available!")
                print(f"  Endpoint: {endpoint['Address']}")
                print(f"  Port: {endpoint['Port']}")
                return endpoint['Address'], endpoint['Port']

            minutes_elapsed = int(elapsed / 60)
            seconds_elapsed = int(elapsed % 60)
            print(f"  Status: {status} ({minutes_elapsed}m {seconds_elapsed}s elapsed)")
            time.sleep(30)

        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(30)


def update_iam_role_for_rds(iam_client, role_name, account_id):
    """Add RDS permissions to the Lambda IAM role."""
    policy_name = 'RDSAccessPolicy'

    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "rds-db:connect"
                ],
                "Resource": f"arn:aws:rds-db:{REGION}:{account_id}:dbuser:*/*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue"
                ],
                "Resource": f"arn:aws:secretsmanager:{REGION}:{account_id}:secret:{RDS_SECRET_NAME}*"
            }
        ]
    }

    try:
        # Check if policy exists
        try:
            iam_client.get_role_policy(
                RoleName=role_name,
                PolicyName=policy_name
            )
            print(f"✓ IAM policy '{policy_name}' already exists on role")
            return True
        except ClientError as e:
            if 'NoSuchEntity' not in str(e):
                raise

        # Attach inline policy
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document)
        )
        print(f"✓ Added RDS access policy to role '{role_name}'")
        return True

    except Exception as e:
        print(f"✗ Error updating IAM role: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 80)
    print("AWS RDS SETUP - PostgreSQL for Sports Betting Analytics")
    print("=" * 80)
    print()

    # Step 0: Check credentials
    account_id = check_aws_credentials()
    if not account_id:
        return 1

    # Initialize clients
    ec2_client = boto3.client('ec2', region_name=REGION)
    rds_client = boto3.client('rds', region_name=REGION)
    secrets_client = boto3.client('secretsmanager', region_name=REGION)
    iam_client = boto3.client('iam')

    # Step 1: Get default VPC
    print("\n" + "=" * 80)
    print("Step 1: Getting VPC Configuration")
    print("=" * 80)

    vpc_id = get_default_vpc(ec2_client)
    if not vpc_id:
        print("✗ Cannot proceed without VPC")
        return 1

    subnet_ids = get_default_subnets(ec2_client, vpc_id)
    if len(subnet_ids) < 2:
        print("✗ Need at least 2 subnets for RDS subnet group")
        return 1

    # Step 2: Create security group
    print("\n" + "=" * 80)
    print("Step 2: Creating Security Group")
    print("=" * 80)

    security_group_id = create_security_group(ec2_client, vpc_id, RDS_SECURITY_GROUP_NAME)
    if not security_group_id:
        return 1

    # Step 3: Create DB subnet group
    print("\n" + "=" * 80)
    print("Step 3: Creating DB Subnet Group")
    print("=" * 80)

    subnet_group_name = create_db_subnet_group(rds_client, subnet_ids)
    if not subnet_group_name:
        return 1

    # Step 4: Generate password and create RDS instance
    print("\n" + "=" * 80)
    print("Step 4: Creating RDS Instance")
    print("=" * 80)

    password = generate_password()

    host, port = create_rds_instance(
        rds_client,
        RDS_INSTANCE_ID,
        RDS_DB_NAME,
        RDS_USERNAME,
        password,
        security_group_id,
        subnet_group_name
    )

    if not host:
        print("✗ Failed to create RDS instance")
        return 1

    # Step 5: Store credentials
    print("\n" + "=" * 80)
    print("Step 5: Storing Credentials in Secrets Manager")
    print("=" * 80)

    if not store_credentials_in_secrets_manager(
        secrets_client, RDS_SECRET_NAME, RDS_USERNAME, password, host, port, RDS_DB_NAME
    ):
        return 1

    # Step 6: Update IAM role
    print("\n" + "=" * 80)
    print("Step 6: Updating IAM Role for RDS Access")
    print("=" * 80)

    if not update_iam_role_for_rds(iam_client, ROLE_NAME, account_id):
        print("⚠ Warning: IAM role update failed, Lambda may not be able to access RDS")

    # Summary
    print("\n" + "=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print()
    print("RDS Instance Details:")
    print(f"  Instance ID: {RDS_INSTANCE_ID}")
    print(f"  Endpoint: {host}")
    print(f"  Port: {port}")
    print(f"  Database: {RDS_DB_NAME}")
    print(f"  Username: {RDS_USERNAME}")
    print(f"  Secret: {RDS_SECRET_NAME}")
    print()
    print("Connection URL (stored in Secrets Manager):")
    print(f"  postgresql://{RDS_USERNAME}:****@{host}:{port}/{RDS_DB_NAME}")
    print()
    print("Next Steps:")
    print("  1. Run data migration:")
    print(f"     cd backend")
    print(f"     PYTHONPATH=. python -m migrations.migrate_sqlite_to_pg \\")
    print(f"         --sqlite-path data/analytics.db \\")
    print(f"         --pg-url 'postgresql://{RDS_USERNAME}:<password>@{host}:{port}/{RDS_DB_NAME}'")
    print()
    print("  2. Update Lambda environment variables:")
    print(f"     DATABASE_URL=postgresql://{RDS_USERNAME}:<password>@{host}:{port}/{RDS_DB_NAME}")
    print()
    print("  To get the password, run:")
    print(f"     aws secretsmanager get-secret-value --secret-id {RDS_SECRET_NAME} --query SecretString --output text | jq -r .password")
    print()
    print("⚠ SECURITY NOTE:")
    print("  The RDS instance is publicly accessible for development convenience.")
    print("  For production, configure Lambda VPC access and disable public accessibility.")

    return 0


if __name__ == "__main__":
    exit(main())
