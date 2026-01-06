# AWS Setup Guide - Phase 1

This guide walks you through setting up AWS for the Sports Betting Analytics cloud automation.

## Prerequisites

- AWS account (free tier eligible)
- AWS CLI installed and configured
- Python 3.12+ installed
- boto3 Python package installed

## Step 1: Create AWS Account

### 1.1 Sign Up for AWS
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the sign-up process
4. You'll need:
   - Email address
   - Credit card (for verification, won't be charged for free tier usage)
   - Phone number for verification

### 1.2 Access AWS Console
1. Once account is created, log in to [console.aws.amazon.com](https://console.aws.amazon.com)
2. Select a region (recommended: `us-east-1` for best free tier coverage)

### 1.3 Enable Free Tier Monitoring
1. Go to Billing Dashboard
2. Enable "Free Tier Usage Alerts"
3. Set up billing alerts to monitor costs

## Step 2: Install and Configure AWS CLI

### 2.1 Install AWS CLI
**macOS:**
```bash
brew install awscli
```

**Or download from:**
https://aws.amazon.com/cli/

### 2.2 Configure AWS CLI
```bash
aws configure
```

You'll need:
- **AWS Access Key ID**: Get from IAM Console → Users → Security Credentials
- **AWS Secret Access Key**: Get from same location
- **Default region**: `us-east-1` (or your preferred region)
- **Default output format**: `json`

### 2.3 Create IAM User (Recommended)
1. Go to IAM Console → Users
2. Click "Add users"
3. Username: `sports-betting-analytics`
4. Select "Programmatic access"
5. Attach policy: `AdministratorAccess` (for setup) or create custom policy
6. Save the Access Key ID and Secret Access Key

**Note:** For production, use least-privilege IAM policies instead of AdministratorAccess.

## Step 3: Install Python Dependencies

```bash
pip3 install boto3 awscli
```

## Step 4: Run Setup Script

Once AWS CLI is configured, run the setup script:

```bash
python3 scripts/aws_setup.py
```

This script will:
- Create S3 bucket
- Upload existing data files
- Set up Secrets Manager
- Create IAM roles for Lambda

## Step 5: Verify Setup

Run the verification script:

```bash
python3 scripts/aws_verify_setup.py
```

## Manual Setup (Alternative)

If you prefer to set up manually through AWS Console:

### Create S3 Bucket
1. Go to S3 Console
2. Click "Create bucket"
3. Bucket name: `sports-betting-analytics-data` (must be globally unique)
4. Region: `us-east-1`
5. Block all public access: **Enabled**
6. Versioning: **Enabled**
7. Click "Create bucket"

### Upload Data Files
1. Navigate to bucket
2. Create folder structure:
   - `data/results/`
   - `predictions/`
   - `scores/`
3. Upload Excel and Parquet files to `data/results/`

### Set Up Secrets Manager
1. Go to Secrets Manager Console
2. Click "Store a new secret"
3. Secret type: "Other type of secret"
4. Key: `ODDS_API_KEY`
5. Value: Your Odds API key
6. Secret name: `odds-api-key`
7. Click "Store"

### Create IAM Role for Lambda
1. Go to IAM Console → Roles
2. Click "Create role"
3. Trusted entity: AWS service → Lambda
4. Attach policies:
   - `AmazonS3FullAccess` (or custom S3 policy)
   - `SecretsManagerReadWrite` (or custom Secrets Manager policy)
5. Role name: `SportsBettingLambdaRole`
6. Click "Create role"

## Troubleshooting

### AWS CLI Not Found
- Make sure AWS CLI is installed: `aws --version`
- Add to PATH if needed

### Access Denied Errors
- Check IAM user has necessary permissions
- Verify AWS credentials: `aws sts get-caller-identity`

### Bucket Name Already Exists
- S3 bucket names must be globally unique
- Try a different name like `sports-betting-analytics-data-[yourname]`

### Region Issues
- Make sure all resources are in the same region
- Default region is set in `~/.aws/config`

## Next Steps

After Phase 1 is complete:
- Proceed to Phase 2: Convert Scripts to Lambda Functions
- See `CLOUD_AUTOMATION_PLAN.md` for details

## Cost Monitoring

- Set up CloudWatch billing alerts
- Monitor free tier usage in Billing Dashboard
- Expected cost: ~$1.60/month after free tier

---

**Last Updated**: 2025-01-15



