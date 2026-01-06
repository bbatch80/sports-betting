# Cloud-Based Automation Plan

## Overview

This document outlines the recommended approach for automating data collection, predictive analytics, and mobile app integration using cloud-based services. The goal is to create a fully automated system that runs independently of the local machine.

## Current State

### What's Automated (Local)
- **Data Collection**: `collect_yesterday_games.py` runs daily via `launchd` at 6:00 AM EST
- **Data Storage**: Excel/Parquet files in `data/results/` directory

### What's Manual
- **Predictive Analytics**: Notebooks (`ncaam_predictive_analysis.ipynb`, `nba_predictive_analysis.ipynb`, `nfl_predictive_analysis.ipynb`) run manually
- **Mobile App Integration**: None

### Problem
- Local automation requires the computer to be on at 6:00 AM EST
- User is not awake to turn on the computer
- Need cloud-based solution that runs independently

## Recommended Architecture: AWS Lambda + S3

### Platform Choice: AWS (Amazon Web Services)

**Why AWS:**
- Fully managed, no server maintenance
- Scales automatically
- Reliable scheduling
- Generous free tier
- Industry standard for serverless applications

### Architecture Diagram

```
6:00 AM EST Daily:
┌─────────────────────────────────────┐
│ AWS EventBridge (Scheduler)        │
│ - Triggers at 6:00 AM EST daily    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Lambda Function 1:                  │
│ collect_yesterday_games.py          │
│ - Calls Odds API                    │
│ - Processes games for NFL, NBA, NCAAM│
│ - Saves to S3 (Excel + Parquet)     │
│ - Updates score storage              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Lambda Function 2:                  │
│ generate_predictions.py             │
│ - Loads data from S3                │
│ - Runs analytics for each sport     │
│ - Generates predictions             │
│ - Saves predictions to S3 (JSON)     │
│ - Updates API/database              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ S3 Bucket: sports-betting-data     │
│ Structure:                          │
│   /data/results/                    │
│     - nfl_season_results.xlsx       │
│     - nba_season_results.xlsx       │
│     - ncaam_season_results.xlsx     │
│     - nfl_season_results.parquet    │
│     - nba_season_results.parquet    │
│     - ncaam_season_results.parquet  │
│   /predictions/                     │
│     - predictions_nfl.json          │
│     - predictions_nba.json          │
│     - predictions_ncaam.json        │
│   /scores/                          │
│     - nfl_scores.json               │
│     - nba_scores.json               │
│     - ncaam_scores.json             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ API Gateway + Lambda:              │
│ REST API Endpoints:                 │
│ - GET /api/predictions/nfl          │
│ - GET /api/predictions/nba          │
│ - GET /api/predictions/ncaam        │
│ - GET /api/predictions/all          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Mobile App                          │
│ - Fetches predictions from API      │
│ - Displays betting opportunities    │
│ - Shows handicap advantages         │
└─────────────────────────────────────┘
```

## Components Breakdown

### 1. AWS EventBridge (Scheduler)
- **Purpose**: Trigger Lambda functions on schedule
- **Schedule**: Daily at 6:00 AM EST (11:00 AM UTC)
- **Cost**: Free (first 14 custom events/month)

### 2. Lambda Function 1: Data Collection
- **Name**: `collect-yesterday-games`
- **Runtime**: Python 3.12
- **Trigger**: EventBridge schedule
- **Functionality**:
  - Calls Odds API for NFL, NBA, NCAAM
  - Collects yesterday's completed games
  - Gets closing spreads
  - Saves to S3 (Excel + Parquet)
  - Updates score storage JSON files
- **Timeout**: 15 minutes (max)
- **Memory**: 512 MB
- **Environment Variables**:
  - `ODDS_API_KEY`: Stored in AWS Secrets Manager
  - `S3_BUCKET_NAME`: Name of S3 bucket

### 3. Lambda Function 2: Generate Predictions
- **Name**: `generate-predictions`
- **Runtime**: Python 3.12
- **Trigger**: EventBridge schedule (runs 30 minutes after collection)
- **Functionality**:
  - Loads historical data from S3
  - Calculates team statistics
  - Calculates handicap statistics:
    - NFL: 5-point handicap
    - NBA: 9-point handicap
    - NCAAM: 10-point handicap
  - Fetches today's games from Odds API
  - Merges today's games with historical stats
  - Identifies betting opportunities
  - Saves predictions to S3 as JSON
- **Timeout**: 10 minutes
- **Memory**: 1024 MB (for pandas operations)
- **Dependencies**: pandas, numpy, requests, openpyxl, pyarrow

### 4. S3 Bucket: Data Storage
- **Name**: `sports-betting-analytics-data` (or similar)
- **Structure**: Organized by type (results, predictions, scores)
- **Versioning**: Enabled (for backup/recovery)
- **Lifecycle**: Keep all versions (or 30-day retention)
- **Access**: Private (only Lambda functions can write, API can read)

### 5. API Gateway + Lambda: Mobile App API
- **Name**: `predictions-api`
- **Runtime**: Python 3.12
- **Trigger**: API Gateway HTTP requests
- **Endpoints**:
  - `GET /api/predictions/nfl` - Returns NFL predictions
  - `GET /api/predictions/nba` - Returns NBA predictions
  - `GET /api/predictions/ncaam` - Returns NCAAM predictions
  - `GET /api/predictions/all` - Returns all predictions
- **Response Format**: JSON
- **CORS**: Enabled for mobile app
- **Authentication**: Optional (API key or JWT for future)

### 6. AWS Secrets Manager
- **Purpose**: Securely store Odds API key
- **Key Name**: `odds-api-key`
- **Access**: Only Lambda functions can read

## Implementation Steps

### Phase 1: AWS Setup
1. Create AWS account
2. Create S3 bucket
3. Upload existing data files to S3
4. Set up AWS Secrets Manager for API key
5. Create IAM roles for Lambda functions

### Phase 2: Convert Scripts to Lambda Functions
1. **Data Collection Lambda**:
   - Convert `collect_yesterday_games.py` to Lambda format
   - Add S3 upload functionality
   - Handle environment variables
   - Add error handling and logging

2. **Predictions Lambda**:
   - Convert notebook logic to Python script
   - Add S3 read/write functionality
   - Handle all three sports (NFL, NBA, NCAAM)
   - Generate JSON output

### Phase 3: Set Up Automation
1. Create EventBridge rules for scheduling
2. Configure Lambda triggers
3. Test end-to-end flow
4. Set up CloudWatch alarms for failures

### Phase 4: Create API
1. Create API Gateway
2. Create Lambda function for API
3. Configure CORS
4. Test endpoints

### Phase 5: Mobile App Integration
1. Build mobile app (React Native/Flutter/PWA)
2. Integrate API calls
3. Display predictions
4. Add refresh/pull-to-refresh functionality

## Technical Details

### Lambda Function Structure

#### Data Collection Lambda
```python
import json
import boto3
from datetime import datetime, timedelta, timezone
# ... other imports

def lambda_handler(event, context):
    """
    Collects yesterday's games and saves to S3
    """
    s3 = boto3.client('s3')
    secrets = boto3.client('secretsmanager')
    
    # Get API key from Secrets Manager
    api_key = secrets.get_secret_value(SecretId='odds-api-key')['SecretString']
    
    # Collect games for each sport
    sports = ['nfl', 'nba', 'ncaam']
    
    for sport in sports:
        # Run collection logic
        # Save to S3
        pass
    
    return {
        'statusCode': 200,
        'body': json.dumps('Collection complete')
    }
```

#### Predictions Lambda
```python
import json
import boto3
import pandas as pd
# ... other imports

def lambda_handler(event, context):
    """
    Generates predictions for today's games
    """
    s3 = boto3.client('s3')
    
    # Load historical data from S3
    # Run analytics
    # Generate predictions
    # Save to S3
    
    return {
        'statusCode': 200,
        'body': json.dumps('Predictions generated')
    }
```

### S3 File Structure
```
sports-betting-analytics-data/
├── data/
│   └── results/
│       ├── nfl_season_results.xlsx
│       ├── nba_season_results.xlsx
│       ├── ncaam_season_results.xlsx
│       ├── nfl_season_results.parquet
│       ├── nba_season_results.parquet
│       └── ncaam_season_results.parquet
├── predictions/
│   ├── predictions_nfl.json
│   ├── predictions_nba.json
│   └── predictions_ncaam.json
└── scores/
    ├── nfl_scores.json
    ├── nba_scores.json
    └── ncaam_scores.json
```

### Prediction JSON Format
```json
{
  "sport": "nfl",
  "generated_at": "2025-01-15T06:35:00Z",
  "games": [
    {
      "game_time_est": "2025-01-15 13:00:00 EST-05:00",
      "away_team": "Team A",
      "home_team": "Team B",
      "current_spread": -7.5,
      "away_cover_pct_handicap_5": 45.2,
      "home_cover_pct_handicap_5": 62.8,
      "handicap_pct_difference": 17.6,
      "opportunity": true
    }
  ],
  "summary": {
    "total_games": 5,
    "opportunities": 2,
    "average_difference": 12.3
  }
}
```

## Cost Estimates

### AWS Free Tier (First 12 Months)
- **Lambda**: 1M requests/month free, 400,000 GB-seconds compute free
- **S3**: 5 GB storage free, 20,000 GET requests free
- **API Gateway**: 1M requests/month free
- **EventBridge**: 14 custom events/month free
- **Secrets Manager**: $0.40/secret/month (not in free tier)

### Estimated Monthly Cost (After Free Tier)
- **Lambda**: ~$0.20 (very low usage)
- **S3**: ~$0.50 (storage + requests)
- **API Gateway**: Free (low traffic)
- **EventBridge**: Free
- **Secrets Manager**: $0.40
- **CloudWatch Logs**: ~$0.50
- **Total**: ~$1.60/month

### Cost Optimization
- Use Lambda Layers for dependencies (reduces package size)
- Enable S3 lifecycle policies (archive old data)
- Use CloudWatch Logs retention (7 days)

## Security Considerations

### API Key Management
- Store Odds API key in AWS Secrets Manager
- Never commit keys to code or Git
- Rotate keys periodically

### S3 Access Control
- Private bucket (no public access)
- IAM roles for Lambda functions
- API Lambda has read-only access

### API Security (Future)
- Add API key authentication
- Rate limiting
- CORS configuration for mobile app

## Monitoring & Alerts

### CloudWatch Metrics
- Lambda execution duration
- Lambda errors
- S3 operations
- API Gateway requests

### CloudWatch Alarms
- Lambda function failures
- API errors
- S3 upload failures

### Notifications
- Email alerts on failures (via SNS)
- Optional: Slack/Discord webhooks

## Migration Strategy

### Step 1: Parallel Run
- Keep local automation running
- Set up cloud automation
- Compare results for accuracy
- Run both for 1-2 weeks

### Step 2: Cloud Primary
- Make cloud the primary system
- Keep local as backup
- Monitor cloud performance

### Step 3: Full Migration
- Disable local automation
- Use cloud exclusively
- Keep local scripts for manual runs if needed

## Future Enhancements

### Phase 2 Features
- Database integration (PostgreSQL/RDS)
- Historical prediction tracking
- Prediction accuracy metrics
- User accounts and preferences

### Phase 3 Features
- Push notifications for new predictions
- Multiple prediction models
- Confidence scores
- Historical performance analytics
- Web dashboard

## Dependencies

### Python Packages (Lambda)
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `requests>=2.31.0`
- `openpyxl>=3.1.0`
- `pyarrow>=14.0.0`
- `boto3>=1.28.0` (AWS SDK)

### Lambda Layers
- Create layer with pandas, numpy, pyarrow (large dependencies)
- Keep Lambda function code small

## Testing Strategy

### Unit Tests
- Test data collection logic
- Test analytics calculations
- Test S3 operations

### Integration Tests
- Test Lambda functions end-to-end
- Test API endpoints
- Test mobile app integration

### Monitoring
- CloudWatch dashboards
- Error tracking
- Performance metrics

## Rollback Plan

### If Cloud Fails
- Keep local scripts as backup
- Can manually run collection
- Can manually generate predictions
- Gradual migration allows easy rollback

## Documentation Updates

### Code Documentation
- Inline comments in Lambda functions
- README for each Lambda function
- API documentation (OpenAPI/Swagger)

### User Documentation
- Mobile app user guide
- How predictions are generated
- Understanding handicap advantages

## Success Criteria

### Phase 1 Complete When:
- ✅ Data collection runs automatically in cloud
- ✅ Predictions generated automatically
- ✅ API serves predictions to mobile app
- ✅ Mobile app displays predictions
- ✅ System runs without manual intervention

### Quality Metrics:
- Data collection success rate: >99%
- Prediction generation time: <5 minutes
- API response time: <500ms
- Mobile app load time: <2 seconds

## Next Steps

1. **Review and approve this plan**
2. **Set up AWS account** (if not already)
3. **Create S3 bucket and upload existing data**
4. **Convert collection script to Lambda function**
5. **Convert analytics to Lambda function**
6. **Set up EventBridge scheduling**
7. **Create API Gateway and Lambda**
8. **Build mobile app**
9. **Test end-to-end**
10. **Deploy and monitor**

## Notes

- All times are in EST (Eastern Standard Time)
- Lambda functions use UTC internally, convert as needed
- Keep local scripts as backup during migration
- Document any deviations from this plan
- Update this document as implementation progresses

---

**Last Updated**: 2025-01-15
**Status**: Planning Phase
**Next Review**: After Phase 1 implementation

