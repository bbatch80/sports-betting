# Sports Betting Backend Review

## Overview

Your sports betting backend is a production-ready AWS-based system that collects, processes, and serves sports betting data. Here's a comprehensive breakdown of how it works and how to integrate with it.

---

## Architecture Components

### 1. Data Collection Layer
- **Location**: `sports-betting-backend/scripts/`
- **Purpose**: Automated daily collection of betting lines from The Odds API
- **Key Scripts**:
  - `collect_yesterday_games.py` - Daily automated collection (runs at 6 AM EST)
  - `collect_nfl_season.py` - Full NFL season collection
  - `collect_nba_season.py` - Full NBA season collection
  - `collect_ncaam_season.py` - Full NCAAM season collection

### 2. Data Storage
- **S3 Bucket**: `sports-betting-analytics-data`
- **Structure**:
  ```
  s3://sports-betting-analytics-data/
  ├── data/
  │   └── results/
  │       ├── nfl_season_results.parquet
  │       ├── nfl_season_results.xlsx
  │       ├── nba_season_results.parquet
  │       ├── nba_season_results.xlsx
  │       ├── ncaam_season_results.parquet
  │       └── ncaam_season_results.xlsx
  ├── predictions/
  │   ├── predictions_nfl.json
  │   ├── predictions_nba.json
  │   ├── predictions_ncaam.json
  │   └── predictions_{sport}_{date}.json (date-specific)
  └── scores/
      ├── nfl_scores.json
      ├── nba_scores.json
      └── ncaam_scores.json
  ```

### 3. Lambda Functions

#### Predictions API Lambda
- **Function Name**: `predictions-api`
- **Location**: `lambda_functions/predictions_api/`
- **Purpose**: Serves predictions from S3
- **Configuration**:
  - Runtime: Python 3.12
  - Memory: 256 MB
  - Timeout: 30 seconds
  - Handler: `lambda_function.lambda_handler`

#### Results API Lambda
- **Function Name**: `results-api`
- **Location**: `lambda_functions/results_api/`
- **Purpose**: Serves historical game results from S3
- **Configuration**:
  - Runtime: Python 3.12
  - Memory: 512 MB (for pandas operations)
  - Timeout: 30 seconds
  - Handler: `lambda_function.lambda_handler`
  - Dependencies: pandas, numpy, openpyxl, pyarrow

### 4. API Gateway
- **API Name**: `sports-betting-predictions-api`
- **Stage**: `prod`
- **Region**: `us-east-1`
- **URL Format**: `https://{api-id}.execute-api.us-east-1.amazonaws.com/prod`

---

## API Endpoints

### Predictions Endpoints

#### Get Predictions by Sport
```
GET /api/predictions/{sport}
```

**Path Parameters:**
- `sport`: `nfl`, `nba`, or `ncaam`

**Query Parameters:**
- `date` (optional): `YYYY-MM-DD` format for date-specific predictions

**Examples:**
```bash
# Today's NBA predictions
GET /api/predictions/nba

# NFL predictions for specific date
GET /api/predictions/nfl?date=2025-01-28

# All sports predictions
GET /api/predictions/all
```

**Response Structure:**
```json
{
  "sport": "nba",
  "sport_name": "NBA",
  "handicap_points": 9,
  "generated_at": "2025-12-12T19:28:22Z",
  "games": [
    {
      "game_date": "2025-01-28",
      "home_team": "Lakers",
      "away_team": "Warriors",
      "closing_spread": -3.5,
      "predicted_advantage": 2.1,
      "coverage_percentage": 65.5
    }
  ],
  "opportunities": [
    {
      "game_date": "2025-01-28",
      "team": "Lakers",
      "opponent": "Warriors",
      "spread": -3.5,
      "advantage": 2.1,
      "confidence": "high"
    }
  ],
  "summary": {
    "total_games": 10,
    "opportunities": 5,
    "high_confidence": 3
  }
}
```

### Results Endpoints

#### Get Results by Sport
```
GET /api/results/{sport}
```

**Path Parameters:**
- `sport`: `nfl`, `nba`, or `ncaam`

**Query Parameters:**
- `start_date` (optional): Filter from date (YYYY-MM-DD, inclusive)
- `end_date` (optional): Filter to date (YYYY-MM-DD, inclusive)

**Examples:**
```bash
# All NBA results
GET /api/results/nba

# NFL results for January 2025
GET /api/results/nfl?start_date=2025-01-01&end_date=2025-01-31

# All sports results
GET /api/results/all
```

**Response Structure:**
```json
{
  "sport": "nba",
  "sport_name": "NBA",
  "total_games": 150,
  "last_updated": "2025-01-28T12:00:00",
  "games": [
    {
      "game_date": "2025-01-27",
      "home_team": "Lakers",
      "away_team": "Warriors",
      "closing_spread": -3.5,
      "home_score": 110,
      "away_score": 105,
      "spread_result_difference": 1.5
    }
  ]
}
```

---

## Data Models

### Prediction Data Model
- **sport**: Sport identifier (nfl, nba, ncaam)
- **sport_name**: Human-readable sport name
- **handicap_points**: Handicap threshold used for predictions
- **generated_at**: ISO timestamp of when predictions were generated
- **games**: Array of all games with predictions
- **opportunities**: Array of betting opportunities (filtered games with advantages)
- **summary**: Summary statistics

### Game Result Data Model
- **game_date**: Date of the game (YYYY-MM-DD)
- **home_team**: Home team name
- **away_team**: Away team name
- **closing_spread**: Closing spread from DraftKings
- **home_score**: Final home team score (if available)
- **away_score**: Final away team score (if available)
- **spread_result_difference**: Difference between actual result and spread

---

## CORS Configuration

All API endpoints support CORS with:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token`

This means your mobile app can make direct requests without CORS issues.

---

## Authentication

Currently, the APIs are **public** (no authentication required). This is fine for development, but consider adding authentication for production:
- API Keys
- AWS Cognito
- JWT tokens

---

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": "Invalid sport",
  "message": "Valid sports are: nfl, nba, ncaam",
  "received": "invalid_sport"
}
```

**404 Not Found:**
```json
{
  "sport": "nba",
  "error": "No predictions available",
  "message": "Predictions for NBA are not yet available"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Internal server error",
  "message": "Error details..."
}
```

---

## Rate Limiting

Currently, there's no explicit rate limiting, but:
- API Gateway has default limits (10,000 requests/second per region)
- Lambda has concurrency limits
- S3 has request rate limits

For production, consider implementing:
- API Gateway throttling
- Request rate limiting per user/IP
- Caching strategies

---

## Deployment

### Deploying the APIs

1. **Deploy Lambda Functions:**
   ```bash
   cd sports-betting-backend
   python3 scripts/deploy_lambda_functions.py
   ```

2. **Deploy API Gateway:**
   ```bash
   python3 scripts/deploy_api_gateway.py
   ```

The deployment script will output the API Gateway URL.

### Updating APIs

When you update Lambda functions:
1. Update the code in `lambda_functions/{api_name}/lambda_function.py`
2. Run the deployment script
3. API Gateway will automatically use the new Lambda version

---

## Monitoring & Logging

### CloudWatch Logs
- Lambda functions log to CloudWatch
- Log group: `/aws/lambda/{function-name}`
- View logs in AWS Console or via CLI

### Metrics
- API Gateway metrics: Request count, latency, errors
- Lambda metrics: Invocations, duration, errors
- S3 metrics: Request count, data transfer

---

## Cost Considerations

### Free Tier (First Year)
- Lambda: 1M requests/month free
- API Gateway: 1M requests/month free
- S3: 5GB storage, 20K GET requests/month free

### Estimated Costs (After Free Tier)
- Lambda: ~$0.20 per 1M requests
- API Gateway: ~$3.50 per 1M requests
- S3: ~$0.023 per GB storage
- **Total**: ~$4-5/month for moderate usage

---

## Best Practices for Frontend Integration

### 1. API Client Setup
- Use a base URL constant
- Implement request timeout (10-15 seconds)
- Add retry logic for failed requests
- Handle network errors gracefully

### 2. Caching Strategy
- Cache predictions for 5-10 minutes (they update daily)
- Cache results for longer (historical data doesn't change)
- Use React Query or SWR for automatic caching

### 3. Error Handling
- Show user-friendly error messages
- Implement retry mechanisms
- Handle offline scenarios
- Log errors for debugging

### 4. Performance
- Lazy load data
- Implement pagination for large result sets
- Use FlatList for efficient rendering
- Optimize images and assets

### 5. User Experience
- Show loading states
- Implement pull-to-refresh
- Add skeleton screens
- Provide feedback for actions

---

## Testing the APIs

### Using curl
```bash
# Get NBA predictions
curl https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/api/predictions/nba

# Get NFL results for January
curl "https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/api/results/nfl?start_date=2025-01-01&end_date=2025-01-31"
```

### Using Postman
1. Create a new request
2. Set method to GET
3. Enter API Gateway URL
4. Add path (e.g., `/api/predictions/nba`)
5. Send request

### Using JavaScript/React Native
```javascript
const response = await fetch('https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/api/predictions/nba');
const data = await response.json();
console.log(data);
```

---

## Next Steps for Frontend Development

1. **Get API Gateway URL**
   - Check AWS Console
   - Or run deployment script

2. **Set up Expo project**
   - Follow the EXPO_APP_GUIDE.md

3. **Implement API service layer**
   - Create axios/fetch wrapper
   - Add error handling
   - Implement caching

4. **Build UI components**
   - Prediction cards
   - Results list
   - Loading states

5. **Add navigation**
   - Tab navigation for sports
   - Screen navigation for details

6. **Test thoroughly**
   - Test on real devices
   - Test error scenarios
   - Test offline behavior

---

## Support & Resources

- **Backend Code**: `/Users/robertbatchelor/Documents/Projects/sports-betting-backend/`
- **API Documentation**: See README files in `lambda_functions/` directories
- **AWS Documentation**: 
  - [API Gateway](https://docs.aws.amazon.com/apigateway/)
  - [Lambda](https://docs.aws.amazon.com/lambda/)
  - [S3](https://docs.aws.amazon.com/s3/)

---

**Last Updated**: 2025-01-28

