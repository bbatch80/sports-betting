# Lambda Function: Results API

## Purpose

This Lambda function serves game results from S3 via API Gateway for mobile app consumption.

## Endpoints

- `GET /api/results/nfl` - Returns NFL game results
- `GET /api/results/nba` - Returns NBA game results
- `GET /api/results/ncaam` - Returns NCAAM game results
- `GET /api/results/all` - Returns all results

## Query Parameters

- `start_date` (optional): Filter results from this date (YYYY-MM-DD format, inclusive)
- `end_date` (optional): Filter results to this date (YYYY-MM-DD format, inclusive)

Example: `/api/results/nba?start_date=2025-01-01&end_date=2025-01-31`

## Response Format

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
    },
    ...
  ]
}
```

## Data Source

Results are read from S3:
- Primary: `s3://sports-betting-analytics-data/data/results/{sport}_season_results.parquet`
- Fallback: `s3://sports-betting-analytics-data/data/results/{sport}_season_results.xlsx`

## CORS

CORS is enabled for all origins to allow mobile app access.

## Lambda Configuration

- **Runtime**: Python 3.12
- **Timeout**: 30 seconds
- **Memory**: 512 MB (more memory for pandas operations)
- **IAM Role**: `SportsBettingLambdaRole` (needs S3 read permissions)

