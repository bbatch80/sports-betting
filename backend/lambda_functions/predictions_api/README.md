# Lambda Function: Predictions API

## Purpose

This Lambda function serves predictions from S3 via API Gateway for mobile app consumption.

## Endpoints

- `GET /api/predictions/nfl` - Returns NFL predictions
- `GET /api/predictions/nba` - Returns NBA predictions
- `GET /api/predictions/ncaam` - Returns NCAAM predictions
- `GET /api/predictions/all` - Returns all predictions

## Response Format

```json
{
  "sport": "nba",
  "sport_name": "NBA",
  "handicap_points": 9,
  "generated_at": "2025-12-12T19:28:22Z",
  "games": [...],
  "opportunities": [...],
  "summary": {...}
}
```

## CORS

CORS is enabled for all origins to allow mobile app access.

## Lambda Configuration

- **Runtime**: Python 3.12
- **Timeout**: 30 seconds
- **Memory**: 256 MB
- **IAM Role**: `SportsBettingLambdaRole` (needs S3 read permissions)



