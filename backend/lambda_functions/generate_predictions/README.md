# Lambda Function: Generate Predictions

## Purpose

This Lambda function generates betting opportunity predictions for today's games by analyzing historical team performance.

## How It Works

1. **Triggered Daily**: EventBridge schedule runs this at 6:30 AM EST (after data collection)
2. **Loads Historical Data**: Reads Excel files from S3 for each sport
3. **Calculates Team Statistics**:
   - Standard spread coverage percentage
   - Handicap-adjusted coverage percentage:
     - NFL: 5-point handicap
     - NBA: 9-point handicap
     - NCAAM: 10-point handicap
4. **Fetches Today's Games**: Gets today's scheduled games from Odds API
5. **Generates Predictions**: Identifies betting opportunities based on sport-specific focus:
   - NFL & NCAAM: Away teams with better handicap coverage
   - NBA: Home teams with better handicap coverage
6. **Saves to S3**: Writes predictions as JSON files for mobile app to read

## What It Generates

For each sport, creates a JSON file with:
- All today's games with team statistics
- Betting opportunities (sport-specific: away teams for NFL/NCAAM, home teams for NBA)
- Summary statistics (total games, opportunities, average difference)

## S3 Output

Predictions are saved to:
- `s3://sports-betting-analytics-data/predictions/predictions_nfl.json`
- `s3://sports-betting-analytics-data/predictions/predictions_nba.json`
- `s3://sports-betting-analytics-data/predictions/predictions_ncaam.json`

## Lambda Configuration

- **Runtime**: Python 3.12
- **Timeout**: 10 minutes
- **Memory**: 1024 MB (for pandas operations)
- **IAM Role**: `SportsBettingLambdaRole`

## JSON Output Format

```json
{
  "sport": "nfl",
  "sport_name": "NFL",
  "handicap_points": 5,
  "generated_at": "2025-01-15T06:35:00Z",
  "games": [...],
  "opportunities": [...],
  "summary": {
    "total_games": 5,
    "opportunities": 2,
    "average_difference": 12.3
  }
}
```



