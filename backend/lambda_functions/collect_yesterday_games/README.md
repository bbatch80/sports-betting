# Lambda Function: Collect Yesterday's Games

## Purpose

This Lambda function automatically collects completed games from yesterday, gets their closing spreads, and updates the data files in S3.

## How It Works

1. **Triggered Daily**: EventBridge schedule runs this at 6:00 AM EST
2. **Gets API Key**: Retrieves Odds API key from AWS Secrets Manager
3. **Reads Existing Data**: Loads current Excel files from S3
4. **Collects New Games**: Fetches yesterday's completed games from Odds API
5. **Gets Closing Spreads**: Retrieves closing spread for each game (5 minutes before game start)
6. **Merges Data**: Combines new games with existing data (preserves manual entries)
7. **Saves to S3**: Updates Excel and Parquet files in S3

## What It Processes

- **Sports**: NFL, NBA, NCAAM
- **Data Collected**:
  - Game date (EST)
  - Home team
  - Away team
  - Closing spread (DraftKings)
  - Final scores (if available)
  - Spread result difference

## S3 Structure

Files are stored at:
- `s3://sports-betting-analytics-data/data/results/{sport}_season_results.xlsx`
- `s3://sports-betting-analytics-data/data/results/{sport}_season_results.parquet`
- `s3://sports-betting-analytics-data/scores/{sport}_scores.json`

## Lambda Configuration

- **Runtime**: Python 3.12
- **Timeout**: 15 minutes (max)
- **Memory**: 512 MB
- **IAM Role**: `SportsBettingLambdaRole`
- **Environment Variables**: None (uses Secrets Manager)

## Dependencies

See `requirements.txt` for Python packages needed.

## Deployment

This function will be deployed in Phase 3 when we set up the automation.



