# Collect Today's Games Lambda

Fetches today's scheduled games and current spreads from The Odds API and stores them in PostgreSQL.

## Schedule

Runs daily at **6:30 AM EST** via EventBridge, after `collect_yesterday_games` completes.

## What It Does

1. Calls The Odds API `/v4/sports/{sport}/odds` endpoint for each sport (NFL, NBA, NCAAM)
2. Filters games to today's date (EST timezone)
3. Extracts current spreads (prefers DraftKings)
4. Upserts games into `todays_games` table
5. Cleans up games older than 7 days

## Database Table

```sql
todays_games (
    id, sport, game_date, commence_time,
    home_team, away_team, spread, spread_source,
    created_at, updated_at
)
```

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string (or uses AWS Secrets Manager)

## Local Testing

```bash
cd backend
export DATABASE_URL='postgresql://...'
python3 -c "
from lambda_functions.collect_todays_games.lambda_function import lambda_handler
print(lambda_handler({}, None))
"
```

## Deployment

Included in `scripts/deploy_lambda_functions.py` deployment script.
