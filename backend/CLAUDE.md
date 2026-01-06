# Sports Betting Analytics Backend

## Claude Behavior Guidelines

### Research Before Action
Do not jump into implementation or change files unless clearly instructed to make changes. When the user's intent is ambiguous, default to providing information, doing research, and providing recommendations rather than taking action. Only proceed with edits, modifications, or implementations when the user explicitly requests them.

### Never Speculate About Unread Code
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.

### Parallel Tool Execution
If you intend to call multiple tools and there are no dependencies between the tool calls, make all of the independent tool calls in parallel. Prioritize calling tools simultaneously whenever the actions can be done in parallel rather than sequentially. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. Maximize use of parallel tool calls where possible to increase speed and efficiency. However, if some tool calls depend on previous calls to inform dependent values like the parameters, do not call these tools in parallel and instead call them sequentially. Never use placeholders or guess missing parameters in tool calls.

---

## Project Overview
Automated system for collecting historical betting line data from The Odds API, storing results in AWS S3, and generating spread predictions for NFL, NBA, and NCAAM games.

## Tech Stack
- **Language**: Python 3.12+
- **Cloud**: AWS (S3, Lambda, Secrets Manager, EventBridge, API Gateway)
- **Data**: pandas, openpyxl, pyarrow
- **API**: The Odds API (https://the-odds-api.com)
- **Local Automation**: macOS launchd

## Directory Structure
```
├── src/                  # Core modules
│   ├── config.py         # Sport definitions, API settings
│   ├── odds_api_client.py # The Odds API client
│   └── score_storage.py  # Local score caching
│
├── scripts/              # Local scripts
│   ├── collect_*.py      # Data collection scripts
│   ├── deploy_*.py       # AWS deployment scripts
│   ├── aws_setup.py      # AWS infrastructure setup
│   └── sync_excel_from_s3.py # S3 sync utility
│
├── lambda_functions/     # AWS Lambda functions
│   ├── collect_yesterday_games/  # Daily collection (6 AM EST)
│   ├── generate_predictions/     # Prediction generation
│   ├── predictions_api/          # GET predictions endpoint
│   └── results_api/              # GET results endpoint
│
├── notebooks/            # Jupyter analysis notebooks
│   ├── *_analysis.ipynb         # Historical analysis
│   └── *_predictive_analysis.ipynb  # Prediction models
│
├── data/                 # Local data storage
│   ├── results/          # Excel and Parquet files
│   └── *.json            # Score cache files
│
├── automation/           # macOS launchd automation
├── docs/                 # Documentation
└── logs/                 # Log files
```

## Common Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run daily collection locally
python3 scripts/collect_yesterday_games.py

# Test predictions locally
python3 scripts/test_predictions_local.py

# Sync Excel files from S3
python3 scripts/sync_excel_from_s3.py
```

### AWS Deployment
```bash
# Initial AWS setup
python3 scripts/aws_setup.py

# Verify AWS setup
python3 scripts/aws_verify_setup.py

# Deploy Lambda functions
python3 scripts/deploy_lambda_functions.py

# Deploy API Gateway
python3 scripts/deploy_api_gateway.py
```

### Lambda Package Build
```bash
# Build Lambda deployment package (from lambda function directory)
pip install -r requirements.txt -t package/
cd package && zip -r ../function.zip . && cd ..
zip function.zip lambda_function.py
```

## Environment Variables
- `ODDS_API_KEY` - The Odds API key (local development)
- AWS credentials via `~/.aws/credentials` or environment

## AWS Resources
- **S3 Bucket**: `sports-betting-analytics-data`
- **Secret**: `odds-api-key` (Secrets Manager)
- **Region**: `us-east-1`
- **Lambda Role**: `SportsBettingLambdaRole`

## Sports Configuration
| Sport | API Key | Data File |
|-------|---------|-----------|
| NFL | `americanfootball_nfl` | `nfl_season_results.xlsx` |
| NBA | `basketball_nba` | `nba_season_results.xlsx` |
| NCAAM | `basketball_ncaab` | `ncaam_season_results.xlsx` |

## Code Style
- Use snake_case for functions and variables
- Use PascalCase for classes
- Type hints on function signatures
- Docstrings for public functions and classes
- Logging via Python `logging` module (CloudWatch compatible)

## Data Formats

### Results DataFrame Columns
- `game_date` - Date of the game
- `home_team` / `away_team` - Team names
- `closing_spread` - Closing spread from DraftKings
- `home_score` / `away_score` - Final scores
- `spread_result_difference` - Actual result vs spread

### API Response Handling
- Always check for rate limits (1 second delay between requests)
- Handle `OddsAPIError` exceptions
- Log API quota usage from response headers

## Testing
```bash
# Test collection completeness
python3 scripts/test_collection_completeness.py

# Test AWS connections
python3 lambda_functions/test_aws_connections.py

# Quick collection test
python3 lambda_functions/test_collect_quick.py
```

## Important Notes
- Lambda packages (`package/` directories and `.zip` files) are gitignored - rebuild from `requirements.txt`
- Daily collection runs at 6:00 AM EST via EventBridge
- API rate limit: 1 second between requests
- DraftKings is the primary bookmaker for spread data
- Excel files are the source of truth; Parquet is for efficient storage

## Notebooks
Analysis notebooks require additional dependencies:
```bash
pip install -r notebooks/requirements_notebooks.txt
```
