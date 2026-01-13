# Sports Betting Analytics

Monorepo containing the backend (Python/AWS) and frontend (React Native/Expo) for sports betting predictions.

---

## CRITICAL: Strategy Performance Tracking

**This is a core requirement of the project.** Every betting strategy must have its performance tracked over time from season start to current date.

### Requirements

1. **Track Every Prediction**: When a strategy identifies an opportunity, record:
   - Date, sport, strategy name
   - Teams involved, spread, handicap applied
   - The bet recommendation (which team to bet on)
   - Confidence/strength metrics

2. **Match Predictions to Outcomes**: After games complete, determine:
   - Did the recommended bet win or lose?
   - By how many points?
   - Store win/loss result linked to the original prediction

3. **Cumulative Performance Visualization**: Display line graphs showing:
   - X-axis: Days of the season (game dates)
   - Y-axis: Cumulative win rate OR cumulative units won
   - One line per strategy for comparison
   - Similar to "Cumulative Percentage Over Time" graphs in analysis notebooks

4. **Key Metrics Per Strategy**:
   - Total predictions made
   - Wins / Losses / Win Rate
   - ROI (units won/lost assuming -110 odds)
   - Performance by confidence level
   - Streak tracking (current win/loss streak)

### Current Strategies to Track

| Strategy | Handicap | Logic |
|----------|----------|-------|
| `home_focus` | Variable | Home team has higher handicap coverage % |
| `away_focus` | Variable | Away team has higher handicap coverage % |
| `coverage_based` | 0 | One team has ≥10% better spread coverage |
| `elite_team` | 0 | Top 25% team by win% in good recent form |
| `hot_vs_cold` | 11 pts | Hot team (60%+ last 5) vs cold team (40%-) |
| `opponent_perfect_form` | 11 pts | Bet against teams with perfect 5/5 form |
| `common_opponent` | 0 | NCAAM only - based on shared opponents |

### Data Storage (S3)

```
s3://sports-betting-analytics-data/
├── predictions/                    # Daily predictions (existing)
│   └── predictions_{sport}_{date}.json
├── strategy_tracking/              # NEW: Strategy results
│   ├── results_{sport}_{date}.json # Daily matched results
│   └── performance/                # Aggregated stats
│       └── {sport}_strategy_performance.json
```

### Implementation Status

- [ ] Lambda function to match predictions to outcomes (runs after games complete)
- [ ] S3 storage for strategy results
- [ ] Aggregation logic for cumulative performance
- [ ] API endpoint for strategy performance data
- [ ] Frontend: Strategy Performance screen with line graphs
- [ ] Frontend: Individual strategy detail views

**When implementing ANY new strategy, always include the tracking infrastructure.**

---

## Claude Behavior Guidelines

### Research Before Action
Do not jump into implementation or change files unless clearly instructed to make changes. When the user's intent is ambiguous, default to providing information, doing research, and providing recommendations rather than taking action. Only proceed with edits, modifications, or implementations when the user explicitly requests them.

### Never Speculate About Unread Code
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.

### Parallel Tool Execution
If you intend to call multiple tools and there are no dependencies between the tool calls, make all of the independent tool calls in parallel. Prioritize calling tools simultaneously whenever the actions can be done in parallel rather than sequentially. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. Maximize use of parallel tool calls where possible to increase speed and efficiency. However, if some tool calls depend on previous calls to inform dependent values like the parameters, do not call these tools in parallel and instead call them sequentially. Never use placeholders or guess missing parameters in tool calls.

## Remind about Github
Occasionally remind the user or ask whether he/she wants to commit new changes to github

## Sensitive Data Security
Prior to pushing any new code to github, ensure that no sensitive data is being included

---

## Project Structure

```
sports-betting/
├── backend/                    # Python backend
│   ├── src/
│   │   ├── config.py           # Sport configs, DatabaseConfig
│   │   ├── database.py         # DB facade (backward compat)
│   │   ├── odds_api_client.py  # The Odds API client
│   │   ├── db/                 # SQLAlchemy layer
│   │   │   ├── engine.py       # Connection pooling
│   │   │   ├── models.py       # Table definitions
│   │   │   └── repository.py   # Query methods
│   │   └── analysis/           # Prediction engine
│   │       ├── metrics.py      # Team ratings
│   │       ├── insights.py     # Streak patterns
│   │       └── network_ratings.py
│   ├── dashboard/
│   │   └── app.py              # Streamlit analytics dashboard
│   ├── lambda_functions/       # AWS Lambda (5 functions)
│   ├── scripts/                # Deployment & operations
│   ├── migrations/             # DB migrations
│   ├── notebooks/              # Jupyter analysis
│   └── data/                   # Local SQLite + Excel
│
├── frontend/                   # React Native/Expo mobile app
│   └── src/
│       ├── screens/            # HomeScreen, SportTabScreen, etc.
│       ├── components/         # PredictionCard, StrategyOpportunityCard
│       ├── services/           # apiService.js
│       └── constants/          # api.js
│
└── CLAUDE.md
```

---

## Architecture Overview

### System Diagram
```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                             │
├─────────────────────────────────────────────────────────────────────┤
│  Mobile App (React Native)     │  Analytics Dashboard (Streamlit)  │
│  - Predictions display         │  - Strategy Overview              │
│  - Strategy selection          │  - Today's Picks                  │
│  - Performance charts          │  - Power Rankings                 │
│  localhost:19000 (Expo)        │  - Backtest Strategies            │
│                                │  localhost:8501                   │
└───────────────┬────────────────┴──────────────┬────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│       AWS API Gateway         │   │    Local SQLite Database      │
│  GET /predictions/{sport}     │   │    backend/data/analytics.db  │
│  GET /results/{sport}         │   │                               │
│  GET /strategy-performance    │   │         OR (production)       │
└───────────────┬───────────────┘   │                               │
                │                   │    AWS RDS PostgreSQL         │
                ▼                   │    sports-betting-analytics   │
┌───────────────────────────────┐   └───────────────────────────────┘
│      AWS Lambda Functions     │
├───────────────────────────────┤
│ collect_yesterday_games (6AM) │──→ The Odds API
│ generate_predictions (6:30AM) │
│ evaluate_strategy_results(3AM)│
│ predictions_api               │
│ results_api                   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│          AWS S3               │
│ predictions/*.json            │
│ data/results/*.xlsx           │
│ strategy_tracking/*.json      │
└───────────────────────────────┘
```

### Database Schema
```
┌─────────────────────────────────────────────────────────────────┐
│                         games                                   │
├─────────────────────────────────────────────────────────────────┤
│ id              INTEGER PRIMARY KEY                             │
│ sport           VARCHAR(50)   -- NFL, NBA, NCAAM                │
│ game_date       DATE                                            │
│ home_team       VARCHAR(100)                                    │
│ away_team       VARCHAR(100)                                    │
│ closing_spread  FLOAT         -- negative = home favored        │
│ home_score      INTEGER                                         │
│ away_score      INTEGER                                         │
│ spread_result   FLOAT         -- margin vs spread               │
├─────────────────────────────────────────────────────────────────┤
│ UNIQUE (sport, game_date, home_team, away_team)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    historical_ratings                           │
├─────────────────────────────────────────────────────────────────┤
│ id              INTEGER PRIMARY KEY                             │
│ sport           VARCHAR(50)                                     │
│ snapshot_date   DATE          -- when calculated                │
│ team            VARCHAR(100)                                    │
│ win_rating      FLOAT         -- strength by wins               │
│ ats_rating      FLOAT         -- strength by ATS                │
│ market_gap      FLOAT         -- perception gap                 │
│ games_analyzed  INTEGER                                         │
│ win_rank        INTEGER                                         │
│ ats_rank        INTEGER                                         │
├─────────────────────────────────────────────────────────────────┤
│ UNIQUE (sport, snapshot_date, team)                             │
└─────────────────────────────────────────────────────────────────┘
```

### Lambda Schedule (EST)
| Time | Function | Purpose |
|------|----------|---------|
| 3:00 AM | evaluate_strategy_results | Match predictions to outcomes |
| 6:00 AM | collect_yesterday_games | Get scores + closing spreads |
| 6:30 AM | generate_predictions | Create today's opportunities |
| On-demand | predictions_api | Serve predictions to apps |
| On-demand | results_api | Serve historical results |

### Streamlit Dashboard Pages
Run with: `cd backend && streamlit run dashboard/app.py`

| Page | Purpose |
|------|---------|
| Strategy Overview | All strategies at a glance |
| Today's Picks | Live recommendations |
| Power Rankings | Team strength ratings |
| Backtest Strategies | Historical performance |
| Macro Trends | League-wide ATS analysis |
| Micro Analysis | Team-specific deep dive |
| Streak Analysis | Streak patterns |
| Exploration | Ad-hoc queries |

## Tech Stack

### Backend
- **Language**: Python 3.12+
- **Database**: SQLite (dev) / PostgreSQL (prod via AWS RDS)
- **ORM**: SQLAlchemy Core
- **Cloud**: AWS (S3, Lambda, RDS, Secrets Manager, EventBridge, API Gateway)
- **Data**: pandas, openpyxl, pyarrow
- **API**: The Odds API
- **Dashboard**: Streamlit (localhost:8501)

### Frontend
- **Framework**: React Native with Expo
- **Navigation**: React Navigation (stack, bottom tabs)
- **HTTP Client**: Axios
- **Charts**: react-native-chart-kit
- **Platform**: iOS, Android, Web

## Common Commands

### Backend
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run dashboard/app.py    # Opens http://localhost:8501

# Run daily collection locally
python3 scripts/collect_yesterday_games.py

# Deploy Lambda functions
python3 scripts/deploy_lambda_functions.py

# Deploy API Gateway
python3 scripts/deploy_api_gateway.py

# AWS RDS setup
python3 scripts/aws_rds_setup.py
python3 scripts/aws_update_lambda_env.py
```

### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android

# Run on web
npm run web
```

## API Integration

The frontend consumes the backend's REST API:

| Endpoint | Purpose |
|----------|---------|
| `GET /predictions/{sport}` | Get predictions for a sport |
| `GET /results/{sport}` | Get historical results |

API base URL is configured in `frontend/src/constants/api.js`

## Sports Covered
- **NFL** - Pro football
- **NBA** - Pro basketball
- **NCAAM** - College basketball

## Environment Variables

### Backend
- `ODDS_API_KEY` - The Odds API key (or use AWS Secrets Manager)
- AWS credentials via `~/.aws/credentials`

### Frontend
- API URL configured in `src/constants/api.js`

## AWS Resources
- **S3 Bucket**: `sports-betting-analytics-data`
- **RDS**: `sports-betting-analytics` (PostgreSQL 15, db.t3.micro)
- **Secrets**: `odds-api-key`, `sports-betting-db-credentials`
- **Region**: `us-east-1`

## Code Style

### Python (Backend)
- snake_case for functions and variables
- PascalCase for classes
- Type hints on function signatures
- Docstrings for public functions

### JavaScript (Frontend)
- camelCase for functions and variables
- PascalCase for components
- Functional components with hooks
- Async/await for API calls

## Workflow

Typical feature development:
1. Add prediction logic in `backend/`
2. Deploy Lambda / API changes
3. Update `frontend/src/services/apiService.js` if needed
4. Add new screen or component in `frontend/src/`
5. Test end-to-end
