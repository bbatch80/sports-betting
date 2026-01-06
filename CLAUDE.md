# Sports Betting Analytics

Monorepo containing the backend (Python/AWS) and frontend (React Native/Expo) for sports betting predictions.

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
├── backend/              # Python backend (AWS Lambda, data collection, predictions)
│   ├── src/              # Core modules (API client, config)
│   ├── scripts/          # Collection and deployment scripts
│   ├── lambda_functions/ # AWS Lambda functions
│   ├── notebooks/        # Jupyter analysis notebooks
│   ├── data/             # Local data storage
│   ├── automation/       # macOS launchd automation
│   └── docs/             # Backend documentation
│
├── frontend/             # React Native/Expo mobile app
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── screens/      # App screens
│   │   ├── services/     # API service layer
│   │   ├── constants/    # API endpoints, config
│   │   └── utils/        # Utility functions
│   └── assets/           # Images, fonts
│
└── CLAUDE.md             # This file
```

## Tech Stack

### Backend
- **Language**: Python 3.12+
- **Cloud**: AWS (S3, Lambda, Secrets Manager, EventBridge, API Gateway)
- **Data**: pandas, openpyxl, pyarrow
- **API**: The Odds API

### Frontend
- **Framework**: React Native with Expo
- **Navigation**: React Navigation (stack, bottom tabs)
- **HTTP Client**: Axios
- **Platform**: iOS, Android, Web

## Common Commands

### Backend
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run daily collection locally
python3 scripts/collect_yesterday_games.py

# Deploy Lambda functions
python3 scripts/deploy_lambda_functions.py

# Deploy API Gateway
python3 scripts/deploy_api_gateway.py
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
- **Secret**: `odds-api-key` (Secrets Manager)
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
