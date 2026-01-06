# Sports Betting Analytics

Full-stack application for collecting sports betting data and generating spread predictions.

## Overview

- **Backend**: Python + AWS Lambda for data collection and prediction APIs
- **Frontend**: React Native/Expo mobile app for viewing predictions

## Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
export ODDS_API_KEY=your_api_key
python3 scripts/collect_yesterday_games.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## Documentation

- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
- [AWS Setup Guide](backend/docs/AWS_SETUP_GUIDE.md)

## Sports Covered

- NFL (Pro Football)
- NBA (Pro Basketball)
- NCAAM (College Basketball)
