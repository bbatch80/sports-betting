# Sports Betting Analytics

Automated collection system for historical betting line data from The Odds API.

## Directory Structure

```
.
├── scripts/              # Collection scripts
│   ├── collect_yesterday_games.py    # Daily automated collection
│   ├── collect_nfl_season.py         # NFL season collection
│   ├── collect_nba_season.py         # NBA season collection
│   ├── collect_ncaam_season.py       # NCAAM season collection
│   └── test_collection_completeness.py  # Verification script
│
├── src/                  # Core modules
│   ├── odds_api_client.py    # API client
│   ├── config.py              # Configuration
│   └── score_storage.py       # Score storage
│
├── data/                 # Data storage
│   ├── results/          # Excel and Parquet files
│   │   ├── nfl_season_results.xlsx
│   │   ├── nba_season_results.xlsx
│   │   └── ncaam_season_results.xlsx
│   └── *.json            # Score storage files
│
├── automation/           # Automation files
│   ├── com.sportsbetting.dailycollection.plist
│   └── setup_automation.sh
│
├── docs/                 # Documentation
│   └── AUTOMATION_SETUP.md
│
├── logs/                 # Log files
│
└── requirements.txt      # Python dependencies
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key:**
   ```bash
   export ODDS_API_KEY=your_api_key
   ```

3. **Run daily collection:**
   ```bash
   python3 scripts/collect_yesterday_games.py
   ```

4. **Set up automation:**
   ```bash
   cd automation
   ./setup_automation.sh
   ```

## Collection Scripts

- **`collect_yesterday_games.py`** - Main daily script (automated at 6 AM EST)
  - Collects missed games from last collection through yesterday
  - Automatically catches up if Mac was off
  
- **`collect_*_season.py`** - Full season collection scripts
  - Run once per season to collect all historical games
  - Creates initial Excel and Parquet files

## Data Files

All results are stored in `data/results/`:
- Excel files (`.xlsx`) - Human-readable format
- Parquet files (`.parquet`) - Efficient storage format

Columns in results:
- `game_date` - Date of the game
- `home_team` - Home team name
- `away_team` - Away team name
- `closing_spread` - Closing spread from DraftKings
- `home_score` - Final home team score (if available)
- `away_score` - Final away team score (if available)
- `spread_result_difference` - Difference between actual result and spread

## Automation

The system runs automatically at 6:00 AM EST daily. See `docs/AUTOMATION_SETUP.md` for details.

## API Usage

The system is optimized for conservative API usage:
- Uses historical endpoints efficiently
- Caches scores locally
- Only collects missing dates

