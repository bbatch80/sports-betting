# Daily Collection Automation Setup

This automation runs `collect_yesterday_games.py` daily at 6:00 AM EST to collect yesterday's games and append them to existing Excel files.

## Quick Setup

Run the setup script:
```bash
./setup_automation.sh
```

This will:
1. Create a logs directory
2. Copy the plist file to `~/Library/LaunchAgents/`
3. Update paths in the plist file
4. Load the launch agent

## Manual Setup

If you prefer to set it up manually:

1. **Copy the plist file:**
   ```bash
   cp com.sportsbetting.dailycollection.plist ~/Library/LaunchAgents/
   ```

2. **Update the API key in the plist file** (if needed):
   ```bash
   nano ~/Library/LaunchAgents/com.sportsbetting.dailycollection.plist
   ```
   Update the `ODDS_API_KEY` value in the `EnvironmentVariables` section.

3. **Update the script path** in the plist file to match your actual path.

4. **Load the launch agent:**
   ```bash
   launchctl load ~/Library/LaunchAgents/com.sportsbetting.dailycollection.plist
   ```

## Managing the Automation

**Check if it's running:**
```bash
launchctl list | grep sportsbetting
```

**View logs:**
```bash
tail -f logs/daily_collection.log
tail -f logs/daily_collection_error.log
```

**Stop the automation:**
```bash
launchctl unload ~/Library/LaunchAgents/com.sportsbetting.dailycollection.plist
```

**Start the automation:**
```bash
launchctl load ~/Library/LaunchAgents/com.sportsbetting.dailycollection.plist
```

**Test the script manually:**
```bash
export ODDS_API_KEY=your_api_key
python3 collect_yesterday_games.py
```

## Schedule

- **Time:** 6:00 AM EST daily
- **What it does:** Collects yesterday's completed games and appends to Excel files
- **Leagues:** NFL, NBA, NCAAM

## Notes

- The script will only add games that aren't already in the Excel files
- If a game already exists, it will update it with the latest data (scores/spreads)
- Logs are saved to `logs/daily_collection.log` and `logs/daily_collection_error.log`
- The automation runs in the background and doesn't require you to be logged in

