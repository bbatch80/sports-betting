"""
Daily collection script to gather yesterday's games and append to existing Excel files

This script:
1. Collects completed games from yesterday
2. Gets closing spreads for those games
3. Loads existing Excel files
4. Appends new games (avoiding duplicates)
5. Saves updated Excel files

Can be run daily via cron/scheduler
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from odds_api_client import OddsAPIClient, OddsAPIError
import config
import score_storage


def get_closing_spread(client: OddsAPIClient, sport_key: str, event_id: str, 
                       home_team: str, away_team: str, commence_time: str) -> Optional[Dict[str, Any]]:
    """Get closing spread from DraftKings using historical event odds endpoint"""
    api_sport_key = config.get_sport_api_key(sport_key)
    
    try:
        # Get snapshot 5 minutes before game start
        game_start = pd.to_datetime(commence_time)
        if game_start.tzinfo is None:
            game_start = game_start.replace(tzinfo=timezone.utc)
        snapshot_time = game_start - timedelta(minutes=5)
        snapshot_date = snapshot_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        time.sleep(config.API_RATE_LIMIT_DELAY)
        
        # Get historical odds
        historical_odds = client.get_historical_event_odds(
            sport=api_sport_key,
            event_id=event_id,
            date=snapshot_date,
            markets=['spreads'],
            regions=config.DEFAULT_REGIONS
        )
        
        if not historical_odds or not historical_odds.get('data'):
            return None
        
        event_odds = historical_odds['data']
        bookmakers = event_odds.get('bookmakers', [])
        
        # Find DraftKings
        for bookmaker in bookmakers:
            if 'draftkings' in bookmaker.get('key', '').lower():
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'spreads':
                        outcomes = market.get('outcomes', [])
                        if len(outcomes) < 2:
                            continue
                        
                        # Find home team outcome
                        for outcome in outcomes:
                            outcome_name = outcome.get('name', '')
                            if home_team.lower() in outcome_name.lower() or outcome_name.lower() in home_team.lower():
                                return {
                                    'spread_point': outcome.get('point'),
                                    'odds': outcome.get('price')
                                }
                        
                        # If can't match, use first outcome
                        return {
                            'spread_point': outcomes[0].get('point'),
                            'odds': outcomes[0].get('price')
                        }
        
        return None
    except Exception as e:
        return None


def extract_scores(game_data: Dict[str, Any], home_team: str, away_team: str) -> tuple:
    """Extract home and away scores from game data"""
    scores_list = game_data.get('scores') or []
    home_score = None
    away_score = None
    
    for score_entry in scores_list:
        if not isinstance(score_entry, dict):
            continue
        team_name = score_entry.get('name', '')
        score_value = score_entry.get('score')
        
        if score_value is None:
            continue
        
        if home_team and team_name:
            if home_team.lower() in team_name.lower() or team_name.lower() in home_team.lower():
                home_score = score_value
        if away_team and team_name:
            if away_team.lower() in team_name.lower() or team_name.lower() in away_team.lower():
                away_score = score_value
    
    return home_score, away_score


def collect_games_for_date(sport_key: str, target_date: datetime) -> pd.DataFrame:
    """
    Collect completed games for a specific date with closing spreads and scores
    
    Args:
        sport_key: Sport key (e.g., 'nfl', 'nba')
        target_date: Target date (datetime object in UTC, but we'll compare using EST)
    
    Returns DataFrame with: game_date, home_team, away_team, closing_spread, 
    home_score, away_score, spread_result_difference
    """
    client = OddsAPIClient()
    api_sport_key = config.get_sport_api_key(sport_key)
    
    # Convert target date to EST for comparison (EST is UTC-5)
    est = timezone(timedelta(hours=-5))
    target_date_est = target_date.astimezone(est)
    target_date_only = target_date_est.date()
    
    date_str = target_date_only.strftime('%Y-%m-%d')
    print(f"  Collecting games from {date_str} (EST)...")
    sys.stdout.flush()
    
    all_games = []
    event_ids_seen = set()
    
    # Get games from scores endpoint (last 3 days covers yesterday)
    try:
        time.sleep(config.API_RATE_LIMIT_DELAY)
        scores = client.get_scores(sport=api_sport_key, days_from=3)
        
        for game in scores:
            if game.get('completed') == True:
                event_id = game.get('id')
                commence_time = game.get('commence_time')
                
                if event_id and commence_time:
                    try:
                        event_time_utc = pd.to_datetime(commence_time)
                        if event_time_utc.tzinfo is None:
                            event_time_utc = event_time_utc.replace(tzinfo=timezone.utc)
                        elif isinstance(event_time_utc, pd.Timestamp):
                            if event_time_utc.tz is None:
                                event_time_utc = event_time_utc.tz_localize('UTC').to_pydatetime()
                            else:
                                event_time_utc = event_time_utc.to_pydatetime()
                        
                        # Convert to EST for date comparison
                        event_time_est = event_time_utc.astimezone(est)
                        event_date_est = event_time_est.date()
                        
                        # Only include games from target date (using EST date)
                        if event_date_est == target_date_only:
                            if event_id not in event_ids_seen:
                                event_ids_seen.add(event_id)
                                all_games.append(game)
                                
                                # Store score locally
                                home_team = game.get('home_team', '')
                                away_team = game.get('away_team', '')
                                home_score, away_score = extract_scores(game, home_team, away_team)
                                
                                if home_score is not None and away_score is not None:
                                    try:
                                        # Use EST date for storage
                                        game_date = event_date_est.isoformat()
                                        score_storage.save_score(
                                            sport_key=sport_key,
                                            event_id=event_id,
                                            home_score=int(home_score),
                                            away_score=int(away_score),
                                            home_team=home_team,
                                            away_team=away_team,
                                            game_date=game_date
                                        )
                                    except:
                                        pass
                    except:
                        pass
        
        print(f"  ✓ Found {len(all_games)} completed games from {date_str}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()
    
    if len(all_games) == 0:
        print(f"  No games found for {date_str}")
        return pd.DataFrame()
    
    # Load stored scores
    stored_scores = score_storage.load_stored_scores(sport_key)
    
    # Process games to get closing spreads
    print(f"  Getting closing spreads for {len(all_games)} games...")
    sys.stdout.flush()
    
    results = []
    successful = 0
    no_spread = 0
    
    for i, game in enumerate(all_games, 1):
        event_id = game.get('id')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')
        
        # Get scores
        home_score, away_score = extract_scores(game, home_team, away_team)
        
        # Try stored scores if not found
        if home_score is None or away_score is None:
            stored = stored_scores.get(event_id)
            if stored:
                home_score = stored.get('home_score')
                away_score = stored.get('away_score')
        
        # Convert to numeric
        try:
            home_score = int(home_score) if home_score is not None else None
            away_score = int(away_score) if away_score is not None else None
        except:
            home_score = None
            away_score = None
        
        # Get closing spread
        spread_data = get_closing_spread(client, sport_key, event_id, home_team, away_team, commence_time)
        
        if spread_data and spread_data.get('spread_point') is not None:
            closing_spread = float(spread_data['spread_point'])
            successful += 1
        else:
            closing_spread = None
            no_spread += 1
        
        # Parse game date (use EST date)
        try:
            est = timezone(timedelta(hours=-5))
            event_time_utc = pd.to_datetime(commence_time)
            if event_time_utc.tzinfo is None:
                event_time_utc = event_time_utc.replace(tzinfo=timezone.utc)
            elif isinstance(event_time_utc, pd.Timestamp):
                if event_time_utc.tz is None:
                    event_time_utc = event_time_utc.tz_localize('UTC').to_pydatetime()
                else:
                    event_time_utc = event_time_utc.to_pydatetime()
            # Convert to EST for date
            event_time_est = event_time_utc.astimezone(est)
            game_date = event_time_est.date()
        except:
            game_date = None
        
        # Calculate spread_result_difference
        if home_score is not None and away_score is not None and closing_spread is not None:
            spread_result_difference = (home_score - away_score) + closing_spread
        else:
            spread_result_difference = None
        
        results.append({
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'closing_spread': closing_spread,
            'home_score': home_score,
            'away_score': away_score,
            'spread_result_difference': spread_result_difference
        })
    
    print(f"  ✓ Got spreads for {successful}/{len(all_games)} games")
    sys.stdout.flush()
    
    df = pd.DataFrame(results)
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)
    
    return df


def collect_yesterday_games(sport_key: str) -> pd.DataFrame:
    """
    Collect yesterday's completed games with closing spreads and scores
    
    Returns DataFrame with: game_date, home_team, away_team, closing_spread, 
    home_score, away_score, spread_result_difference
    """
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    return collect_games_for_date(sport_key, yesterday)


def get_last_collection_date(sport_key: str) -> Optional[datetime]:
    """
    Get the date of the most recent game in the Excel file
    
    Returns None if no file exists or no games found
    """
    existing_df = load_existing_data(sport_key)
    
    if len(existing_df) == 0 or 'game_date' not in existing_df.columns:
        return None
    
    try:
        # Get the most recent game date
        max_date = existing_df['game_date'].max()
        if pd.isna(max_date):
            return None
        
        # Convert to datetime if it's a date
        if isinstance(max_date, pd.Timestamp):
            max_date = max_date.to_pydatetime()
        elif isinstance(max_date, datetime):
            pass
        else:
            max_date = pd.to_datetime(max_date).to_pydatetime()
        
        # Ensure timezone-aware
        if max_date.tzinfo is None:
            max_date = max_date.replace(tzinfo=timezone.utc)
        
        return max_date
    except Exception as e:
        return None


def get_missed_dates(sport_key: str) -> List[datetime]:
    """
    Determine which dates need to be collected (from last collection to yesterday)
    Uses EST dates for consistency with user's timezone
    
    Returns list of datetime objects for dates that need collection
    """
    # Use EST for date calculations
    est = timezone(timedelta(hours=-5))
    today_est = datetime.now(est)
    yesterday_est = (today_est - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Convert to UTC for API calls (but we'll compare using EST dates)
    yesterday = yesterday_est.astimezone(timezone.utc)
    
    last_collection = get_last_collection_date(sport_key)
    
    # If no previous collection, only collect yesterday
    if last_collection is None:
        return [yesterday]
    
    # Convert last_collection to EST for comparison
    last_collection_est = last_collection.astimezone(est)
    last_collection_date_est = last_collection_est.date()
    yesterday_date_est = yesterday_est.date()
    
    # If last collection was yesterday or later, nothing to collect
    if last_collection_date_est >= yesterday_date_est:
        return []
    
    # Collect all dates from day after last collection through yesterday (in EST)
    missed_dates = []
    current_date_est = last_collection_date_est + timedelta(days=1)
    
    while current_date_est <= yesterday_date_est:
        # Create datetime at start of day in EST, then convert to UTC
        date_start_est = datetime.combine(current_date_est, datetime.min.time()).replace(tzinfo=est)
        date_start_utc = date_start_est.astimezone(timezone.utc)
        missed_dates.append(date_start_utc)
        current_date_est += timedelta(days=1)
    
    return missed_dates


def load_existing_data(sport_key: str) -> pd.DataFrame:
    """Load existing Excel file if it exists"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
    excel_file = os.path.join(data_dir, f"{sport_key}_season_results.xlsx")
    
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file)
            # Convert game_date back to date if it's datetime
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date']).dt.date
            return df
        except Exception as e:
            print(f"  ⚠ Warning: Could not load existing file: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def append_to_excel(sport_key: str, new_games_df: pd.DataFrame):
    """Append new games to existing Excel file, avoiding duplicates and preserving manually filled data"""
    if len(new_games_df) == 0:
        print(f"  No new games to add")
        return
    
    # Load existing data
    existing_df = load_existing_data(sport_key)
    
    if len(existing_df) > 0:
        # Create a key for matching games
        existing_df['_match_key'] = (
            existing_df['game_date'].astype(str) + '|' + 
            existing_df['home_team'].astype(str) + '|' + 
            existing_df['away_team'].astype(str)
        )
        new_games_df['_match_key'] = (
            new_games_df['game_date'].astype(str) + '|' + 
            new_games_df['home_team'].astype(str) + '|' + 
            new_games_df['away_team'].astype(str)
        )
        
        # Find games that exist in both
        existing_keys = set(existing_df['_match_key'])
        new_keys = set(new_games_df['_match_key'])
        duplicate_keys = existing_keys & new_keys
        
        # For duplicate games, preserve manually filled scores from existing data
        if len(duplicate_keys) > 0:
            for key in duplicate_keys:
                existing_row = existing_df[existing_df['_match_key'] == key].iloc[0]
                new_row_idx = new_games_df[new_games_df['_match_key'] == key].index[0]
                
                # If existing row has scores and new row doesn't, preserve existing scores
                if (pd.notna(existing_row.get('home_score')) and pd.notna(existing_row.get('away_score')) and
                    (pd.isna(new_games_df.loc[new_row_idx, 'home_score']) or pd.isna(new_games_df.loc[new_row_idx, 'away_score']))):
                    new_games_df.loc[new_row_idx, 'home_score'] = existing_row['home_score']
                    new_games_df.loc[new_row_idx, 'away_score'] = existing_row['away_score']
                    # Recalculate spread_result_difference if we have all values
                    if pd.notna(new_games_df.loc[new_row_idx, 'closing_spread']):
                        new_games_df.loc[new_row_idx, 'spread_result_difference'] = (
                            new_games_df.loc[new_row_idx, 'home_score'] - 
                            new_games_df.loc[new_row_idx, 'away_score'] + 
                            new_games_df.loc[new_row_idx, 'closing_spread']
                        )
        
        # Remove match key columns
        existing_df = existing_df.drop(columns=['_match_key'])
        new_games_df = new_games_df.drop(columns=['_match_key'])
        
        # Get games that only exist in existing (not in new)
        existing_only = existing_df[~existing_df.apply(
            lambda row: f"{row['game_date']}|{row['home_team']}|{row['away_team']}" in new_keys,
            axis=1
        )]
        
        # Combine: existing-only games + new games (which now have preserved scores)
        combined_df = pd.concat([existing_only, new_games_df], ignore_index=True)
        
        print(f"  Existing games: {len(existing_df)}")
        print(f"  New games: {len(new_games_df)}")
        print(f"  Preserved {len(duplicate_keys)} manually filled scores")
        print(f"  Total after merge: {len(combined_df)}")
    else:
        combined_df = new_games_df
        print(f"  No existing file found, creating new file with {len(new_games_df)} games")
    
    # Sort by date
    if 'game_date' in combined_df.columns:
        combined_df = combined_df.sort_values('game_date').reset_index(drop=True)
    
    # Save to Excel
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
    os.makedirs(data_dir, exist_ok=True)
    excel_file = os.path.join(data_dir, f"{sport_key}_season_results.xlsx")
    excel_df = combined_df.copy()
    
    # Ensure columns are in the right order
    columns = ['game_date', 'home_team', 'away_team', 'closing_spread', 
               'home_score', 'away_score', 'spread_result_difference']
    excel_df = excel_df[columns].copy()
    
    # Convert game_date to date format for Excel
    if 'game_date' in excel_df.columns:
        excel_df['game_date'] = pd.to_datetime(excel_df['game_date']).dt.date
    
    # Format numeric columns
    if 'closing_spread' in excel_df.columns:
        excel_df['closing_spread'] = excel_df['closing_spread'].round(1)
    if 'spread_result_difference' in excel_df.columns:
        excel_df['spread_result_difference'] = excel_df['spread_result_difference'].round(1)
    
    # Save to Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        excel_df.to_excel(writer, sheet_name=f'{sport_key.upper()} Season', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets[f'{sport_key.upper()} Season']
        for idx, col in enumerate(excel_df.columns, 1):
            max_length = max(
                excel_df[col].astype(str).map(len).max(),
                len(str(col))
            )
            worksheet.column_dimensions[chr(64 + idx)].width = min(max_length + 2, 50)
    
    print(f"  ✓ Updated {excel_file} with {len(combined_df)} total games")
    
    # Also update parquet backup
    parquet_file = os.path.join(data_dir, f"{sport_key}_season_results.parquet")
    combined_df.to_parquet(parquet_file, compression='snappy', index=False)
    print(f"  ✓ Updated backup {parquet_file}")


def main():
    """Main function - Collect missed games (from last collection to yesterday) for all leagues"""
    print("="*80)
    print("COLLECTING MISSED GAMES")
    print("="*80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        client = OddsAPIClient()
        usage = client.get_usage()
        print(f"API Usage: {usage.get('requests_used')} used, {usage.get('requests_remaining')} remaining\n")
    except:
        pass
    
    # Collect for all leagues
    sports = ['nfl', 'nba', 'ncaam']
    all_results = {}
    
    for sport_key in sports:
        print(f"\n{'='*80}")
        print(f"Processing {sport_key.upper()}")
        print(f"{'='*80}")
        
        try:
            # Check for missed dates
            missed_dates = get_missed_dates(sport_key)
            
            if len(missed_dates) == 0:
                print(f"  ✓ No missed dates - data is up to date")
                all_results[sport_key] = pd.DataFrame()
                continue
            
            last_collection = get_last_collection_date(sport_key)
            if last_collection:
                last_str = last_collection.strftime('%Y-%m-%d')
                print(f"  Last collection: {last_str}")
            else:
                print(f"  No previous collection found")
            
            print(f"  Collecting {len(missed_dates)} missed day(s): {', '.join([d.strftime('%Y-%m-%d') for d in missed_dates])}")
            sys.stdout.flush()
            
            # Collect games for each missed date
            all_new_games = []
            for date in missed_dates:
                date_games_df = collect_games_for_date(sport_key, date)
                if len(date_games_df) > 0:
                    all_new_games.append(date_games_df)
            
            # Combine all new games
            if len(all_new_games) > 0:
                new_games_df = pd.concat(all_new_games, ignore_index=True)
                new_games_df = new_games_df.sort_values('game_date').reset_index(drop=True)
            else:
                new_games_df = pd.DataFrame()
            
            all_results[sport_key] = new_games_df
            
            if len(new_games_df) > 0:
                append_to_excel(sport_key, new_games_df)
                print(f"  ✓ Added {len(new_games_df)} new games to {sport_key}_season_results.xlsx")
            else:
                print(f"  No new games found for missed dates")
            
            try:
                usage = client.get_usage()
                print(f"  API remaining: {usage.get('requests_remaining')}")
            except:
                pass
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[sport_key] = pd.DataFrame()
    
    # Summary
    print(f"\n{'='*80}")
    print("COLLECTION SUMMARY")
    print(f"{'='*80}")
    total_new = 0
    for sport_key, df in all_results.items():
        count = len(df)
        total_new += count
        print(f"  {sport_key.upper()}: {count} new games collected")
    
    if total_new == 0:
        print("  All leagues are up to date!")
    
    try:
        usage = client.get_usage()
        print(f"\nFinal API Usage: {usage.get('requests_used')} used, {usage.get('requests_remaining')} remaining")
    except:
        pass
    
    print("="*80)


if __name__ == "__main__":
    main()

