"""
NCAAM Season Data Collection
Collects closing spreads and scores for the entire 2025 NCAAM season
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
                        
                        # If can't match, use first outcome (assuming it's home team perspective)
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


def collect_season_games(sport_key: str, season_start_date: datetime = None) -> pd.DataFrame:
    """
    Collect closing spreads and scores for games from season start to today
    
    Args:
        sport_key: Sport key (e.g., 'ncaam')
        season_start_date: Season start date (default: Nov 1, 2025 for NCAAM)
    
    Returns DataFrame with: game_date, home_team, away_team, closing_spread, 
    home_score, away_score, spread_result_difference
    """
    if season_start_date is None:
        # Default to 2025 NCAAM season start
        season_start_date = datetime(2025, 11, 1, tzinfo=timezone.utc)
    
    print(f"\n{'='*80}")
    print(f"Collecting {sport_key.upper()} Results (Entire Season)")
    print(f"{'='*80}")
    print(f"Season start: {season_start_date.strftime('%Y-%m-%d')}")
    print(f"Today: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    
    client = OddsAPIClient()
    api_sport_key = config.get_sport_api_key(sport_key)
    current_date = datetime.now(timezone.utc)
    start_date = season_start_date
    
    all_games = []
    event_ids_seen = set()
    
    # STEP 1: Get games from scores endpoint (days 1-3) - AUTHORITATIVE SOURCE
    print(f"\nStep 1: Getting completed games from scores endpoint (last 3 days)...")
    sys.stdout.flush()
    
    try:
        time.sleep(config.API_RATE_LIMIT_DELAY)
        scores = client.get_scores(sport=api_sport_key, days_from=3)
        
        scores_games = 0
        for game in scores:
            if game.get('completed') == True:
                event_id = game.get('id')
                commence_time = game.get('commence_time')
                
                if event_id and commence_time:
                    try:
                        event_time = pd.to_datetime(commence_time)
                        if event_time.tzinfo is None:
                            event_time = event_time.replace(tzinfo=timezone.utc)
                        elif isinstance(event_time, pd.Timestamp):
                            if event_time.tz is None:
                                event_time = event_time.tz_localize('UTC').to_pydatetime()
                            else:
                                event_time = event_time.to_pydatetime()
                        
                        # Only include games from our date range
                        if event_time >= start_date and event_time < current_date:
                            if event_id not in event_ids_seen:
                                event_ids_seen.add(event_id)
                                all_games.append(game)
                                scores_games += 1
                                
                                # Store score locally
                                home_team = game.get('home_team', '')
                                away_team = game.get('away_team', '')
                                home_score, away_score = extract_scores(game, home_team, away_team)
                                
                                if home_score is not None and away_score is not None:
                                    try:
                                        # Use EST date for consistency
                                        est = timezone(timedelta(hours=-5))
                                        event_time_est = event_time.astimezone(est)
                                        game_date = event_time_est.date().isoformat()
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
        
        print(f"  ✓ Found {scores_games} completed games from scores endpoint")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # STEP 2: Get games from historical events (entire season, querying daily)
    print(f"\nStep 2: Getting games from historical events (entire season)...")
    sys.stdout.flush()
    
    three_days_ago = current_date - timedelta(days=3)
    historical_games = 0
    
    # Query daily from season start to 3 days ago (scores endpoint covers last 3 days)
    # Query at end of each day (23:59) to capture all games from that day
    query_date = start_date.replace(tzinfo=None)
    total_days = (three_days_ago - start_date).days + 1
    
    print(f"  Querying {total_days} days of data (daily queries to capture all games)...")
    sys.stdout.flush()
    
    api_calls = 0
    while query_date <= three_days_ago.replace(tzinfo=None):
        try:
            time.sleep(config.API_RATE_LIMIT_DELAY)
            # Query at end of day (23:59) to capture all games from that day
            date_str = query_date.replace(hour=23, minute=59, second=0).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            response = client.get_historical_events(sport=api_sport_key, date=date_str)
            api_calls += 1
            
            if isinstance(response, dict) and 'data' in response:
                events = response['data']
            elif isinstance(response, list):
                events = response
            else:
                events = []
            
            events_added = 0
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                event_id = event.get('id')
                commence_time = event.get('commence_time')
                
                if event_id and event_id not in event_ids_seen and commence_time:
                    try:
                        event_time = pd.to_datetime(commence_time)
                        if event_time.tzinfo is None:
                            event_time = event_time.replace(tzinfo=timezone.utc)
                        elif isinstance(event_time, pd.Timestamp):
                            if event_time.tz is None:
                                event_time = event_time.tz_localize('UTC').to_pydatetime()
                            else:
                                event_time = event_time.to_pydatetime()
                        
                        # Only include games from season start to 3 days ago
                        if event_time >= start_date and event_time < three_days_ago:
                            event_ids_seen.add(event_id)
                            all_games.append(event)
                            historical_games += 1
                            events_added += 1
                    except:
                        pass
            
            if events_added > 0 and api_calls % 10 == 0:
                print(f"    Day {api_calls}/{total_days}: +{events_added} games (Total: {len(all_games)})")
                sys.stdout.flush()
            
            # Move to next day
            query_date = query_date + timedelta(days=1)
            
        except Exception as e:
            # On error, still move to next day
            query_date = query_date + timedelta(days=1)
            continue
    
    print(f"  ✓ Found {historical_games} additional games from historical events")
    print(f"  Made {api_calls} API calls for historical events")
    print(f"  Total games to process: {len(all_games)}")
    sys.stdout.flush()
    
    # STEP 3: Load stored scores
    stored_scores = score_storage.load_stored_scores(sport_key)
    print(f"\nStep 3: Loaded {len(stored_scores)} stored scores")
    sys.stdout.flush()
    
    # STEP 4: Process games to get closing spreads
    print(f"\nStep 4: Getting closing spreads for {len(all_games)} games...")
    sys.stdout.flush()
    
    results = []
    successful = 0
    no_spread = 0
    
    for i, game in enumerate(all_games, 1):
        event_id = game.get('id')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')
        
        if i % 100 == 0 or i == 1:
            print(f"    [{i}/{len(all_games)}] {away_team} @ {home_team}...", end='', flush=True)
        
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
            if i % 100 == 0 or i == 1:
                score_str = f"{away_score}-{home_score}" if (home_score and away_score) else "no score"
                print(f" ✓ (Spread: {closing_spread}, Score: {score_str})")
        else:
            closing_spread = None
            no_spread += 1
            if i % 100 == 0 or i == 1:
                print(f" ⚠ (no spread)")
        
        # Parse game date (use EST date for consistency)
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
        
        # Calculate spread_result_difference = (home_score - away_score) + closing_spread
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
    
    print(f"\n  Summary:")
    print(f"    Total games: {len(all_games)}")
    print(f"    With spreads: {successful}")
    print(f"    Without spreads: {no_spread}")
    print(f"    With scores: {sum(1 for r in results if r.get('home_score') is not None)}")
    print(f"    Without scores: {sum(1 for r in results if r.get('home_score') is None)}")
    sys.stdout.flush()
    
    df = pd.DataFrame(results)
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)
    
    return df


def save_results(sport_key: str, df: pd.DataFrame):
    """Save to Excel with specified format"""
    if len(df) == 0:
        print(f"  No data to save for {sport_key.upper()}")
        return
    
    # Prepare Excel output with exact column order and names
    excel_df = df.copy()
    
    # Ensure columns are in the right order
    columns = ['game_date', 'home_team', 'away_team', 'closing_spread', 
               'home_score', 'away_score', 'spread_result_difference']
    
    # Reorder and rename columns for clarity
    excel_df = excel_df[columns].copy()
    
    # Convert game_date to date format for Excel
    if 'game_date' in excel_df.columns:
        excel_df['game_date'] = pd.to_datetime(excel_df['game_date']).dt.date
    
    # Format closing_spread to show as number (not scientific notation)
    if 'closing_spread' in excel_df.columns:
        excel_df['closing_spread'] = excel_df['closing_spread'].round(1)
    
    # Format spread_result_difference
    if 'spread_result_difference' in excel_df.columns:
        excel_df['spread_result_difference'] = excel_df['spread_result_difference'].round(1)
    
    # Save to data/results directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
    os.makedirs(data_dir, exist_ok=True)
    excel_file = os.path.join(data_dir, f"{sport_key}_season_results.xlsx")
    
    # Save to Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        excel_df.to_excel(writer, sheet_name=f'{sport_key.upper()} Season', index=False)
        
        # Get the worksheet to format
        worksheet = writer.sheets[f'{sport_key.upper()} Season']
        
        # Auto-adjust column widths
        for idx, col in enumerate(excel_df.columns, 1):
            max_length = max(
                excel_df[col].astype(str).map(len).max(),
                len(str(col))
            )
            worksheet.column_dimensions[chr(64 + idx)].width = min(max_length + 2, 50)
    
    print(f"  ✓ Saved {len(excel_df)} games to {excel_file}")
    
    # Also save parquet for backup
    parquet_file = os.path.join(data_dir, f"{sport_key}_season_results.parquet")
    df.to_parquet(parquet_file, compression='snappy', index=False)
    print(f"  ✓ Saved backup to {parquet_file}")


def main():
    """Main collection function - Collect entire 2025 NCAAM season"""
    print("="*80)
    print("COLLECTING 2025 NCAAM SEASON DATA")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        client = OddsAPIClient()
        usage = client.get_usage()
        print(f"API Usage: {usage.get('requests_used')} used, {usage.get('requests_remaining')} remaining\n")
    except:
        pass
    
    # Collect NCAAM season (Nov 1, 2025 - Today)
    sport_key = 'ncaam'
    season_start = datetime(2025, 11, 1, tzinfo=timezone.utc)
    
    try:
        df = collect_season_games(sport_key, season_start_date=season_start)
        
        if len(df) > 0:
            save_results(sport_key, df)
            
            # Summary
            print(f"\n{'='*80}")
            print("COLLECTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total games: {len(df)}")
            print(f"Games with spreads: {df['closing_spread'].notna().sum()}")
            print(f"Games with scores: {df['home_score'].notna().sum()}")
            print(f"Games without scores: {df['home_score'].isna().sum()}")
            print(f"Games with spread difference calculated: {df['spread_result_difference'].notna().sum()}")
            
            # Show date range
            if 'game_date' in df.columns:
                dates = pd.to_datetime(df['game_date'].dropna())
                if len(dates) > 0:
                    print(f"\nDate range: {dates.min().date()} to {dates.max().date()}")
        else:
            print("No games collected")
        
        try:
            usage = client.get_usage()
            print(f"\nFinal API Usage: {usage.get('requests_used')} used, {usage.get('requests_remaining')} remaining")
        except:
            pass
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)


if __name__ == "__main__":
    main()

