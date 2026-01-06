"""
Test script to verify we're capturing ALL games and that closing lines are available

This script:
1. Gets games from scores endpoint (authoritative source for last 3 days)
2. Gets games from historical events endpoint
3. Compares to ensure no games are missing
4. Tests if closing spreads are available for each game
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from odds_api_client import OddsAPIClient, OddsAPIError
import config


def get_games_from_scores(client, sport_key, days=3):
    """Get all completed games from scores endpoint - AUTHORITATIVE SOURCE"""
    api_sport_key = config.get_sport_api_key(sport_key)
    
    print(f"\n{'='*70}")
    print(f"STEP 1: Getting games from SCORES endpoint (authoritative source)")
    print(f"{'='*70}")
    
    scores = client.get_scores(sport=api_sport_key, days_from=days)
    
    completed_games = {}
    current_date = datetime.now(timezone.utc)
    start_date = current_date - timedelta(days=days)
    
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
                    
                    if event_time >= start_date and event_time < current_date:
                        completed_games[event_id] = {
                            'event': game,
                            'commence_time': event_time,
                            'home_team': game.get('home_team'),
                            'away_team': game.get('away_team')
                        }
                except:
                    pass
    
    print(f"✓ Found {len(completed_games)} completed games from scores endpoint")
    print(f"\nSample games:")
    for i, (event_id, data) in enumerate(list(completed_games.items())[:5], 1):
        print(f"  {i}. {data['away_team']} @ {data['home_team']} ({data['commence_time'].strftime('%Y-%m-%d %H:%M')})")
    
    return completed_games


def get_games_from_historical_events(client, sport_key, start_date=None, end_date=None):
    """Get games from historical events endpoint for a date range"""
    api_sport_key = config.get_sport_api_key(sport_key)
    current_date = datetime.now(timezone.utc)
    
    if start_date is None:
        # Default to 2025 NFL season start (September 5, 2025)
        start_date = datetime(2025, 9, 5, tzinfo=timezone.utc)
    if end_date is None:
        end_date = current_date
    
    print(f"\n{'='*70}")
    print(f"STEP 2: Getting games from HISTORICAL EVENTS endpoint")
    print(f"{'='*70}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_events = {}
    event_ids_seen = set()
    
    # Query weekly to balance completeness vs API usage
    query_date = start_date.replace(tzinfo=None)
    total_days = (end_date - start_date).days
    weeks_to_query = (total_days // 7) + 1
    
    print(f"Querying {weeks_to_query} weeks of data (weekly queries)...")
    sys.stdout.flush()
    
    api_calls = 0
    while query_date < end_date.replace(tzinfo=None):
        try:
            # Query at end of week (Sunday 23:59) to capture all games from that week
            # For NFL, most games are on Sunday, so this should capture most games
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
                        
                        # Only include games in our date range that have completed
                        if event_time >= start_date and event_time < end_date:
                            event_ids_seen.add(event_id)
                            all_events[event_id] = {
                                'event': event,
                                'commence_time': event_time,
                                'home_team': event.get('home_team'),
                                'away_team': event.get('away_team')
                            }
                            events_added += 1
                    except:
                        pass
            
            if events_added > 0 and api_calls % 5 == 0:
                print(f"  Week {api_calls}/{weeks_to_query}: +{events_added} games (Total: {len(all_events)})")
                sys.stdout.flush()
            
            # Move to next week
            query_date = query_date + timedelta(days=7)
            
        except Exception as e:
            query_date = query_date + timedelta(days=7)
            continue
    
    print(f"\n✓ Found {len(all_events)} games from historical events endpoint")
    print(f"  Made {api_calls} API calls")
    if len(all_events) > 0:
        print(f"\nSample games (first and last):")
        sorted_events = sorted(all_events.items(), key=lambda x: x[1]['commence_time'])
        for i, (event_id, data) in enumerate([sorted_events[0], sorted_events[-1]], 1):
            print(f"  {i}. {data['away_team']} @ {data['home_team']} ({data['commence_time'].strftime('%Y-%m-%d %H:%M')})")
    
    return all_events


def test_closing_spread_availability(client, sport_key, event_id, home_team, away_team, commence_time):
    """Test if closing spread is available for a game"""
    api_sport_key = config.get_sport_api_key(sport_key)
    
    try:
        # Get snapshot 5 minutes before game start
        game_start = pd.to_datetime(commence_time)
        if game_start.tzinfo is None:
            game_start = game_start.replace(tzinfo=timezone.utc)
        snapshot_time = game_start - timedelta(minutes=5)
        snapshot_date = snapshot_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        historical_odds = client.get_historical_event_odds(
            sport=api_sport_key,
            event_id=event_id,
            date=snapshot_date,
            markets=['spreads'],
            regions=['us']
        )
        
        if not historical_odds or not historical_odds.get('data'):
            return False, "No odds data"
        
        event_odds = historical_odds['data']
        bookmakers = event_odds.get('bookmakers', [])
        
        # Check for DraftKings
        for bookmaker in bookmakers:
            if 'draftkings' in bookmaker.get('key', '').lower():
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'spreads':
                        outcomes = market.get('outcomes', [])
                        if len(outcomes) >= 2:
                            return True, f"Available (spread: {outcomes[0].get('point')})"
        
        return False, "DraftKings not found"
    except OddsAPIError as e:
        return False, f"API Error: {str(e)[:50]}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def verify_completeness(scores_games, historical_games):
    """Verify that all games from scores endpoint are in historical events"""
    print(f"\n{'='*70}")
    print(f"STEP 3: VERIFICATION - Checking completeness")
    print(f"{'='*70}")
    
    scores_event_ids = set(scores_games.keys())
    historical_event_ids = set(historical_games.keys())
    
    missing_in_historical = scores_event_ids - historical_event_ids
    only_in_historical = historical_event_ids - scores_event_ids
    
    print(f"\nScores endpoint games: {len(scores_event_ids)}")
    print(f"Historical events games: {len(historical_event_ids)}")
    print(f"Games in both: {len(scores_event_ids & historical_event_ids)}")
    print(f"Missing in historical: {len(missing_in_historical)}")
    print(f"Only in historical: {len(only_in_historical)}")
    
    if missing_in_historical:
        print(f"\n⚠ WARNING: {len(missing_in_historical)} games from scores endpoint NOT found in historical events:")
        for event_id in list(missing_in_historical)[:5]:
            game = scores_games[event_id]
            print(f"  - {game['away_team']} @ {game['home_team']} ({game['commence_time'].strftime('%Y-%m-%d')})")
    
    if only_in_historical:
        print(f"\nℹ Games only in historical events (from days 4-7): {len(only_in_historical)}")
    
    return {
        'scores_count': len(scores_event_ids),
        'historical_count': len(historical_event_ids),
        'missing': len(missing_in_historical),
        'coverage': (len(scores_event_ids & historical_event_ids) / len(scores_event_ids) * 100) if scores_event_ids else 0
    }


def test_closing_spreads(client, sport_key, games_dict, max_test=10):
    """Test closing spread availability for a sample of games"""
    print(f"\n{'='*70}")
    print(f"STEP 4: Testing closing spread availability")
    print(f"{'='*70}")
    
    test_games = list(games_dict.items())[:max_test]
    available = 0
    unavailable = 0
    
    print(f"\nTesting {len(test_games)} games...")
    
    for event_id, game_data in test_games:
        event = game_data['event']
        home_team = game_data['home_team']
        away_team = game_data['away_team']
        commence_time = game_data['commence_time']
        
        is_available, reason = test_closing_spread_availability(
            client, sport_key, event_id, home_team, away_team, commence_time
        )
        
        if is_available:
            available += 1
            print(f"  ✓ {away_team} @ {home_team}: {reason}")
        else:
            unavailable += 1
            print(f"  ✗ {away_team} @ {home_team}: {reason}")
    
    print(f"\nResults: {available}/{len(test_games)} games have closing spreads available")
    print(f"  Available: {available}")
    print(f"  Unavailable: {unavailable}")
    
    return available, unavailable


def main():
    """Main test function"""
    print("="*70)
    print("COLLECTION COMPLETENESS TEST (2025 NFL SEASON)")
    print("="*70)
    print(f"Testing if we can capture ALL games and get closing spreads")
    print(f"Testing period: 2025 NFL Season (Sept 5, 2025 - Today)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test with NFL first (smaller dataset)
    sport_key = 'nfl'
    
    try:
        client = OddsAPIClient()
        
        # Step 1: Get games from scores endpoint (authoritative) - last 3 days only
        scores_games = get_games_from_scores(client, sport_key, days=3)
        
        # Step 2: Get games from historical events (entire 2025 season)
        season_start = datetime(2025, 9, 5, tzinfo=timezone.utc)  # NFL season typically starts early September
        historical_games = get_games_from_historical_events(client, sport_key, start_date=season_start)
        
        # Step 3: Verify completeness
        verification = verify_completeness(scores_games, historical_games)
        
        # Step 4: Test closing spread availability
        # Combine both sources for testing
        all_games = {**scores_games, **historical_games}
        available, unavailable = test_closing_spreads(client, sport_key, all_games, max_test=min(10, len(all_games)))
        
        # Final summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        print(f"✓ Scores endpoint: {verification['scores_count']} games")
        print(f"✓ Historical events: {verification['historical_count']} games")
        print(f"✓ Coverage: {verification['coverage']:.1f}%")
        print(f"✓ Closing spreads available: {available}/{available+unavailable} tested")
        
        # Calculate expected games (NFL has ~16 games per week, ~17 weeks in season)
        # We're testing from Sept 5 to today (about 3 months = ~12-13 weeks)
        weeks_elapsed = (datetime.now(timezone.utc) - datetime(2025, 9, 5, tzinfo=timezone.utc)).days / 7
        expected_games_approx = int(weeks_elapsed * 16)  # Rough estimate
        
        print(f"\nExpected games (approx): ~{expected_games_approx} (based on ~16 games/week)")
        print(f"Actual games found: {verification['historical_count']}")
        
        if verification['missing'] == 0 and available > 0:
            print(f"\n✅ SUCCESS: All games captured and closing spreads are available!")
        elif verification['missing'] > 0:
            print(f"\n⚠ WARNING: {verification['missing']} games missing from historical events")
        elif available == 0:
            print(f"\n⚠ WARNING: No closing spreads available for tested games")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

