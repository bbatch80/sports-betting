"""
Elite Teams Today - Identify elite teams in good form with games today

This script identifies teams that are:
1. Elite (top 25% by win percentage)
2. In good recent form (last 5 games spread performance > 3)
3. Have games scheduled for today

Usage:
    python scripts/elite_teams_today.py                    # All sports
    python scripts/elite_teams_today.py --sport nfl        # Specific sport
    python scripts/elite_teams_today.py --local            # Use local files
"""

import os
import sys
import argparse
import json
import pandas as pd
import io
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import AWS clients (optional for local mode)
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from odds_api_client import OddsAPIClient
    ODDS_CLIENT_AVAILABLE = True
except ImportError:
    ODDS_CLIENT_AVAILABLE = False


# Configuration
SPORTS_CONFIG = {
    'nfl': {
        'api_key': 'americanfootball_nfl',
        'handicap_points': 5,
        'name': 'NFL'
    },
    'nba': {
        'api_key': 'basketball_nba',
        'handicap_points': 9,
        'name': 'NBA'
    },
    'ncaam': {
        'api_key': 'basketball_ncaab',
        'handicap_points': 10,
        'name': 'NCAAM'
    }
}

BUCKET_NAME = 'sports-betting-analytics-data'
DEFAULT_REGIONS = ['us']

# Elite team criteria
ELITE_PERCENTILE = 0.75  # Top 25%
GOOD_FORM_THRESHOLD = 3  # Last 5 games avg spread performance > 3


def load_historical_data_local(sport_key: str) -> pd.DataFrame:
    """Load historical data from local Excel file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, '..', 'data', 'results', f'{sport_key}_season_results.xlsx')

    if not os.path.exists(data_file):
        print(f"  ⚠ Local file not found: {data_file}")
        return pd.DataFrame()

    df = pd.read_excel(data_file)
    print(f"  ✓ Loaded {len(df)} games from local file")
    return df


def load_historical_data_s3(sport_key: str) -> pd.DataFrame:
    """Load historical data from S3"""
    if not AWS_AVAILABLE:
        print("  ⚠ AWS boto3 not available, cannot load from S3")
        return pd.DataFrame()

    s3_client = boto3.client('s3')
    s3_key = f"data/results/{sport_key}_season_results.xlsx"

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        excel_data = response['Body'].read()
        df = pd.read_excel(io.BytesIO(excel_data))
        print(f"  ✓ Loaded {len(df)} games from S3")
        return df
    except Exception as e:
        print(f"  ⚠ Error loading from S3: {e}")
        return pd.DataFrame()


def calculate_team_standings(df_completed: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team standings including win %, point differential, and tier

    Returns DataFrame with columns:
    - team, games, wins, win_pct, point_diff_avg, tier (Elite/Mid/Bottom)
    """
    all_teams = set(df_completed['home_team'].unique()) | set(df_completed['away_team'].unique())

    team_stats = []
    for team in all_teams:
        home_games = df_completed[df_completed['home_team'] == team]
        away_games = df_completed[df_completed['away_team'] == team]

        # Calculate wins
        home_wins = (home_games['home_score'] > home_games['away_score']).sum()
        away_wins = (away_games['away_score'] > away_games['home_score']).sum()

        # Calculate point differential
        home_diff = (home_games['home_score'] - home_games['away_score']).sum()
        away_diff = (away_games['away_score'] - away_games['home_score']).sum()

        total_games = len(home_games) + len(away_games)
        if total_games > 0:
            team_stats.append({
                'team': team,
                'games': total_games,
                'wins': home_wins + away_wins,
                'win_pct': (home_wins + away_wins) / total_games,
                'point_diff_avg': (home_diff + away_diff) / total_games
            })

    df_teams = pd.DataFrame(team_stats).sort_values('win_pct', ascending=False)

    # Classify tiers
    q75 = df_teams['win_pct'].quantile(ELITE_PERCENTILE)
    q25 = df_teams['win_pct'].quantile(1 - ELITE_PERCENTILE)

    df_teams['tier'] = df_teams['win_pct'].apply(
        lambda x: 'Elite' if x >= q75 else ('Bottom' if x <= q25 else 'Mid')
    )

    return df_teams


def calculate_recent_form(df_completed: pd.DataFrame, team_tier_map: Dict[str, str]) -> pd.DataFrame:
    """
    Calculate recent form (last 5 games) for each team

    Returns DataFrame with columns:
    - team, tier, last_5_avg_spread, games_count, is_good_form
    """
    df_sorted = df_completed.sort_values('game_date')
    all_teams = set(df_completed['home_team'].unique()) | set(df_completed['away_team'].unique())

    form_data = []
    for team in all_teams:
        games = []
        for _, row in df_sorted.iterrows():
            if row['home_team'] == team:
                # Home team: spread_result_difference is already from home perspective
                games.append(row['spread_result_difference'])
            elif row['away_team'] == team:
                # Away team: flip the sign
                games.append(-row['spread_result_difference'])

        if len(games) >= 5:
            last_5_avg = sum(games[-5:]) / 5
            form_data.append({
                'team': team,
                'tier': team_tier_map.get(team, 'Unknown'),
                'last_5_avg_spread': last_5_avg,
                'games_count': len(games),
                'is_good_form': last_5_avg > GOOD_FORM_THRESHOLD
            })

    return pd.DataFrame(form_data)


def get_todays_games_from_api(api_key: str, sport_key: str) -> Tuple[List[Dict], str]:
    """
    Fetch today's games from Odds API

    Returns: (list of games, today's date string)
    """
    import requests

    api_sport_key = SPORTS_CONFIG[sport_key]['api_key']
    est = timezone(timedelta(hours=-5))
    today_est = datetime.now(est).date()
    today_date_str = today_est.isoformat()

    url = f"https://api.the-odds-api.com/v4/sports/{api_sport_key}/odds"
    params = {
        'apiKey': api_key,
        'regions': ','.join(DEFAULT_REGIONS),
        'markets': 'spreads',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        all_games = response.json()

        # Filter to today's games
        today_games = []
        for game in all_games:
            commence_time = game.get('commence_time')
            if commence_time:
                try:
                    event_time = pd.to_datetime(commence_time)
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    event_time_est = event_time.astimezone(est)

                    if event_time_est.date() == today_est:
                        today_games.append(game)
                except:
                    pass

        return today_games, today_date_str

    except Exception as e:
        print(f"  ⚠ Error fetching games from API: {e}")
        return [], today_date_str


def extract_spread_from_game(game: Dict) -> Tuple[Optional[float], Optional[str]]:
    """Extract current spread from game data (DraftKings preferred)"""
    home_team = game.get('home_team', '')
    bookmakers = game.get('bookmakers', [])

    for bookmaker in bookmakers:
        if 'draftkings' in bookmaker.get('key', '').lower():
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'spreads':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        if outcome.get('name') == home_team:
                            return outcome.get('point'), 'DraftKings'
                    # Fallback to first outcome if home team not found
                    if outcomes:
                        return outcomes[0].get('point'), 'DraftKings'

    # Fallback to any bookmaker
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market.get('key') == 'spreads':
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name') == home_team:
                        return outcome.get('point'), bookmaker.get('title', 'Unknown')
                if outcomes:
                    return outcomes[0].get('point'), bookmaker.get('title', 'Unknown')

    return None, None


def format_game_time(commence_time: str) -> str:
    """Format commence time to readable EST string"""
    try:
        est = timezone(timedelta(hours=-5))
        event_time = pd.to_datetime(commence_time)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        event_time_est = event_time.astimezone(est)
        return event_time_est.strftime('%I:%M %p EST')
    except:
        return 'TBD'


def find_elite_teams_with_games_today(
    df_standings: pd.DataFrame,
    df_form: pd.DataFrame,
    today_games: List[Dict]
) -> List[Dict]:
    """
    Find elite teams in good form that have games today

    Returns list of opportunities with game details
    """
    # Get elite teams in good form
    elite_good_form = df_form[
        (df_form['tier'] == 'Elite') &
        (df_form['is_good_form'] == True)
    ]['team'].tolist()

    if not elite_good_form:
        return []

    opportunities = []

    for game in today_games:
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        # Check if either team is elite + good form
        elite_teams_in_game = []

        if home_team in elite_good_form:
            elite_teams_in_game.append(('home', home_team))
        if away_team in elite_good_form:
            elite_teams_in_game.append(('away', away_team))

        if not elite_teams_in_game:
            continue

        # Get spread and game details
        spread, bookmaker = extract_spread_from_game(game)
        game_time = format_game_time(game.get('commence_time', ''))

        # Get team stats
        for position, elite_team in elite_teams_in_game:
            opponent = away_team if position == 'home' else home_team

            # Get standings info
            team_standings = df_standings[df_standings['team'] == elite_team].iloc[0] if len(df_standings[df_standings['team'] == elite_team]) > 0 else None
            opp_standings = df_standings[df_standings['team'] == opponent].iloc[0] if len(df_standings[df_standings['team'] == opponent]) > 0 else None

            # Get form info
            team_form = df_form[df_form['team'] == elite_team].iloc[0] if len(df_form[df_form['team'] == elite_team]) > 0 else None

            opportunities.append({
                'elite_team': elite_team,
                'elite_team_position': position,
                'opponent': opponent,
                'opponent_tier': opp_standings['tier'] if opp_standings is not None else 'Unknown',
                'home_team': home_team,
                'away_team': away_team,
                'game_time': game_time,
                'spread': spread,
                'spread_source': bookmaker,
                'elite_team_win_pct': round(team_standings['win_pct'] * 100, 1) if team_standings is not None else None,
                'elite_team_point_diff': round(team_standings['point_diff_avg'], 1) if team_standings is not None else None,
                'elite_team_last_5_spread': round(team_form['last_5_avg_spread'], 1) if team_form is not None else None,
                'opponent_win_pct': round(opp_standings['win_pct'] * 100, 1) if opp_standings is not None else None
            })

    # Sort by elite team's recent form (best form first)
    opportunities.sort(key=lambda x: x.get('elite_team_last_5_spread', 0) or 0, reverse=True)

    return opportunities


def get_api_key() -> Optional[str]:
    """Get API key from environment or AWS Secrets Manager"""
    # Try environment variable first
    api_key = os.environ.get('ODDS_API_KEY')
    if api_key:
        return api_key

    # Try AWS Secrets Manager
    if AWS_AVAILABLE:
        try:
            secrets_client = boto3.client('secretsmanager')
            response = secrets_client.get_secret_value(SecretId='odds-api-key')
            return response['SecretString']
        except Exception as e:
            print(f"  ⚠ Could not get API key from Secrets Manager: {e}")

    return None


def process_sport(sport_key: str, use_local: bool = False, api_key: str = None) -> Dict[str, Any]:
    """
    Process a single sport and return elite team opportunities
    """
    config = SPORTS_CONFIG[sport_key]
    print(f"\n{'='*60}")
    print(f"Processing {config['name']}")
    print(f"{'='*60}")

    # Load historical data
    print("\n📊 Loading historical data...")
    if use_local:
        df_historical = load_historical_data_local(sport_key)
    else:
        df_historical = load_historical_data_s3(sport_key)
        if len(df_historical) == 0:
            print("  Falling back to local data...")
            df_historical = load_historical_data_local(sport_key)

    if len(df_historical) == 0:
        return {'sport': sport_key, 'error': 'No historical data available'}

    # Filter to completed games
    df_completed = df_historical[
        (df_historical['home_score'].notna()) &
        (df_historical['away_score'].notna()) &
        (df_historical['closing_spread'].notna()) &
        (df_historical['spread_result_difference'].notna())
    ].copy()

    print(f"  ✓ {len(df_completed)} completed games with spread data")

    # Calculate standings
    print("\n📈 Calculating team standings...")
    df_standings = calculate_team_standings(df_completed)
    elite_teams = df_standings[df_standings['tier'] == 'Elite']
    print(f"  ✓ {len(elite_teams)} elite teams (top 25% by win %)")

    # Calculate recent form
    print("\n🔥 Calculating recent form...")
    team_tier_map = df_standings.set_index('team')['tier'].to_dict()
    df_form = calculate_recent_form(df_completed, team_tier_map)

    elite_good_form = df_form[(df_form['tier'] == 'Elite') & (df_form['is_good_form'] == True)]
    print(f"  ✓ {len(elite_good_form)} elite teams in good form (last 5 avg > {GOOD_FORM_THRESHOLD})")

    if len(elite_good_form) > 0:
        print("\n  Elite teams in good form:")
        for _, row in elite_good_form.sort_values('last_5_avg_spread', ascending=False).head(10).iterrows():
            print(f"    • {row['team']}: L5 avg {row['last_5_avg_spread']:+.1f}")

    # Get today's games
    print("\n📅 Fetching today's games...")
    if api_key:
        today_games, today_date = get_todays_games_from_api(api_key, sport_key)
        print(f"  ✓ {len(today_games)} games scheduled for {today_date}")
    else:
        print("  ⚠ No API key available, cannot fetch today's games")
        today_games, today_date = [], datetime.now().strftime('%Y-%m-%d')

    # Find opportunities
    print("\n🎯 Finding elite team opportunities...")
    opportunities = find_elite_teams_with_games_today(df_standings, df_form, today_games)
    print(f"  ✓ {len(opportunities)} opportunities found")

    return {
        'sport': sport_key,
        'sport_name': config['name'],
        'date': today_date,
        'total_games_today': len(today_games),
        'elite_teams_count': len(elite_teams),
        'elite_good_form_count': len(elite_good_form),
        'opportunities': opportunities,
        'elite_teams': elite_good_form[['team', 'last_5_avg_spread', 'games_count']].to_dict('records')
    }


def print_opportunities(results: Dict[str, Any]):
    """Pretty print the opportunities"""
    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")
        return

    opps = results.get('opportunities', [])
    if not opps:
        print(f"\n📭 No elite team opportunities for {results['sport_name']} today")
        return

    print(f"\n{'='*70}")
    print(f"🏆 {results['sport_name']} ELITE TEAM OPPORTUNITIES ({results['date']})")
    print(f"{'='*70}")
    print(f"Total games today: {results['total_games_today']}")
    print(f"Elite teams in good form: {results['elite_good_form_count']}")
    print(f"Opportunities found: {len(opps)}")
    print()

    for i, opp in enumerate(opps, 1):
        print(f"{'─'*70}")
        print(f"#{i} | {opp['game_time']}")
        print(f"   {opp['away_team']} @ {opp['home_team']}")
        if opp['spread'] is not None:
            spread_display = f"+{opp['spread']}" if opp['spread'] > 0 else str(opp['spread'])
            print(f"   Spread: {opp['home_team']} {spread_display} ({opp['spread_source']})")

        print(f"\n   ⭐ ELITE TEAM: {opp['elite_team']} ({opp['elite_team_position'].upper()})")
        print(f"      Win %: {opp['elite_team_win_pct']}%")
        print(f"      Avg Point Diff: {opp['elite_team_point_diff']:+.1f}")
        print(f"      Last 5 Spread Avg: {opp['elite_team_last_5_spread']:+.1f} ← GOOD FORM")

        print(f"\n   vs {opp['opponent']} ({opp['opponent_tier']})")
        if opp['opponent_win_pct']:
            print(f"      Win %: {opp['opponent_win_pct']}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Find elite teams in good form with games today'
    )
    parser.add_argument(
        '--sport',
        choices=['nfl', 'nba', 'ncaam'],
        help='Specific sport to process (default: all)'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local Excel files instead of S3'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of formatted text'
    )

    args = parser.parse_args()

    # Get API key
    print("🔑 Getting API key...")
    api_key = get_api_key()
    if api_key:
        print("  ✓ API key available")
    else:
        print("  ⚠ No API key found - will skip fetching today's games")

    # Process sports
    sports = [args.sport] if args.sport else list(SPORTS_CONFIG.keys())
    all_results = []

    for sport in sports:
        results = process_sport(sport, use_local=args.local, api_key=api_key)
        all_results.append(results)

        if not args.json:
            print_opportunities(results)

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))

    # Summary
    if not args.json and len(sports) > 1:
        print(f"\n{'='*70}")
        print("📊 SUMMARY")
        print(f"{'='*70}")
        total_opps = sum(len(r.get('opportunities', [])) for r in all_results)
        print(f"Total opportunities across all sports: {total_opps}")


if __name__ == "__main__":
    main()
