"""
Lambda Function: Generate Predictions
This function generates betting opportunity predictions for today's games.

Triggered by: EventBridge schedule (daily at 6:30 AM EST, after data collection)
"""

import json
import boto3
import time
import pandas as pd
import io
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import logging

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'
SECRET_NAME = 'odds-api-key'
REGION = 'us-east-1'

# Sports and their handicap values (from notebooks)
SPORTS_CONFIG = {
    'nfl': {
        'api_key': 'americanfootball_nfl',
        'handicap_points': 5,
        'name': 'NFL',
        'focus_team': 'away'  # Focus on away teams
    },
    'nba': {
        'api_key': 'basketball_nba',
        'handicap_points': 9,
        'name': 'NBA',
        'focus_team': 'home'  # Focus on home teams
    },
    'ncaam': {
        'api_key': 'basketball_ncaab',
        'handicap_points': 10,
        'name': 'NCAAM',
        'focus_team': 'away'  # Focus on away teams
    }
}

DEFAULT_REGIONS = ['us']
API_RATE_LIMIT_DELAY = 1.0


def get_api_key() -> str:
    """Get Odds API key from AWS Secrets Manager"""
    try:
        response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        return response['SecretString']
    except Exception as e:
        logger.error(f"Error getting API key from Secrets Manager: {e}")
        raise


def read_excel_from_s3(sport_key: str) -> pd.DataFrame:
    """Read Excel file from S3"""
    s3_key = f"data/results/{sport_key}_season_results.xlsx"
    
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        excel_data = response['Body'].read()
        df = pd.read_excel(io.BytesIO(excel_data))
        logger.info(f"Read {len(df)} rows from S3: {s3_key}")
        return df
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"No existing file found in S3: {s3_key}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading Excel from S3: {e}")
        return pd.DataFrame()


def write_json_to_s3(data: Dict[str, Any], sport_key: str, date: str = None):
    """
    Write predictions JSON to S3
    
    Args:
        data: Predictions data dictionary
        sport_key: Sport key (nfl, nba, ncaam)
        date: Optional date in YYYY-MM-DD format. If provided, saves as predictions_{sport}_{date}.json
              Also saves as predictions_{sport}.json (default) for today's access
    """
    # Always save as default (for today's access)
    default_key = f"predictions/predictions_{sport_key}.json"
    
    try:
        json_data = json.dumps(data, indent=2, default=str)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=default_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Wrote predictions to S3: {default_key}")
        
        # Also save with date if provided (for historical access)
        if date:
            dated_key = f"predictions/predictions_{sport_key}_{date}.json"
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=dated_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            logger.info(f"Wrote dated predictions to S3: {dated_key}")
    except Exception as e:
        logger.error(f"Error writing JSON to S3: {e}")
        raise


def make_odds_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Any:
    """Make request to Odds API"""
    import requests
    
    url = f"https://api.the-odds-api.com/v4/{endpoint}"
    params['apiKey'] = api_key
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error calling Odds API: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error calling Odds API: {e}")
        raise


def parse_commence_time_to_est(commence_time: str):
    """Parse commence_time from API and convert to EST datetime"""
    try:
        event_time_utc = pd.to_datetime(commence_time)
        if event_time_utc.tzinfo is None:
            event_time_utc = event_time_utc.replace(tzinfo=timezone.utc)
        elif isinstance(event_time_utc, pd.Timestamp):
            if event_time_utc.tz is None:
                event_time_utc = event_time_utc.tz_localize('UTC').to_pydatetime()
            else:
                event_time_utc = event_time_utc.to_pydatetime()
        
        # Convert to EST
        est = timezone(timedelta(hours=-5))
        event_time_est = event_time_utc.astimezone(est)
        return event_time_est
    except Exception as e:
        logger.error(f"Error parsing commence_time: {e}")
        return None


def calculate_team_statistics(df_completed: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team spread coverage statistics with trend analysis
    
    Includes:
    - Overall coverage percentage
    - Average cover margin (when covering)
    - Average failure margin (when not covering)
    - Last 5 games trend (coverage % and improvement indicator)
    """
    team_stats = []
    all_teams = set(df_completed['home_team'].unique()) | set(df_completed['away_team'].unique())
    
    # Sort by date for trend analysis
    df_sorted = df_completed.sort_values('game_date').copy()
    
    for team in all_teams:
        # Games where team was home
        home_games = df_sorted[df_sorted['home_team'] == team].copy()
        # Games where team was away
        away_games = df_sorted[df_sorted['away_team'] == team].copy()
        
        # Combine all games for this team, sorted by date
        team_games_list = []
        for _, row in home_games.iterrows():
            team_games_list.append({
                'game_date': row['game_date'],
                'covered': row['spread_result_difference'] > 0,
                'spread_result_difference': row['spread_result_difference']
            })
        for _, row in away_games.iterrows():
            team_games_list.append({
                'game_date': row['game_date'],
                'covered': row['spread_result_difference'] < 0,
                'spread_result_difference': -row['spread_result_difference']  # Flip for away
            })
        
        team_games_df = pd.DataFrame(team_games_list)
        team_games_df = team_games_df.sort_values('game_date')
        
        total_games = len(team_games_df)
        if total_games == 0:
            continue
        
        # Overall statistics
        total_covers = team_games_df['covered'].sum()
        cover_pct = (total_covers / total_games) * 100 if total_games > 0 else 0
        
        # Average cover margin (when they covered)
        cover_games = team_games_df[team_games_df['covered']]
        avg_cover_margin = cover_games['spread_result_difference'].mean() if len(cover_games) > 0 else 0
        
        # Average failure margin (when they didn't cover)
        non_cover_games = team_games_df[~team_games_df['covered']]
        avg_failure_margin = non_cover_games['spread_result_difference'].mean() if len(non_cover_games) > 0 else 0
        
        # Trend: Last 5 games vs whole season
        last_5_coverage_pct = None
        trend_indicator = 'stable'  # 'improving', 'declining', 'stable'
        coverage_improvement_pct = 0
        
        if len(team_games_df) >= 5:
            last_5_games = team_games_df.tail(5)
            last_5_covers = last_5_games['covered'].sum()
            last_5_coverage_pct = (last_5_covers / len(last_5_games)) * 100
            
            # Calculate improvement (last 5 games vs whole season)
            coverage_improvement_pct = last_5_coverage_pct - cover_pct
            
            # Determine trend indicator (10% threshold for improving)
            if coverage_improvement_pct >= 10:
                trend_indicator = 'improving'
            elif coverage_improvement_pct <= -10:
                trend_indicator = 'declining'
            else:
                trend_indicator = 'stable'
        elif len(team_games_df) > 0:
            # If less than 5 games, use all games as "recent"
            last_5_coverage_pct = cover_pct
            trend_indicator = 'stable'
        
        # For home/away breakdown (keeping existing logic)
        home_covers = (home_games['spread_result_difference'] > 0).sum() if len(home_games) > 0 else 0
        away_covers = (away_games['spread_result_difference'] < 0).sum() if len(away_games) > 0 else 0
        avg_spread_diff = (
            (home_games['spread_result_difference'].sum() if len(home_games) > 0 else 0) +
            (away_games['spread_result_difference'].sum() * -1 if len(away_games) > 0 else 0)
        ) / total_games if total_games > 0 else 0
        
        team_stats.append({
            'team': team,
            'total_games': total_games,
            'covers': total_covers,
            'non_covers': total_games - total_covers,
            'cover_pct': cover_pct,
            'avg_spread_diff': avg_spread_diff,
            'avg_cover_margin': avg_cover_margin,
            'avg_failure_margin': avg_failure_margin,
            'last_5_coverage_pct': last_5_coverage_pct,
            'trend_indicator': trend_indicator,
            'coverage_improvement_pct': coverage_improvement_pct,
            'home_games': len(home_games),
            'away_games': len(away_games)
        })
    
    df_team_stats = pd.DataFrame(team_stats)
    df_team_stats = df_team_stats.sort_values('cover_pct', ascending=False)
    
    return df_team_stats


def calculate_handicap_statistics(df_completed: pd.DataFrame, handicap_points: int) -> pd.DataFrame:
    """
    Calculate team spread coverage with handicap
    
    Same logic as notebooks - adjusts spread by handicap points
    """
    team_stats_handicap = []
    all_teams = set(df_completed['home_team'].unique()) | set(df_completed['away_team'].unique())
    
    for team in all_teams:
        # Games where team was home
        home_games = df_completed[df_completed['home_team'] == team].copy()
        # Games where team was away
        away_games = df_completed[df_completed['away_team'] == team].copy()
        
        # For home games: adjust spread by +handicap (making it easier to cover)
        home_covers_handicap = 0
        if len(home_games) > 0:
            adjusted_spread_result = home_games['spread_result_difference'] + handicap_points
            home_covers_handicap = (adjusted_spread_result > 0).sum()
        
        # For away games: adjust spread by -handicap (making it easier to cover)
        away_covers_handicap = 0
        if len(away_games) > 0:
            adjusted_spread_result = away_games['spread_result_difference'] - handicap_points
            away_covers_handicap = (adjusted_spread_result < 0).sum()
        
        total_games = len(home_games) + len(away_games)
        total_covers_handicap = home_covers_handicap + away_covers_handicap
        
        if total_games > 0:
            cover_pct_handicap = (total_covers_handicap / total_games) * 100
            
            team_stats_handicap.append({
                'team': team,
                'total_games': total_games,
                'covers_handicap': total_covers_handicap,
                'cover_pct_handicap': cover_pct_handicap
            })
    
    df_team_stats_handicap = pd.DataFrame(team_stats_handicap)
    df_team_stats_handicap = df_team_stats_handicap.sort_values('cover_pct_handicap', ascending=False)
    
    return df_team_stats_handicap


def get_team_opponents(team_name: str, df_data: pd.DataFrame) -> List[str]:
    """Get all opponents a team has played"""
    home_games = df_data[df_data['home_team'] == team_name]
    away_games = df_data[df_data['away_team'] == team_name]
    
    opponents = set()
    opponents.update(home_games['away_team'].unique())
    opponents.update(away_games['home_team'].unique())
    
    return sorted(list(opponents))


def find_common_opponents(team_a: str, team_b: str, df_data: pd.DataFrame) -> List[str]:
    """Find common opponents between two teams"""
    opponents_a = set(get_team_opponents(team_a, df_data))
    opponents_b = set(get_team_opponents(team_b, df_data))
    return sorted(list(opponents_a & opponents_b))


def get_team_performance_vs_opponent(team_name: str, opponent_name: str, df_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Get team's performance against a specific opponent"""
    games = df_data[
        ((df_data['home_team'] == team_name) & (df_data['away_team'] == opponent_name)) |
        ((df_data['away_team'] == team_name) & (df_data['home_team'] == opponent_name))
    ].copy()
    
    if len(games) == 0:
        return None
    
    results = []
    for _, game in games.iterrows():
        was_home = game['home_team'] == team_name
        
        if was_home:
            team_score = game['home_score']
            opp_score = game['away_score']
            spread = game['closing_spread']
            spread_result = game['spread_result_difference']
        else:
            team_score = game['away_score']
            opp_score = game['home_score']
            spread = -game['closing_spread']
            spread_result = -game['spread_result_difference']
        
        won = team_score > opp_score
        covered = spread_result > 0
        push = spread_result == 0
        
        results.append({
            'date': game['game_date'],
            'was_home': was_home,
            'team_score': team_score,
            'opp_score': opp_score,
            'spread': spread,
            'spread_result': spread_result,
            'won': won,
            'covered': covered,
            'push': push,
            'margin': team_score - opp_score
        })
    
    return pd.DataFrame(results)


def compare_teams_vs_common_opponents(team_a: str, team_b: str, df_data: pd.DataFrame) -> tuple:
    """Compare two teams' performance against common opponents"""
    common_opponents = find_common_opponents(team_a, team_b, df_data)
    
    if len(common_opponents) == 0:
        return None, None
    
    comparison_data = []
    
    for opponent in common_opponents:
        team_a_games = get_team_performance_vs_opponent(team_a, opponent, df_data)
        team_b_games = get_team_performance_vs_opponent(team_b, opponent, df_data)
        
        if team_a_games is None or team_b_games is None or len(team_a_games) == 0 or len(team_b_games) == 0:
            continue
        
        # Team A stats
        team_a_wins = team_a_games['won'].sum()
        team_a_covers = team_a_games['covered'].sum()
        team_a_pushes = team_a_games['push'].sum()
        team_a_total = len(team_a_games)
        team_a_home_games = team_a_games['was_home'].sum()
        team_a_away_games = team_a_total - team_a_home_games
        team_a_win_pct = (team_a_wins / team_a_total * 100) if team_a_total > 0 else 0
        team_a_cover_pct = (team_a_covers / (team_a_total - team_a_pushes) * 100) if (team_a_total - team_a_pushes) > 0 else 0
        team_a_avg_margin = team_a_games['margin'].mean()
        
        # Calculate adjusted margin for Team A (home = -5, away = +5)
        team_a_adjusted_margins = team_a_games['margin'].copy()
        team_a_adjusted_margins = team_a_adjusted_margins + team_a_games['was_home'].map({True: -5, False: +5})
        team_a_avg_adjusted_margin = team_a_adjusted_margins.mean()
        
        # Team B stats
        team_b_wins = team_b_games['won'].sum()
        team_b_covers = team_b_games['covered'].sum()
        team_b_pushes = team_b_games['push'].sum()
        team_b_total = len(team_b_games)
        team_b_home_games = team_b_games['was_home'].sum()
        team_b_away_games = team_b_total - team_b_home_games
        team_b_win_pct = (team_b_wins / team_b_total * 100) if team_b_total > 0 else 0
        team_b_cover_pct = (team_b_covers / (team_b_total - team_b_pushes) * 100) if (team_b_total - team_b_pushes) > 0 else 0
        team_b_avg_margin = team_b_games['margin'].mean()
        
        # Calculate adjusted margin for Team B (home = -5, away = +5)
        team_b_adjusted_margins = team_b_games['margin'].copy()
        team_b_adjusted_margins = team_b_adjusted_margins + team_b_games['was_home'].map({True: -5, False: +5})
        team_b_avg_adjusted_margin = team_b_adjusted_margins.mean()
        
        # Calculate differentials
        win_pct_diff = team_a_win_pct - team_b_win_pct
        cover_pct_diff = team_a_cover_pct - team_b_cover_pct
        margin_diff = team_a_avg_margin - team_b_avg_margin
        adjusted_margin_diff = team_a_avg_adjusted_margin - team_b_avg_adjusted_margin  # Home/away adjusted
        
        comparison_data.append({
            'opponent': opponent,
            'team_a_games': team_a_total,
            'team_b_games': team_b_total,
            'team_a_home': team_a_home_games,
            'team_a_away': team_a_away_games,
            'team_b_home': team_b_home_games,
            'team_b_away': team_b_away_games,
            'team_a_win_pct': round(team_a_win_pct, 1),
            'team_b_win_pct': round(team_b_win_pct, 1),
            'win_pct_diff': round(win_pct_diff, 1),
            'team_a_cover_pct': round(team_a_cover_pct, 1),
            'team_b_cover_pct': round(team_b_cover_pct, 1),
            'cover_pct_diff': round(cover_pct_diff, 1),
            'team_a_avg_margin': round(team_a_avg_margin, 1),
            'team_b_avg_margin': round(team_b_avg_margin, 1),
            'margin_diff': round(margin_diff, 1),
            'team_a_avg_adjusted_margin': round(team_a_avg_adjusted_margin, 1),
            'team_b_avg_adjusted_margin': round(team_b_avg_adjusted_margin, 1),
            'adjusted_margin_diff': round(adjusted_margin_diff, 1)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate overall summary
    if len(comparison_df) > 0:
        summary = {
            'common_opponents_count': len(common_opponents),
            'avg_win_pct_diff': round(comparison_df['win_pct_diff'].mean(), 1),
            'avg_cover_pct_diff': round(comparison_df['cover_pct_diff'].mean(), 1),
            'avg_margin_diff': round(comparison_df['margin_diff'].mean(), 1),
            'avg_adjusted_margin_diff': round(comparison_df['adjusted_margin_diff'].mean(), 1),  # Home/away adjusted
            'team_a_better_win_pct': (comparison_df['win_pct_diff'] > 0).sum(),
            'team_b_better_win_pct': (comparison_df['win_pct_diff'] < 0).sum(),
            'team_a_better_cover_pct': (comparison_df['cover_pct_diff'] > 0).sum(),
            'team_b_better_cover_pct': (comparison_df['cover_pct_diff'] < 0).sum()
        }
    else:
        summary = None
    
    return comparison_df, summary


def generate_common_opponent_predictions(df_completed: pd.DataFrame, today_games_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate common opponent based predictions for today's games
    
    Only works for games where teams have common opponents
    """
    predictions = []
    
    for _, game in today_games_df.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        if not home_team or not away_team:
            continue
        
        # Find common opponents and compare
        comparison_df, summary = compare_teams_vs_common_opponents(home_team, away_team, df_completed)
        
        if summary is None or summary['common_opponents_count'] == 0:
            # No common opponents - skip this game
            continue
        
        # Calculate projections based on adjusted margins
        projected_margin_neutral = summary['avg_adjusted_margin_diff']  # From home team's perspective
        
        # Calculate home/away scenarios (using home team perspective)
        # If home team is home: their margin increases by 5
        margin_if_home_home = projected_margin_neutral + 5
        # If away team is home (home team is away): home team's margin decreases by 5
        margin_if_away_home = projected_margin_neutral - 5
        
        # Determine winners
        if projected_margin_neutral > 0:
            winner_neutral = home_team
            margin_display_neutral = f"{home_team} by {abs(projected_margin_neutral):.1f}"
        elif projected_margin_neutral < 0:
            winner_neutral = away_team
            margin_display_neutral = f"{away_team} by {abs(projected_margin_neutral):.1f}"
        else:
            winner_neutral = "Tie"
            margin_display_neutral = "Tie"
        
        if margin_if_home_home > 0:
            winner_if_home_home = home_team
            margin_display_home_home = f"{home_team} by {abs(margin_if_home_home):.1f}"
        elif margin_if_home_home < 0:
            winner_if_home_home = away_team
            margin_display_home_home = f"{away_team} by {abs(margin_if_home_home):.1f}"
        else:
            winner_if_home_home = "Tie"
            margin_display_home_home = "Tie"
        
        if margin_if_away_home > 0:
            winner_if_away_home = home_team
            margin_display_away_home = f"{home_team} by {abs(margin_if_away_home):.1f}"
        elif margin_if_away_home < 0:
            winner_if_away_home = away_team
            margin_display_away_home = f"{away_team} by {abs(margin_if_away_home):.1f}"
        else:
            winner_if_away_home = "Tie"
            margin_display_away_home = "Tie"
        
        # Calculate win probabilities
        base_prob = 50.0
        win_prob_adjustment = summary['avg_win_pct_diff'] * 0.5
        home_win_prob = base_prob + win_prob_adjustment
        away_win_prob = 100 - home_win_prob
        
        # ATS projection
        home_cover_prob = 50.0 + (summary['avg_cover_pct_diff'] * 0.5)
        away_cover_prob = 100 - home_cover_prob
        
        # Confidence level
        if summary['common_opponents_count'] >= 5:
            confidence = "High"
        elif summary['common_opponents_count'] >= 3:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        prediction = {
            'game_time_est': game.get('game_time_est'),
            'home_team': home_team,
            'away_team': away_team,
            'current_spread': float(game.get('current_spread')) if pd.notna(game.get('current_spread')) else None,
            'common_opponents_count': summary['common_opponents_count'],
            'common_opponents': find_common_opponents(home_team, away_team, df_completed),
            'projections': {
                'neutral_court': {
                    'winner': winner_neutral,
                    'margin': margin_display_neutral,
                    'projected_margin': round(projected_margin_neutral, 2)
                },
                'actual_game': {
                    # Actual game: home_team is home, so use home_team_home scenario
                    'winner': winner_if_home_home,
                    'margin': margin_display_home_home,
                    'projected_margin': round(margin_if_home_home, 2),
                    'note': f'{home_team} is home for this game'
                },
                'if_away_team_home': {
                    # Hypothetical: what if the game was at away team's court
                    'winner': winner_if_away_home,
                    'margin': margin_display_away_home,
                    'projected_margin': round(margin_if_away_home, 2),
                    'note': f'Hypothetical: {away_team} would be home'
                }
            },
            'win_probabilities': {
                'home_team': round(home_win_prob, 1),
                'away_team': round(away_win_prob, 1)
            },
            'ats_probabilities': {
                'home_team_cover': round(home_cover_prob, 1),
                'away_team_cover': round(away_cover_prob, 1)
            },
            'confidence': confidence,
            'statistics': {
                'avg_win_pct_diff': summary['avg_win_pct_diff'],
                'avg_cover_pct_diff': summary['avg_cover_pct_diff'],
                'avg_margin_diff': summary['avg_margin_diff'],
                'avg_adjusted_margin_diff': summary['avg_adjusted_margin_diff']
            }
        }
        
        predictions.append(prediction)
    
    return predictions


def fetch_todays_games(api_key: str, sport_key: str) -> tuple:
    """
    Fetch today's games from Odds API
    
    Same logic as notebooks - filters to only games scheduled for today (EST)
    
    Returns:
        tuple: (list of games, date string in YYYY-MM-DD format)
    """
    api_sport_key = SPORTS_CONFIG[sport_key]['api_key']
    est = timezone(timedelta(hours=-5))
    today_est = datetime.now(est).date()
    today_date_str = today_est.isoformat()  # YYYY-MM-DD format
    
    logger.info(f"Fetching today's {sport_key.upper()} games for {today_date_str}...")
    
    try:
        time.sleep(API_RATE_LIMIT_DELAY)
        upcoming_odds = make_odds_api_request(
            endpoint=f"sports/{api_sport_key}/odds",
            params={
                "regions": ",".join(DEFAULT_REGIONS),
                "markets": "spreads",
                "oddsFormat": "american",
                "dateFormat": "iso"
            },
            api_key=api_key
        )
        
        # Filter to only games happening today (using EST timezone)
        today_games = []
        if isinstance(upcoming_odds, list):
            for game in upcoming_odds:
                if not isinstance(game, dict):
                    continue
                
                commence_time = game.get('commence_time')
                if commence_time:
                    event_time_est = parse_commence_time_to_est(commence_time)
                    if event_time_est and event_time_est.date() == today_est:
                        today_games.append(game)
        
        logger.info(f"Found {len(today_games)} games scheduled for today ({today_date_str})")
        return today_games, today_date_str
        
    except Exception as e:
        logger.error(f"Error fetching today's games: {e}")
        return [], today_date_str


def process_todays_games(today_games: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process today's games into a DataFrame
    
    Extracts: event_id, game_time_est, home_team, away_team, current_spread
    """
    games_data = []
    
    for game in today_games:
        if not isinstance(game, dict):
            continue
        
        # Extract basic game info
        event_id = game.get('id', '')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')
        
        # Parse commence_time using helper function
        event_time_est = parse_commence_time_to_est(commence_time)
        game_time_str = event_time_est.strftime('%Y-%m-%d %H:%M:%S %Z') if event_time_est else None
        
        # Extract current spread from DraftKings
        current_spread = None
        spread_odds = None
        bookmakers = game.get('bookmakers', [])
        
        for bookmaker in bookmakers:
            if 'draftkings' in bookmaker.get('key', '').lower():
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'spreads':
                        outcomes = market.get('outcomes', [])
                        if len(outcomes) >= 2:
                            # Find home team outcome
                            for outcome in outcomes:
                                outcome_name = outcome.get('name', '')
                                if home_team.lower() in outcome_name.lower() or outcome_name.lower() in home_team.lower():
                                    current_spread = outcome.get('point')
                                    spread_odds = outcome.get('price')
                                    break
                            
                            # If not found, use first outcome
                            if current_spread is None:
                                current_spread = outcomes[0].get('point')
                                spread_odds = outcomes[0].get('price')
                        break
                break
        
        games_data.append({
            'event_id': event_id,
            'game_time_est': game_time_str,
            'home_team': home_team,
            'away_team': away_team,
            'current_spread': current_spread,
            'spread_odds': spread_odds
        })
    
    return pd.DataFrame(games_data)


def generate_predictions_for_sport(api_key: str, sport_key: str) -> Dict[str, Any]:
    """
    Generate predictions for one sport
    
    Returns a dictionary with predictions data
    """
    logger.info(f"Generating predictions for {sport_key.upper()}...")
    
    config = SPORTS_CONFIG[sport_key]
    handicap_points = config['handicap_points']
    focus_team = config['focus_team']  # 'away' or 'home'
    
    # Load historical data from S3
    df_historical = read_excel_from_s3(sport_key)
    
    if len(df_historical) == 0:
        logger.warning(f"No historical data for {sport_key.upper()}")
        return {
            'sport': sport_key,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'games': [],
            'opportunities': [],
            'summary': {
                'total_games': 0,
                'opportunities': 0
            }
        }
    
    # Filter to completed games with all data
    df_completed = df_historical[
        (df_historical['home_score'].notna()) & 
        (df_historical['away_score'].notna()) & 
        (df_historical['closing_spread'].notna()) &
        (df_historical['spread_result_difference'].notna())
    ].copy()
    
    logger.info(f"Processing {len(df_completed)} completed games")
    
    # Calculate team statistics
    df_team_stats = calculate_team_statistics(df_completed)
    logger.info(f"Calculated stats for {len(df_team_stats)} teams")
    
    # Calculate handicap statistics
    df_team_stats_handicap = calculate_handicap_statistics(df_completed, handicap_points)
    logger.info(f"Calculated {handicap_points}-point handicap stats for {len(df_team_stats_handicap)} teams")
    
    # Fetch today's games
    today_games, today_date_str = fetch_todays_games(api_key, sport_key)
    
    if len(today_games) == 0:
        logger.info(f"No games scheduled for today ({today_date_str})")
        return {
            'sport': sport_key,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'prediction_date': today_date_str,  # Add date for historical matching
            'games': [],
            'opportunities': [],
            'summary': {
                'total_games': 0,
                'opportunities': 0
            }
        }
    
    # Process today's games
    df_today = process_todays_games(today_games)
    
    # Merge today's games with team statistics
    games_with_stats = df_today.copy()
    
    # Merge home team standard stats
    games_with_stats = games_with_stats.merge(
        df_team_stats[['team', 'cover_pct', 'total_games']],
        left_on='home_team',
        right_on='team',
        how='left'
    )
    games_with_stats = games_with_stats.rename(columns={
        'cover_pct': 'home_cover_pct',
        'total_games': 'home_total_games'
    })
    games_with_stats = games_with_stats.drop(columns=['team'])
    
    # Merge home team handicap stats
    games_with_stats = games_with_stats.merge(
        df_team_stats_handicap[['team', 'cover_pct_handicap']],
        left_on='home_team',
        right_on='team',
        how='left'
    )
    games_with_stats = games_with_stats.rename(columns={'cover_pct_handicap': 'home_cover_pct_handicap'})
    games_with_stats = games_with_stats.drop(columns=['team'])
    
    # Merge away team standard stats
    games_with_stats = games_with_stats.merge(
        df_team_stats[['team', 'cover_pct', 'total_games']],
        left_on='away_team',
        right_on='team',
        how='left'
    )
    games_with_stats = games_with_stats.rename(columns={
        'cover_pct': 'away_cover_pct',
        'total_games': 'away_total_games'
    })
    games_with_stats = games_with_stats.drop(columns=['team'])
    
    # Merge away team handicap stats
    games_with_stats = games_with_stats.merge(
        df_team_stats_handicap[['team', 'cover_pct_handicap']],
        left_on='away_team',
        right_on='team',
        how='left'
    )
    games_with_stats = games_with_stats.rename(columns={'cover_pct_handicap': 'away_cover_pct_handicap'})
    games_with_stats = games_with_stats.drop(columns=['team'])
    
    # Merge team trend statistics (for coverage-based strategy)
    # Note: home_cover_pct and away_cover_pct already exist from earlier merge, so we use _full suffix
    games_with_stats = games_with_stats.merge(
        df_team_stats[['team', 'cover_pct', 'avg_cover_margin', 'avg_failure_margin', 
                       'last_5_coverage_pct', 'trend_indicator', 'coverage_improvement_pct']],
        left_on='home_team',
        right_on='team',
        how='left',
        suffixes=('', '_home_full')
    )
    games_with_stats = games_with_stats.rename(columns={
        'cover_pct': 'home_cover_pct_full',
        'avg_cover_margin': 'home_avg_cover_margin',
        'avg_failure_margin': 'home_avg_failure_margin',
        'last_5_coverage_pct': 'home_last_5_coverage_pct',
        'trend_indicator': 'home_trend_indicator',
        'coverage_improvement_pct': 'home_coverage_improvement_pct'
    })
    if 'team' in games_with_stats.columns:
        games_with_stats = games_with_stats.drop(columns=['team'])
    
    games_with_stats = games_with_stats.merge(
        df_team_stats[['team', 'cover_pct', 'avg_cover_margin', 'avg_failure_margin',
                       'last_5_coverage_pct', 'trend_indicator', 'coverage_improvement_pct']],
        left_on='away_team',
        right_on='team',
        how='left',
        suffixes=('', '_away_full')
    )
    games_with_stats = games_with_stats.rename(columns={
        'cover_pct': 'away_cover_pct_full',
        'avg_cover_margin': 'away_avg_cover_margin',
        'avg_failure_margin': 'away_avg_failure_margin',
        'last_5_coverage_pct': 'away_last_5_coverage_pct',
        'trend_indicator': 'away_trend_indicator',
        'coverage_improvement_pct': 'away_coverage_improvement_pct'
    })
    if 'team' in games_with_stats.columns:
        games_with_stats = games_with_stats.drop(columns=['team'])
    
    # Calculate handicap advantage based on focus team
    if focus_team == 'away':
        # Away team advantage (positive when away team has better coverage)
        games_with_stats['team_handicap_advantage'] = (
            games_with_stats['away_cover_pct_handicap'] - 
            games_with_stats['home_cover_pct_handicap']
        )
    else:  # focus_team == 'home'
        # Home team advantage (positive when home team has better coverage)
        games_with_stats['team_handicap_advantage'] = (
            games_with_stats['home_cover_pct_handicap'] - 
            games_with_stats['away_cover_pct_handicap']
        )
    
    # Filter for games with enough data (at least 5 games per team)
    games_with_enough_data = games_with_stats[
        (games_with_stats['home_total_games'] >= 5) & 
        (games_with_stats['away_total_games'] >= 5)
    ].copy()
    
    # Generate opportunities for each strategy
    
    # Strategy 1: Home Focus (bet on home team when they have better handicap coverage)
    home_focus_opportunities = games_with_enough_data[
        (games_with_enough_data['home_cover_pct_handicap'].notna()) &
        (games_with_enough_data['away_cover_pct_handicap'].notna()) &
        (games_with_enough_data['home_cover_pct_handicap'] > games_with_enough_data['away_cover_pct_handicap'])
    ].copy()
    if len(home_focus_opportunities) > 0:
        home_focus_opportunities['handicap_pct_difference'] = (
            home_focus_opportunities['home_cover_pct_handicap'] - 
            home_focus_opportunities['away_cover_pct_handicap']
        )
        home_focus_opportunities = home_focus_opportunities.sort_values('handicap_pct_difference', ascending=False)
    
    # Strategy 2: Away Focus (bet on away team when they have better handicap coverage)
    away_focus_opportunities = games_with_enough_data[
        (games_with_enough_data['home_cover_pct_handicap'].notna()) &
        (games_with_enough_data['away_cover_pct_handicap'].notna()) &
        (games_with_enough_data['away_cover_pct_handicap'] > games_with_enough_data['home_cover_pct_handicap'])
    ].copy()
    if len(away_focus_opportunities) > 0:
        away_focus_opportunities['handicap_pct_difference'] = (
            away_focus_opportunities['away_cover_pct_handicap'] - 
            away_focus_opportunities['home_cover_pct_handicap']
        )
        away_focus_opportunities = away_focus_opportunities.sort_values('handicap_pct_difference', ascending=False)
    
    # Strategy 3: Coverage Based (bet on team with better coverage %, difference > 10%)
    # Calculate coverage difference (absolute difference in coverage percentages)
    games_with_enough_data['coverage_pct_difference'] = (
        games_with_enough_data['home_cover_pct_full'] - 
        games_with_enough_data['away_cover_pct_full']
    ).abs()
    
    coverage_based_opportunities = games_with_enough_data[
        (games_with_enough_data['home_cover_pct_full'].notna()) &
        (games_with_enough_data['away_cover_pct_full'].notna()) &
        (games_with_enough_data['coverage_pct_difference'] > 10)  # Minimum 10% difference
    ].copy()
    
    if len(coverage_based_opportunities) > 0:
        # Determine which team has better coverage
        coverage_based_opportunities['better_team'] = coverage_based_opportunities.apply(
            lambda row: 'home' if row['home_cover_pct_full'] > row['away_cover_pct_full'] else 'away',
            axis=1
        )
        coverage_based_opportunities['coverage_pct_difference_absolute'] = coverage_based_opportunities['coverage_pct_difference']
        coverage_based_opportunities = coverage_based_opportunities.sort_values('coverage_pct_difference', ascending=False)
    
    # Keep original focus_team opportunities for backward compatibility
    if focus_team == 'away':
        opportunities = away_focus_opportunities
    else:  # focus_team == 'home'
        opportunities = home_focus_opportunities
    
    # Calculate difference for opportunities (backward compatibility)
    if len(opportunities) > 0:
        if focus_team == 'away':
            opportunities['handicap_pct_difference'] = (
                opportunities['away_cover_pct_handicap'] - 
                opportunities['home_cover_pct_handicap']
            )
        else:  # focus_team == 'home'
            opportunities['handicap_pct_difference'] = (
                opportunities['home_cover_pct_handicap'] - 
                opportunities['away_cover_pct_handicap']
            )
        opportunities = opportunities.sort_values('handicap_pct_difference', ascending=False)
    
    # Helper function to convert opportunities to JSON format
    def opportunities_to_json(opp_df, strategy_type):
        opp_list = []
        for _, row in opp_df.iterrows():
            opp_dict = {
                'game_time_est': row.get('game_time_est'),
                'away_team': row.get('away_team'),
                'home_team': row.get('home_team'),
                'current_spread': float(row.get('current_spread')) if pd.notna(row.get('current_spread')) else None,
            }
            
            if strategy_type in ['home_focus', 'away_focus']:
                # Handicap-based strategies
                opp_dict.update({
                    'away_cover_pct_handicap': float(row.get('away_cover_pct_handicap')) if pd.notna(row.get('away_cover_pct_handicap')) else None,
                    'home_cover_pct_handicap': float(row.get('home_cover_pct_handicap')) if pd.notna(row.get('home_cover_pct_handicap')) else None,
                    'handicap_pct_difference': float(row.get('handicap_pct_difference')) if pd.notna(row.get('handicap_pct_difference')) else None
                })
            elif strategy_type == 'coverage_based':
                # Coverage-based strategy
                opp_dict.update({
                    'better_team': row.get('better_team'),
                    'away_cover_pct': float(row.get('away_cover_pct_full')) if pd.notna(row.get('away_cover_pct_full')) else None,
                    'home_cover_pct': float(row.get('home_cover_pct_full')) if pd.notna(row.get('home_cover_pct_full')) else None,
                    'coverage_pct_difference': float(row.get('coverage_pct_difference_absolute')) if pd.notna(row.get('coverage_pct_difference_absolute')) else None,
                    'away_avg_cover_margin': float(row.get('away_avg_cover_margin')) if pd.notna(row.get('away_avg_cover_margin')) else None,
                    'home_avg_cover_margin': float(row.get('home_avg_cover_margin')) if pd.notna(row.get('home_avg_cover_margin')) else None,
                    'away_trend_indicator': row.get('away_trend_indicator'),
                    'home_trend_indicator': row.get('home_trend_indicator'),
                    'away_coverage_improvement_pct': float(row.get('away_coverage_improvement_pct')) if pd.notna(row.get('away_coverage_improvement_pct')) else None,
                    'home_coverage_improvement_pct': float(row.get('home_coverage_improvement_pct')) if pd.notna(row.get('home_coverage_improvement_pct')) else None,
                    'away_last_5_coverage_pct': float(row.get('away_last_5_coverage_pct')) if pd.notna(row.get('away_last_5_coverage_pct')) else None,
                    'home_last_5_coverage_pct': float(row.get('home_last_5_coverage_pct')) if pd.notna(row.get('home_last_5_coverage_pct')) else None
                })
            
            opp_list.append(opp_dict)
        return opp_list
    
    # Convert to JSON-serializable format
    games_list = []
    for _, row in games_with_stats.iterrows():
        games_list.append({
            'game_time_est': row.get('game_time_est'),
            'away_team': row.get('away_team'),
            'home_team': row.get('home_team'),
            'current_spread': float(row.get('current_spread')) if pd.notna(row.get('current_spread')) else None,
            'away_cover_pct': float(row.get('away_cover_pct')) if pd.notna(row.get('away_cover_pct')) else None,
            'home_cover_pct': float(row.get('home_cover_pct')) if pd.notna(row.get('home_cover_pct')) else None,
            'away_cover_pct_handicap': float(row.get('away_cover_pct_handicap')) if pd.notna(row.get('away_cover_pct_handicap')) else None,
            'home_cover_pct_handicap': float(row.get('home_cover_pct_handicap')) if pd.notna(row.get('home_cover_pct_handicap')) else None,
            'team_handicap_advantage': float(row.get('team_handicap_advantage')) if pd.notna(row.get('team_handicap_advantage')) else None,
            'focus_team': focus_team
        })
    
    # Generate opportunities for each strategy
    home_focus_list = opportunities_to_json(home_focus_opportunities, 'home_focus')
    away_focus_list = opportunities_to_json(away_focus_opportunities, 'away_focus')
    coverage_based_list = opportunities_to_json(coverage_based_opportunities, 'coverage_based')
    
    # Generate common opponent predictions (only for NCAAM)
    common_opponent_predictions = []
    if sport_key == 'ncaam' and len(df_today) > 0:
        logger.info("Generating common opponent predictions for NCAAM...")
        common_opponent_predictions = generate_common_opponent_predictions(df_completed, df_today)
        logger.info(f"Generated {len(common_opponent_predictions)} common opponent predictions")
    
    # Legacy opportunities list (for backward compatibility)
    opportunities_list = opportunities_to_json(opportunities, focus_team + '_focus')
    
    # Build strategies dictionary
    strategies = {
        'home_focus': {
            'opportunities': home_focus_list,
            'summary': {
                'opportunities': len(home_focus_list),
                'average_difference': float(home_focus_opportunities['handicap_pct_difference'].mean()) if len(home_focus_opportunities) > 0 else None,
                'largest_difference': float(home_focus_opportunities['handicap_pct_difference'].max()) if len(home_focus_opportunities) > 0 else None
            }
        },
        'away_focus': {
            'opportunities': away_focus_list,
            'summary': {
                'opportunities': len(away_focus_list),
                'average_difference': float(away_focus_opportunities['handicap_pct_difference'].mean()) if len(away_focus_opportunities) > 0 else None,
                'largest_difference': float(away_focus_opportunities['handicap_pct_difference'].max()) if len(away_focus_opportunities) > 0 else None
            }
        },
        'coverage_based': {
            'opportunities': coverage_based_list,
            'summary': {
                'opportunities': len(coverage_based_list),
                'average_coverage_difference': float(coverage_based_opportunities['coverage_pct_difference_absolute'].mean()) if len(coverage_based_opportunities) > 0 else None,
                'largest_coverage_difference': float(coverage_based_opportunities['coverage_pct_difference_absolute'].max()) if len(coverage_based_opportunities) > 0 else None
            }
        }
    }
    
    # Add common_opponent strategy (only for NCAAM)
    if sport_key == 'ncaam':
        strategies['common_opponent'] = {
            'opportunities': common_opponent_predictions,
            'summary': {
                'opportunities': len(common_opponent_predictions),
                'games_with_common_opponents': len(common_opponent_predictions),
                'games_without_common_opponents': len(df_today) - len(common_opponent_predictions) if len(df_today) > 0 else 0
            }
        }
    
    return {
        'sport': sport_key,
        'sport_name': config['name'],
        'handicap_points': handicap_points,
        'focus_team': focus_team,  # Legacy field
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'prediction_date': today_date_str,
        'games': games_list,
        'opportunities': opportunities_list,  # Legacy field (for backward compatibility)
        'strategies': strategies,
        'summary': {
            'total_games': len(games_with_stats),
            'games_with_sufficient_data': len(games_with_enough_data),
            'opportunities': len(opportunities),  # Legacy field
            'average_difference': float(opportunities['handicap_pct_difference'].mean()) if len(opportunities) > 0 else None,
            'largest_difference': float(opportunities['handicap_pct_difference'].max()) if len(opportunities) > 0 else None,
            'smallest_difference': float(opportunities['handicap_pct_difference'].min()) if len(opportunities) > 0 else None
        }
    }


def lambda_handler(event, context):
    """
    Lambda handler function - entry point AWS calls
    
    Generates predictions for all sports and saves to S3
    """
    logger.info("="*80)
    logger.info("Starting prediction generation")
    logger.info("="*80)
    
    try:
        # Get API key from Secrets Manager
        api_key = get_api_key()
        logger.info("✓ Retrieved API key from Secrets Manager")
        
        # Generate predictions for each sport
        all_predictions = {}
        results = []
        
        for sport_key in SPORTS_CONFIG.keys():
            try:
                logger.info(f"\nProcessing {sport_key.upper()}...")
                predictions = generate_predictions_for_sport(api_key, sport_key)
                
                # Get the prediction date from the predictions data
                prediction_date = predictions.get('prediction_date')
                
                # Save to S3 (both default and dated versions)
                write_json_to_s3(predictions, sport_key, prediction_date)
                
                all_predictions[sport_key] = predictions
                results.append({
                    'sport': sport_key,
                    'total_games': predictions['summary']['total_games'],
                    'opportunities': predictions['summary']['opportunities'],
                    'prediction_date': prediction_date
                })
                
                logger.info(f"✓ {sport_key.upper()}: {predictions['summary']['opportunities']} opportunities found for {prediction_date}")
                
            except Exception as e:
                logger.error(f"✗ Error processing {sport_key.upper()}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Return summary
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Predictions generated',
                'results': results,
                'generated_at': datetime.now(timezone.utc).isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Fatal error in Lambda function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }



