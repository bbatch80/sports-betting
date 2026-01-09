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

# Home/Away Focus strategy configuration
HOME_AWAY_HANDICAP = 11  # Fixed 11-point handicap for home_focus/away_focus strategies

# Elite team strategy configuration
ELITE_PERCENTILE = 0.75  # Top 25% by win percentage
GOOD_FORM_THRESHOLD = 3  # Last 5 games avg spread performance > 3
ELITE_HANDICAP_POINTS = 11  # 11-point handicap for elite team strategies

# Form-based strategy configuration (Hot vs Cold, Perfect Form)
FORM_HANDICAP_POINTS = 11   # 11-point handicap for form-based strategies
FORM_GAMES_LOOKBACK = 5     # Default lookback for perfect form strategy

# Hot vs Cold configuration - multiple lookback windows
HOT_COLD_CONFIG = {
    3: {'hot_min': 2, 'cold_max': 0},   # Hot: ≥2/3 (67%+), Cold: 0/3 (0%)
    5: {'hot_min': 3, 'cold_max': 2},   # Hot: ≥3/5 (60%+), Cold: ≤2/5 (40%)
    7: {'hot_min': 4, 'cold_max': 3},   # Hot: ≥4/7 (57%+), Cold: ≤3/7 (43%)
}

# Legacy thresholds (for backward compatibility in calculate_team_form_for_game)
HOT_FORM_THRESHOLD = 0.60   # 60% coverage = hot (3+ of last 5)
COLD_FORM_THRESHOLD = 0.40  # 40% coverage = cold (2 or less of last 5)


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

    # Classify tiers based on win percentage
    q75 = df_teams['win_pct'].quantile(ELITE_PERCENTILE)
    q25 = df_teams['win_pct'].quantile(1 - ELITE_PERCENTILE)

    df_teams['tier'] = df_teams['win_pct'].apply(
        lambda x: 'Elite' if x >= q75 else ('Bottom' if x <= q25 else 'Mid')
    )

    return df_teams


def calculate_team_standings_by_coverage(df_completed: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team standings based on spread coverage percentage.

    Returns DataFrame with columns:
    - team, games, covers, cover_pct, tier (Elite/Mid/Bottom)
    """
    all_teams = set(df_completed['home_team'].unique()) | set(df_completed['away_team'].unique())

    team_stats = []
    for team in all_teams:
        home_games = df_completed[df_completed['home_team'] == team]
        away_games = df_completed[df_completed['away_team'] == team]

        # Calculate spread covers
        home_covers = (home_games['spread_result_difference'] > 0).sum() if len(home_games) > 0 else 0
        away_covers = (away_games['spread_result_difference'] < 0).sum() if len(away_games) > 0 else 0

        total_games = len(home_games) + len(away_games)
        total_covers = home_covers + away_covers

        if total_games > 0:
            team_stats.append({
                'team': team,
                'games': total_games,
                'covers': total_covers,
                'cover_pct': total_covers / total_games
            })

    df_teams = pd.DataFrame(team_stats).sort_values('cover_pct', ascending=False)

    # Classify tiers based on coverage percentage
    q75 = df_teams['cover_pct'].quantile(ELITE_PERCENTILE)
    q25 = df_teams['cover_pct'].quantile(1 - ELITE_PERCENTILE)

    df_teams['tier'] = df_teams['cover_pct'].apply(
        lambda x: 'Elite' if x >= q75 else ('Bottom' if x <= q25 else 'Mid')
    )

    return df_teams


def calculate_recent_form(df_completed: pd.DataFrame, team_tier_map: Dict[str, str]) -> pd.DataFrame:
    """
    Calculate recent form (last 5 games spread performance) for each team

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


def generate_elite_team_opportunities(
    df_standings: pd.DataFrame,
    df_form: pd.DataFrame,
    today_games_df: pd.DataFrame,
    strategy_type: str = 'winpct',
    df_standings_winpct: pd.DataFrame = None
) -> List[Dict[str, Any]]:
    """
    Generate elite team opportunities - elite teams in good form with games today

    Args:
        df_standings: Standings DataFrame (used for elite tier determination)
        df_form: Form DataFrame with team tier and good form status
        today_games_df: Today's games
        strategy_type: 'winpct' or 'coverage' - determines elite ranking method
        df_standings_winpct: Win% standings for showing opponent win% (always passed for reference)

    Returns list of opportunities with game and team details
    """
    # Get elite teams in good form
    elite_good_form = df_form[
        (df_form['tier'] == 'Elite') &
        (df_form['is_good_form'] == True)
    ]['team'].tolist()

    if not elite_good_form:
        return []

    opportunities = []

    for _, game in today_games_df.iterrows():
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

        for position, elite_team in elite_teams_in_game:
            opponent = away_team if position == 'home' else home_team

            # Get standings info based on strategy type
            team_standings = df_standings[df_standings['team'] == elite_team]
            opp_standings = df_standings[df_standings['team'] == opponent]

            # Get form info
            team_form = df_form[df_form['team'] == elite_team]

            opp_tier = opp_standings.iloc[0]['tier'] if len(opp_standings) > 0 else 'Unknown'

            # Build opportunity dict based on strategy type
            opp_dict = {
                'game_time_est': game.get('game_time_est'),
                'home_team': home_team,
                'away_team': away_team,
                'current_spread': float(game.get('current_spread')) if pd.notna(game.get('current_spread')) else None,
                'elite_team': elite_team,
                'elite_team_position': position,
                'opponent': opponent,
                'opponent_tier': opp_tier,
                'elite_team_last_5_spread': float(team_form.iloc[0]['last_5_avg_spread']) if len(team_form) > 0 else None,
                'handicap_points': ELITE_HANDICAP_POINTS,
                'strategy_type': strategy_type
            }

            if strategy_type == 'winpct':
                opp_dict['elite_team_win_pct'] = float(team_standings.iloc[0]['win_pct'] * 100) if len(team_standings) > 0 else None
                opp_dict['elite_team_point_diff'] = float(team_standings.iloc[0]['point_diff_avg']) if len(team_standings) > 0 else None
                # Get opponent win %
                if df_standings_winpct is not None:
                    opp_wp = df_standings_winpct[df_standings_winpct['team'] == opponent]
                    opp_dict['opponent_win_pct'] = float(opp_wp.iloc[0]['win_pct'] * 100) if len(opp_wp) > 0 else None
                else:
                    opp_dict['opponent_win_pct'] = float(opp_standings.iloc[0]['win_pct'] * 100) if len(opp_standings) > 0 and 'win_pct' in opp_standings.columns else None
            else:  # coverage
                opp_dict['elite_team_cover_pct'] = float(team_standings.iloc[0]['cover_pct'] * 100) if len(team_standings) > 0 else None
                # Get opponent cover %
                opp_dict['opponent_cover_pct'] = float(opp_standings.iloc[0]['cover_pct'] * 100) if len(opp_standings) > 0 else None
                # Also include win % for reference
                if df_standings_winpct is not None:
                    team_wp = df_standings_winpct[df_standings_winpct['team'] == elite_team]
                    opp_wp = df_standings_winpct[df_standings_winpct['team'] == opponent]
                    opp_dict['elite_team_win_pct'] = float(team_wp.iloc[0]['win_pct'] * 100) if len(team_wp) > 0 else None
                    opp_dict['opponent_win_pct'] = float(opp_wp.iloc[0]['win_pct'] * 100) if len(opp_wp) > 0 else None

            opportunities.append(opp_dict)

    # Sort by elite team's recent form (best form first)
    opportunities.sort(key=lambda x: x.get('elite_team_last_5_spread') or 0, reverse=True)

    return opportunities


def calculate_team_form_for_game(
    df_completed: pd.DataFrame,
    team: str,
    as_of_date: datetime,
    lookback: int = FORM_GAMES_LOOKBACK,
    hot_min: int = None,
    cold_max: int = None
) -> Dict[str, Any]:
    """
    Calculate a team's form (last 5 games spread coverage) as of a specific date.

    This prevents look-ahead bias by only using games before the given date.

    Args:
        df_completed: Historical completed games DataFrame
        team: Team name to calculate form for
        as_of_date: Calculate form only using games before this date
        lookback: Number of games to look back (default: FORM_GAMES_LOOKBACK)
        hot_min: Minimum covers to be considered "hot" (default: None, uses 60% threshold)
        cold_max: Maximum covers to be considered "cold" (default: None, uses 40% threshold)

    Returns:
        Dictionary with form stats:
        - last_n_games: list of coverage results (True/False)
        - last_n_coverage_pct: percentage of games covered (0.0 to 1.0)
        - last_n_results: string like "4/5"
        - games_count: number of games used (may be < lookback)
        - covers_count: number of covers
        - is_hot: True if meets hot threshold
        - is_cold: True if meets cold threshold
        - is_perfect: True if coverage == 100%
    """
    # Filter to games before the as_of_date
    # Convert as_of_date to pandas Timestamp for comparison with datetime64 column
    as_of_ts = pd.Timestamp(as_of_date).tz_localize(None) if hasattr(as_of_date, 'tzinfo') and as_of_date.tzinfo else pd.Timestamp(as_of_date)
    df_before = df_completed[df_completed['game_date'] < as_of_ts].copy()
    df_before = df_before.sort_values('game_date')

    # Get team's games (both home and away)
    team_games = []

    for _, row in df_before.iterrows():
        if row['home_team'] == team:
            # Home team: spread_result_difference > 0 means covered
            covered = row['spread_result_difference'] > 0
            team_games.append({
                'date': row['game_date'],
                'covered': covered
            })
        elif row['away_team'] == team:
            # Away team: spread_result_difference < 0 means covered
            covered = row['spread_result_difference'] < 0
            team_games.append({
                'date': row['game_date'],
                'covered': covered
            })

    # Sort by date and get last n games
    team_games = sorted(team_games, key=lambda x: x['date'])
    last_n = team_games[-lookback:] if len(team_games) >= lookback else team_games

    games_count = len(last_n)
    if games_count == 0:
        return {
            'last_n_games': [],
            'last_n_coverage_pct': None,
            'last_n_results': '0/0',
            'games_count': 0,
            'covers_count': 0,
            'is_hot': False,
            'is_cold': False,
            'is_perfect': False
        }

    covers = sum(1 for g in last_n if g['covered'])
    coverage_pct = covers / games_count

    # Determine hot/cold status using provided thresholds or percentage-based defaults
    if hot_min is not None and cold_max is not None:
        # Use count-based thresholds (for Hot vs Cold variants)
        is_hot = covers >= hot_min and games_count >= lookback
        is_cold = covers <= cold_max and games_count >= lookback
    else:
        # Use percentage-based thresholds (legacy behavior)
        is_hot = coverage_pct >= HOT_FORM_THRESHOLD
        is_cold = coverage_pct <= COLD_FORM_THRESHOLD

    return {
        'last_n_games': [g['covered'] for g in last_n],
        'last_n_coverage_pct': coverage_pct,
        'last_n_results': f"{covers}/{games_count}",
        'games_count': games_count,
        'covers_count': covers,
        'is_hot': is_hot,
        'is_cold': is_cold,
        'is_perfect': coverage_pct == 1.0 and games_count >= lookback,
        # Legacy fields for backward compatibility
        'last_5_games': [g['covered'] for g in last_n],
        'last_5_coverage_pct': coverage_pct,
        'last_5_results': f"{covers}/{games_count}"
    }


def calculate_current_streak(
    df_completed: pd.DataFrame,
    team: str,
    as_of_date: datetime
) -> Dict[str, Any]:
    """
    Calculate a team's current win/loss streak for spread coverage.

    Args:
        df_completed: Historical completed games DataFrame
        team: Team name
        as_of_date: Calculate streak only using games before this date

    Returns:
        Dictionary with streak info:
        - streak_type: 'W' for winning streak, 'L' for losing streak
        - streak_count: number of consecutive games
        - streak_display: string like "W3" or "L2"
    """
    # Filter to games before the as_of_date
    # Convert as_of_date to pandas Timestamp for comparison with datetime64 column
    as_of_ts = pd.Timestamp(as_of_date).tz_localize(None) if hasattr(as_of_date, 'tzinfo') and as_of_date.tzinfo else pd.Timestamp(as_of_date)
    df_before = df_completed[df_completed['game_date'] < as_of_ts].copy()
    df_before = df_before.sort_values('game_date', ascending=False)  # Most recent first

    # Get team's games in reverse chronological order
    team_games = []
    for _, row in df_before.iterrows():
        if row['home_team'] == team:
            covered = row['spread_result_difference'] > 0
            team_games.append(covered)
        elif row['away_team'] == team:
            covered = row['spread_result_difference'] < 0
            team_games.append(covered)

    if not team_games:
        return {
            'streak_type': None,
            'streak_count': 0,
            'streak_display': '-'
        }

    # Count current streak (how many consecutive same results from most recent)
    current_result = team_games[0]  # Most recent game result
    streak_count = 0
    for result in team_games:
        if result == current_result:
            streak_count += 1
        else:
            break

    streak_type = 'W' if current_result else 'L'
    streak_display = f"{streak_type}{streak_count}"

    return {
        'streak_type': streak_type,
        'streak_count': streak_count,
        'streak_display': streak_display
    }


def generate_hot_vs_cold_opportunities(
    df_completed: pd.DataFrame,
    today_games_df: pd.DataFrame,
    lookback: int = None
) -> List[Dict[str, Any]]:
    """
    Generate hot vs cold opportunities for a specific lookback window.

    Strategy: When a "hot" team plays a "cold" team, bet on the hot team
    to cover an 11-point handicap.

    Args:
        df_completed: Historical completed games
        today_games_df: Today's games DataFrame
        lookback: Number of games to look back (3, 5, or 7). If None, generates for all 3.

    Returns:
        List of opportunities with hot/cold team info
    """
    opportunities = []

    # Use today's date for form calculation
    est = timezone(timedelta(hours=-5))
    today_date = datetime.now(est)

    # Determine which lookback windows to process
    if lookback is not None:
        lookbacks_to_process = [lookback]
    else:
        lookbacks_to_process = list(HOT_COLD_CONFIG.keys())

    for lb in lookbacks_to_process:
        config = HOT_COLD_CONFIG.get(lb, {'hot_min': 3, 'cold_max': 2})
        hot_min = config['hot_min']
        cold_max = config['cold_max']

        for _, game in today_games_df.iterrows():
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')

            if not home_team or not away_team:
                continue

            # Calculate form for both teams with specified lookback
            home_form = calculate_team_form_for_game(
                df_completed, home_team, today_date,
                lookback=lb, hot_min=hot_min, cold_max=cold_max
            )
            away_form = calculate_team_form_for_game(
                df_completed, away_team, today_date,
                lookback=lb, hot_min=hot_min, cold_max=cold_max
            )

            # Skip if either team doesn't have enough games
            if home_form['games_count'] < lb or away_form['games_count'] < lb:
                continue

            # Check for hot vs cold matchup
            hot_team = None
            cold_team = None
            hot_form_data = None
            cold_form_data = None

            if home_form['is_hot'] and away_form['is_cold']:
                hot_team = home_team
                cold_team = away_team
                hot_form_data = home_form
                cold_form_data = away_form
                hot_position = 'home'
                cold_position = 'away'
            elif away_form['is_hot'] and home_form['is_cold']:
                hot_team = away_team
                cold_team = home_team
                hot_form_data = away_form
                cold_form_data = home_form
                hot_position = 'away'
                cold_position = 'home'

            # Only create opportunity if we have a clear hot vs cold matchup
            if hot_team and cold_team:
                form_differential = (hot_form_data['last_n_coverage_pct'] - cold_form_data['last_n_coverage_pct']) * 100

                # Calculate current streaks for both teams
                hot_streak = calculate_current_streak(df_completed, hot_team, today_date)
                cold_streak = calculate_current_streak(df_completed, cold_team, today_date)

                opportunities.append({
                    'game_time_est': game.get('game_time_est'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'current_spread': float(game.get('current_spread')) if pd.notna(game.get('current_spread')) else None,
                    'hot_team': hot_team,
                    'hot_team_position': hot_position,
                    'hot_team_last_n_coverage_pct': round(hot_form_data['last_n_coverage_pct'] * 100, 1),
                    'hot_team_last_n_results': hot_form_data['last_n_results'],
                    'hot_team_current_streak': hot_streak['streak_display'],
                    'cold_team': cold_team,
                    'cold_team_position': cold_position,
                    'cold_team_last_n_coverage_pct': round(cold_form_data['last_n_coverage_pct'] * 100, 1),
                    'cold_team_last_n_results': cold_form_data['last_n_results'],
                    'cold_team_current_streak': cold_streak['streak_display'],
                    'form_differential': round(form_differential, 1),
                    'bet_on': hot_team,
                    'handicap_points': FORM_HANDICAP_POINTS,
                    'lookback_games': lb,
                    'hot_threshold': f">={hot_min}/{lb}",
                    'cold_threshold': f"<={cold_max}/{lb}"
                })

    # Sort by form differential (largest advantage first)
    opportunities.sort(key=lambda x: x.get('form_differential', 0), reverse=True)

    return opportunities


def generate_hot_vs_cold_opportunities_all_variants(
    df_completed: pd.DataFrame,
    today_games_df: pd.DataFrame
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate hot vs cold opportunities for all 3 variants (3, 5, 7 game lookbacks).

    Returns a dictionary with keys 'hot_vs_cold_3', 'hot_vs_cold_5', 'hot_vs_cold_7'
    """
    return {
        'hot_vs_cold_3': generate_hot_vs_cold_opportunities(df_completed, today_games_df, lookback=3),
        'hot_vs_cold_5': generate_hot_vs_cold_opportunities(df_completed, today_games_df, lookback=5),
        'hot_vs_cold_7': generate_hot_vs_cold_opportunities(df_completed, today_games_df, lookback=7),
    }


def generate_opponent_perfect_form_opportunities(
    df_completed: pd.DataFrame,
    today_games_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generate opponent perfect form (regression) opportunities.

    Strategy: When a team has covered their spread in ALL of their last 5 games
    (100% = 5/5), their NEXT opponent is likely to cover the 11-point handicap
    due to regression to mean.

    Args:
        df_completed: Historical completed games
        today_games_df: Today's games DataFrame

    Returns:
        List of opportunities
    """
    opportunities = []

    # Use today's date for form calculation
    est = timezone(timedelta(hours=-5))
    today_date = datetime.now(est)

    for _, game in today_games_df.iterrows():
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        if not home_team or not away_team:
            continue

        # Calculate form for both teams
        home_form = calculate_team_form_for_game(df_completed, home_team, today_date)
        away_form = calculate_team_form_for_game(df_completed, away_team, today_date)

        # Check if home team has perfect form
        if home_form['is_perfect']:
            # Bet on away team (opponent of perfect form team)
            opportunities.append({
                'game_time_est': game.get('game_time_est'),
                'home_team': home_team,
                'away_team': away_team,
                'current_spread': float(game.get('current_spread')) if pd.notna(game.get('current_spread')) else None,
                'perfect_form_team': home_team,
                'perfect_form_team_position': 'home',
                'perfect_form_team_coverage_pct': 100.0,
                'perfect_form_team_results': home_form['last_5_results'],
                'opponent': away_team,
                'opponent_position': 'away',
                'opponent_coverage_pct': round(away_form['last_5_coverage_pct'] * 100, 1) if away_form['last_5_coverage_pct'] else None,
                'opponent_results': away_form['last_5_results'],
                'bet_on': away_team,
                'handicap_points': FORM_HANDICAP_POINTS,
                'rationale': 'Regression to mean - betting against perfect 5/5 streak'
            })

        # Check if away team has perfect form
        if away_form['is_perfect']:
            # Bet on home team (opponent of perfect form team)
            opportunities.append({
                'game_time_est': game.get('game_time_est'),
                'home_team': home_team,
                'away_team': away_team,
                'current_spread': float(game.get('current_spread')) if pd.notna(game.get('current_spread')) else None,
                'perfect_form_team': away_team,
                'perfect_form_team_position': 'away',
                'perfect_form_team_coverage_pct': 100.0,
                'perfect_form_team_results': away_form['last_5_results'],
                'opponent': home_team,
                'opponent_position': 'home',
                'opponent_coverage_pct': round(home_form['last_5_coverage_pct'] * 100, 1) if home_form['last_5_coverage_pct'] else None,
                'opponent_results': home_form['last_5_results'],
                'bet_on': home_team,
                'handicap_points': FORM_HANDICAP_POINTS,
                'rationale': 'Regression to mean - betting against perfect 5/5 streak'
            })

    return opportunities


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
    
    # Calculate handicap statistics (sport-specific, for legacy compatibility)
    df_team_stats_handicap = calculate_handicap_statistics(df_completed, handicap_points)
    logger.info(f"Calculated {handicap_points}-point handicap stats for {len(df_team_stats_handicap)} teams")

    # Calculate fixed 11pt handicap statistics for home_focus/away_focus strategies
    df_team_stats_handicap_11pt = calculate_handicap_statistics(df_completed, HOME_AWAY_HANDICAP)
    logger.info(f"Calculated {HOME_AWAY_HANDICAP}-point handicap stats for home/away focus strategies")

    # Calculate team standings for elite team strategy (by win %)
    df_standings_winpct = calculate_team_standings(df_completed)
    elite_teams_winpct_count = len(df_standings_winpct[df_standings_winpct['tier'] == 'Elite'])
    logger.info(f"Calculated win% standings for {len(df_standings_winpct)} teams ({elite_teams_winpct_count} elite)")

    # Calculate team standings by spread coverage % (for elite_team_coverage variant)
    df_standings_coverage = calculate_team_standings_by_coverage(df_completed)
    elite_teams_coverage_count = len(df_standings_coverage[df_standings_coverage['tier'] == 'Elite'])
    logger.info(f"Calculated coverage standings for {len(df_standings_coverage)} teams ({elite_teams_coverage_count} elite)")

    # Calculate recent form for win% elite teams
    team_tier_map_winpct = df_standings_winpct.set_index('team')['tier'].to_dict()
    df_form_winpct = calculate_recent_form(df_completed, team_tier_map_winpct)
    elite_good_form_winpct_count = len(df_form_winpct[(df_form_winpct['tier'] == 'Elite') & (df_form_winpct['is_good_form'] == True)])
    logger.info(f"Calculated win% form: {elite_good_form_winpct_count} elite teams in good form")

    # Calculate recent form for coverage elite teams
    team_tier_map_coverage = df_standings_coverage.set_index('team')['tier'].to_dict()
    df_form_coverage = calculate_recent_form(df_completed, team_tier_map_coverage)
    elite_good_form_coverage_count = len(df_form_coverage[(df_form_coverage['tier'] == 'Elite') & (df_form_coverage['is_good_form'] == True)])
    logger.info(f"Calculated coverage form: {elite_good_form_coverage_count} elite teams in good form")

    # Legacy aliases for backward compatibility
    df_standings = df_standings_winpct
    df_form = df_form_winpct
    elite_teams_count = elite_teams_winpct_count
    elite_good_form_count = elite_good_form_winpct_count

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

    # Merge 11pt handicap stats for home_focus/away_focus strategies
    games_with_stats = games_with_stats.merge(
        df_team_stats_handicap_11pt[['team', 'cover_pct_handicap']],
        left_on='home_team',
        right_on='team',
        how='left'
    )
    games_with_stats = games_with_stats.rename(columns={'cover_pct_handicap': 'home_cover_pct_handicap_11pt'})
    games_with_stats = games_with_stats.drop(columns=['team'])

    games_with_stats = games_with_stats.merge(
        df_team_stats_handicap_11pt[['team', 'cover_pct_handicap']],
        left_on='away_team',
        right_on='team',
        how='left'
    )
    games_with_stats = games_with_stats.rename(columns={'cover_pct_handicap': 'away_cover_pct_handicap_11pt'})
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

    # Strategy 1: Home Focus (bet on home team when they have better 11pt handicap coverage)
    home_focus_opportunities = games_with_enough_data[
        (games_with_enough_data['home_cover_pct_handicap_11pt'].notna()) &
        (games_with_enough_data['away_cover_pct_handicap_11pt'].notna()) &
        (games_with_enough_data['home_cover_pct_handicap_11pt'] > games_with_enough_data['away_cover_pct_handicap_11pt'])
    ].copy()
    if len(home_focus_opportunities) > 0:
        home_focus_opportunities['handicap_pct_difference'] = (
            home_focus_opportunities['home_cover_pct_handicap_11pt'] -
            home_focus_opportunities['away_cover_pct_handicap_11pt']
        )
        home_focus_opportunities = home_focus_opportunities.sort_values('handicap_pct_difference', ascending=False)

    # Strategy 2: Away Focus (bet on away team when they have better 11pt handicap coverage)
    away_focus_opportunities = games_with_enough_data[
        (games_with_enough_data['home_cover_pct_handicap_11pt'].notna()) &
        (games_with_enough_data['away_cover_pct_handicap_11pt'].notna()) &
        (games_with_enough_data['away_cover_pct_handicap_11pt'] > games_with_enough_data['home_cover_pct_handicap_11pt'])
    ].copy()
    if len(away_focus_opportunities) > 0:
        away_focus_opportunities['handicap_pct_difference'] = (
            away_focus_opportunities['away_cover_pct_handicap_11pt'] -
            away_focus_opportunities['home_cover_pct_handicap_11pt']
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
                opportunities['away_cover_pct_handicap_11pt'] -
                opportunities['home_cover_pct_handicap_11pt']
            )
        else:  # focus_team == 'home'
            opportunities['handicap_pct_difference'] = (
                opportunities['home_cover_pct_handicap_11pt'] -
                opportunities['away_cover_pct_handicap_11pt']
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
                # Handicap-based strategies (using 11pt handicap)
                opp_dict.update({
                    'away_cover_pct_handicap': float(row.get('away_cover_pct_handicap_11pt')) if pd.notna(row.get('away_cover_pct_handicap_11pt')) else None,
                    'home_cover_pct_handicap': float(row.get('home_cover_pct_handicap_11pt')) if pd.notna(row.get('home_cover_pct_handicap_11pt')) else None,
                    'handicap_pct_difference': float(row.get('handicap_pct_difference')) if pd.notna(row.get('handicap_pct_difference')) else None,
                    'handicap_points': HOME_AWAY_HANDICAP
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

    # Generate elite team opportunities (Win % variant)
    elite_team_winpct_opportunities = generate_elite_team_opportunities(
        df_standings_winpct, df_form_winpct, df_today,
        strategy_type='winpct', df_standings_winpct=df_standings_winpct
    )
    logger.info(f"Generated {len(elite_team_winpct_opportunities)} elite team (win%) opportunities")

    # Generate elite team opportunities (Coverage % variant)
    elite_team_coverage_opportunities = generate_elite_team_opportunities(
        df_standings_coverage, df_form_coverage, df_today,
        strategy_type='coverage', df_standings_winpct=df_standings_winpct
    )
    logger.info(f"Generated {len(elite_team_coverage_opportunities)} elite team (coverage%) opportunities")

    # Legacy: combine both for backward compatibility
    elite_team_opportunities = elite_team_winpct_opportunities

    # Generate hot vs cold opportunities (all 3 variants)
    hot_vs_cold_variants = generate_hot_vs_cold_opportunities_all_variants(df_completed, df_today)
    hot_vs_cold_3_opportunities = hot_vs_cold_variants['hot_vs_cold_3']
    hot_vs_cold_5_opportunities = hot_vs_cold_variants['hot_vs_cold_5']
    hot_vs_cold_7_opportunities = hot_vs_cold_variants['hot_vs_cold_7']
    logger.info(f"Generated hot vs cold opportunities: 3-game={len(hot_vs_cold_3_opportunities)}, 5-game={len(hot_vs_cold_5_opportunities)}, 7-game={len(hot_vs_cold_7_opportunities)}")

    # Legacy: keep combined for backward compatibility
    hot_vs_cold_opportunities = hot_vs_cold_5_opportunities

    # Generate opponent perfect form (regression) opportunities
    opponent_perfect_form_opportunities = generate_opponent_perfect_form_opportunities(
        df_completed, df_today
    )
    logger.info(f"Generated {len(opponent_perfect_form_opportunities)} opponent perfect form opportunities")

    # Legacy opportunities list (for backward compatibility)
    opportunities_list = opportunities_to_json(opportunities, focus_team + '_focus')
    
    # Build strategies dictionary
    strategies = {
        'home_focus': {
            'opportunities': home_focus_list,
            'summary': {
                'opportunities': len(home_focus_list),
                'average_difference': float(home_focus_opportunities['handicap_pct_difference'].mean()) if len(home_focus_opportunities) > 0 else None,
                'largest_difference': float(home_focus_opportunities['handicap_pct_difference'].max()) if len(home_focus_opportunities) > 0 else None,
                'handicap_points': HOME_AWAY_HANDICAP
            }
        },
        'away_focus': {
            'opportunities': away_focus_list,
            'summary': {
                'opportunities': len(away_focus_list),
                'average_difference': float(away_focus_opportunities['handicap_pct_difference'].mean()) if len(away_focus_opportunities) > 0 else None,
                'largest_difference': float(away_focus_opportunities['handicap_pct_difference'].max()) if len(away_focus_opportunities) > 0 else None,
                'handicap_points': HOME_AWAY_HANDICAP
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

    # Add elite_team_winpct strategy (top 25% by win %)
    strategies['elite_team_winpct'] = {
        'opportunities': elite_team_winpct_opportunities,
        'summary': {
            'opportunities': len(elite_team_winpct_opportunities),
            'elite_teams_total': elite_teams_winpct_count,
            'elite_teams_good_form': elite_good_form_winpct_count,
            'elite_percentile': ELITE_PERCENTILE,
            'good_form_threshold': GOOD_FORM_THRESHOLD,
            'handicap_points': ELITE_HANDICAP_POINTS,
            'ranking_method': 'win_percentage'
        }
    }

    # Add elite_team_coverage strategy (top 25% by spread coverage %)
    strategies['elite_team_coverage'] = {
        'opportunities': elite_team_coverage_opportunities,
        'summary': {
            'opportunities': len(elite_team_coverage_opportunities),
            'elite_teams_total': elite_teams_coverage_count,
            'elite_teams_good_form': elite_good_form_coverage_count,
            'elite_percentile': ELITE_PERCENTILE,
            'good_form_threshold': GOOD_FORM_THRESHOLD,
            'handicap_points': ELITE_HANDICAP_POINTS,
            'ranking_method': 'spread_coverage'
        }
    }

    # Legacy: keep elite_team for backward compatibility (uses win%)
    strategies['elite_team'] = strategies['elite_team_winpct']

    # Add hot_vs_cold_3 strategy (3-game lookback)
    strategies['hot_vs_cold_3'] = {
        'opportunities': hot_vs_cold_3_opportunities,
        'summary': {
            'opportunities': len(hot_vs_cold_3_opportunities),
            'lookback_games': 3,
            'hot_threshold': f">={HOT_COLD_CONFIG[3]['hot_min']}/3",
            'cold_threshold': f"<={HOT_COLD_CONFIG[3]['cold_max']}/3",
            'handicap_points': FORM_HANDICAP_POINTS
        }
    }

    # Add hot_vs_cold_5 strategy (5-game lookback)
    strategies['hot_vs_cold_5'] = {
        'opportunities': hot_vs_cold_5_opportunities,
        'summary': {
            'opportunities': len(hot_vs_cold_5_opportunities),
            'lookback_games': 5,
            'hot_threshold': f">={HOT_COLD_CONFIG[5]['hot_min']}/5",
            'cold_threshold': f"<={HOT_COLD_CONFIG[5]['cold_max']}/5",
            'handicap_points': FORM_HANDICAP_POINTS
        }
    }

    # Add hot_vs_cold_7 strategy (7-game lookback)
    strategies['hot_vs_cold_7'] = {
        'opportunities': hot_vs_cold_7_opportunities,
        'summary': {
            'opportunities': len(hot_vs_cold_7_opportunities),
            'lookback_games': 7,
            'hot_threshold': f">={HOT_COLD_CONFIG[7]['hot_min']}/7",
            'cold_threshold': f"<={HOT_COLD_CONFIG[7]['cold_max']}/7",
            'handicap_points': FORM_HANDICAP_POINTS
        }
    }

    # Legacy: keep hot_vs_cold for backward compatibility (uses 5-game lookback)
    strategies['hot_vs_cold'] = strategies['hot_vs_cold_5']

    # Add opponent_perfect_form strategy (regression-based)
    strategies['opponent_perfect_form'] = {
        'opportunities': opponent_perfect_form_opportunities,
        'summary': {
            'opportunities': len(opponent_perfect_form_opportunities),
            'perfect_form_games': FORM_GAMES_LOOKBACK,
            'handicap_points': FORM_HANDICAP_POINTS,
            'rationale': 'Regression to mean - betting against perfect streaks'
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



