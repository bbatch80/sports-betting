"""
Sports Betting Analytics Dashboard

Interactive Streamlit dashboard for exploring betting trends.
Run with: streamlit run dashboard/app.py
"""

import re
import sys
import time
import functools
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file (for DATABASE_URL, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Performance Profiling Utilities
# =============================================================================

def timed(func):
    """
    Decorator to measure function execution time.
    Only displays timing when debug_timing is enabled in session state.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        if st.session_state.get('debug_timing'):
            st.caption(f"⏱️ {func.__name__}: {elapsed:.1f}ms")
        return result
    return wrapper


def db_ping(conn) -> float:
    """Measure database round-trip latency."""
    from sqlalchemy import text
    start = time.perf_counter()
    conn.execute(text("SELECT 1"))
    return (time.perf_counter() - start) * 1000

from src.database import get_connection, get_games, get_all_teams, get_sports, get_date_range, get_game_count
from src.analysis.metrics import (
    ats_cover_rate,
    ats_record,
    spread_margin_avg,
    handicap_cover_rate,
    time_series_ats,
    team_ats_cover_rate,
    team_ats_record,
    streak_continuation_analysis,
    streak_summary_all_lengths,
    get_streak_situations_detail,
    baseline_handicap_coverage,
    ou_cover_rate,
    ou_record,
    team_ou_cover_rate,
    team_ou_record,
    # O/U streak analysis
    ou_streak_continuation_analysis,
    baseline_ou_coverage,
    ou_streak_summary_all_lengths,
    # Team totals streak analysis
    tt_streak_summary_all_lengths,
    tt_streak_continuation_analysis,
    tt_baseline_coverage,
    convergent_gt_analysis,
)
from src.analysis.aggregations import (
    macro_ats_summary,
    macro_time_series,
    macro_by_spread_bucket,
    micro_team_summary,
    micro_all_teams,
    macro_ou_summary,
    macro_ou_time_series,
    macro_ou_by_total_bucket,
    micro_ou_all_teams,
)
from src.analysis.insights import (
    detect_patterns,
    get_current_streaks,
    get_cached_streaks,
    get_cached_patterns,
    get_cached_ou_patterns,
    get_cached_tt_patterns,
    get_current_ou_streaks,
    get_cached_ou_streaks,
    get_current_tt_streaks,
    get_cached_tt_streaks,
)
from src.analysis.network_ratings import (
    get_team_rankings,
    get_cached_rankings,
    get_rankings_dataframe,
)
from src.analysis.tier_matchups import get_tier
from src.analysis.todays_recommendations import (
    generate_recommendations,
    get_cached_recommendations,
    get_closing_lines,
    get_combined_confidence,
    get_games_last_updated,
)
from src.ai import ask_question, LLMConfig, QueryResult
from src.db.engine import get_engine


# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Sports Betting Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Database Connection (cached)
# =============================================================================

@st.cache_resource
def get_db_connection():
    """Get cached database connection with AUTOCOMMIT to prevent idle-in-transaction locks."""
    conn = get_connection(check_same_thread=False)
    # AUTOCOMMIT prevents idle-in-transaction locks that block DDL (ALTER TABLE) and other queries.
    # Without this, each SELECT opens a transaction that stays open until commit/close,
    # which can block migrations and cause cascading lock pileups.
    return conn.execution_options(isolation_level="AUTOCOMMIT")


@st.cache_data(ttl=3600)
def load_games(sport: str = None):
    """Load games with caching."""
    conn = get_db_connection()
    return get_games(conn, sport=sport)


@st.cache_data(ttl=3600)
def load_teams(sport: str):
    """Load teams list with caching."""
    conn = get_db_connection()
    return get_all_teams(conn, sport)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_streak_continuation(_conn, sport: str, streak_length: int, streak_type: str):
    """Cached streak continuation analysis."""
    return streak_continuation_analysis(_conn, sport, streak_length, streak_type)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_baseline_coverage(_conn, sport: str):
    """Cached baseline handicap coverage."""
    return baseline_handicap_coverage(_conn, sport)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_streak_summary(_conn, sport: str):
    """Cached streak summary for all lengths."""
    return streak_summary_all_lengths(_conn, sport)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ou_streak_summary(_conn, sport: str):
    """Cached O/U streak summary for all lengths."""
    return ou_streak_summary_all_lengths(_conn, sport)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ou_streak_continuation(_conn, sport: str, streak_length: int, streak_type: str):
    """Cached O/U streak continuation analysis."""
    return ou_streak_continuation_analysis(_conn, sport, streak_length, streak_type)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_baseline_ou_coverage(_conn, sport: str):
    """Cached baseline O/U coverage."""
    return baseline_ou_coverage(_conn, sport)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_tt_streak_summary(_conn, sport: str):
    """Cached team totals streak summary."""
    return tt_streak_summary_all_lengths(_conn, sport)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_tt_streak_continuation(_conn, sport: str, streak_length: int, streak_type: str, direction: str):
    """Cached team totals streak continuation analysis."""
    return tt_streak_continuation_analysis(_conn, sport, streak_length, streak_type, direction=direction)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_tt_baseline_coverage(_conn, sport: str, direction: str):
    """Cached team totals baseline coverage."""
    return tt_baseline_coverage(_conn, sport, direction=direction)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_convergent_gt(_conn, sport: str, combo_dir: str, min_streak_a: int, min_streak_b: int, direction: str):
    """Cached convergent game total analysis."""
    return convergent_gt_analysis(_conn, sport, combo_dir, min_streak_a, min_streak_b, direction=direction)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_historical_ratings(_conn, sport: str) -> pd.DataFrame:
    """Cached historical ratings query for trajectory/momentum pages."""
    from sqlalchemy import text

    query = text("""
        SELECT snapshot_date, team, ats_rating, ats_rank
        FROM historical_ratings
        WHERE sport = :sport
        ORDER BY snapshot_date, team
    """)

    try:
        result = _conn.execute(query, {'sport': sport})
        rows = result.fetchall()
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['snapshot_date', 'team', 'ats_rating', 'ats_rank'])
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date']).dt.date
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def cached_team_slopes(_historical_df, sport: str) -> pd.DataFrame:
    """
    Pre-compute slopes for all teams at once (vectorized where possible).
    Returns DataFrame with Team, ATS Rating, Rank, and slope columns.
    """
    from scipy.stats import linregress

    if _historical_df.empty:
        return pd.DataFrame()

    teams = sorted(_historical_df['team'].unique())
    latest_date = pd.to_datetime(_historical_df['snapshot_date']).max()

    results = []
    for team in teams:
        team_data = _historical_df[_historical_df['team'] == team].sort_values('snapshot_date')
        if team_data.empty:
            continue

        current_rating = team_data['ats_rating'].iloc[-1]
        current_rank = int(team_data['ats_rank'].iloc[-1]) if pd.notna(team_data['ats_rank'].iloc[-1]) else None

        slopes = {}
        for label, days in [('1w', 7), ('2w', 14), ('3w', 21)]:
            cutoff = latest_date - pd.Timedelta(days=days)
            window = team_data[pd.to_datetime(team_data['snapshot_date']) >= cutoff]

            if len(window) >= 3:
                x = (pd.to_datetime(window['snapshot_date']) - pd.to_datetime(window['snapshot_date']).min()).dt.days
                y = window['ats_rating']
                slope, _, _, _, _ = linregress(x, y)
                slopes[label] = slope
            else:
                slopes[label] = None

        results.append({
            'Team': team,
            'ATS Rating': current_rating,
            'Rank': current_rank,
            '1W Slope': slopes['1w'],
            '2W Slope': slopes['2w'],
            '3W Slope': slopes['3w'],
        })

    return pd.DataFrame(results)


# =============================================================================
# Sidebar - Navigation & Filters
# =============================================================================

st.sidebar.title("📊 Analytics Dashboard")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🎲 Today's Picks", "📈 Prediction Results", "🏆 Power Rankings", "🔥 Current Streaks", "League-Wide Trends", "Team Trends", "Streak Analysis", "Verification", "🤖 AI Research"]
)

st.sidebar.markdown("---")

# Global sport filter
conn = get_db_connection()
sports = get_sports(conn)
selected_sport = st.sidebar.selectbox("Sport", sports if sports else ["NBA"])

# Show data summary
if sports:
    count = get_game_count(conn, selected_sport)
    date_range = get_date_range(conn, selected_sport)
    st.sidebar.caption(f"{count} games • {date_range[0]} to {date_range[1]}")


# =============================================================================
# Page: League-Wide Trends
# =============================================================================

def page_macro_trends():
    st.title("Macro Trends (League-Wide)")
    st.markdown(f"League-wide analysis for **{selected_sport}**")

    games = load_games(selected_sport)

    if len(games) == 0:
        st.warning(f"No games found for {selected_sport}")
        return

    # Top-level Spread/Totals toggle
    analysis_type = st.tabs(["Spread Analysis", "Totals Analysis"])

    # =========================================================================
    # SPREAD ANALYSIS
    # =========================================================================
    with analysis_type[0]:
        tab1, tab2, tab3 = st.tabs(["ATS Summary", "Time Series", "Spread Buckets"])

        # ----- Tab 1: ATS Summary by Handicap -----
        with tab1:
            st.subheader("ATS Cover Rate by Handicap")

            col1, col2 = st.columns(2)

            with col1:
                handicaps = st.multiselect(
                    "Handicaps to analyze",
                    options=list(range(0, 16)),
                    default=[0, 3, 5, 7, 10, 11, 13, 15]
                )

            summary = macro_ats_summary(games, handicaps=handicaps)

            # Create comparison chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=summary['handicap'],
                y=summary['home_pct'] * 100,
                mode='lines+markers',
                name='Home Teams',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=summary['handicap'],
                y=summary['away_pct'] * 100,
                mode='lines+markers',
                name='Away Teams',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8)
            ))

            # 50% reference line
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")

            fig.update_layout(
                title=f"{selected_sport} ATS Cover Rate by Handicap",
                xaxis_title="Handicap (points)",
                yaxis_title="Cover Rate (%)",
                yaxis=dict(range=[40, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            with st.expander("View Data Table"):
                display_df = summary.copy()
                display_df['home_pct'] = (display_df['home_pct'] * 100).round(1).astype(str) + '%'
                display_df['away_pct'] = (display_df['away_pct'] * 100).round(1).astype(str) + '%'
                st.dataframe(display_df, hide_index=True)

        # ----- Tab 2: Time Series -----
        with tab2:
            st.subheader("ATS Cover Rate Over Time (Cumulative)")

            col1, col2 = st.columns(2)
            with col1:
                ts_handicap = st.slider("Handicap", 0, 15, 0, key="ts_handicap")
            with col2:
                ts_perspective = st.selectbox("Perspective", ["home", "away"], key="ts_perspective")

            ts_data = macro_time_series(games, handicap=ts_handicap, perspective=ts_perspective)

            if len(ts_data) > 0:
                fig = px.line(
                    ts_data,
                    x='game_date',
                    y='cover_pct',
                    title=f"Cumulative {ts_perspective.title()} Team Cover Rate (+{ts_handicap}pt handicap)",
                    labels={'game_date': 'Date', 'cover_pct': 'Cover Rate'}
                )

                fig.update_traces(line_color='#1f77b4')
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.update_yaxes(tickformat='.1%')
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)

                # Summary stats
                col1, col2, col3 = st.columns(3)
                latest = ts_data.iloc[-1]
                col1.metric("Total Games", int(latest['games']))
                col2.metric("Total Covers", int(latest['covers']))
                col3.metric("Cover Rate", f"{latest['cover_pct']:.1%}")

        # ----- Tab 3: Spread Buckets -----
        with tab3:
            st.subheader("ATS Performance by Spread Size")

            bucket_data = macro_by_spread_bucket(games)

            if len(bucket_data) > 0:
                # Bar chart
                fig = px.bar(
                    bucket_data,
                    x='bucket',
                    y='home_pct',
                    title="Home Team Cover Rate by Spread Bucket",
                    labels={'bucket': 'Spread Bucket', 'home_pct': 'Home Cover Rate'},
                    text=bucket_data['home_pct'].apply(lambda x: f'{x:.1%}')
                )

                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.update_traces(textposition='outside')
                fig.update_layout(height=450)

                st.plotly_chart(fig, use_container_width=True)

                # Data table
                with st.expander("View Data Table"):
                    display_df = bucket_data[['bucket', 'games', 'home_covers', 'home_pct']].copy()
                    display_df['home_pct'] = (display_df['home_pct'] * 100).round(1).astype(str) + '%'
                    st.dataframe(display_df, hide_index=True)

    # =========================================================================
    # TOTALS ANALYSIS
    # =========================================================================
    with analysis_type[1]:
        tab1, tab2, tab3 = st.tabs(["O/U Summary", "Time Series", "Total Buckets"])

        # ----- Tab 1: O/U Summary by Handicap -----
        with tab1:
            st.subheader("Over/Under Cover Rate by Handicap")

            col1, col2 = st.columns(2)

            with col1:
                ou_handicaps = st.multiselect(
                    "Handicaps to analyze",
                    options=list(range(0, 21)),
                    default=[0, 3, 5, 7, 10, 13, 15, 17, 20],
                    key="ou_handicaps"
                )

            ou_summary = macro_ou_summary(games, handicaps=ou_handicaps)

            if len(ou_summary) == 0:
                st.warning("No totals data available for this sport")
            else:
                # Create comparison chart
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=ou_summary['handicap'],
                    y=ou_summary['over_pct'] * 100,
                    mode='lines+markers',
                    name='OVER',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=8)
                ))

                fig.add_trace(go.Scatter(
                    x=ou_summary['handicap'],
                    y=ou_summary['under_pct'] * 100,
                    mode='lines+markers',
                    name='UNDER',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=8)
                ))

                # 50% reference line
                fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")

                fig.update_layout(
                    title=f"{selected_sport} O/U Cover Rate by Handicap",
                    xaxis_title="Handicap (points)",
                    yaxis_title="Cover Rate (%)",
                    yaxis=dict(range=[20, 80]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show data table
                with st.expander("View Data Table"):
                    display_df = ou_summary.copy()
                    display_df['over_pct'] = (display_df['over_pct'] * 100).round(1).astype(str) + '%'
                    display_df['under_pct'] = (display_df['under_pct'] * 100).round(1).astype(str) + '%'
                    st.dataframe(display_df[['handicap', 'games', 'over_wins', 'over_pct', 'under_wins', 'under_pct']], hide_index=True)

        # ----- Tab 2: Time Series -----
        with tab2:
            st.subheader("O/U Cover Rate Over Time (Cumulative)")

            col1, col2 = st.columns(2)
            with col1:
                ou_ts_handicap = st.slider("Handicap", 0, 20, 0, key="ou_ts_handicap")
            with col2:
                ou_ts_direction = st.selectbox("Direction", ["over", "under"], key="ou_ts_direction")

            ou_ts_data = macro_ou_time_series(games, handicap=ou_ts_handicap, direction=ou_ts_direction)

            if len(ou_ts_data) > 0:
                fig = px.line(
                    ou_ts_data,
                    x='game_date',
                    y='cover_pct',
                    title=f"Cumulative {ou_ts_direction.upper()} Cover Rate (+{ou_ts_handicap}pt handicap)",
                    labels={'game_date': 'Date', 'cover_pct': 'Cover Rate'}
                )

                color = '#2ecc71' if ou_ts_direction == 'over' else '#e74c3c'
                fig.update_traces(line_color=color)
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.update_yaxes(tickformat='.1%')
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)

                # Summary stats
                col1, col2, col3 = st.columns(3)
                latest = ou_ts_data.iloc[-1]
                col1.metric("Total Games", int(latest['games']))
                col2.metric(f"Total {ou_ts_direction.upper()}s", int(latest['covers']))
                col3.metric("Cover Rate", f"{latest['cover_pct']:.1%}")
            else:
                st.warning("No totals data available for this sport")

        # ----- Tab 3: Total Buckets -----
        with tab3:
            st.subheader("O/U Performance by Total Size")

            ou_bucket_data = macro_ou_by_total_bucket(games)

            if len(ou_bucket_data) > 0:
                # Bar chart
                fig = px.bar(
                    ou_bucket_data,
                    x='bucket',
                    y='over_pct',
                    title="OVER Rate by Total Bucket",
                    labels={'bucket': 'Total Bucket', 'over_pct': 'OVER Rate'},
                    text=ou_bucket_data['over_pct'].apply(lambda x: f'{x:.1%}')
                )

                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.update_traces(textposition='outside', marker_color='#2ecc71')
                fig.update_layout(height=450)

                st.plotly_chart(fig, use_container_width=True)

                # Data table
                with st.expander("View Data Table"):
                    display_df = ou_bucket_data[['bucket', 'games', 'overs', 'unders', 'over_pct']].copy()
                    display_df['over_pct'] = (display_df['over_pct'] * 100).round(1).astype(str) + '%'
                    st.dataframe(display_df, hide_index=True)
            else:
                st.warning("No totals data available for this sport")


# =============================================================================
# Page: Micro (Team) Analysis
# =============================================================================

@timed
def page_micro_analysis():
    st.title("Micro Analysis (Team-Specific)")
    st.markdown(f"Individual team analysis for **{selected_sport}**")

    teams = load_teams(selected_sport)

    if not teams:
        st.warning(f"No teams found for {selected_sport}")
        return

    # Top-level Spread/Totals toggle
    analysis_type = st.tabs(["Spread Analysis", "Totals Analysis"])

    # =========================================================================
    # SPREAD ANALYSIS
    # =========================================================================
    with analysis_type[0]:
        tab1, tab2 = st.tabs(["All Teams Comparison", "Individual Team Deep Dive"])

        # ----- Tab 1: All Teams -----
        with tab1:
            st.subheader("All Teams ATS Performance")

            col1, col2 = st.columns(2)
            with col1:
                team_handicap = st.slider("Handicap", 0, 15, 0, key="team_handicap")
            with col2:
                min_games = st.slider("Minimum Games", 1, 20, 5, key="min_games")

            teams_df = micro_all_teams(conn, selected_sport, handicap=team_handicap, min_games=min_games)

            if len(teams_df) > 0:
                # Chart - Top and Bottom 10
                top_10 = teams_df.head(10)
                bottom_10 = teams_df.tail(10)

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Top 10 by ATS %", "Bottom 10 by ATS %")
                )

                fig.add_trace(
                    go.Bar(
                        y=top_10['team'],
                        x=top_10['ats_pct'] * 100,
                        orientation='h',
                        marker_color='#2ecc71',
                        text=top_10['ats_pct'].apply(lambda x: f'{x:.1%}'),
                        textposition='auto',
                        name='Top 10'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(
                        y=bottom_10['team'],
                        x=bottom_10['ats_pct'] * 100,
                        orientation='h',
                        marker_color='#e74c3c',
                        text=bottom_10['ats_pct'].apply(lambda x: f'{x:.1%}'),
                        textposition='auto',
                        name='Bottom 10'
                    ),
                    row=1, col=2
                )

                fig.add_vline(x=50, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_vline(x=50, line_dash="dash", line_color="gray", row=1, col=2)

                fig.update_layout(
                    height=500,
                    showlegend=False,
                    title_text=f"Team ATS Rankings (+{team_handicap}pt handicap, min {min_games} games)"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Full table
                with st.expander("View All Teams"):
                    display_df = teams_df.copy()
                    display_df['ats_pct'] = (display_df['ats_pct'] * 100).round(1).astype(str) + '%'
                    display_df['record'] = display_df.apply(
                        lambda r: f"{int(r['ats_wins'])}-{int(r['ats_losses'])}-{int(r['ats_pushes'])}",
                        axis=1
                    )
                    st.dataframe(
                        display_df[['team', 'games', 'record', 'ats_pct']],
                        hide_index=True,
                        use_container_width=True
                    )

        # ----- Tab 2: Individual Team -----
        with tab2:
            st.subheader("Individual Team Deep Dive")

            selected_team = st.selectbox("Select Team", teams, key="ats_team_select")

            if selected_team:
                # Get team's games
                team_games = get_games(conn, sport=selected_sport, team=selected_team)

                if len(team_games) > 0:
                    home_games = team_games[team_games['is_home'] == True]
                    away_games = team_games[team_games['is_home'] == False]

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    overall_w, overall_l, overall_p = team_ats_record(team_games, handicap=0)
                    overall_pct = team_ats_cover_rate(team_games, handicap=0)

                    col1.metric("Total Games", len(team_games))
                    col2.metric("ATS Record", f"{overall_w}-{overall_l}-{overall_p}")
                    col3.metric("ATS Cover %", f"{overall_pct:.1%}")
                    col4.metric("Avg Spread Margin", f"{spread_margin_avg(team_games, 'home'):.1f}")

                    # Home vs Away breakdown
                    st.markdown("---")
                    st.markdown("**Home vs Away Breakdown**")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Home Games**")
                        if len(home_games) > 0:
                            h_w, h_l, h_p = team_ats_record(home_games, handicap=0)
                            h_pct = team_ats_cover_rate(home_games, handicap=0)
                            st.write(f"Games: {len(home_games)}")
                            st.write(f"Record: {h_w}-{h_l}-{h_p}")
                            st.write(f"Cover %: {h_pct:.1%}")
                        else:
                            st.write("No home games")

                    with col2:
                        st.markdown("**Away Games**")
                        if len(away_games) > 0:
                            a_w, a_l, a_p = team_ats_record(away_games, handicap=0)
                            a_pct = team_ats_cover_rate(away_games, handicap=0)
                            st.write(f"Games: {len(away_games)}")
                            st.write(f"Record: {a_w}-{a_l}-{a_p}")
                            st.write(f"Cover %: {a_pct:.1%}")
                        else:
                            st.write("No away games")

                    # Handicap coverage table
                    st.markdown("---")
                    st.markdown("**Handicap Coverage Analysis**")

                    handicap_results = []
                    for h in range(0, 16):
                        w, l, p = team_ats_record(team_games, handicap=h)
                        total = w + l + p
                        pct = w / total if total > 0 else 0
                        handicap_results.append({
                            'handicap': h,
                            'wins': w,
                            'losses': l,
                            'pushes': p,
                            'cover_pct': pct
                        })

                    hc_df = pd.DataFrame(handicap_results)

                    fig = px.line(
                        hc_df,
                        x='handicap',
                        y='cover_pct',
                        title=f"{selected_team} Cover Rate by Handicap",
                        markers=True
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                    fig.update_yaxes(tickformat='.0%')
                    fig.update_layout(height=350)

                    st.plotly_chart(fig, use_container_width=True)

                    # Recent games
                    with st.expander("View All Games"):
                        display_games = team_games[['game_date', 'home_team', 'away_team',
                                                    'closing_spread', 'home_score', 'away_score',
                                                    'spread_result', 'is_home', 'team_covered']].copy()
                        display_games = display_games.sort_values('game_date', ascending=False)
                        display_games['result'] = display_games['team_covered'].apply(
                            lambda x: '✅ Cover' if x else '❌ Loss'
                        )
                        st.dataframe(display_games, hide_index=True, use_container_width=True)
                else:
                    st.warning(f"No games found for {selected_team}")

    # =========================================================================
    # TOTALS ANALYSIS
    # =========================================================================
    with analysis_type[1]:
        tab1, tab2 = st.tabs(["All Teams O/U", "Individual Team O/U"])

        # ----- Tab 1: All Teams O/U -----
        with tab1:
            st.subheader("All Teams O/U Performance")

            col1, col2 = st.columns(2)
            with col1:
                ou_team_handicap = st.slider("Handicap", 0, 20, 0, key="ou_team_handicap")
            with col2:
                ou_min_games = st.slider("Minimum Games", 1, 20, 5, key="ou_min_games")

            ou_teams_df = micro_ou_all_teams(conn, selected_sport, handicap=ou_team_handicap, min_games=ou_min_games)

            if len(ou_teams_df) > 0:
                # Chart - Top and Bottom 10 by OVER %
                top_10 = ou_teams_df.head(10)
                bottom_10 = ou_teams_df.tail(10)

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Top 10 by OVER %", "Bottom 10 by OVER %")
                )

                fig.add_trace(
                    go.Bar(
                        y=top_10['team'],
                        x=top_10['over_pct'] * 100,
                        orientation='h',
                        marker_color='#2ecc71',
                        text=top_10['over_pct'].apply(lambda x: f'{x:.1%}'),
                        textposition='auto',
                        name='Top 10'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(
                        y=bottom_10['team'],
                        x=bottom_10['over_pct'] * 100,
                        orientation='h',
                        marker_color='#e74c3c',
                        text=bottom_10['over_pct'].apply(lambda x: f'{x:.1%}'),
                        textposition='auto',
                        name='Bottom 10'
                    ),
                    row=1, col=2
                )

                fig.add_vline(x=50, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_vline(x=50, line_dash="dash", line_color="gray", row=1, col=2)

                fig.update_layout(
                    height=500,
                    showlegend=False,
                    title_text=f"Team O/U Rankings (+{ou_team_handicap}pt handicap, min {ou_min_games} games)"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Full table
                with st.expander("View All Teams"):
                    display_df = ou_teams_df.copy()
                    display_df['over_pct'] = (display_df['over_pct'] * 100).round(1).astype(str) + '%'
                    display_df['record'] = display_df.apply(
                        lambda r: f"{int(r['overs'])}-{int(r['unders'])}-{int(r['pushes'])}",
                        axis=1
                    )
                    display_df['avg_margin'] = display_df['avg_total_margin'].apply(lambda x: f"{x:+.1f}")
                    st.dataframe(
                        display_df[['team', 'games', 'record', 'over_pct', 'avg_margin']],
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.warning("No totals data available for this sport")

        # ----- Tab 2: Individual Team O/U -----
        with tab2:
            st.subheader("Individual Team O/U Deep Dive")

            ou_selected_team = st.selectbox("Select Team", teams, key="ou_team_select")

            if ou_selected_team:
                # Get team's games
                team_games = get_games(conn, sport=selected_sport, team=ou_selected_team)

                if len(team_games) > 0 and 'total_result' in team_games.columns:
                    # Filter games with totals data
                    valid_games = team_games[team_games['total_result'].notna()]

                    if len(valid_games) > 0:
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        ou_w, ou_l, ou_p = team_ou_record(valid_games, handicap=0)
                        ou_pct = team_ou_cover_rate(valid_games, handicap=0)
                        avg_total_margin = valid_games['total_result'].mean()

                        col1.metric("Games w/ Totals", len(valid_games))
                        col2.metric("O/U Record", f"{ou_w}-{ou_l}-{ou_p}")
                        col3.metric("OVER Rate", f"{ou_pct:.1%}")
                        col4.metric("Avg Total Margin", f"{avg_total_margin:+.1f}")

                        # O/U bias indicator
                        st.markdown("---")
                        if avg_total_margin > 2:
                            st.success(f"**OVER Bias**: This team's games average {avg_total_margin:.1f} points OVER the total")
                        elif avg_total_margin < -2:
                            st.error(f"**UNDER Bias**: This team's games average {abs(avg_total_margin):.1f} points UNDER the total")
                        else:
                            st.info(f"**Neutral**: This team's games are close to the total (avg {avg_total_margin:+.1f})")

                        # Handicap coverage analysis
                        st.markdown("---")
                        st.markdown("**Handicap Coverage Analysis (OVER)**")

                        ou_handicap_results = []
                        for h in range(0, 21):
                            w, l, p = team_ou_record(valid_games, handicap=h)
                            total = w + l + p
                            pct = w / total if total > 0 else 0
                            ou_handicap_results.append({
                                'handicap': h,
                                'overs': w,
                                'unders': l,
                                'pushes': p,
                                'over_pct': pct
                            })

                        ou_hc_df = pd.DataFrame(ou_handicap_results)

                        fig = px.line(
                            ou_hc_df,
                            x='handicap',
                            y='over_pct',
                            title=f"{ou_selected_team} OVER Rate by Handicap",
                            markers=True
                        )
                        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                        fig.update_yaxes(tickformat='.0%')
                        fig.update_traces(line_color='#2ecc71')
                        fig.update_layout(height=350)

                        st.plotly_chart(fig, use_container_width=True)

                        # Recent games with totals
                        with st.expander("View All Games"):
                            display_cols = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score']
                            if 'closing_total' in valid_games.columns:
                                display_cols.append('closing_total')
                            display_cols.append('total_result')

                            display_games = valid_games[display_cols].copy()
                            display_games = display_games.sort_values('game_date', ascending=False)
                            display_games['result'] = display_games['total_result'].apply(
                                lambda x: '🟢 OVER' if x > 0 else ('🔴 UNDER' if x < 0 else '⚪ PUSH')
                            )
                            st.dataframe(display_games, hide_index=True, use_container_width=True)
                    else:
                        st.warning(f"No totals data found for {ou_selected_team}")
                else:
                    st.warning(f"No games found for {ou_selected_team}")


# =============================================================================
# Page: Streak Analysis
# =============================================================================

def page_streak_analysis():
    st.title("Streak Continuation Analysis")

    # Top-level tabs for each market type
    streak_type_tabs = st.tabs(["Spread Streaks (ATS)", "Game Total O/U Streaks", "Team Total Streaks"])

    # =========================================================================
    # SPREAD STREAKS
    # =========================================================================
    with streak_type_tabs[0]:
        st.markdown(f"""
        **Question:** After a team covers/loses X games in a row, how does their next game perform across different handicaps (0-15 points)?

        Select a streak length and type, then see coverage rates at every handicap level for **{selected_sport}**.
        """)

        # First, show available streak data (cached)
        with st.spinner("Loading streak summary..."):
            summary = cached_streak_summary(conn, selected_sport)

        if len(summary) == 0:
            st.warning("No streak data found")
        else:
            # Show summary table
            st.subheader("Available Streak Data")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**WIN Streaks**")
                win_summary = summary[summary['streak_type'] == 'WIN'][['streak_length', 'situations']]
                win_summary.columns = ['Streak Length', 'Situations']
                st.dataframe(win_summary, hide_index=True)

            with col2:
                st.markdown("**LOSS Streaks**")
                loss_summary = summary[summary['streak_type'] == 'LOSS'][['streak_length', 'situations']]
                loss_summary.columns = ['Streak Length', 'Situations']
                st.dataframe(loss_summary, hide_index=True)

            # Select streak to analyze
            st.markdown("---")
            st.subheader("Analyze Handicap Coverage After Streak")

            col1, col2 = st.columns(2)
            with col1:
                streak_length = st.selectbox("Streak Length", list(range(2, 11)), index=0, key="streak_len")
            with col2:
                streak_type = st.selectbox("Streak Type", ["WIN", "LOSS"], key="streak_type")

            # Check sample size
            selected_summary = summary[(summary['streak_length'] == streak_length) & (summary['streak_type'] == streak_type)]
            if len(selected_summary) > 0:
                sample_size = selected_summary['situations'].values[0]
                st.info(f"Sample size: **{sample_size}** situations where a team had a {streak_length}-game {streak_type} streak")

                # Run handicap analysis (cached for speed)
                with st.spinner("Analyzing handicap coverage..."):
                    handicap_data = cached_streak_continuation(conn, selected_sport, streak_length, streak_type)
                    baseline_data = cached_baseline_coverage(conn, selected_sport)

                if len(handicap_data) > 0:
                    # Merge baseline into handicap data
                    handicap_data = handicap_data.merge(baseline_data[['handicap', 'baseline_cover_pct']], on='handicap')
                    handicap_data['edge_vs_baseline'] = handicap_data['cover_pct'] - handicap_data['baseline_cover_pct']

                    # Chart: Cover rate by handicap with baseline comparison
                    st.subheader(f"Next Game Cover Rate by Handicap (After {streak_length}-Game {streak_type} Streak)")

                    fig = go.Figure()

                    # Bars for streak cover rate - color based on vs baseline
                    fig.add_trace(go.Bar(
                        name=f'After {streak_length}-Game {streak_type} Streak',
                        x=handicap_data['handicap'],
                        y=handicap_data['cover_pct'] * 100,
                        marker_color=['#e74c3c' if edge < 0 else '#2ecc71' for edge in handicap_data['edge_vs_baseline']],
                        text=handicap_data['cover_pct'].apply(lambda x: f"{x:.1%}"),
                        textposition='outside'
                    ))

                    # Baseline trend line
                    fig.add_trace(go.Scatter(
                        name=f'{selected_sport} Baseline',
                        x=baseline_data['handicap'],
                        y=baseline_data['baseline_cover_pct'] * 100,
                        mode='lines+markers',
                        line=dict(color='#3498db', width=3, dash='solid'),
                        marker=dict(size=8)
                    ))

                    fig.update_layout(
                        xaxis_title="Handicap (points added to spread)",
                        yaxis_title="Cover Rate (%)",
                        yaxis=dict(range=[0, 100]),
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        barmode='overlay'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    st.markdown(f"""
                    **How to read this chart:**
                    - **Bars** = Cover rate after a {streak_length}-game {streak_type} streak
                    - **Blue line** = League-wide baseline cover rate at each handicap
                    - **Green bars** = streak situations OUTPERFORM the baseline (potential edge)
                    - **Red bars** = streak situations UNDERPERFORM the baseline (fade opportunity)

                    **Key insight:** Where the bars exceed the blue line, there may be a betting edge.
                    """)

                    # Data table
                    with st.expander("View Data Table"):
                        display = handicap_data.copy()
                        display['cover_pct_fmt'] = display['cover_pct'].apply(lambda x: f"{x:.1%}")
                        display['baseline_fmt'] = display['baseline_cover_pct'].apply(lambda x: f"{x:.1%}")
                        display['edge_fmt'] = display['edge_vs_baseline'].apply(lambda x: f"{x*100:+.1f}%")
                        display = display.rename(columns={
                            'handicap': 'Handicap',
                            'covers': 'Covers',
                            'total': 'Total',
                            'cover_pct_fmt': 'Streak Cover %',
                            'baseline_fmt': 'Baseline %',
                            'edge_fmt': 'Edge vs Baseline'
                        })
                        st.dataframe(display[['Handicap', 'Total', 'Covers', 'Streak Cover %', 'Baseline %', 'Edge vs Baseline']], hide_index=True)

                    # Drill-down to individual games
                    st.markdown("---")
                    st.subheader("View Individual Situations")

                    if st.button("Load All Situations", key="load_ats_situations"):
                        with st.spinner("Loading..."):
                            situations = get_streak_situations_detail(conn, selected_sport, streak_length, streak_type, handicap=0)

                        if len(situations) > 0:
                            st.success(f"Found {len(situations)} situations")

                            # Summary at 0 handicap
                            cover_pct = situations['next_covered'].mean()
                            st.metric(
                                f"Cover Rate at 0 Handicap",
                                f"{cover_pct:.1%}",
                                f"{(cover_pct - 0.5) * 100:+.1f}% vs baseline"
                            )

                            # Table
                            st.dataframe(
                                situations[['team', 'next_game_date', 'next_opponent', 'next_is_home', 'next_spread', 'result']].sort_values('next_game_date', ascending=False),
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.warning("No situations found")
                else:
                    st.warning("No data found")
            else:
                st.warning(f"No {streak_length}-game {streak_type} streaks found")

    # =========================================================================
    # GAME TOTAL O/U STREAKS
    # =========================================================================
    with streak_type_tabs[1]:
        ou_subtabs = st.tabs(["O/U Streak Analysis", "Convergent Matchups"])

        # =================================================================
        # O/U Streak Analysis (Ride / Fade)
        # =================================================================
        with ou_subtabs[0]:
            st.markdown(f"""
            **Game Total O/U** tracks the combined score of both teams vs the game total line.

            After a game total goes OVER/UNDER N+ games in a row for a team,
            how does the **next game's total** perform? Analyzed with both
            **RIDE** (bet with streak) and **FADE** (bet against) strategies for **{selected_sport}**.
            """)

            # Load O/U streak summary
            with st.spinner("Loading game total O/U streak summary..."):
                ou_summary = cached_ou_streak_summary(conn, selected_sport)

            if len(ou_summary) == 0:
                st.warning("No game total O/U streak data found.")
            else:
                st.subheader("Available Game Total O/U Streak Data")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**OVER Streaks**")
                    ou_over = ou_summary[ou_summary['streak_type'] == 'OVER'][['streak_length', 'situations']]
                    ou_over.columns = ['Streak Length', 'Situations']
                    st.dataframe(ou_over, hide_index=True)

                with col2:
                    st.markdown("**UNDER Streaks**")
                    ou_under = ou_summary[ou_summary['streak_type'] == 'UNDER'][['streak_length', 'situations']]
                    ou_under.columns = ['Streak Length', 'Situations']
                    st.dataframe(ou_under, hide_index=True)

                # Controls
                st.markdown("---")
                st.subheader("Analyze Game Total Coverage After Streak")

                col1, col2 = st.columns(2)
                with col1:
                    ou_streak_len = st.selectbox("Streak Length", list(range(2, 11)), index=0, key="ou_streak_len")
                with col2:
                    ou_streak_type = st.selectbox("Streak Type", ["OVER", "UNDER"], key="ou_streak_type")

                ou_opposite = "UNDER" if ou_streak_type == "OVER" else "OVER"

                # Sample size
                ou_sel = ou_summary[(ou_summary['streak_length'] == ou_streak_len) & (ou_summary['streak_type'] == ou_streak_type)]
                if len(ou_sel) > 0:
                    ou_sample = int(ou_sel['situations'].values[0])
                    st.info(f"Sample size: **{ou_sample}** situations where a team had a {ou_streak_len}-game O/U {ou_streak_type} streak")

                    # Load analysis data
                    with st.spinner("Analyzing game total O/U coverage..."):
                        # Ride = bet same direction as streak
                        ou_ride = ou_streak_continuation_analysis(
                            conn, selected_sport, ou_streak_len, ou_streak_type,
                            direction=ou_streak_type.lower())
                        # For ride: baseline in same direction as streak
                        ou_bl_ride = baseline_ou_coverage(conn, selected_sport, handicap_range=(0, 20),
                                                          direction=ou_streak_type.lower())
                        # For fade: baseline in opposite direction
                        ou_bl_fade = baseline_ou_coverage(conn, selected_sport, handicap_range=(0, 20),
                                                          direction=ou_opposite.lower())

                    # --- RIDE Chart ---
                    if len(ou_ride) > 0 and len(ou_bl_ride) > 0:
                        # ou_streak_continuation returns ride data by default
                        ou_ride_m = ou_ride.merge(ou_bl_ride[['handicap', 'baseline_cover_pct']], on='handicap')
                        ou_ride_m['edge'] = ou_ride_m['cover_pct'] - ou_ride_m['baseline_cover_pct']

                        st.subheader(f"After {ou_streak_len}-Game O/U {ou_streak_type} Streak — Bet {ou_streak_type} (Ride)")

                        fig_ou_r = go.Figure()
                        fig_ou_r.add_trace(go.Bar(
                            name=f'O/U {ou_streak_type} Coverage (Ride)',
                            x=ou_ride_m['handicap'],
                            y=ou_ride_m['cover_pct'] * 100,
                            marker_color=['#2ecc71' if e >= 0 else '#e74c3c' for e in ou_ride_m['edge']],
                            text=ou_ride_m['cover_pct'].apply(lambda x: f"{x:.1%}"),
                            textposition='outside'
                        ))
                        fig_ou_r.add_trace(go.Scatter(
                            name=f'{selected_sport} O/U Baseline ({ou_streak_type})',
                            x=ou_bl_ride['handicap'],
                            y=ou_bl_ride['baseline_cover_pct'] * 100,
                            mode='lines+markers',
                            line=dict(color='#3498db', width=3),
                            marker=dict(size=8)
                        ))
                        fig_ou_r.update_layout(
                            xaxis_title="Handicap (points)",
                            yaxis_title="Cover Rate (%)",
                            yaxis=dict(range=[0, 100]),
                            height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            barmode='overlay'
                        )
                        st.plotly_chart(fig_ou_r, use_container_width=True)

                        with st.expander("View Ride Data Table"):
                            d = ou_ride_m.copy()
                            d['cover_pct_fmt'] = d['cover_pct'].apply(lambda x: f"{x:.1%}")
                            d['baseline_fmt'] = d['baseline_cover_pct'].apply(lambda x: f"{x:.1%}")
                            d['edge_fmt'] = d['edge'].apply(lambda x: f"{x*100:+.1f}%")
                            d = d.rename(columns={'handicap': 'Handicap', 'covers': 'Covers', 'total': 'Total',
                                                   'cover_pct_fmt': 'Cover %', 'baseline_fmt': 'Baseline %', 'edge_fmt': 'Edge'})
                            st.dataframe(d[['Handicap', 'Total', 'Covers', 'Cover %', 'Baseline %', 'Edge']], hide_index=True)

                    else:
                        st.warning("No ride data found for this streak configuration")
                else:
                    st.warning(f"No {ou_streak_len}-game O/U {ou_streak_type} streaks found")

        # =================================================================
        # Convergent Matchups (moved from old Totals tab)
        # =================================================================
        with ou_subtabs[1]:
            st.markdown(f"""
            **Convergent Matchups** — When **both teams** enter a game on same-direction
            team total streaks, how does the **game total** (combined score) perform?

            Streak lengths can differ — e.g., one team on a 3-game TT UNDER streak
            and the other on a 2-game streak. Analyzed for **{selected_sport}**.
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                conv_dir = st.selectbox("Both Teams On", ["OVER", "UNDER"], key="conv_dir")
            with col2:
                conv_min_a = st.selectbox("Team A Min Streak", list(range(1, 6)), index=1, key="conv_min_a")
            with col3:
                conv_min_b = st.selectbox("Team B Min Streak", list(range(1, 6)), index=1, key="conv_min_b")

            conv_opposite = "UNDER" if conv_dir == "OVER" else "OVER"

            # Sort so the lower streak is always first for cache consistency
            lo_streak, hi_streak = sorted([conv_min_a, conv_min_b])
            streak_label = f"{hi_streak}+ / {lo_streak}+" if lo_streak != hi_streak else f"{lo_streak}+"

            with st.spinner("Analyzing convergent matchups..."):
                conv_ride = cached_convergent_gt(conn, selected_sport, conv_dir, lo_streak, hi_streak, 'ride')
                conv_fade = cached_convergent_gt(conn, selected_sport, conv_dir, lo_streak, hi_streak, 'fade')

            if len(conv_ride) > 0:
                n_games = conv_ride['n_games'].iloc[0]
                st.info(f"Found **{n_games}** games where both teams were on TT {conv_dir} streaks ({streak_label})")

                gt_bl_ride_dir = 'over' if conv_dir == 'OVER' else 'under'
                gt_bl_fade_dir = 'under' if conv_dir == 'OVER' else 'over'
                gt_bl_ride = baseline_ou_coverage(conn, selected_sport, handicap_range=(0, 20), direction=gt_bl_ride_dir)
                gt_bl_fade = baseline_ou_coverage(conn, selected_sport, handicap_range=(0, 20), direction=gt_bl_fade_dir)

                # --- RIDE Chart ---
                if len(gt_bl_ride) > 0:
                    conv_ride_m = conv_ride.merge(gt_bl_ride[['handicap', 'baseline_cover_pct']], on='handicap')
                    conv_ride_m['edge'] = conv_ride_m['cover_pct'] - conv_ride_m['baseline_cover_pct']

                    st.subheader(f"Both {conv_dir} {streak_label} — Game Total: Bet {conv_dir} (Ride)")

                    fig_cr = go.Figure()
                    fig_cr.add_trace(go.Bar(
                        name=f'GT {conv_dir} Coverage (n={n_games})',
                        x=conv_ride_m['handicap'],
                        y=conv_ride_m['cover_pct'] * 100,
                        marker_color=['#2ecc71' if e >= 0 else '#e74c3c' for e in conv_ride_m['edge']],
                        text=conv_ride_m['cover_pct'].apply(lambda x: f"{x:.1%}"),
                        textposition='outside'
                    ))
                    fig_cr.add_trace(go.Scatter(
                        name=f'{selected_sport} GT Baseline ({conv_dir})',
                        x=gt_bl_ride['handicap'],
                        y=gt_bl_ride['baseline_cover_pct'] * 100,
                        mode='lines+markers',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8)
                    ))
                    fig_cr.update_layout(
                        xaxis_title="Handicap (points)",
                        yaxis_title="Cover Rate (%)",
                        yaxis=dict(range=[0, 100]),
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        barmode='overlay'
                    )
                    st.plotly_chart(fig_cr, use_container_width=True)

                    with st.expander("View Ride Data Table"):
                        d = conv_ride_m.copy()
                        d['cover_pct_fmt'] = d['cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['baseline_fmt'] = d['baseline_cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['edge_fmt'] = d['edge'].apply(lambda x: f"{x*100:+.1f}%")
                        d = d.rename(columns={'handicap': 'Handicap', 'covers': 'Covers', 'total': 'Total',
                                               'cover_pct_fmt': 'Cover %', 'baseline_fmt': 'Baseline %', 'edge_fmt': 'Edge'})
                        st.dataframe(d[['Handicap', 'Total', 'Covers', 'Cover %', 'Baseline %', 'Edge']], hide_index=True)

                # --- FADE Chart ---
                if len(conv_fade) > 0 and len(gt_bl_fade) > 0:
                    conv_fade_m = conv_fade.merge(gt_bl_fade[['handicap', 'baseline_cover_pct']], on='handicap')
                    conv_fade_m['edge'] = conv_fade_m['cover_pct'] - conv_fade_m['baseline_cover_pct']

                    st.subheader(f"Both {conv_dir} {streak_label} — Game Total: Bet {conv_opposite} (Fade)")

                    fig_cf = go.Figure()
                    fig_cf.add_trace(go.Bar(
                        name=f'GT {conv_opposite} Coverage (n={n_games})',
                        x=conv_fade_m['handicap'],
                        y=conv_fade_m['cover_pct'] * 100,
                        marker_color=['#2ecc71' if e >= 0 else '#e74c3c' for e in conv_fade_m['edge']],
                        text=conv_fade_m['cover_pct'].apply(lambda x: f"{x:.1%}"),
                        textposition='outside'
                    ))
                    fig_cf.add_trace(go.Scatter(
                        name=f'{selected_sport} GT Baseline ({conv_opposite})',
                        x=gt_bl_fade['handicap'],
                        y=gt_bl_fade['baseline_cover_pct'] * 100,
                        mode='lines+markers',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8)
                    ))
                    fig_cf.update_layout(
                        xaxis_title="Handicap (points)",
                        yaxis_title="Cover Rate (%)",
                        yaxis=dict(range=[0, 100]),
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        barmode='overlay'
                    )
                    st.plotly_chart(fig_cf, use_container_width=True)

                    with st.expander("View Fade Data Table"):
                        d = conv_fade_m.copy()
                        d['cover_pct_fmt'] = d['cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['baseline_fmt'] = d['baseline_cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['edge_fmt'] = d['edge'].apply(lambda x: f"{x*100:+.1f}%")
                        d = d.rename(columns={'handicap': 'Handicap', 'covers': 'Covers', 'total': 'Total',
                                               'cover_pct_fmt': 'Cover %', 'baseline_fmt': 'Baseline %', 'edge_fmt': 'Edge'})
                        st.dataframe(d[['Handicap', 'Total', 'Covers', 'Cover %', 'Baseline %', 'Edge']], hide_index=True)
            else:
                st.warning(f"No convergent matchups found where both teams were on TT {conv_dir} streaks ({streak_label})")

    # =========================================================================
    # TEAM TOTAL STREAKS
    # =========================================================================
    with streak_type_tabs[2]:
        st.markdown(f"""
        **Team Totals** track each team's individual score vs their own O/U line
        (e.g., "Lakers O/U 115.5"), NOT the combined game total.

        After a team goes OVER/UNDER their team total N+ games in a row,
        how does their **next game's team total** perform? Analyzed with both
        **RIDE** (bet with streak) and **FADE** (bet against) strategies for **{selected_sport}**.
        """)

        # Load TT streak summary
        with st.spinner("Loading team totals streak summary..."):
            tt_summary = cached_tt_streak_summary(conn, selected_sport)

        if len(tt_summary) == 0:
            st.warning("No team totals streak data found. Team total lines may not be available yet.")
        else:
            # Show summary tables
            st.subheader("Available Team Totals Streak Data")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**OVER Streaks**")
                tt_over = tt_summary[tt_summary['streak_type'] == 'OVER'][['streak_length', 'situations']]
                tt_over.columns = ['Streak Length', 'Situations']
                st.dataframe(tt_over, hide_index=True)

            with col2:
                st.markdown("**UNDER Streaks**")
                tt_under = tt_summary[tt_summary['streak_type'] == 'UNDER'][['streak_length', 'situations']]
                tt_under.columns = ['Streak Length', 'Situations']
                st.dataframe(tt_under, hide_index=True)

            # Controls
            st.markdown("---")
            st.subheader("Analyze Team Total Coverage After Streak")

            col1, col2 = st.columns(2)
            with col1:
                tt_streak_len = st.selectbox("Streak Length", list(range(1, 11)), index=2, key="tt_streak_len")
            with col2:
                tt_streak_type = st.selectbox("Streak Type", ["OVER", "UNDER"], key="tt_streak_type")

            tt_opposite = "UNDER" if tt_streak_type == "OVER" else "OVER"

            # Sample size
            tt_sel = tt_summary[(tt_summary['streak_length'] == tt_streak_len) & (tt_summary['streak_type'] == tt_streak_type)]
            if len(tt_sel) > 0:
                # Sum situations for this length and above (N+ analysis)
                tt_n_plus = tt_summary[(tt_summary['streak_length'] >= tt_streak_len) & (tt_summary['streak_type'] == tt_streak_type)]
                tt_sample = int(tt_n_plus['situations'].sum())
                st.info(f"Sample size: **{tt_sample}** situations where a team had a {tt_streak_len}+ game TT {tt_streak_type} streak")

                # Load analysis data
                with st.spinner("Analyzing team totals coverage..."):
                    tt_ride = cached_tt_streak_continuation(conn, selected_sport, tt_streak_len, tt_streak_type, 'ride')
                    tt_fade = cached_tt_streak_continuation(conn, selected_sport, tt_streak_len, tt_streak_type, 'fade')
                    tt_bl_over = cached_tt_baseline_coverage(conn, selected_sport, 'over')
                    tt_bl_under = cached_tt_baseline_coverage(conn, selected_sport, 'under')

                # Pick the correct baseline for each direction
                ride_bl = tt_bl_over if tt_streak_type == "OVER" else tt_bl_under
                fade_bl = tt_bl_under if tt_streak_type == "OVER" else tt_bl_over

                # --- RIDE Chart ---
                if len(tt_ride) > 0 and len(ride_bl) > 0:
                    tt_ride = tt_ride.merge(ride_bl[['handicap', 'baseline_cover_pct']], on='handicap')
                    tt_ride['edge'] = tt_ride['cover_pct'] - tt_ride['baseline_cover_pct']

                    st.subheader(f"After {tt_streak_len}+ Game TT {tt_streak_type} Streak — Bet {tt_streak_type} (Ride)")

                    fig_ride = go.Figure()
                    fig_ride.add_trace(go.Bar(
                        name=f'TT {tt_streak_type} Coverage (Ride)',
                        x=tt_ride['handicap'],
                        y=tt_ride['cover_pct'] * 100,
                        marker_color=['#2ecc71' if e >= 0 else '#e74c3c' for e in tt_ride['edge']],
                        text=tt_ride['cover_pct'].apply(lambda x: f"{x:.1%}"),
                        textposition='outside'
                    ))
                    fig_ride.add_trace(go.Scatter(
                        name=f'{selected_sport} TT Baseline ({tt_streak_type})',
                        x=ride_bl['handicap'],
                        y=ride_bl['baseline_cover_pct'] * 100,
                        mode='lines+markers',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8)
                    ))
                    fig_ride.update_layout(
                        xaxis_title="Handicap (points)",
                        yaxis_title="Cover Rate (%)",
                        yaxis=dict(range=[0, 100]),
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        barmode='overlay'
                    )
                    st.plotly_chart(fig_ride, use_container_width=True)

                    with st.expander("View Ride Data Table"):
                        d = tt_ride.copy()
                        d['cover_pct_fmt'] = d['cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['baseline_fmt'] = d['baseline_cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['edge_fmt'] = d['edge'].apply(lambda x: f"{x*100:+.1f}%")
                        d = d.rename(columns={'handicap': 'Handicap', 'covers': 'Covers', 'total': 'Total',
                                               'cover_pct_fmt': 'Cover %', 'baseline_fmt': 'Baseline %', 'edge_fmt': 'Edge'})
                        st.dataframe(d[['Handicap', 'Total', 'Covers', 'Cover %', 'Baseline %', 'Edge']], hide_index=True)

                # --- FADE Chart ---
                if len(tt_fade) > 0 and len(fade_bl) > 0:
                    tt_fade = tt_fade.merge(fade_bl[['handicap', 'baseline_cover_pct']], on='handicap')
                    tt_fade['edge'] = tt_fade['cover_pct'] - tt_fade['baseline_cover_pct']

                    st.subheader(f"After {tt_streak_len}+ Game TT {tt_streak_type} Streak — Bet {tt_opposite} (Fade)")

                    fig_fade = go.Figure()
                    fig_fade.add_trace(go.Bar(
                        name=f'TT {tt_opposite} Coverage (Fade)',
                        x=tt_fade['handicap'],
                        y=tt_fade['cover_pct'] * 100,
                        marker_color=['#2ecc71' if e >= 0 else '#e74c3c' for e in tt_fade['edge']],
                        text=tt_fade['cover_pct'].apply(lambda x: f"{x:.1%}"),
                        textposition='outside'
                    ))
                    fig_fade.add_trace(go.Scatter(
                        name=f'{selected_sport} TT Baseline ({tt_opposite})',
                        x=fade_bl['handicap'],
                        y=fade_bl['baseline_cover_pct'] * 100,
                        mode='lines+markers',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8)
                    ))
                    fig_fade.update_layout(
                        xaxis_title="Handicap (points)",
                        yaxis_title="Cover Rate (%)",
                        yaxis=dict(range=[0, 100]),
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        barmode='overlay'
                    )
                    st.plotly_chart(fig_fade, use_container_width=True)

                    with st.expander("View Fade Data Table"):
                        d = tt_fade.copy()
                        d['cover_pct_fmt'] = d['cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['baseline_fmt'] = d['baseline_cover_pct'].apply(lambda x: f"{x:.1%}")
                        d['edge_fmt'] = d['edge'].apply(lambda x: f"{x*100:+.1f}%")
                        d = d.rename(columns={'handicap': 'Handicap', 'covers': 'Covers', 'total': 'Total',
                                               'cover_pct_fmt': 'Cover %', 'baseline_fmt': 'Baseline %', 'edge_fmt': 'Edge'})
                        st.dataframe(d[['Handicap', 'Total', 'Covers', 'Cover %', 'Baseline %', 'Edge']], hide_index=True)

                if len(tt_ride) == 0 and len(tt_fade) == 0:
                    st.warning("No data found for this streak configuration")
            else:
                st.warning(f"No {tt_streak_len}-game TT {tt_streak_type} streaks found")


# =============================================================================
# Page: Today's Picks (Primary Navigation)
# =============================================================================

@timed
def page_todays_picks():
    st.title("🎲 Today's Picks")
    st.markdown("""
    **Today's Picks** shows streak-based betting recommendations for games scheduled today.
    Recommendations are generated by cross-referencing today's games against detected streak patterns.
    """)

    # Use a fresh connection — the global cached conn can go stale with long-lived Streamlit sessions
    _engine = get_engine()
    _conn = _engine.connect().execution_options(isolation_level="AUTOCOMMIT")

    # Get pre-computed patterns (fast database lookup)
    patterns = get_cached_patterns(_conn, min_sample=30, min_edge=0.05)

    # Show last updated timestamp
    last_updated = get_games_last_updated(_conn)
    if last_updated:
        st.caption(f"Games last updated: {last_updated.strftime('%Y-%m-%d %I:%M %p')} UTC")
    else:
        st.warning("No games loaded yet. Run the collect_todays_games Lambda to fetch today's schedule.")

    # Sport filter and handicap view
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        picks_sport = st.selectbox(
            "Sport",
            ["All", "NFL", "NBA", "NCAAM"],
            key="todays_picks_sport_main"
        )
    with filter_col2:
        handicap_options = ["All Handicaps", "H=0", "H=2 / H=3", "H=5 / H=10", "H=8 / H=14", "H=10"]
        handicap_view = st.selectbox("Handicap View", handicap_options, index=0, key="handicap_view")

    sport_to_use = picks_sport if picks_sport != "All" else "All"

    # Get pre-computed recommendations (fast - reads from cache table)
    # Falls back to live computation if cache is empty
    with st.spinner("Loading today's picks..."):
        game_recommendations = get_cached_recommendations(_conn, sport_to_use)
        closing_lines = get_closing_lines(_conn, sport_to_use)

    # Filter to only games with detected edges
    games_with_edges = [g for g in game_recommendations if g.recommendations]

    if not games_with_edges:
        if not game_recommendations:
            st.info(f"No games scheduled today for {picks_sport}.")
        else:
            st.info(f"No betting edges detected in {len(game_recommendations)} games today.")
    else:
        st.markdown(f"**{len(games_with_edges)} games with betting edges** (out of {len(game_recommendations)} total)")

        # Sort by max edge descending
        def get_max_edge(game):
            return max(rec.edge for rec in game.recommendations)

        games_sorted = sorted(games_with_edges, key=get_max_edge, reverse=True)

        for game in games_sorted:
            # Game card
            with st.container():
                # Header with team names and ratings
                st.markdown(f"### {game.away_team} @ {game.home_team}")

                # Team ratings row - prominent display
                rating_col1, rating_col2, rating_col3 = st.columns([2, 2, 1])

                with rating_col1:
                    # Away team rating
                    if game.away_ats_rating is not None and game.away_tier:
                        st.markdown(f"**🚗 {game.away_team}:** {game.away_tier} ({game.away_ats_rating:.3f})")
                    else:
                        st.markdown(f"**🚗 {game.away_team}:** No rating")

                with rating_col2:
                    # Home team rating
                    if game.home_ats_rating is not None and game.home_tier:
                        st.markdown(f"**🏠 {game.home_team}:** {game.home_tier} ({game.home_ats_rating:.3f})")
                    else:
                        st.markdown(f"**🏠 {game.home_team}:** No rating")

                with rating_col3:
                    st.caption(f"{game.sport} | {game.game_time}")

                # Lines info (spread, total, team totals) with closing line movement
                closing = closing_lines.get((game.sport, game.home_team, game.away_team), {})
                lines_parts = []
                if game.spread is not None:
                    spread_display = f"+{game.spread}" if game.spread > 0 else str(game.spread)
                    spread_text = f"Spread: {game.home_team} {spread_display}"
                    cl_spread = closing.get('closing_spread')
                    if cl_spread is not None:
                        cl_display = f"+{cl_spread}" if cl_spread > 0 else str(cl_spread)
                        arrow = " ↓" if cl_spread < game.spread else (" ↑" if cl_spread > game.spread else "")
                        spread_text += f" → {cl_display}{arrow}"
                    spread_text += f" ({game.spread_source or 'Unknown'})"
                    lines_parts.append(spread_text)
                if game.total is not None:
                    total_text = f"Total: {game.total}"
                    cl_total = closing.get('closing_total')
                    if cl_total is not None:
                        arrow = " ↑" if cl_total > game.total else (" ↓" if cl_total < game.total else "")
                        total_text += f" → {cl_total}{arrow}"
                    total_text += f" ({game.total_source or 'Unknown'})"
                    lines_parts.append(total_text)
                if game.home_team_total is not None or game.away_team_total is not None:
                    tt_parts = []
                    cl_away_tt = closing.get('closing_away_tt')
                    cl_home_tt = closing.get('closing_home_tt')
                    if game.away_team_total is not None:
                        att = f"{game.away_team} {game.away_team_total}"
                        if cl_away_tt is not None:
                            arrow = " ↑" if cl_away_tt > game.away_team_total else (" ↓" if cl_away_tt < game.away_team_total else "")
                            att += f" → {cl_away_tt}{arrow}"
                        tt_parts.append(att)
                    if game.home_team_total is not None:
                        htt = f"{game.home_team} {game.home_team_total}"
                        if cl_home_tt is not None:
                            arrow = " ↑" if cl_home_tt > game.home_team_total else (" ↓" if cl_home_tt < game.home_team_total else "")
                            htt += f" → {cl_home_tt}{arrow}"
                        tt_parts.append(htt)
                    lines_parts.append(f"Team Totals: {' | '.join(tt_parts)}")
                if lines_parts:
                    st.caption(" | ".join(lines_parts))

                # Recommendations - group by bet_on
                teams_recommended = {}
                for rec in game.recommendations:
                    if rec.bet_on not in teams_recommended:
                        teams_recommended[rec.bet_on] = []
                    teams_recommended[rec.bet_on].append(rec)

                for bet_team, recs in teams_recommended.items():
                    combined_conf = get_combined_confidence(recs)
                    conf_color = "🟢" if combined_conf == "HIGH" else ("🟡" if combined_conf == "MEDIUM" else "⚪")

                    st.markdown(f"#### {conf_color} BET ON: {bet_team}")

                    for rec in recs:
                        source_icons = {
                            'streak': '📈',
                            'ou_streak': '🔄',
                            'tt_streak': '👤',
                        }
                        source_icon = source_icons.get(rec.source, '📊')
                        st.markdown(f"- {source_icon} **{rec.source.replace('_', ' ').title()}**: {rec.rationale}")

                        # Analysis link hint
                        analysis_tab = {'streak': 'Spread Streaks (ATS)', 'ou_streak': 'Game Total O/U Streaks', 'tt_streak': 'Team Total Streaks'}
                        tab_name = analysis_tab.get(rec.source, 'Streak Analysis')
                        # Extract streak length from rationale (e.g., "3-game WIN streak")
                        streak_match = re.search(r'(\d+)-game\s+(WIN|LOSS|OVER|UNDER)', rec.rationale or '')
                        if streak_match:
                            s_len, s_type = streak_match.group(1), streak_match.group(2)
                            st.caption(f"  See analysis: Streak Analysis > {tab_name} > {s_len}-game {s_type} streak")

                        # Handicap tier breakdown
                        if rec.tier_data:
                            # Determine which tiers to show based on handicap_view selector
                            show_tiers = rec.tier_data
                            highlight_idx = None
                            if handicap_view != "All Handicaps":
                                # Map selector to handicap values for highlighting
                                view_map = {
                                    "H=0": [0], "H=2 / H=3": [2, 3],
                                    "H=5 / H=10": [5, 10], "H=8 / H=14": [8, 14],
                                    "H=10": [10],
                                }
                                highlight_handicaps = view_map.get(handicap_view, [])
                                for idx, td in enumerate(show_tiers):
                                    if td['handicap'] in highlight_handicaps:
                                        highlight_idx = idx
                                        break

                            tier_cols = st.columns(len(show_tiers))
                            for i, td in enumerate(show_tiers):
                                with tier_cols[i]:
                                    display_edge = abs(td['edge'])
                                    has_edge = display_edge >= 0.03
                                    icon = "+" if has_edge else " "
                                    is_highlighted = (highlight_idx == i)
                                    label = f"{icon} {td['label']}"
                                    if is_highlighted:
                                        label = f">> {td['label']}"
                                    st.metric(
                                        label=label,
                                        value=f"{td['cover_rate']:.0%}",
                                        delta=f"{display_edge:.1%} edge" if has_edge else "No edge",
                                    )

                    st.caption(f"Combined Confidence: {combined_conf}")

                # Team Details and streak info in expander
                with st.expander("Team Details & Streaks"):
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.markdown(f"**{game.home_team}** (Home)")
                        if game.home_ats_rating is not None:
                            st.write(f"ATS Rating: {game.home_ats_rating:.3f} ({game.home_tier})")
                        if game.home_streak:
                            st.write(f"ATS Streak: {game.home_streak[0]}-game {game.home_streak[1].lower()}")
                        else:
                            st.write("No current ATS streak")
                        # O/U streak
                        if game.home_ou_streak:
                            ou_icon = "🟢" if game.home_ou_streak[1] == 'OVER' else "🔴"
                            st.write(f"O/U Streak: {ou_icon} {game.home_ou_streak[0]}-game {game.home_ou_streak[1]}")
                        else:
                            st.write("No current O/U streak")
                        # TT streak
                        if game.home_tt_streak:
                            tt_icon = "🟢" if game.home_tt_streak[1] == 'OVER' else "🔴"
                            st.write(f"TT Streak: {tt_icon} {game.home_tt_streak[0]}-game {game.home_tt_streak[1]}")
                        else:
                            st.write("No current TT streak")

                    with detail_col2:
                        st.markdown(f"**{game.away_team}** (Away)")
                        if game.away_ats_rating is not None:
                            st.write(f"ATS Rating: {game.away_ats_rating:.3f} ({game.away_tier})")
                        if game.away_streak:
                            st.write(f"ATS Streak: {game.away_streak[0]}-game {game.away_streak[1].lower()}")
                        else:
                            st.write("No current ATS streak")
                        # O/U streak
                        if game.away_ou_streak:
                            ou_icon = "🟢" if game.away_ou_streak[1] == 'OVER' else "🔴"
                            st.write(f"O/U Streak: {ou_icon} {game.away_ou_streak[0]}-game {game.away_ou_streak[1]}")
                        else:
                            st.write("No current O/U streak")
                        # TT streak
                        if game.away_tt_streak:
                            tt_icon = "🟢" if game.away_tt_streak[1] == 'OVER' else "🔴"
                            st.write(f"TT Streak: {tt_icon} {game.away_tt_streak[0]}-game {game.away_tt_streak[1]}")
                        else:
                            st.write("No current TT streak")

                st.markdown("---")

    # Quick-ask AI widget — collapsible
    with st.expander("🤖 **Ask AI** — query the database in plain English", expanded=False):
        quick_q = st.text_input(
            "Ask anything...",
            key="quick_ask_input",
            placeholder="e.g., What's the Knicks' ATS record at home this season?",
            label_visibility="collapsed",
        )
        if quick_q:
            with st.spinner("Thinking..."):
                engine = get_engine()
                result = ask_question(quick_q, engine)
            st.markdown(result.answer)
            if result.sql:
                with st.expander("View SQL"):
                    st.code(result.sql, language="sql")
            if result.dataframe is not None and not result.dataframe.empty:
                with st.expander("View raw data"):
                    st.dataframe(result.dataframe, hide_index=True, use_container_width=True)

    # Current team streaks (collapsible) - use cached for speed
    with st.expander("📊 All Current Team Streaks (ATS)"):
        for sport in ['NFL', 'NBA', 'NCAAM']:
            streaks = get_cached_streaks(_conn, sport)
            if streaks:
                st.markdown(f"**{sport}**")
                streak_data = []
                for team, info in sorted(streaks.items(), key=lambda x: -x[1]['streak_length']):
                    streak_data.append({
                        'Team': team,
                        'Streak': f"{info['streak_length']} {info['streak_type']}",
                        'Length': info['streak_length'],
                        'Type': info['streak_type']
                    })
                streak_df = pd.DataFrame(streak_data)
                st.dataframe(streak_df[['Team', 'Streak']], hide_index=True, use_container_width=True)
                st.markdown("")

    # O/U streaks (collapsible) - use cached for speed
    with st.expander("🔄 All Current O/U Streaks"):
        for sport in ['NFL', 'NBA', 'NCAAM']:
            ou_streaks = get_cached_ou_streaks(_conn, sport)
            if ou_streaks:
                st.markdown(f"**{sport}**")
                ou_streak_data = []
                for team, info in sorted(ou_streaks.items(), key=lambda x: -x[1]['streak_length']):
                    ou_icon = "🟢" if info['streak_type'] == 'OVER' else "🔴"
                    ou_streak_data.append({
                        'Team': team,
                        'Streak': f"{ou_icon} {info['streak_length']} {info['streak_type']}",
                        'Length': info['streak_length'],
                        'Type': info['streak_type']
                    })
                ou_streak_df = pd.DataFrame(ou_streak_data)
                st.dataframe(ou_streak_df[['Team', 'Streak']], hide_index=True, use_container_width=True)
                st.markdown("")

    # TT streaks (collapsible) - use cached for speed
    with st.expander("👤 All Current Team Total Streaks"):
        for sport in ['NFL', 'NBA', 'NCAAM']:
            tt_streaks = get_cached_tt_streaks(_conn, sport)
            if tt_streaks:
                st.markdown(f"**{sport}**")
                tt_streak_data = []
                for team, info in sorted(tt_streaks.items(), key=lambda x: -x[1]['streak_length']):
                    tt_icon = "🟢" if info['streak_type'] == 'OVER' else "🔴"
                    tt_streak_data.append({
                        'Team': team,
                        'Streak': f"{tt_icon} {info['streak_length']} {info['streak_type']}",
                        'Length': info['streak_length'],
                        'Type': info['streak_type']
                    })
                tt_streak_df = pd.DataFrame(tt_streak_data)
                st.dataframe(tt_streak_df[['Team', 'Streak']], hide_index=True, use_container_width=True)
                st.markdown("")

    # Close the fresh connection
    _conn.close()


# =============================================================================
# Page: Power Rankings
# =============================================================================

@timed
def page_power_rankings():
    st.title("🏆 Power Rankings")

    # Top-level Spread/Totals toggle
    rankings_type = st.tabs(["ATS Rankings", "O/U Bias Rankings"])

    # =========================================================================
    # ATS RANKINGS
    # =========================================================================
    with rankings_type[0]:
        st.markdown(f"""
        **Network-based team ratings** using iterative strength propagation for **{selected_sport}**.

        Two rating types are computed:
        - **Win Rating**: True team strength based on game outcomes
        - **ATS Rating**: Market-beating ability based on spread coverage

        The **Market Gap** (ATS - Win) reveals market efficiency:
        - 🟢 **Positive gap**: Market undervalues this team (potential betting edge)
        - 🔴 **Negative gap**: Market overvalues this team (fade candidate)
        """)

        # Settings
        col1, col2, col3 = st.columns(3)
        with col1:
            min_games = st.slider("Min Games for Reliable", 1, 20, 5, key="pr_min_games")
        with col2:
            sort_by = st.selectbox("Sort By", ["Market Gap", "Win Rating", "ATS Rating"], key="pr_sort")
        with col3:
            show_unreliable = st.checkbox("Show teams < min games", value=False, key="pr_unreliable")

        # Get rankings (from pre-computed cache for fast loads)
        with st.spinner("Loading power rankings..."):
            rankings = get_cached_rankings(conn, selected_sport, min_games=min_games)

        if not rankings:
            st.warning(f"No ranking data found for {selected_sport}")
        else:
            # Filter unreliable if needed
            if not show_unreliable:
                display_rankings = [r for r in rankings if r.is_reliable]
            else:
                display_rankings = rankings

            if not display_rankings:
                st.warning("No teams meet the minimum games threshold")
            else:
                # Sort based on selection
                if sort_by == "Win Rating":
                    display_rankings = sorted(display_rankings, key=lambda x: x.win_rating, reverse=True)
                elif sort_by == "ATS Rating":
                    display_rankings = sorted(display_rankings, key=lambda x: x.ats_rating, reverse=True)
                # Market Gap is already default sort

                # Summary metrics
                reliable_count = len([r for r in rankings if r.is_reliable])
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Teams", len(rankings))
                col2.metric(f"Reliable ({min_games}+ games)", reliable_count)
                col3.metric("Most Undervalued", display_rankings[0].team if display_rankings else "N/A")

                # Main rankings table
                st.markdown("---")
                st.subheader("Rankings Table")

                # Build dataframe for display
                table_data = []
                for r in display_rankings:
                    gap_color = "🟢" if r.market_gap > 0.02 else ("🔴" if r.market_gap < -0.02 else "⚪")
                    table_data.append({
                        "Team": r.team,
                        "Win Rtg": f"{r.win_rating:.3f}",
                        "Win Rank": r.win_rank,
                        "ATS Rtg": f"{r.ats_rating:.3f}",
                        "ATS Rank": r.ats_rank,
                        "Gap": f"{gap_color} {r.market_gap:+.3f}",
                        "Record": r.win_record,
                        "ATS Rec": r.ats_record,
                        "Games": r.games_analyzed,
                    })

                df = pd.DataFrame(table_data)
                st.dataframe(df, hide_index=True, use_container_width=True)

                # Visualization
                st.markdown("---")
                st.subheader("Rating Comparison")

                viz_tab1, viz_tab2 = st.tabs(["Market Gap Chart", "Win vs ATS Scatter"])

                with viz_tab1:
                    # Bar chart of market gap (top and bottom)
                    top_n = min(15, len(display_rankings))

                    fig = go.Figure()

                    # Top by market gap (most undervalued)
                    top_teams = display_rankings[:top_n]
                    fig.add_trace(go.Bar(
                        y=[r.team for r in top_teams],
                        x=[r.market_gap * 100 for r in top_teams],
                        orientation='h',
                        marker_color=['#2ecc71' if r.market_gap > 0 else '#e74c3c' for r in top_teams],
                        text=[f"{r.market_gap:+.1%}" for r in top_teams],
                        textposition='auto',
                        name='Market Gap'
                    ))

                    fig.add_vline(x=0, line_dash="dash", line_color="gray")

                    fig.update_layout(
                        title=f"Top {top_n} Teams by Market Gap ({selected_sport})",
                        xaxis_title="Market Gap (ATS Rating - Win Rating) %",
                        yaxis_title="",
                        height=max(400, top_n * 30),
                        yaxis=dict(autorange="reversed")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with viz_tab2:
                    # Scatter plot: Win Rating vs ATS Rating
                    scatter_data = pd.DataFrame([{
                        'team': r.team,
                        'win_rating': r.win_rating,
                        'ats_rating': r.ats_rating,
                        'market_gap': r.market_gap,
                        'games': r.games_analyzed
                    } for r in display_rankings])

                    fig = px.scatter(
                        scatter_data,
                        x='win_rating',
                        y='ats_rating',
                        hover_name='team',
                        color='market_gap',
                        color_continuous_scale=['#e74c3c', '#f0f0f0', '#2ecc71'],
                        color_continuous_midpoint=0,
                        size='games',
                        title=f"Win Rating vs ATS Rating ({selected_sport})",
                        labels={
                            'win_rating': 'Win Rating (True Strength)',
                            'ats_rating': 'ATS Rating (Market-Beating)',
                            'market_gap': 'Market Gap'
                        }
                    )

                    # Add diagonal line (where Win = ATS)
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        name='Win = ATS',
                        showlegend=False
                    ))

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("""
                    **How to read this chart:**
                    - Points **above** the diagonal line have higher ATS rating than Win rating → undervalued by market
                    - Points **below** the diagonal line have lower ATS rating than Win rating → overvalued by market
                    - Color intensity shows the magnitude of the gap
                    - Size represents games played
                    """)

    # =========================================================================
    # O/U BIAS RANKINGS
    # =========================================================================
    with rankings_type[1]:
        st.markdown(f"""
        **O/U Bias Rankings** for **{selected_sport}** - Which teams tend to go OVER or UNDER?

        - 🟢 **Positive Avg Margin**: Games tend to go OVER (high-scoring offense or poor defense)
        - 🔴 **Negative Avg Margin**: Games tend to go UNDER (strong defense or slow pace)
        """)

        # Settings
        col1, col2 = st.columns(2)
        with col1:
            ou_min_games = st.slider("Minimum Games", 1, 20, 5, key="ou_pr_min_games")
        with col2:
            ou_sort_by = st.selectbox("Sort By", ["OVER Bias (Highest)", "UNDER Bias (Lowest)", "OVER %"], key="ou_pr_sort")

        # Get O/U data for all teams
        ou_teams_df = micro_ou_all_teams(conn, selected_sport, handicap=0, min_games=ou_min_games)

        if len(ou_teams_df) == 0:
            st.warning(f"No totals data found for {selected_sport}")
        else:
            # Sort based on selection
            if ou_sort_by == "OVER Bias (Highest)":
                ou_teams_df = ou_teams_df.sort_values('avg_total_margin', ascending=False)
            elif ou_sort_by == "UNDER Bias (Lowest)":
                ou_teams_df = ou_teams_df.sort_values('avg_total_margin', ascending=True)
            else:  # OVER %
                ou_teams_df = ou_teams_df.sort_values('over_pct', ascending=False)

            # Summary metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Teams", len(ou_teams_df))

            over_biased = ou_teams_df[ou_teams_df['avg_total_margin'] > 2]
            under_biased = ou_teams_df[ou_teams_df['avg_total_margin'] < -2]
            col2.metric("OVER Biased Teams", len(over_biased))
            col3.metric("UNDER Biased Teams", len(under_biased))

            # Main rankings table
            st.markdown("---")
            st.subheader("O/U Bias Rankings")

            display_df = ou_teams_df.copy()
            display_df['bias'] = display_df['avg_total_margin'].apply(
                lambda x: '🟢 OVER' if x > 2 else ('🔴 UNDER' if x < -2 else '⚪ Neutral')
            )
            display_df['over_pct_str'] = (display_df['over_pct'] * 100).round(1).astype(str) + '%'
            display_df['avg_margin_str'] = display_df['avg_total_margin'].apply(lambda x: f"{x:+.1f}")
            display_df['record'] = display_df.apply(
                lambda r: f"{int(r['overs'])}-{int(r['unders'])}-{int(r['pushes'])}",
                axis=1
            )

            st.dataframe(
                display_df[['team', 'games', 'record', 'over_pct_str', 'avg_margin_str', 'bias']].rename(columns={
                    'over_pct_str': 'OVER %',
                    'avg_margin_str': 'Avg Margin',
                    'bias': 'Bias'
                }),
                hide_index=True,
                use_container_width=True
            )

            # Visualization - O/U Bias Bar Chart
            st.markdown("---")
            st.subheader("O/U Bias Visualization")

            top_n = min(20, len(ou_teams_df))
            chart_df = ou_teams_df.head(top_n)

            fig = go.Figure()

            fig.add_trace(go.Bar(
                y=chart_df['team'],
                x=chart_df['avg_total_margin'],
                orientation='h',
                marker_color=['#2ecc71' if x > 0 else '#e74c3c' for x in chart_df['avg_total_margin']],
                text=chart_df['avg_total_margin'].apply(lambda x: f"{x:+.1f}"),
                textposition='auto',
                name='Avg Total Margin'
            ))

            fig.add_vline(x=0, line_dash="dash", line_color="gray")

            fig.update_layout(
                title=f"Top {top_n} Teams by O/U Bias ({selected_sport})",
                xaxis_title="Avg Total Margin (Positive = OVER bias)",
                yaxis_title="",
                height=max(400, top_n * 25),
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption("""
            **How to read this chart:**
            - **Positive margin**: Games average this many points OVER the total → look for OVER bets
            - **Negative margin**: Games average this many points UNDER the total → look for UNDER bets
            """)

    # Interpretation guide
    with st.expander("📖 How to Use Power Rankings"):
        st.markdown("""
        ### Understanding the Ratings

        **Win Rating** measures true team strength based on game outcomes:
        - Higher = better team (wins more games, by larger margins)
        - Based on actual wins/losses, weighted by opponent strength and recency

        **ATS Rating** measures market-beating ability:
        - Higher = team covers spreads more often
        - Since spreads account for team strength, this reveals market mispricing

        ### The Key Insight: Market Gap

        The **Market Gap** (ATS Rating - Win Rating) reveals market efficiency:

        | Gap | Meaning | Action |
        |-----|---------|--------|
        | Large Positive (>5%) | Market consistently undervalues | Look to bet ON this team |
        | Small (+/- 2%) | Market has them priced correctly | No edge |
        | Large Negative (<-5%) | Market consistently overvalues | Look to fade this team |

        ### Important Notes

        - **Sample size matters**: Teams with <5 games may have unreliable ratings
        - **Recency weighted**: Recent games count more (0.92 decay per week)
        - **Network-based**: Ratings consider opponent strength transitively
        - **Margin capped**: Blowouts capped at 20 points to reduce noise
        """)






def calculate_team_slopes(df: pd.DataFrame, team: str) -> dict:
    """Calculate ATS rating slopes for various time windows using linear regression."""
    from scipy.stats import linregress

    team_data = df[df['team'] == team].sort_values('snapshot_date')
    if team_data.empty:
        return {'1w': None, '2w': None, '3w': None}

    latest_date = pd.to_datetime(team_data['snapshot_date']).max()

    slopes = {}
    for label, days in [('1w', 7), ('2w', 14), ('3w', 21)]:
        cutoff = latest_date - pd.Timedelta(days=days)
        window = team_data[pd.to_datetime(team_data['snapshot_date']) >= cutoff]

        if len(window) >= 3:
            x = (pd.to_datetime(window['snapshot_date']) - pd.to_datetime(window['snapshot_date']).min()).dt.days
            y = window['ats_rating']
            slope, _, _, _, _ = linregress(x, y)
            slopes[label] = slope
        else:
            slopes[label] = None

    return slopes


# =============================================================================
# Page: Verification
# =============================================================================

def page_verification():
    st.title("Calculation Verification")
    st.markdown("Drill down into the raw data behind any metric to verify calculations")

    teams = load_teams(selected_sport)
    games = load_games(selected_sport)

    st.subheader("Select a Metric to Verify")

    metric_type = st.selectbox(
        "Metric Type",
        ["League ATS Rate", "Team ATS Rate", "Spread Bucket Performance"]
    )

    if metric_type == "League ATS Rate":
        col1, col2 = st.columns(2)
        with col1:
            v_handicap = st.slider("Handicap", 0, 15, 0, key="v_handicap")
        with col2:
            v_perspective = st.selectbox("Perspective", ["home", "away"], key="v_perspective")

        # Calculate
        rate = ats_cover_rate(games, handicap=v_handicap, perspective=v_perspective)
        w, l, p = ats_record(games, handicap=v_handicap, perspective=v_perspective)

        st.markdown("---")
        st.markdown(f"### Result: {rate:.2%} ({w}-{l}-{p})")

        # Show calculation
        st.markdown("**Calculation Method:**")
        if v_perspective == 'home':
            st.code(f"""
Home team covers when: spread_result + {v_handicap} >= 0

spread_result = home_score - away_score - closing_spread

Example: If spread is -7 (home favored by 7):
  - Home wins by 10 → spread_result = 10 - (-7) = 3 → COVER
  - Home wins by 5 → spread_result = 5 - (-7) = -2 → LOSS
            """)
        else:
            st.code(f"""
Away team covers when: spread_result - {v_handicap} <= 0

spread_result = home_score - away_score - closing_spread

Example: If spread is -7 (home favored by 7):
  - Home wins by 10 → spread_result = 3 → Away LOSS
  - Home wins by 5 → spread_result = -2 → Away COVER
            """)

        # Sample games
        st.markdown("**Sample Games (first 20):**")
        sample = games.head(20).copy()
        sample['adjusted'] = sample['spread_result'] + v_handicap if v_perspective == 'home' else -(sample['spread_result'] - v_handicap)
        sample['covered'] = sample['adjusted'] > 0
        sample['result'] = sample['covered'].apply(lambda x: '✅' if x else '❌')

        st.dataframe(
            sample[['game_date', 'home_team', 'away_team', 'closing_spread',
                    'home_score', 'away_score', 'spread_result', 'adjusted', 'result']],
            hide_index=True,
            use_container_width=True
        )

    elif metric_type == "Team ATS Rate":
        selected_team = st.selectbox("Select Team", teams, key="v_team")
        v_handicap = st.slider("Handicap", 0, 15, 0, key="v_team_handicap")

        team_games = get_games(conn, sport=selected_sport, team=selected_team)

        if len(team_games) > 0:
            rate = team_ats_cover_rate(team_games, handicap=v_handicap)
            w, l, p = team_ats_record(team_games, handicap=v_handicap)

            st.markdown("---")
            st.markdown(f"### {selected_team}: {rate:.2%} ({w}-{l}-{p})")

            st.markdown("**All Games:**")
            display = team_games.copy()
            display['handicap_adjusted'] = display.apply(
                lambda r: r['spread_result'] + v_handicap if r['is_home'] else -(r['spread_result'] - v_handicap),
                axis=1
            )
            display['covered'] = display['handicap_adjusted'] > 0
            display['result'] = display['covered'].apply(lambda x: '✅' if x else '❌')
            display = display.sort_values('game_date', ascending=False)

            st.dataframe(
                display[['game_date', 'home_team', 'away_team', 'is_home', 'closing_spread',
                         'home_score', 'away_score', 'spread_result', 'handicap_adjusted', 'result']],
                hide_index=True,
                use_container_width=True
            )

    else:  # Spread Bucket
        bucket_data = macro_by_spread_bucket(games)
        st.markdown("---")
        st.dataframe(bucket_data, hide_index=True, use_container_width=True)

        selected_bucket = st.selectbox("Select Bucket to Inspect", bucket_data['bucket'].tolist())

        if selected_bucket:
            bucket_row = bucket_data[bucket_data['bucket'] == selected_bucket].iloc[0]
            min_s, max_s = bucket_row['min_spread'], bucket_row['max_spread']

            bucket_games = games[(games['closing_spread'] >= min_s) & (games['closing_spread'] <= max_s)]

            st.markdown(f"**Games in '{selected_bucket}' (spread {min_s} to {max_s}):**")
            display = bucket_games[['game_date', 'home_team', 'away_team',
                                    'closing_spread', 'home_score', 'away_score', 'spread_result']].copy()
            display['home_covered'] = display['spread_result'] >= 0
            display['result'] = display['home_covered'].apply(lambda x: '✅ Home' if x else '❌ Away')
            display = display.sort_values('game_date', ascending=False)

            st.dataframe(display, hide_index=True, use_container_width=True)


# =============================================================================
# Page: Current Streaks (ATS + Totals)
# =============================================================================

def page_current_streaks():
    """Display current streaks for all teams in the selected sport."""
    st.title("🔥 Current Streaks")
    st.caption(f"Track which {selected_sport} teams are currently hot or cold")

    # Add tabs for ATS and Totals streaks
    streak_tabs = st.tabs(["ATS Streaks", "Totals Streaks"])

    # =========================================================================
    # ATS STREAKS TAB
    # =========================================================================
    with streak_tabs[0]:
        st.subheader("🔥 Current ATS Streaks")
        st.caption(f"Track which {selected_sport} teams are currently hot or cold against the spread")

        # Use cached streaks for fast loads
        streaks = get_cached_streaks(conn, selected_sport)
        if not streaks:
            st.warning(f"No ATS streak data available for {selected_sport}")
        else:
            # Get ratings as dict for lookup (also cached for fast loads)
            rankings = {r.team: r for r in (get_cached_rankings(conn, selected_sport) or [])}

            # Build dataframe
            rows = []
            for team, info in streaks.items():
                r = rankings.get(team)
                rows.append({
                    'Team': team,
                    'Length': info['streak_length'],
                    'Streak': f"{info['streak_length']} {info['streak_type']}",
                    'Type': info['streak_type'],
                    'Win Rtg': round(r.win_rating, 3) if r else None,
                    'ATS Rtg': round(r.ats_rating, 3) if r else None,
                })
            df = pd.DataFrame(rows)

            sort_by = st.selectbox("Sort by", ["Length", "Team", "Win Rtg", "ATS Rtg"], key="ats_sort")
            cols = ['Team', 'Length', 'Streak', 'Win Rtg', 'ATS Rtg']

            # Hot streaks
            st.subheader("🔥 Hot Streaks (2+ ATS Wins)")
            hot = df[(df['Type'] == 'WIN') & (df['Length'] >= 2)].sort_values('Length', ascending=False)
            if not hot.empty:
                st.dataframe(hot[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No teams currently on 2+ game ATS win streaks")

            # Cold streaks
            st.subheader("❄️ Cold Streaks (2+ ATS Losses)")
            cold = df[(df['Type'] == 'LOSS') & (df['Length'] >= 2)].sort_values('Length', ascending=False)
            if not cold.empty:
                st.dataframe(cold[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No teams currently on 2+ game ATS loss streaks")

            # All teams
            st.subheader("📊 All Teams")
            df_sorted = df.sort_values(sort_by, ascending=(sort_by == "Team"), na_position='last')
            st.dataframe(df_sorted[cols], use_container_width=True, hide_index=True)

    # =========================================================================
    # TEAM TOTALS STREAKS TAB
    # =========================================================================
    with streak_tabs[1]:
        st.subheader("🎯 Current Team Totals Streaks")
        st.caption(f"Track which {selected_sport} teams are running OVER or UNDER their individual team total line")

        # Compute TT streaks (live — no pre-computed table yet)
        tt_streaks = get_current_tt_streaks(conn, selected_sport)
        if not tt_streaks:
            st.warning(f"No team totals streak data available for {selected_sport}. Team total lines may not be collected yet.")
        else:
            rows = []
            for team, info in tt_streaks.items():
                rows.append({
                    'Team': team,
                    'Length': info['streak_length'],
                    'Streak': f"{info['streak_length']} {info['streak_type']}",
                    'Type': info['streak_type'],
                })
            df = pd.DataFrame(rows)

            sort_by = st.selectbox("Sort by", ["Length", "Team"], key="tt_sort")
            cols = ['Team', 'Length', 'Streak']

            # OVER streaks
            st.subheader("🔥 TT OVER Streaks (2+)")
            hot = df[(df['Type'] == 'OVER') & (df['Length'] >= 2)].sort_values('Length', ascending=False)
            if not hot.empty:
                st.dataframe(hot[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No teams currently on 2+ game TT OVER streaks")

            # UNDER streaks
            st.subheader("❄️ TT UNDER Streaks (2+)")
            cold = df[(df['Type'] == 'UNDER') & (df['Length'] >= 2)].sort_values('Length', ascending=False)
            if not cold.empty:
                st.dataframe(cold[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No teams currently on 2+ game TT UNDER streaks")

            # All teams
            st.subheader("📊 All Teams")
            df_sorted = df.sort_values(sort_by, ascending=(sort_by == "Team"), na_position='last')
            st.dataframe(df_sorted[cols], use_container_width=True, hide_index=True)


# =============================================================================
# Page: AI Research
# =============================================================================

@timed
def page_ai_research():
    """Full chat interface for AI-powered database queries."""
    st.title("🤖 AI Research")
    st.caption(
        "Ask questions about games, teams, and betting trends in plain English. "
        "Powered by AI text-to-SQL."
    )

    # Initialize chat history in session state
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    # Clear chat button
    if st.session_state.ai_chat_history:
        if st.button("Clear chat", type="secondary"):
            st.session_state.ai_chat_history = []
            st.rerun()

    # Display chat history
    for entry in st.session_state.ai_chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("sql"):
                with st.expander("View SQL"):
                    st.code(entry["sql"], language="sql")
            if entry.get("has_data"):
                with st.expander("View raw data"):
                    st.dataframe(
                        pd.DataFrame(entry["data"]),
                        hide_index=True,
                        use_container_width=True,
                    )

    # Chat input
    if prompt := st.chat_input("Ask about games, teams, or trends..."):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                engine = get_engine()
                result = ask_question(
                    prompt,
                    engine,
                    chat_history=st.session_state.ai_chat_history,
                )

            st.markdown(result.answer)
            if result.sql:
                with st.expander("View SQL"):
                    st.code(result.sql, language="sql")
            if result.dataframe is not None and not result.dataframe.empty:
                with st.expander("View raw data"):
                    st.dataframe(result.dataframe, hide_index=True, use_container_width=True)

        # Save to chat history
        entry = {
            "question": prompt,
            "answer": result.answer,
            "sql": result.sql,
            "has_data": result.dataframe is not None and not result.dataframe.empty,
            "data": result.dataframe.to_dict("records") if result.dataframe is not None and not result.dataframe.empty else [],
        }
        st.session_state.ai_chat_history.append(entry)


# =============================================================================
# Page: Prediction Results
# =============================================================================

@timed
def page_prediction_results():
    st.title("Prediction Results")
    st.markdown("Track prediction outcomes and analyze what makes winning picks.")

    engine = get_engine()
    if engine is None:
        st.error("Could not connect to database")
        return

    # Load all resolved + unresolved predictions
    import pandas as pd
    from sqlalchemy import text as sa_text

    with engine.connect() as db_conn:
        all_preds = pd.read_sql(sa_text("SELECT * FROM prediction_results ORDER BY game_date DESC, id"), db_conn)

    if all_preds.empty:
        st.info("No predictions tracked yet. Predictions are captured automatically when the daily precompute Lambda runs.")
        return

    # --- Filters (inline, uses global selected_sport from sidebar) ---
    filtered = all_preds[all_preds['sport'] == selected_sport].copy()

    if filtered.empty:
        st.info(f"No predictions tracked for {selected_sport}.")
        return

    sources = ["All"] + sorted(filtered['source'].dropna().unique().tolist())
    confidences = ["All"] + sorted(filtered['confidence'].dropna().unique().tolist())
    min_date = pd.to_datetime(filtered['game_date']).min().date()
    max_date = pd.to_datetime(filtered['game_date']).max().date()

    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        sel_source = st.selectbox("Source", sources, key="pred_source")
    with fcol2:
        sel_conf = st.selectbox("Confidence", confidences, key="pred_conf")
    with fcol3:
        all_time = st.checkbox("All-Time", value=True, key="pred_all_time")
        if not all_time:
            date_range = st.date_input("Date Range", value=(min_date, max_date), key="pred_dates")

    if sel_source != "All":
        filtered = filtered[filtered['source'] == sel_source]
    if sel_conf != "All":
        filtered = filtered[filtered['confidence'] == sel_conf]
    if not all_time and len(date_range) == 2:
        filtered = filtered[
            (pd.to_datetime(filtered['game_date']).dt.date >= date_range[0]) &
            (pd.to_datetime(filtered['game_date']).dt.date <= date_range[1])
        ]

    # --- Personal Handicap Toggle ---
    # Typical handicap ranges per sport + market type (from -400 to -550 lines)
    PERSONAL_H_RANGES = {
        'NCAAM': {'ats': (7, 10), 'ou': (12, 16), 'tt': (7, 10)},
        'NBA':   {'ats': (9, 11), 'ou': (14, 18), 'tt': (9, 11)},
    }

    use_personal = st.toggle("Use My Handicaps", value=False, key="use_personal_h")

    if use_personal:
        # --- Average Raw Margin by Category (cross-sport) ---
        # Shows how much directional cushion each bet category has at H=0
        all_resolved = all_preds[all_preds['outcome'].notna()].copy()
        all_resolved['raw_margin'] = all_resolved['margin'] - all_resolved['handicap'].fillna(0)

        # Derive bet direction from bet_on
        def _bet_direction(row):
            bo = str(row.get('bet_on', '')).upper()
            if row['market_type'] == 'ats':
                return 'Spread'
            elif 'OVER' in bo:
                return 'Over'
            elif 'UNDER' in bo:
                return 'Under'
            return 'Other'

        all_resolved['direction'] = all_resolved.apply(_bet_direction, axis=1)
        all_resolved['category'] = all_resolved['sport'] + ' ' + all_resolved['market_type'].map(
            {'ats': 'Spread', 'ou': 'Game Total', 'tt': 'Team Total'}
        ).fillna('') + ' ' + all_resolved['direction']
        # Clean up: "NCAAM Spread Spread" → "NCAAM Spread"
        all_resolved['category'] = all_resolved['category'].str.replace('Spread Spread', 'Spread')

        margin_summary = all_resolved.groupby('category').apply(
            lambda g: pd.Series({
                'N': len(g),
                'Avg Raw Margin': round(g['raw_margin'].mean(), 1),
                'Median': round(g['raw_margin'].median(), 1),
                'Std Dev': round(g['raw_margin'].std(), 1) if len(g) > 1 else 0.0,
                'Win% (H=0)': f"{(g['raw_margin'] > 0).sum() / max((g['raw_margin'] != 0).sum(), 1) * 100:.0f}%",
            })
        ).reset_index().rename(columns={'category': 'Category'})
        margin_summary = margin_summary.sort_values('Category')

        st.markdown("**Avg raw margin by category (H=0 baseline, all sports):**")
        st.caption("Raw margin = how the bet performed with zero handicap. Positive = bet went your way.")
        st.dataframe(margin_summary, hide_index=True, use_container_width=True)

    # Split into resolved vs unresolved
    resolved = filtered[filtered['outcome'].notna()].copy()
    unresolved = filtered[filtered['outcome'].isna()]

    if resolved.empty:
        st.warning(f"No resolved predictions yet for {selected_sport}. Outcomes are filled after games complete.")
        if not unresolved.empty:
            st.info(f"{len(unresolved)} predictions pending resolution.")
        return

    # Pre-compute shared data
    resolved['game_date_dt'] = pd.to_datetime(resolved['game_date'])

    # Compute raw margin (strip out system handicap) — reusable for personal H
    resolved['raw_margin'] = resolved['margin'] - resolved['handicap'].fillna(0)

    # Re-evaluate outcomes at personal handicaps
    selected_personal_h = None
    if use_personal:
        ranges = PERSONAL_H_RANGES.get(selected_sport, {'ats': (0, 0), 'ou': (0, 0), 'tt': (0, 0)})

        # Build a performance table for each H value in the range
        # Collect all unique H values across all three market types
        all_h_values = set()
        for mkt, (lo, hi) in ranges.items():
            all_h_values.update(range(lo, hi + 1))
        all_h_values = sorted(all_h_values)

        perf_rows = []
        for h_val in all_h_values:
            # For each market type, use h_val if it's in that market's range, else use the range bound
            h_map = {}
            for mkt, (lo, hi) in ranges.items():
                h_map[mkt] = min(max(h_val, lo), hi)

            new_h = resolved['market_type'].map(h_map).fillna(0)
            new_margin = (resolved['raw_margin'] + new_h).round(1)
            w = int((new_margin > 0).sum())
            l = int((new_margin < 0).sum())
            p = int((new_margin == 0).sum())
            wr = w / max(w + l, 1) * 100
            units = round(w * 1.0 + l * -1.1, 1)
            perf_rows.append({
                'H': h_val,
                'ATS': h_map.get('ats', '-'),
                'O/U': h_map.get('ou', '-'),
                'TT': h_map.get('tt', '-'),
                'W': w, 'L': l, 'P': p,
                'Win%': f"{wr:.1f}%",
                'Units': f"{units:+.1f}",
            })

        perf_df = pd.DataFrame(perf_rows)

        st.markdown("**Performance at each handicap in your range:**")

        def highlight_perf_row(row):
            units_val = float(row['Units'])
            if units_val > 0:
                return ['background-color: rgba(46, 204, 113, 0.15)'] * len(row)
            elif units_val < 0:
                return ['background-color: rgba(231, 76, 60, 0.15)'] * len(row)
            return [''] * len(row)

        st.dataframe(perf_df.style.apply(highlight_perf_row, axis=1), hide_index=True, use_container_width=True)

        # Let user pick which H to apply to the rest of the page
        selected_h_idx = st.selectbox(
            "Apply handicap to page",
            range(len(all_h_values)),
            format_func=lambda i: (
                f"H={all_h_values[i]} — "
                f"ATS:{perf_rows[i]['ATS']} O/U:{perf_rows[i]['O/U']} TT:{perf_rows[i]['TT']} — "
                f"{perf_rows[i]['W']}W-{perf_rows[i]['L']}L ({perf_rows[i]['Win%']}) {perf_rows[i]['Units']}u"
            ),
            key="personal_h_select",
        )
        selected_personal_h = perf_rows[selected_h_idx]

        # Apply selected handicap to resolved data
        h_map_apply = {
            'ats': selected_personal_h['ATS'],
            'ou': selected_personal_h['O/U'],
            'tt': selected_personal_h['TT'],
        }
        personal_h = resolved['market_type'].map(h_map_apply).fillna(0)
        resolved['margin'] = (resolved['raw_margin'] + personal_h).round(1)
        resolved['handicap'] = personal_h.astype(int)
        resolved.loc[resolved['margin'] > 0, 'outcome'] = 'WIN'
        resolved.loc[resolved['margin'] < 0, 'outcome'] = 'LOSS'
        resolved.loc[resolved['margin'] == 0, 'outcome'] = 'PUSH'

    # =====================================================================
    # Two-tab layout
    # =====================================================================
    tab_overview, tab_daily, tab_losses = st.tabs(["All-Time Overview", "Daily Results", "All Losses"])

    # =================================================================
    # Tab 1: All-Time Overview
    # =================================================================
    with tab_overview:

        # --- Summary Metrics ---
        st.subheader("Performance Summary")

        wins = (resolved['outcome'] == 'WIN').sum()
        losses = (resolved['outcome'] == 'LOSS').sum()
        pushes = (resolved['outcome'] == 'PUSH').sum()
        total = wins + losses + pushes
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        units_total = wins * 1.0 + losses * (-1.1) + pushes * 0
        roi = units_total / total * 100 if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolved", f"{total}", help=f"+{len(unresolved)} pending")
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{wins}W-{losses}L-{pushes}P")
        with col3:
            st.metric("ROI (-110)", f"{roi:+.1f}%", delta=f"{units_total:+.1f} units")
        with col4:
            avg_edge_win = resolved[resolved['outcome'] == 'WIN']['edge'].mean()
            avg_edge_loss = resolved[resolved['outcome'] == 'LOSS']['edge'].mean()
            st.metric("Avg Edge (W)", f"{avg_edge_win:.1%}" if pd.notna(avg_edge_win) else "N/A",
                      help=f"Losers avg: {avg_edge_loss:.1%}" if pd.notna(avg_edge_loss) else "")

        # --- Cumulative Performance Chart ---
        st.subheader("Cumulative Performance")

        daily = resolved.sort_values('game_date_dt').copy()
        daily['unit_result'] = daily['outcome'].map({'WIN': 1.0, 'LOSS': -1.1, 'PUSH': 0.0})
        daily['is_win'] = (daily['outcome'] == 'WIN').astype(int)
        daily['is_decided'] = (daily['outcome'].isin(['WIN', 'LOSS'])).astype(int)

        by_date = daily.groupby('game_date_dt').agg(
            units=('unit_result', 'sum'),
            wins=('is_win', 'sum'),
            decided=('is_decided', 'sum'),
        ).reset_index()
        by_date['cum_units'] = by_date['units'].cumsum()
        by_date['cum_wins'] = by_date['wins'].cumsum()
        by_date['cum_decided'] = by_date['decided'].cumsum()
        by_date['cum_win_rate'] = (by_date['cum_wins'] / by_date['cum_decided'] * 100).round(1)

        chart_metric = st.radio("Chart metric", ["Cumulative Units", "Cumulative Win Rate %"], horizontal=True, key="perf_metric")

        fig = go.Figure()
        if chart_metric == "Cumulative Units":
            fig.add_trace(go.Scatter(
                x=by_date['game_date_dt'], y=by_date['cum_units'],
                mode='lines+markers', name='Cumulative Units',
                line=dict(color='#2ecc71' if by_date['cum_units'].iloc[-1] >= 0 else '#e74c3c', width=3),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_yaxes(title_text="Units")
        else:
            fig.add_trace(go.Scatter(
                x=by_date['game_date_dt'], y=by_date['cum_win_rate'],
                mode='lines+markers', name='Win Rate %',
                line=dict(color='#3498db', width=3),
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
            fig.update_yaxes(title_text="Win Rate %")

        fig.update_layout(xaxis_title="Date", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Daily Record Table ---
        st.subheader("Daily Record")

        daily_summary = resolved.groupby('game_date_dt').apply(
            lambda g: pd.Series({
                'Picks': len(g),
                'W': int((g['outcome'] == 'WIN').sum()),
                'L': int((g['outcome'] == 'LOSS').sum()),
                'Win%': (g['outcome'] == 'WIN').sum() / max((g['outcome'].isin(['WIN', 'LOSS'])).sum(), 1) * 100,
                'Units': round((g['outcome'] == 'WIN').sum() * 1.0 + (g['outcome'] == 'LOSS').sum() * -1.1, 1),
            })
        ).reset_index().rename(columns={'game_date_dt': 'Date'})
        daily_summary = daily_summary.sort_values('Date', ascending=False)
        daily_summary['Date'] = daily_summary['Date'].dt.strftime('%Y-%m-%d')
        daily_summary['Win%'] = daily_summary['Win%'].apply(lambda x: f"{x:.0f}%")
        daily_summary['Units'] = daily_summary['Units'].apply(lambda x: f"{x:+.1f}")

        def highlight_daily_row(row):
            units_val = float(row['Units'])
            if units_val > 0:
                return ['background-color: rgba(46, 204, 113, 0.15)'] * len(row)
            elif units_val < 0:
                return ['background-color: rgba(231, 76, 60, 0.15)'] * len(row)
            return [''] * len(row)

        st.dataframe(
            daily_summary.style.apply(highlight_daily_row, axis=1),
            hide_index=True,
            use_container_width=True,
        )

        # --- Win Rate by Dimension ---
        st.subheader("Win Rate by Dimension")

        dimension_options = {
            "confidence": "Confidence",
            "source": "Source",
            "market_type": "Market Type",
            "pattern_type": "Pattern (Ride/Fade)",
            "tier_matchup": "Tier Matchup",
            "spread_bucket": "Spread Bucket",
            "bet_is_home": "Home/Away Bet",
            "handicap": "Handicap",
        }
        sel_dim = st.selectbox("Group by", list(dimension_options.keys()),
                               format_func=lambda x: dimension_options[x], key="pred_dim")

        dim_data = resolved.copy()
        if sel_dim == "bet_is_home":
            dim_data['bet_is_home'] = dim_data['bet_is_home'].map({1: 'Home', 0: 'Away'})

        dim_data = dim_data[dim_data[sel_dim].notna()]
        if not dim_data.empty:
            grouped = dim_data.groupby(sel_dim).apply(
                lambda g: pd.Series({
                    'wins': (g['outcome'] == 'WIN').sum(),
                    'losses': (g['outcome'] == 'LOSS').sum(),
                    'total': len(g),
                    'win_rate': (g['outcome'] == 'WIN').sum() / max((g['outcome'].isin(['WIN', 'LOSS'])).sum(), 1) * 100,
                })
            ).reset_index()

            fig_dim = go.Figure()
            fig_dim.add_trace(go.Bar(
                x=grouped[sel_dim].astype(str),
                y=grouped['win_rate'],
                text=[f"{wr:.0f}%<br>(n={n})" for wr, n in zip(grouped['win_rate'], grouped['total'])],
                textposition='outside',
                marker_color=['#2ecc71' if wr >= 50 else '#e74c3c' for wr in grouped['win_rate']],
            ))
            fig_dim.add_hline(y=50, line_dash="dash", line_color="gray")
            fig_dim.update_layout(
                xaxis_title=dimension_options[sel_dim],
                yaxis_title="Win Rate %",
                yaxis=dict(range=[0, max(grouped['win_rate'].max() + 15, 60)]),
                height=400,
            )
            st.plotly_chart(fig_dim, use_container_width=True)
        else:
            st.info(f"No data for dimension: {dimension_options[sel_dim]}")

    # =================================================================
    # Tab 2: Daily Results
    # =================================================================
    with tab_daily:

        # --- Date Selector ---
        available_dates = sorted(resolved['game_date_dt'].dt.date.unique(), reverse=True)
        date_labels = {}
        for d in available_dates:
            day_data = resolved[resolved['game_date_dt'].dt.date == d]
            w = int((day_data['outcome'] == 'WIN').sum())
            l = int((day_data['outcome'] == 'LOSS').sum())
            date_labels[d] = f"{d.strftime('%b %d, %Y')} ({len(day_data)} resolved — {w}W-{l}L)"

        selected_date = st.selectbox(
            "Select Date", available_dates,
            format_func=lambda d: date_labels[d],
            key="daily_results_date",
        )

        # --- Day Summary Bar ---
        day_preds = resolved[resolved['game_date_dt'].dt.date == selected_date].copy()
        day_w = int((day_preds['outcome'] == 'WIN').sum())
        day_l = int((day_preds['outcome'] == 'LOSS').sum())
        day_total_decided = day_w + day_l
        day_wr = day_w / day_total_decided * 100 if day_total_decided > 0 else 0
        day_units = round(day_w * 1.0 + day_l * -1.1, 1)
        units_color = "green" if day_units >= 0 else "red"

        st.markdown(
            f"**{selected_date.strftime('%B %d, %Y')}** — {selected_sport}: "
            f"{day_w}-{day_l} ({day_wr:.1f}%) | "
            f":{units_color}[{day_units:+.1f} units]"
        )

        # --- Game-by-Game Cards ---
        source_icons = {'streak': '📈', 'ou_streak': '🔄', 'tt_streak': '👤'}

        # Sort games: most predictions first, then alphabetically by away team
        game_groups = list(day_preds.groupby(['home_team', 'away_team']))
        game_groups.sort(key=lambda x: (-len(x[1]), x[0][1]))

        for (home, away), game_group in game_groups:
            g_wins = int((game_group['outcome'] == 'WIN').sum())
            g_losses = int((game_group['outcome'] == 'LOSS').sum())
            if g_losses == 0 and g_wins > 0:
                game_icon = "✅"
            elif g_wins == 0 and g_losses > 0:
                game_icon = "❌"
            else:
                game_icon = "🔀"

            with st.container():
                st.markdown(f"#### {game_icon} {away} @ {home}")

                # Score + lines info
                row0 = game_group.iloc[0]
                score_parts = []
                if pd.notna(row0.get('away_score')) and pd.notna(row0.get('home_score')):
                    score_parts.append(f"{away} {int(row0['away_score'])}, {home} {int(row0['home_score'])}")
                if pd.notna(row0.get('closing_spread')):
                    sp = row0['closing_spread']
                    score_parts.append(f"Spread: {home} {'+' if sp > 0 else ''}{sp}")
                elif pd.notna(row0.get('spread')):
                    sp = row0['spread']
                    score_parts.append(f"Spread: {home} {'+' if sp > 0 else ''}{sp}")
                if pd.notna(row0.get('closing_total')):
                    score_parts.append(f"Total: {row0['closing_total']}")
                elif pd.notna(row0.get('total')):
                    score_parts.append(f"Total: {row0['total']}")
                tt_parts = []
                if pd.notna(row0.get('closing_away_tt')):
                    tt_parts.append(f"{away} {row0['closing_away_tt']}")
                elif pd.notna(row0.get('away_team_total')):
                    tt_parts.append(f"{away} {row0['away_team_total']}")
                if pd.notna(row0.get('closing_home_tt')):
                    tt_parts.append(f"{home} {row0['closing_home_tt']}")
                elif pd.notna(row0.get('home_team_total')):
                    tt_parts.append(f"{home} {row0['home_team_total']}")
                if tt_parts:
                    score_parts.append(f"TT: {' / '.join(tt_parts)}")
                if score_parts:
                    st.caption(" | ".join(score_parts))

                # Each prediction as a line
                for _, pred in game_group.iterrows():
                    result_icon = "✅" if pred['outcome'] == 'WIN' else ("❌" if pred['outcome'] == 'LOSS' else "⚪")
                    src_icon = source_icons.get(pred['source'], '📊')
                    margin_str = f"{pred['margin']:+.1f}" if pd.notna(pred.get('margin')) else ""
                    edge_str = f"{pred['edge']:.1%}" if pd.notna(pred.get('edge')) else "N/A"
                    h_val = int(pred['handicap']) if pd.notna(pred.get('handicap')) else 0

                    bet_line = f"{src_icon} **{pred['bet_on']}** — {result_icon} **{pred['outcome']}**"
                    if margin_str:
                        bet_line += f" ({margin_str} margin)"
                    detail_line = (
                        f"{pred['rationale'] or ''} | "
                        f"Edge: {edge_str} | Conf: {pred['confidence']} | H={h_val}"
                    )

                    if pred['outcome'] == 'LOSS':
                        st.markdown(
                            f'<div style="background-color: rgba(231, 76, 60, 0.12); '
                            f'border-left: 3px solid #e74c3c; padding: 8px 12px; '
                            f'border-radius: 4px; margin-bottom: 8px;">'
                            f'{bet_line}<br>'
                            f'<span style="color: #888; font-size: 0.85em;">{detail_line}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(bet_line)
                        st.caption(detail_line)

                st.markdown("---")

        # --- Pending predictions for this date ---
        day_unresolved = unresolved[pd.to_datetime(unresolved['game_date']).dt.date == selected_date]
        if not day_unresolved.empty:
            with st.expander(f"Pending Predictions ({len(day_unresolved)})"):
                pending_cols = ['game_date', 'home_team', 'away_team', 'bet_on', 'source', 'confidence', 'edge']
                st.dataframe(day_unresolved[pending_cols], hide_index=True, use_container_width=True)

    # =================================================================
    # Tab 3: All Losses
    # =================================================================
    with tab_losses:

        losses_df = resolved[resolved['outcome'] == 'LOSS'].copy()

        if losses_df.empty:
            st.info("No losses recorded yet.")
        else:
            st.markdown(f"**{len(losses_df)} losses** — look for patterns below")

            # --- Quick breakdown metrics ---
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                avg_margin = losses_df['margin'].mean()
                st.metric("Avg Margin", f"{avg_margin:+.1f}" if pd.notna(avg_margin) else "N/A")
            with lc2:
                avg_edge = losses_df['edge'].mean()
                st.metric("Avg Edge", f"{avg_edge:.1%}" if pd.notna(avg_edge) else "N/A")
            with lc3:
                top_conf = losses_df['confidence'].value_counts().idxmax() if losses_df['confidence'].notna().any() else "N/A"
                st.metric("Most Common Conf", top_conf)
            with lc4:
                top_source = losses_df['source'].value_counts().idxmax() if losses_df['source'].notna().any() else "N/A"
                st.metric("Most Common Source", top_source)

            # --- Loss breakdown charts ---
            loss_dim_col1, loss_dim_col2 = st.columns(2)

            with loss_dim_col1:
                st.markdown("**Losses by Source**")
                src_counts = losses_df['source'].value_counts()
                fig_src = go.Figure(go.Bar(
                    x=src_counts.index.tolist(),
                    y=src_counts.values.tolist(),
                    marker_color='#e74c3c',
                    text=src_counts.values.tolist(),
                    textposition='outside',
                ))
                fig_src.update_layout(height=300, xaxis_title="Source", yaxis_title="Losses", margin=dict(t=10))
                st.plotly_chart(fig_src, use_container_width=True)

            with loss_dim_col2:
                st.markdown("**Losses by Confidence**")
                conf_counts = losses_df['confidence'].value_counts()
                fig_conf = go.Figure(go.Bar(
                    x=conf_counts.index.tolist(),
                    y=conf_counts.values.tolist(),
                    marker_color='#e74c3c',
                    text=conf_counts.values.tolist(),
                    textposition='outside',
                ))
                fig_conf.update_layout(height=300, xaxis_title="Confidence", yaxis_title="Losses", margin=dict(t=10))
                st.plotly_chart(fig_conf, use_container_width=True)

            # --- Full losses table ---
            st.markdown("**All Loss Details**")
            loss_display = losses_df[[
                'game_date', 'away_team', 'home_team', 'bet_on', 'source',
                'confidence', 'edge', 'handicap', 'margin', 'rationale',
                'pattern_type', 'tier_matchup', 'spread_bucket',
            ]].copy()
            loss_display['edge'] = loss_display['edge'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
            loss_display['margin'] = loss_display['margin'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "")
            loss_display.columns = [
                'Date', 'Away', 'Home', 'Bet On', 'Source',
                'Conf', 'Edge', 'H', 'Margin', 'Rationale',
                'Pattern', 'Tier Matchup', 'Spread Bucket',
            ]
            loss_display = loss_display.sort_values('Date', ascending=False)

            st.dataframe(
                loss_display,
                hide_index=True,
                use_container_width=True,
            )


# =============================================================================
# Main Router
# =============================================================================

if page == "🎲 Today's Picks":
    page_todays_picks()
elif page == "📈 Prediction Results":
    page_prediction_results()
elif page == "🏆 Power Rankings":
    page_power_rankings()
elif page == "🔥 Current Streaks":
    page_current_streaks()
elif page == "League-Wide Trends":
    page_macro_trends()
elif page == "Team Trends":
    page_micro_analysis()
elif page == "Streak Analysis":
    page_streak_analysis()
elif page == "Verification":
    page_verification()
elif page == "🤖 AI Research":
    page_ai_research()


# =============================================================================
# Footer
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.caption("Sports Betting Analytics v0.1")
