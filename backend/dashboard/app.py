"""
Sports Betting Analytics Dashboard

Interactive Streamlit dashboard for exploring betting trends.
Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
)
from src.analysis.aggregations import (
    macro_ats_summary,
    macro_time_series,
    macro_by_spread_bucket,
    micro_team_summary,
    micro_all_teams,
)
from src.analysis.insights import (
    detect_patterns,
    find_opportunities,
    get_current_streaks,
    get_pattern_summary,
    get_opportunity_summary,
)
from src.analysis.network_ratings import (
    get_team_rankings,
    get_rankings_dataframe,
)
from src.analysis.backtest_ratings import (
    get_games_with_ratings,
    backtest_gap_strategy,
    backtest_rank_strategy,
    analyze_gap_thresholds,
    generate_backtest_report,
    get_game_by_game_results,
    BREAKEVEN_WIN_RATE,
)
from src.analysis.tier_matchups import (
    detect_tier_matchup_patterns,
    detect_all_tier_matchup_patterns,
    find_tier_matchup_opportunities,
    get_tier_pattern_summary,
    get_tier_opportunity_summary,
    get_tier,
    TIERS,
)
from src.analysis.todays_recommendations import (
    generate_recommendations,
    get_combined_confidence,
    get_games_last_updated,
)


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
    """Get cached database connection."""
    return get_connection(check_same_thread=False)


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


# =============================================================================
# Sidebar - Navigation & Filters
# =============================================================================

st.sidebar.title("📊 Analytics Dashboard")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["📋 Strategy Overview", "🎲 Today's Picks", "🏆 Power Rankings", "📈 Backtest Strategies", "Macro Trends", "Micro (Team) Analysis", "Streak Analysis", "Exploration", "Verification"]
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
# Page: Macro Trends
# =============================================================================

def page_macro_trends():
    st.title("Macro Trends (League-Wide)")
    st.markdown(f"League-wide ATS analysis for **{selected_sport}**")

    games = load_games(selected_sport)

    if len(games) == 0:
        st.warning(f"No games found for {selected_sport}")
        return

    # Tabs for different views
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


# =============================================================================
# Page: Micro (Team) Analysis
# =============================================================================

def page_micro_analysis():
    st.title("Micro Analysis (Team-Specific)")
    st.markdown(f"Individual team ATS analysis for **{selected_sport}**")

    teams = load_teams(selected_sport)

    if not teams:
        st.warning(f"No teams found for {selected_sport}")
        return

    # Tabs
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

        selected_team = st.selectbox("Select Team", teams)

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


# =============================================================================
# Page: Streak Analysis
# =============================================================================

def page_streak_analysis():
    st.title("Streak Continuation Analysis")
    st.markdown(f"""
    **Question:** After a team covers/loses X games in a row, how does their next game perform across different handicaps (0-15 points)?

    Select a streak length and type, then see coverage rates at every handicap level for **{selected_sport}**.
    """)

    # First, show available streak data
    with st.spinner("Loading streak summary..."):
        summary = streak_summary_all_lengths(conn, selected_sport)

    if len(summary) == 0:
        st.warning("No streak data found")
        return

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
    else:
        st.warning(f"No {streak_length}-game {streak_type} streaks found")
        return

    # Run handicap analysis
    with st.spinner("Analyzing handicap coverage..."):
        handicap_data = streak_continuation_analysis(conn, selected_sport, streak_length, streak_type)
        baseline_data = baseline_handicap_coverage(conn, selected_sport)

    if len(handicap_data) == 0:
        st.warning("No data found")
        return

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

    if st.button("Load All Situations"):
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


# =============================================================================
# Page: Strategy Overview
# =============================================================================

def page_matchup_opportunities():
    st.title("📋 Strategy Overview")
    st.markdown("""
    **Real-time analysis results** showing which betting strategies are currently
    profitable based on backtest results and streak pattern detection.
    """)

    # Sport selector
    sport = st.selectbox("Select Sport", ["NBA", "NFL", "NCAAM"], key="strategy_sport")

    # -------------------------
    # Section 1: Backtest Strategies
    # -------------------------
    st.header("📊 Backtest Strategies")
    st.caption("Rating-based strategies tested against historical data")

    # Get backtest report for selected sport
    @st.cache_data(ttl=600, show_spinner=False)
    def get_backtest_data(_conn, sport):
        from src.analysis.backtest_ratings import generate_backtest_report
        return generate_backtest_report(_conn, sport)

    with st.spinner("Running backtests..."):
        report = get_backtest_data(conn, sport)

    if 'error' not in report:
        col1, col2 = st.columns(2)

        # Gap Strategy Card
        with col1:
            st.subheader("🎯 Market Gap Strategy")
            if report.get('best_gap_strategy'):
                gap = report['best_gap_strategy']
                is_profitable = gap['roi'] > 0
                icon = "✅" if is_profitable else "⚠️"
                st.markdown(f"**{icon} Threshold: {gap['threshold']:.2f}**")
                st.metric("Win Rate", f"{gap['win_rate']:.1%}", delta=f"{gap['edge']:+.1%} edge")
                st.metric("ROI", f"{gap['roi']:.1%}")
                st.caption(f"{gap['bets']} total bets analyzed")
            else:
                st.info("No profitable gap threshold found")

        # Rank Strategy Card
        with col2:
            st.subheader("🏆 Rank Strategy")
            if report.get('best_rank_strategy'):
                rank = report['best_rank_strategy']
                is_profitable = rank['roi'] > 0
                icon = "✅" if is_profitable else "⚠️"
                st.markdown(f"**{icon} {rank['strategy_name']}**")
                st.metric("Win Rate", f"{rank['win_rate']:.1%}", delta=f"{rank['edge']:+.1%} edge")
                st.metric("ROI", f"{rank['roi']:.1%}")
                st.caption(f"{rank['total_bets']} total bets analyzed")
            else:
                st.info("No profitable rank strategy found")

        st.caption(f"Analysis period: {report['date_range']['start']} to {report['date_range']['end']} | {report['games_analyzed']} games")
    else:
        st.warning(f"No backtest data: {report.get('error', 'Unknown error')}")

    st.markdown("---")

    # -------------------------
    # Section 2: Streak Patterns
    # -------------------------
    st.header("🔥 Streak Patterns")
    st.caption("Statistically significant edges from streak analysis")

    @st.cache_data(ttl=600, show_spinner=False)
    def get_streak_patterns(_conn):
        from src.analysis.insights import detect_patterns
        return detect_patterns(_conn, min_sample=30, min_edge=0.05)

    with st.spinner("Detecting streak patterns..."):
        patterns = get_streak_patterns(conn)

    # Filter to selected sport
    sport_patterns = [p for p in patterns if p.sport == sport]

    if sport_patterns:
        # Separate ride vs fade patterns
        ride_patterns = [p for p in sport_patterns if p.pattern_type == 'streak_ride']
        fade_patterns = [p for p in sport_patterns if p.pattern_type == 'streak_fade']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Ride Patterns")
            st.caption("Bet ON teams after these streaks")
            if ride_patterns:
                for p in sorted(ride_patterns, key=lambda x: x.edge, reverse=True)[:5]:
                    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[p.confidence]
                    st.markdown(f"**{conf_icon} {p.streak_length}+ {p.streak_type.lower()} streak at +{p.handicap}**")
                    st.caption(f"{p.cover_rate:.1%} cover vs {p.baseline_rate:.1%} baseline = **{p.edge:+.1%} edge** | {p.sample_size} samples")
                if len(ride_patterns) > 5:
                    with st.expander(f"Show all {len(ride_patterns)} ride patterns"):
                        for p in sorted(ride_patterns, key=lambda x: x.edge, reverse=True):
                            conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[p.confidence]
                            st.markdown(f"**{conf_icon} {p.streak_length}+ {p.streak_type.lower()} streak at +{p.handicap}**")
                            st.caption(f"{p.cover_rate:.1%} cover vs {p.baseline_rate:.1%} baseline = **{p.edge:+.1%} edge** | {p.sample_size} samples")
            else:
                st.info("No ride patterns detected")

        with col2:
            st.subheader("📉 Fade Patterns")
            st.caption("Bet AGAINST teams after these streaks")
            if fade_patterns:
                for p in sorted(fade_patterns, key=lambda x: abs(x.edge), reverse=True)[:5]:
                    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[p.confidence]
                    st.markdown(f"**{conf_icon} {p.streak_length}+ {p.streak_type.lower()} streak at +{p.handicap}**")
                    st.caption(f"{p.cover_rate:.1%} cover vs {p.baseline_rate:.1%} baseline = **{p.edge:+.1%} edge** | {p.sample_size} samples")
                if len(fade_patterns) > 5:
                    with st.expander(f"Show all {len(fade_patterns)} fade patterns"):
                        for p in sorted(fade_patterns, key=lambda x: abs(x.edge), reverse=True):
                            conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[p.confidence]
                            st.markdown(f"**{conf_icon} {p.streak_length}+ {p.streak_type.lower()} streak at +{p.handicap}**")
                            st.caption(f"{p.cover_rate:.1%} cover vs {p.baseline_rate:.1%} baseline = **{p.edge:+.1%} edge** | {p.sample_size} samples")
            else:
                st.info("No fade patterns detected")

        # Summary stats
        st.markdown("---")
        total_patterns = len(sport_patterns)
        high_conf = len([p for p in sport_patterns if p.confidence == 'high'])
        st.caption(f"**{total_patterns} patterns detected** for {sport} ({high_conf} high confidence)")
    else:
        st.info(f"No significant streak patterns found for {sport}")


# =============================================================================
# Page: Today's Picks (Primary Navigation)
# =============================================================================

def page_todays_picks():
    st.title("🎲 Today's Picks")
    st.markdown("""
    **Today's Picks** combines streak and tier matchup insights for games scheduled today.
    Recommendations are generated by cross-referencing today's games against detected patterns.
    """)

    # Detect streak patterns (needed for recommendations)
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_detect_patterns(_conn):
        return detect_patterns(_conn, min_sample=30, min_edge=0.05)

    with st.spinner("Detecting patterns..."):
        patterns = cached_detect_patterns(conn)

    # Show last updated timestamp
    last_updated = get_games_last_updated(conn)
    if last_updated:
        st.caption(f"Games last updated: {last_updated.strftime('%Y-%m-%d %I:%M %p')} UTC")
    else:
        st.warning("No games loaded yet. Run the collect_todays_games Lambda to fetch today's schedule.")

    # Sport filter
    picks_sport = st.selectbox(
        "Sport",
        ["All", "NFL", "NBA", "NCAAM"],
        key="todays_picks_sport_main"
    )

    sport_to_use = picks_sport if picks_sport != "All" else "All"

    # Get tier patterns
    @st.cache_data(ttl=600, show_spinner=False)
    def get_cached_tier_patterns(_conn, sport_key):
        if sport_key == "All":
            return detect_all_tier_matchup_patterns(_conn, min_sample=30, min_edge=0.05)
        return detect_tier_matchup_patterns(_conn, sport_key, min_sample=30, min_edge=0.05)

    tier_patterns = get_cached_tier_patterns(conn, sport_to_use)

    # Generate recommendations (games fetched from database)
    with st.spinner("Analyzing today's games..."):
        game_recommendations = generate_recommendations(
            conn,
            sport_to_use,
            patterns,
            tier_patterns,
        )

    if not game_recommendations:
        st.info(f"No games scheduled today for {picks_sport}.")
    else:
        # Count games with recommendations
        games_with_recs = [g for g in game_recommendations if g.recommendations]
        st.markdown(f"**Found {len(game_recommendations)} games today** ({len(games_with_recs)} with betting edges)")

        # Sort by max edge descending (games with no recommendations go to bottom)
        def get_max_edge(game):
            if not game.recommendations:
                return -1  # No recommendations, sort to bottom
            return max(rec.edge for rec in game.recommendations)

        game_recommendations_sorted = sorted(game_recommendations, key=get_max_edge, reverse=True)

        for game in game_recommendations_sorted:
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

                # Spread info
                if game.spread is not None:
                    spread_display = f"+{game.spread}" if game.spread > 0 else str(game.spread)
                    st.caption(f"Spread: {game.home_team} {spread_display} ({game.spread_source or 'Unknown'})")

                # Recommendations
                if game.recommendations:
                    # Group by bet_on team
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
                            source_icon = "📈" if rec.source == "streak" else "🎯"
                            st.markdown(f"- {source_icon} **{rec.source.replace('_', ' ').title()}**: {rec.rationale}")

                        st.caption(f"Combined Confidence: {combined_conf}")

                else:
                    st.markdown("⚪ **NO EDGE DETECTED**")
                    st.caption("Neither team matches profitable patterns")

                # Streak info in expander
                with st.expander("Team Details"):
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.markdown(f"**{game.home_team}** (Home)")
                        if game.home_ats_rating is not None:
                            st.write(f"ATS Rating: {game.home_ats_rating:.3f} ({game.home_tier})")
                        if game.home_streak:
                            st.write(f"Streak: {game.home_streak[0]}-game ATS {game.home_streak[1].lower()}")
                        else:
                            st.write("No current streak")

                    with detail_col2:
                        st.markdown(f"**{game.away_team}** (Away)")
                        if game.away_ats_rating is not None:
                            st.write(f"ATS Rating: {game.away_ats_rating:.3f} ({game.away_tier})")
                        if game.away_streak:
                            st.write(f"Streak: {game.away_streak[0]}-game ATS {game.away_streak[1].lower()}")
                        else:
                            st.write("No current streak")

                st.markdown("---")

    # Current team streaks (collapsible)
    with st.expander("📊 All Current Team Streaks"):
        for sport in ['NFL', 'NBA', 'NCAAM']:
            streaks = get_current_streaks(conn, sport)
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


# =============================================================================
# Page: Opportunities (Dynamic Insight Engine)
# =============================================================================

def page_opportunities():
    st.title("🎯 Today's Opportunities")
    st.markdown("""
    **Dynamic pattern detection** identifies statistically significant betting edges
    and matches them against current team states to surface actionable opportunities.
    """)

    # Settings in expander
    with st.expander("⚙️ Detection Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_sample = st.slider(
                "Min Sample Size",
                5, 100, 30,
                help="Minimum streak situations required. Must have at least 5 occurrences to be considered actionable."
            )
        with col2:
            min_edge = st.slider("Min Edge %", 1, 15, 5, help="Minimum edge vs baseline to include")
        with col3:
            min_confidence = st.selectbox("Min Confidence", ["low", "medium", "high"], index=0)

        st.caption("⚠️ Patterns with fewer than 30 samples should be treated with caution. Sample size is shown for each opportunity.")

    # Detect patterns (cached)
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_detect_patterns(_conn, min_sample, min_edge):
        return detect_patterns(_conn, min_sample=min_sample, min_edge=min_edge/100)

    with st.spinner("Detecting patterns..."):
        patterns = cached_detect_patterns(conn, min_sample, min_edge)

    # Find current opportunities
    with st.spinner("Scanning for opportunities..."):
        opportunities = find_opportunities(conn, patterns, min_confidence=min_confidence)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Patterns Detected", len(patterns))
    col2.metric("Current Opportunities", len(opportunities))
    col3.metric(
        "Strongest Edge",
        f"{max([p.edge for p in patterns], key=abs)*100:+.1f}%" if patterns else "N/A"
    )

    st.markdown("---")

    # Tabs for opportunities vs patterns
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Streak Opportunities", "🔍 Streak Patterns", "🎯 Tier Matchup Strategies", "🎲 Today's Picks"])

    # ----- Tab 1: Current Opportunities -----
    with tab1:
        if not opportunities:
            st.info("No opportunities match current team streaks and pattern criteria.")
        else:
            # Filter by sport if desired
            opp_sports = list(set([o.sport for o in opportunities]))
            if len(opp_sports) > 1:
                sport_filter = st.multiselect("Filter by Sport", opp_sports, default=opp_sports)
                filtered_opps = [o for o in opportunities if o.sport in sport_filter]
            else:
                filtered_opps = opportunities

            st.markdown(f"**Showing {len(filtered_opps)} opportunities** (sorted by edge strength)")

            for i, opp in enumerate(filtered_opps[:15]):  # Top 15
                # Color and icon based on recommendation
                if opp.recommendation == "FADE":
                    icon = "🔴"
                    color = "#e74c3c"
                    action = f"Bet AGAINST {opp.team}"
                else:
                    icon = "🟢"
                    color = "#2ecc71"
                    action = f"Bet ON {opp.team}"

                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

                    with col1:
                        st.markdown(f"### {icon} {opp.recommendation} {opp.team}")
                        st.caption(f"{opp.sport} | {opp.current_streak}-game ATS {opp.streak_type.lower()} streak")

                    with col2:
                        st.markdown(f"**Action:** {action}")
                        st.caption(f"Pattern: {opp.pattern.streak_length}+ {opp.pattern.streak_type.lower()} → {opp.pattern.pattern_type.replace('_', ' ')}")

                    with col3:
                        st.metric("Edge", f"{opp.edge_pct:.1%}")

                    with col4:
                        st.metric("Sample", opp.pattern.sample_size)
                        st.caption(opp.pattern.confidence.title())

                    # Expandable details
                    with st.expander(f"Pattern Details for {opp.team}"):
                        detail_col1, detail_col2 = st.columns(2)

                        with detail_col1:
                            st.markdown("**Pattern Statistics:**")
                            st.write(f"- After {opp.pattern.streak_length}+ game {opp.pattern.streak_type.lower()} streaks in {opp.pattern.sport}")
                            st.write(f"- Handicap: +{opp.pattern.handicap} points")
                            st.write(f"- Sample size: {opp.pattern.sample_size} observations")

                        with detail_col2:
                            st.markdown("**Edge Calculation:**")
                            st.write(f"- Cover rate after streak: **{opp.pattern.cover_rate:.1%}**")
                            st.write(f"- League baseline: {opp.pattern.baseline_rate:.1%}")
                            st.write(f"- Edge: **{opp.pattern.edge*100:+.1f}%**")

                        st.markdown("---")
                        st.markdown(f"""
                        **Interpretation:** Teams on {opp.pattern.streak_length}+ game ATS {opp.pattern.streak_type.lower()} streaks
                        in {opp.pattern.sport} cover at {opp.pattern.cover_rate:.1%}, which is
                        {"below" if opp.pattern.edge < 0 else "above"} the league baseline of {opp.pattern.baseline_rate:.1%}.
                        This suggests {"fading" if opp.pattern.edge < 0 else "riding"} these teams.
                        """)

                st.markdown("")  # Spacing

    # ----- Tab 2: All Detected Patterns -----
    with tab2:
        if not patterns:
            st.info("No patterns meet the minimum criteria. Try lowering the thresholds.")
        else:
            st.markdown(f"**Found {len(patterns)} significant patterns** (sorted by edge strength)")

            # Filter controls
            col1, col2, col3 = st.columns(3)
            with col1:
                pattern_sport = st.multiselect(
                    "Sport",
                    ['NFL', 'NBA', 'NCAAM'],
                    default=['NFL', 'NBA', 'NCAAM'],
                    key="pattern_sport"
                )
            with col2:
                pattern_type_filter = st.multiselect(
                    "Pattern Type",
                    ['streak_fade', 'streak_ride'],
                    default=['streak_fade', 'streak_ride'],
                    key="pattern_type"
                )
            with col3:
                pattern_confidence = st.multiselect(
                    "Confidence",
                    ['high', 'medium', 'low'],
                    default=['high', 'medium', 'low'],
                    key="pattern_conf"
                )

            filtered_patterns = [
                p for p in patterns
                if p.sport in pattern_sport
                and p.pattern_type in pattern_type_filter
                and p.confidence in pattern_confidence
            ]

            if filtered_patterns:
                summary_df = get_pattern_summary(filtered_patterns)
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

                # Visualization
                st.markdown("---")
                st.subheader("Pattern Distribution")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # By sport
                    sport_counts = {}
                    for p in filtered_patterns:
                        sport_counts[p.sport] = sport_counts.get(p.sport, 0) + 1

                    fig = px.pie(
                        values=list(sport_counts.values()),
                        names=list(sport_counts.keys()),
                        title="Patterns by Sport"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with chart_col2:
                    # Edge distribution
                    edges = [p.edge * 100 for p in filtered_patterns]
                    fig = px.histogram(
                        x=edges,
                        nbins=20,
                        title="Edge Distribution (%)",
                        labels={'x': 'Edge vs Baseline (%)'}
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No patterns match the filters")

    # ----- Tab 3: Tier Matchup Strategies -----
    with tab3:
        st.markdown("""
        **Tier Matchup Analysis** identifies profitable patterns when teams from different
        rating tiers play each other at specific handicaps.

        *Example: When TOP tier plays BOTTOM tier with +11 handicap, the lower-rated team covers 85% of the time.*
        """)

        # Settings
        tier_col1, tier_col2, tier_col3 = st.columns(3)
        with tier_col1:
            tier_min_sample = st.slider(
                "Min Sample Size",
                10, 100, 30,
                key="tier_min_sample",
                help="Minimum games required for a pattern"
            )
        with tier_col2:
            tier_min_edge = st.slider(
                "Min Edge %",
                1, 20, 5,
                key="tier_min_edge",
                help="Minimum edge vs baseline to include"
            )
        with tier_col3:
            tier_sport_filter = st.selectbox(
                "Sport",
                ["All", "NFL", "NBA", "NCAAM"],
                key="tier_sport"
            )

        # Detect tier matchup patterns (cached)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_tier_patterns(_conn, sport, min_sample, min_edge):
            if sport == "All":
                return detect_all_tier_matchup_patterns(_conn, min_sample=min_sample, min_edge=min_edge/100)
            else:
                return detect_tier_matchup_patterns(_conn, sport, min_sample=min_sample, min_edge=min_edge/100)

        with st.spinner("Analyzing tier matchup patterns..."):
            tier_patterns = cached_tier_patterns(
                conn,
                tier_sport_filter,
                tier_min_sample,
                tier_min_edge
            )

        # Summary metrics
        tier_m1, tier_m2, tier_m3, tier_m4 = st.columns(4)
        tier_m1.metric("Patterns Found", len(tier_patterns))
        high_conf_patterns = [p for p in tier_patterns if p.confidence == 'high']
        tier_m2.metric("High Confidence", len(high_conf_patterns))
        if tier_patterns:
            avg_edge = sum(p.edge for p in tier_patterns) / len(tier_patterns)
            tier_m3.metric("Avg Edge", f"{avg_edge:.1%}")
            best_pattern = max(tier_patterns, key=lambda p: p.edge)
            tier_m4.metric("Best Edge", f"{best_pattern.edge:.1%}")

        st.markdown("---")

        if not tier_patterns:
            st.info("No tier matchup patterns meet the minimum criteria. Try lowering the thresholds.")
        else:
            # Sub-tabs for patterns vs opportunities
            tier_tab1, tier_tab2 = st.tabs(["📊 Detected Patterns", "🎯 Current Opportunities"])

            with tier_tab1:
                st.markdown(f"**{len(tier_patterns)} patterns found** (sorted by edge)")

                # Filters
                tier_filter_col1, tier_filter_col2 = st.columns(2)
                with tier_filter_col1:
                    tier_higher_filter = st.multiselect(
                        "Higher Tier",
                        TIERS,
                        default=TIERS,
                        key="tier_higher"
                    )
                with tier_filter_col2:
                    tier_lower_filter = st.multiselect(
                        "Lower Tier",
                        TIERS,
                        default=TIERS,
                        key="tier_lower"
                    )

                # Filter patterns
                filtered_tier_patterns = [
                    p for p in tier_patterns
                    if p.higher_tier in tier_higher_filter
                    and p.lower_tier in tier_lower_filter
                ]

                if filtered_tier_patterns:
                    # Show patterns table
                    tier_summary_df = get_tier_pattern_summary(filtered_tier_patterns[:50])  # Top 50
                    st.dataframe(tier_summary_df, hide_index=True, use_container_width=True)

                    # Visualization
                    st.markdown("---")
                    st.subheader("Pattern Insights")

                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        # Patterns by matchup type
                        matchup_counts = {}
                        for p in filtered_tier_patterns:
                            key = f"{p.higher_tier} vs {p.lower_tier}"
                            matchup_counts[key] = matchup_counts.get(key, 0) + 1

                        # Sort by count
                        sorted_matchups = sorted(matchup_counts.items(), key=lambda x: -x[1])[:10]

                        fig = go.Figure(go.Bar(
                            y=[m[0] for m in sorted_matchups],
                            x=[m[1] for m in sorted_matchups],
                            orientation='h',
                            marker_color='#3498db'
                        ))
                        fig.update_layout(
                            title="Patterns by Tier Matchup",
                            xaxis_title="Number of Patterns",
                            yaxis=dict(autorange="reversed"),
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with viz_col2:
                        # Edge by bet direction
                        higher_patterns = [p for p in filtered_tier_patterns if p.bet_on == 'HIGHER']
                        lower_patterns = [p for p in filtered_tier_patterns if p.bet_on == 'LOWER']

                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=[p.edge * 100 for p in higher_patterns],
                            name=f"Bet HIGHER ({len(higher_patterns)})",
                            marker_color='#2ecc71'
                        ))
                        fig.add_trace(go.Box(
                            y=[p.edge * 100 for p in lower_patterns],
                            name=f"Bet LOWER ({len(lower_patterns)})",
                            marker_color='#e74c3c'
                        ))
                        fig.update_layout(
                            title="Edge Distribution by Bet Direction",
                            yaxis_title="Edge (%)",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show top individual patterns
                    st.markdown("---")
                    st.subheader("Top 5 High-Confidence Patterns")

                    top_patterns = [p for p in filtered_tier_patterns if p.confidence == 'high'][:5]
                    if not top_patterns:
                        top_patterns = filtered_tier_patterns[:5]

                    for i, p in enumerate(top_patterns):
                        with st.container():
                            icon = "🟢" if p.bet_on == "HIGHER" else "🔴"
                            st.markdown(f"### {icon} {p.sport}: {p.higher_tier} vs {p.lower_tier} (+{p.handicap})")

                            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                            p_col1.metric("Bet On", p.bet_on)
                            p_col2.metric("Cover Rate", f"{p.cover_rate:.1%}")
                            p_col3.metric("Edge", f"{p.edge:+.1%}")
                            p_col4.metric("Sample", p.sample_size)

                            st.caption(f"Record: {p.wins}-{p.losses} | ROI: {p.roi:+.1%} | Baseline: {p.baseline_rate:.1%} | Confidence: {p.confidence.title()}")
                            st.markdown("---")
                else:
                    st.warning("No patterns match the tier filters")

            with tier_tab2:
                st.markdown("""
                **Current Opportunities** show team pairs currently in tier combinations
                with detected profitable patterns. When these teams play each other,
                the pattern suggests a betting edge.
                """)

                # Find opportunities
                tier_opps = find_tier_matchup_opportunities(conn, tier_patterns, min_confidence='medium')

                if not tier_opps:
                    st.info("No current tier matchup opportunities. This means no teams currently in matching tier combinations or no high-confidence patterns detected.")
                else:
                    # Filter by sport if desired
                    opp_sports = list(set([o.sport for o in tier_opps]))
                    if len(opp_sports) > 1:
                        tier_opp_sport = st.multiselect(
                            "Filter by Sport",
                            opp_sports,
                            default=opp_sports,
                            key="tier_opp_sport"
                        )
                        filtered_tier_opps = [o for o in tier_opps if o.sport in tier_opp_sport]
                    else:
                        filtered_tier_opps = tier_opps

                    st.markdown(f"**{len(filtered_tier_opps)} potential matchups** (when these teams play each other)")

                    # Show opportunities
                    for i, opp in enumerate(filtered_tier_opps[:20]):
                        icon = "🟢" if opp.pattern.bet_on == "HIGHER" else "🔴"

                        with st.container():
                            opp_col1, opp_col2, opp_col3 = st.columns([3, 2, 1])

                            with opp_col1:
                                st.markdown(f"### {icon} {opp.higher_rated_team} vs {opp.lower_rated_team}")
                                st.caption(f"{opp.sport} | {opp.higher_tier} ({opp.higher_rating:.2f}) vs {opp.lower_tier} ({opp.lower_rating:.2f})")

                            with opp_col2:
                                st.markdown(f"**{opp.recommendation}**")
                                st.caption(f"Pattern: +{opp.pattern.handicap} handicap | {opp.pattern.sample_size} games")

                            with opp_col3:
                                st.metric("Edge", f"{opp.edge_pct:.1%}")

                        if i < len(filtered_tier_opps) - 1:
                            st.markdown("---")

    # ----- Tab 4: Today's Picks -----
    with tab4:
        st.markdown("""
        **Today's Picks** combines streak and tier matchup insights for games scheduled today.
        Recommendations are generated by cross-referencing today's games against detected patterns.
        """)

        # Check for API key
        api_key = get_api_key()

        if not api_key:
            st.warning("⚠️ Odds API key not configured. Set `ODDS_API_KEY` environment variable or configure AWS Secrets Manager.")
            st.info("The API key is required to fetch today's game schedule from The Odds API.")
        else:
            # Sport filter for today's picks
            picks_sport = st.selectbox(
                "Sport",
                ["All", "NFL", "NBA", "NCAAM"],
                key="todays_picks_sport"
            )

            sport_to_use = picks_sport if picks_sport != "All" else "All"

            # Get tier patterns for all sports or selected sport
            @st.cache_data(ttl=600, show_spinner=False)
            def get_cached_tier_patterns(_conn, sport_key):
                if sport_key == "All":
                    return detect_all_tier_matchup_patterns(_conn, min_sample=30, min_edge=0.05)
                return detect_tier_matchup_patterns(_conn, sport_key, min_sample=30, min_edge=0.05)

            tier_patterns_for_picks = get_cached_tier_patterns(conn, sport_to_use)

            # Generate recommendations
            with st.spinner("Fetching today's games and analyzing..."):
                game_recommendations = generate_recommendations(
                    conn,
                    sport_to_use,
                    patterns,  # Streak patterns from earlier in the page
                    tier_patterns_for_picks,
                    api_key
                )

            if not game_recommendations:
                st.info(f"No games scheduled today for {picks_sport}.")
            else:
                # Count games with recommendations
                games_with_recs = [g for g in game_recommendations if g.recommendations]
                st.markdown(f"**Found {len(game_recommendations)} games today** ({len(games_with_recs)} with betting edges)")

                for game in game_recommendations:
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

                        # Spread info
                        if game.spread is not None:
                            spread_display = f"+{game.spread}" if game.spread > 0 else str(game.spread)
                            st.caption(f"Spread: {game.home_team} {spread_display} ({game.spread_source or 'Unknown'})")

                        # Recommendations
                        if game.recommendations:
                            # Group by bet_on team
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
                                    source_icon = "📈" if rec.source == "streak" else "🎯"
                                    st.markdown(f"- {source_icon} **{rec.source.replace('_', ' ').title()}**: {rec.rationale}")

                                st.caption(f"Combined Confidence: {combined_conf}")

                        else:
                            st.markdown("⚪ **NO EDGE DETECTED**")
                            st.caption("Neither team matches profitable patterns")

                        # Streak info in expander
                        with st.expander("Team Details"):
                            detail_col1, detail_col2 = st.columns(2)

                            with detail_col1:
                                st.markdown(f"**{game.home_team}** (Home)")
                                if game.home_ats_rating is not None:
                                    st.write(f"ATS Rating: {game.home_ats_rating:.3f} ({game.home_tier})")
                                if game.home_streak:
                                    st.write(f"Streak: {game.home_streak[0]}-game ATS {game.home_streak[1].lower()}")
                                else:
                                    st.write("No current streak")

                            with detail_col2:
                                st.markdown(f"**{game.away_team}** (Away)")
                                if game.away_ats_rating is not None:
                                    st.write(f"ATS Rating: {game.away_ats_rating:.3f} ({game.away_tier})")
                                if game.away_streak:
                                    st.write(f"Streak: {game.away_streak[0]}-game ATS {game.away_streak[1].lower()}")
                                else:
                                    st.write("No current streak")

                    st.markdown("---")

    # Current team streaks (collapsible)
    with st.expander("📊 All Current Team Streaks"):
        for sport in ['NFL', 'NBA', 'NCAAM']:
            streaks = get_current_streaks(conn, sport)
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


# =============================================================================
# Page: Power Rankings
# =============================================================================

def page_power_rankings():
    st.title("🏆 Power Rankings")
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

    # Get rankings
    with st.spinner("Computing power rankings..."):
        rankings = get_team_rankings(conn, selected_sport, min_games=min_games)

    if not rankings:
        st.warning(f"No ranking data found for {selected_sport}")
        return

    # Filter unreliable if needed
    if not show_unreliable:
        display_rankings = [r for r in rankings if r.is_reliable]
    else:
        display_rankings = rankings

    if not display_rankings:
        st.warning("No teams meet the minimum games threshold")
        return

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

    tab1, tab2 = st.tabs(["Market Gap Chart", "Win vs ATS Scatter"])

    with tab1:
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

    with tab2:
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


# =============================================================================
# Page: Backtest Strategies
# =============================================================================

def page_backtest():
    st.title("📈 Backtest Strategies")
    st.markdown(f"**Rating-based strategy backtesting** for **{selected_sport}**. Breakeven at -110 odds: **52.4%**")

    games_df = get_games_with_ratings(conn, selected_sport)
    if len(games_df) == 0:
        st.warning(f"No games with historical ratings found. Run `python scripts/generate_historical_ratings.py` first.")
        return

    # Get live baseline cover rates for all handicap levels (0-15) for dynamic color scaling
    baseline_df = baseline_handicap_coverage(conn, selected_sport, handicap_range=(0, 15))

    # Helper: assign rating tier
    def get_tier(rating):
        if rating >= 0.7: return '1-Top (.7+)'
        if rating >= 0.55: return '2-High (.55-.7)'
        if rating >= 0.45: return '3-Mid (.45-.55)'
        if rating >= 0.3: return '4-Low (.3-.45)'
        return '5-Bottom (<.3)'

    # Add tier columns
    games_df['home_tier'] = games_df['home_ats_rating'].apply(get_tier)
    games_df['away_tier'] = games_df['away_ats_rating'].apply(get_tier)

    # Determine higher/lower rated team info
    games_df['higher_team'] = games_df.apply(
        lambda r: r['home_team'] if r['home_ats_rating'] > r['away_ats_rating'] else r['away_team'], axis=1)
    games_df['lower_team'] = games_df.apply(
        lambda r: r['away_team'] if r['home_ats_rating'] > r['away_ats_rating'] else r['home_team'], axis=1)
    games_df['higher_rating'] = games_df[['home_ats_rating', 'away_ats_rating']].max(axis=1)
    games_df['lower_rating'] = games_df[['home_ats_rating', 'away_ats_rating']].min(axis=1)
    games_df['higher_tier'] = games_df['higher_rating'].apply(get_tier)
    games_df['lower_tier'] = games_df['lower_rating'].apply(get_tier)

    # Determine if higher-rated team is home (needed for handicap calculations)
    games_df['higher_is_home'] = games_df['home_ats_rating'] > games_df['away_ats_rating']

    # Did the team with higher ATS rating cover? (at handicap 0)
    # This replaces higher_gap_covered for consistency with tier-based analysis
    games_df['higher_ats_covered'] = (
        (games_df['higher_is_home'] & (games_df['spread_result'] > 0)) |
        (~games_df['higher_is_home'] & (games_df['spread_result'] < 0))
    )

    # Header metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Games Analyzed", len(games_df))
    col2.metric("Avg Rating Diff", f"{(games_df['higher_rating'] - games_df['lower_rating']).mean():.3f}")
    col3.metric("Higher Rated Covers", f"{games_df['higher_ats_covered'].mean():.1%}")

    tab1, tab2, tab3, tab4 = st.tabs(["Tier Matchups", "Tier + Handicap", "Gap Threshold", "Matchup Details"])

    # Tab 1: Rating Tier Matrix
    with tab1:
        st.subheader("Rating Tier Matchup Matrix")
        st.markdown("When the **higher-rated** team plays a **lower-rated** team, does the higher-rated team cover?")

        # Build matrix
        matrix_data = games_df.groupby(['higher_tier', 'lower_tier']).agg(
            games=('higher_ats_covered', 'count'),
            wins=('higher_ats_covered', 'sum')
        ).reset_index()
        matrix_data['win_rate'] = matrix_data['wins'] / matrix_data['games']
        matrix_data['losses'] = matrix_data['games'] - matrix_data['wins']

        # Create pivot for heatmap
        tier_order = ['1-Top (.7+)', '2-High (.55-.7)', '3-Mid (.45-.55)', '4-Low (.3-.45)', '5-Bottom (<.3)']
        pivot_wr = matrix_data.pivot(index='higher_tier', columns='lower_tier', values='win_rate')
        pivot_n = matrix_data.pivot(index='higher_tier', columns='lower_tier', values='games')

        # Reindex to ensure order
        pivot_wr = pivot_wr.reindex(index=tier_order, columns=tier_order)
        pivot_n = pivot_n.reindex(index=tier_order, columns=tier_order)

        # Create heatmap with annotations
        annotations = []
        for i, row_tier in enumerate(tier_order):
            for j, col_tier in enumerate(tier_order):
                wr = pivot_wr.loc[row_tier, col_tier] if pd.notna(pivot_wr.loc[row_tier, col_tier]) else None
                n = pivot_n.loc[row_tier, col_tier] if pd.notna(pivot_n.loc[row_tier, col_tier]) else None
                if wr is not None and n is not None:
                    annotations.append(f"{wr:.0%}<br>({int(n)})")
                else:
                    annotations.append("")

        # Get live baseline at handicap=0 for color scale centering
        baseline_at_0 = baseline_df[baseline_df['handicap'] == 0]['baseline_cover_pct'].iloc[0] * 100
        # Dynamic zmin/zmax centered on baseline (±25 percentage points)
        zmin_0 = max(0, baseline_at_0 - 25)
        zmax_0 = min(100, baseline_at_0 + 25)

        fig = go.Figure(data=go.Heatmap(
            z=pivot_wr.values * 100,
            x=[t.split('-')[1] for t in tier_order],  # Clean labels
            y=[t.split('-')[1] for t in tier_order],
            text=[[annotations[i*5+j] for j in range(5)] for i in range(5)],
            texttemplate="%{text}",
            colorscale=[[0, '#e74c3c'], [0.5, '#f5f5f5'], [1, '#2ecc71']],
            zmid=baseline_at_0,
            zmin=zmin_0, zmax=zmax_0,
            hovertemplate="Higher: %{y}<br>vs Lower: %{x}<br>Win Rate: %{z:.1f}%<extra></extra>"
        ))
        fig.update_layout(
            title="Higher-Rated Team Cover Rate by Tier Matchup",
            xaxis_title="Lower-Rated Team Tier",
            yaxis_title="Higher-Rated Team Tier",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"🎯 League baseline at 0pt handicap: {baseline_at_0:.2f}% | Green = above baseline | Red = below baseline")

        # Tier selector for drill-down
        st.markdown("---")
        st.subheader("Drill Down by Tier")
        col1, col2 = st.columns(2)
        with col1:
            selected_higher = st.selectbox("Higher-Rated Tier", tier_order, index=0)
        with col2:
            selected_lower = st.selectbox("Lower-Rated Tier", tier_order, index=4)

        # Filter and show summary
        filtered = games_df[(games_df['higher_tier'] == selected_higher) & (games_df['lower_tier'] == selected_lower)]
        if len(filtered) > 0:
            wins = filtered['higher_ats_covered'].sum()
            losses = len(filtered) - wins
            wr = wins / len(filtered)
            roi = (wins * 0.91 - losses) / len(filtered)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Games", len(filtered))
            col2.metric("Record", f"{wins}-{losses}")
            col3.metric("Win Rate", f"{wr:.1%}", delta=f"{(wr - BREAKEVEN_WIN_RATE)*100:+.1f}%")
            col4.metric("ROI", f"{roi:+.1%}")

            # Store selection for Tab 3
            st.session_state['backtest_filter'] = {
                'higher_tier': selected_higher,
                'lower_tier': selected_lower
            }
        else:
            st.info("No games match this tier combination.")

    # Tab 2: Tier Matchups with Handicap
    with tab2:
        st.subheader("Rating Tier Matrix with Handicap")
        st.markdown("Analyze cover rates when giving **extra points** to either the higher-rated or lower-rated team.")

        # Controls
        col1, col2 = st.columns([1, 2])
        with col1:
            handicap_pts = st.slider("Handicap Points", 0, 15, 0, key="tier_handicap")
        with col2:
            handicap_to = st.radio(
                "Give handicap to:",
                ["Higher-Rated Team", "Lower-Rated Team"],
                horizontal=True,
                key="handicap_recipient"
            )

        # Calculate coverage with handicap for BOTH sides
        # Higher-rated team with +N handicap:
        #   If HOME: cover if spread_result > -N
        #   If AWAY: cover if spread_result < N
        games_df['higher_covers_with_handicap'] = (
            (games_df['higher_is_home'] & (games_df['spread_result'] > -handicap_pts)) |
            (~games_df['higher_is_home'] & (games_df['spread_result'] < handicap_pts))
        )
        # Lower-rated team with +N handicap:
        #   If HOME (higher is away): cover if spread_result > -N
        #   If AWAY (higher is home): cover if spread_result < N
        games_df['lower_covers_with_handicap'] = (
            (~games_df['higher_is_home'] & (games_df['spread_result'] > -handicap_pts)) |
            (games_df['higher_is_home'] & (games_df['spread_result'] < handicap_pts))
        )

        # Show summary stats for BOTH sides
        st.markdown("---")
        st.subheader(f"Overall Coverage at +{handicap_pts} Handicap")

        total_games = len(games_df)
        higher_wins = games_df['higher_covers_with_handicap'].sum()
        lower_wins = games_df['lower_covers_with_handicap'].sum()
        higher_wr = higher_wins / total_games if total_games > 0 else 0
        lower_wr = lower_wins / total_games if total_games > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Games", total_games)
        col2.metric(
            f"Higher-Rated +{handicap_pts}",
            f"{higher_wr:.1%}",
            delta=f"{(higher_wr - BREAKEVEN_WIN_RATE)*100:+.1f}%"
        )
        col3.metric(
            f"Lower-Rated +{handicap_pts}",
            f"{lower_wr:.1%}",
            delta=f"{(lower_wr - BREAKEVEN_WIN_RATE)*100:+.1f}%"
        )

        # Select which coverage column to use for the heatmap
        coverage_col = 'higher_covers_with_handicap' if handicap_to == "Higher-Rated Team" else 'lower_covers_with_handicap'

        # Build matrix with handicap
        matrix_hc = games_df.groupby(['higher_tier', 'lower_tier']).agg(
            games=(coverage_col, 'count'),
            wins=(coverage_col, 'sum')
        ).reset_index()
        matrix_hc['win_rate'] = matrix_hc['wins'] / matrix_hc['games']

        # Create pivot for heatmap
        tier_order = ['1-Top (.7+)', '2-High (.55-.7)', '3-Mid (.45-.55)', '4-Low (.3-.45)', '5-Bottom (<.3)']
        pivot_wr_hc = matrix_hc.pivot(index='higher_tier', columns='lower_tier', values='win_rate')
        pivot_n_hc = matrix_hc.pivot(index='higher_tier', columns='lower_tier', values='games')

        # Reindex to ensure order
        pivot_wr_hc = pivot_wr_hc.reindex(index=tier_order, columns=tier_order)
        pivot_n_hc = pivot_n_hc.reindex(index=tier_order, columns=tier_order)

        # Create heatmap with annotations
        annotations_hc = []
        for i, row_tier in enumerate(tier_order):
            for j, col_tier in enumerate(tier_order):
                wr = pivot_wr_hc.loc[row_tier, col_tier] if pd.notna(pivot_wr_hc.loc[row_tier, col_tier]) else None
                n = pivot_n_hc.loc[row_tier, col_tier] if pd.notna(pivot_n_hc.loc[row_tier, col_tier]) else None
                if wr is not None and n is not None:
                    annotations_hc.append(f"{wr:.0%}<br>({int(n)})")
                else:
                    annotations_hc.append("")

        team_label = "Higher-Rated" if handicap_to == "Higher-Rated Team" else "Lower-Rated"

        # Get live baseline at selected handicap for color scale centering
        baseline_at_handicap = baseline_df[baseline_df['handicap'] == handicap_pts]['baseline_cover_pct'].iloc[0] * 100
        # Dynamic zmin/zmax centered on baseline (±25 percentage points)
        zmin_hc = max(0, baseline_at_handicap - 25)
        zmax_hc = min(100, baseline_at_handicap + 25)

        fig_hc = go.Figure(data=go.Heatmap(
            z=pivot_wr_hc.values * 100,
            x=[t.split('-')[1] for t in tier_order],
            y=[t.split('-')[1] for t in tier_order],
            text=[[annotations_hc[i*5+j] for j in range(5)] for i in range(5)],
            texttemplate="%{text}",
            colorscale=[[0, '#e74c3c'], [0.5, '#f5f5f5'], [1, '#2ecc71']],
            zmid=baseline_at_handicap,
            zmin=zmin_hc, zmax=zmax_hc,
            hovertemplate="Higher Tier: %{y}<br>vs Lower Tier: %{x}<br>Cover Rate: %{z:.1f}%<extra></extra>"
        ))
        fig_hc.update_layout(
            title=f"{team_label} Team Cover Rate (+{handicap_pts} pts)",
            xaxis_title="Lower-Rated Team Tier",
            yaxis_title="Higher-Rated Team Tier",
            height=500
        )
        st.plotly_chart(fig_hc, use_container_width=True)
        st.caption(f"🎯 League baseline at +{handicap_pts}pt handicap: {baseline_at_handicap:.2f}% | Green = above baseline | Red = below baseline")

        # Summary table
        st.markdown("---")
        st.subheader(f"Summary: {team_label} +{handicap_pts} by Tier Matchup")
        summary_hc = matrix_hc.copy()
        summary_hc['tier_matchup'] = summary_hc['higher_tier'].str.split('-').str[1] + ' vs ' + summary_hc['lower_tier'].str.split('-').str[1]
        summary_hc['Cover Rate'] = summary_hc['win_rate'].apply(lambda x: f"{x:.1%}")
        summary_hc['losses'] = summary_hc['games'] - summary_hc['wins']
        summary_hc['Record'] = summary_hc.apply(lambda r: f"{int(r['wins'])}-{int(r['losses'])}", axis=1)
        summary_hc['ROI'] = summary_hc.apply(lambda r: f"{((r['wins'] * 0.91 - r['losses']) / r['games']):+.1%}" if r['games'] > 0 else "N/A", axis=1)
        summary_hc = summary_hc.sort_values('win_rate', ascending=False)
        st.dataframe(
            summary_hc[['tier_matchup', 'games', 'Record', 'Cover Rate', 'ROI']].rename(columns={'tier_matchup': 'Matchup', 'games': 'Games'}),
            hide_index=True
        )

    # Tab 3: Gap Threshold Analysis
    with tab3:
        st.subheader("Gap Threshold Analysis")
        bet_direction = st.selectbox(
            "Bet Direction", ["higher_gap", "lower_gap"],
            format_func=lambda x: "Bet on Higher Gap Team" if x == "higher_gap" else "Fade Higher Gap Team"
        )

        threshold_df = analyze_gap_thresholds(games_df, bet_on=bet_direction)

        if len(threshold_df) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                name='Win Rate', x=threshold_df['threshold'], y=threshold_df['win_rate'] * 100,
                marker_color=['#2ecc71' if wr > BREAKEVEN_WIN_RATE else '#e74c3c' for wr in threshold_df['win_rate']],
                text=threshold_df['win_rate'].apply(lambda x: f'{x:.1%}'), textposition='outside'
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                name='ROI', x=threshold_df['threshold'], y=threshold_df['roi'] * 100,
                mode='lines+markers', line=dict(color='#3498db', width=3)
            ), secondary_y=True)
            fig.add_hline(y=BREAKEVEN_WIN_RATE * 100, line_dash="dash", line_color="gray")
            fig.update_layout(title=f"Win Rate & ROI by Gap Threshold ({selected_sport})",
                              xaxis_title="Gap Threshold", height=450)
            fig.update_yaxes(title_text="Win Rate (%)", range=[40, 80], secondary_y=False)
            fig.update_yaxes(title_text="ROI (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            display_df = threshold_df.copy()
            display_df['Win Rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1%}")
            display_df['ROI'] = display_df['roi'].apply(lambda x: f"{x:+.1%}")
            st.dataframe(display_df[['threshold', 'total_bets', 'wins', 'losses', 'Win Rate', 'ROI']], hide_index=True)

    # Tab 4: Matchup Details
    with tab4:
        st.subheader("Matchup Details")

        # Get filter from session state or use defaults
        filter_info = st.session_state.get('backtest_filter', {})
        higher_t = filter_info.get('higher_tier', '1-Top (.7+)')
        lower_t = filter_info.get('lower_tier', '5-Bottom (<.3)')

        st.markdown(f"**Showing:** {higher_t.split('-')[1]} vs {lower_t.split('-')[1]}")
        st.caption("Select tiers in the 'Tier Matchups' tab to change filter")

        filtered = games_df[(games_df['higher_tier'] == higher_t) & (games_df['lower_tier'] == lower_t)].copy()
        filtered = filtered.sort_values('game_date', ascending=False)

        if len(filtered) > 0:
            wins = filtered['higher_ats_covered'].sum()
            wr = wins / len(filtered)
            col1, col2 = st.columns(2)
            col1.metric("Win Rate", f"{wr:.1%}")
            col2.metric("Edge vs Breakeven", f"{(wr - BREAKEVEN_WIN_RATE)*100:+.1f}%")

            # Build display table
            display_df = filtered[['game_date', 'higher_team', 'higher_rating', 'lower_team', 'lower_rating',
                                   'closing_spread', 'higher_ats_covered']].copy()
            display_df.columns = ['Date', 'Higher Rated', 'H.Rtg', 'Lower Rated', 'L.Rtg', 'Spread', 'Covered']
            display_df['H.Rtg'] = display_df['H.Rtg'].apply(lambda x: f"{x:.3f}")
            display_df['L.Rtg'] = display_df['L.Rtg'].apply(lambda x: f"{x:.3f}")
            display_df['Result'] = display_df['Covered'].apply(lambda x: '✅' if x else '❌')
            st.dataframe(display_df[['Date', 'Higher Rated', 'H.Rtg', 'Lower Rated', 'L.Rtg', 'Spread', 'Result']],
                         hide_index=True, use_container_width=True)
        else:
            st.info("No games match the selected tier combination.")


# =============================================================================
# Page: Exploration
# =============================================================================

def page_exploration():
    st.title("Custom Exploration")
    st.markdown("Build custom queries with flexible filters")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        perspective = st.selectbox("Perspective", ["home", "away"])
        handicap = st.slider("Handicap Points", 0, 15, 0)

    with col2:
        min_spread = st.number_input("Min Spread", value=-50.0, step=0.5)
        max_spread = st.number_input("Max Spread", value=50.0, step=0.5)

    with col3:
        date_range = get_date_range(conn, selected_sport)
        if date_range[0] and date_range[1]:
            from datetime import datetime
            start = datetime.strptime(date_range[0], '%Y-%m-%d').date() if isinstance(date_range[0], str) else date_range[0]
            end = datetime.strptime(date_range[1], '%Y-%m-%d').date() if isinstance(date_range[1], str) else date_range[1]
        else:
            from datetime import date
            start = date(2024, 1, 1)
            end = date.today()

    # Load filtered games
    games = get_games(
        conn,
        sport=selected_sport,
        min_spread=min_spread if min_spread > -50 else None,
        max_spread=max_spread if max_spread < 50 else None
    )

    if len(games) == 0:
        st.warning("No games match the filters")
        return

    # Calculate metrics
    rate = ats_cover_rate(games, handicap=handicap, perspective=perspective)
    w, l, p = ats_record(games, handicap=handicap, perspective=perspective)
    margin = spread_margin_avg(games, perspective=perspective)

    # Display results
    st.markdown("---")
    st.subheader("Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games Analyzed", len(games))
    col2.metric("ATS Record", f"{w}-{l}-{p}")
    col3.metric("Cover Rate", f"{rate:.1%}")
    col4.metric("Avg Margin", f"{margin:+.1f}")

    # Time series
    ts = time_series_ats(games, handicap=handicap, perspective=perspective, cumulative=True)

    if len(ts) > 0:
        fig = px.line(
            ts,
            x='game_date',
            y='cover_pct',
            title=f"Cumulative {perspective.title()} Cover Rate Over Time",
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_yaxes(tickformat='.1%')

        st.plotly_chart(fig, use_container_width=True)


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
# Main Router
# =============================================================================

if page == "📋 Strategy Overview":
    page_matchup_opportunities()
elif page == "🎲 Today's Picks":
    page_todays_picks()
elif page == "🏆 Power Rankings":
    page_power_rankings()
elif page == "📈 Backtest Strategies":
    page_backtest()
elif page == "Macro Trends":
    page_macro_trends()
elif page == "Micro (Team) Analysis":
    page_micro_analysis()
elif page == "Streak Analysis":
    page_streak_analysis()
elif page == "Exploration":
    page_exploration()
elif page == "Verification":
    page_verification()


# =============================================================================
# Footer
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.caption("Sports Betting Analytics v0.1")
