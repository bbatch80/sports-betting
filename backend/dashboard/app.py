"""
Sports Betting Analytics Dashboard

Interactive Streamlit dashboard for exploring betting trends.
Run with: streamlit run dashboard/app.py
"""

import sys
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
    get_current_streaks,
)
from src.analysis.network_ratings import (
    get_team_rankings,
    get_rankings_dataframe,
)
from src.analysis.tier_matchups import get_tier
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
    ["🎲 Today's Picks", "🏆 Power Rankings", "📉 ATS Rating Trajectory", "📈 ATS Momentum", "🔥 Current Streaks", "Macro Trends", "Micro (Team) Analysis", "Streak Analysis", "Verification"]
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
# Page: Today's Picks (Primary Navigation)
# =============================================================================

def page_todays_picks():
    st.title("🎲 Today's Picks")
    st.markdown("""
    **Today's Picks** shows streak-based betting recommendations for games scheduled today.
    Recommendations are generated by cross-referencing today's games against detected streak patterns.
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

    # Generate recommendations (games fetched from database)
    # Note: Passing empty list for tier_patterns - only streak-based recommendations
    with st.spinner("Analyzing today's games..."):
        game_recommendations = generate_recommendations(
            conn,
            sport_to_use,
            patterns,
            [],  # No tier matchup patterns - streak-based only
        )

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

                # Spread info
                if game.spread is not None:
                    spread_display = f"+{game.spread}" if game.spread > 0 else str(game.spread)
                    st.caption(f"Spread: {game.home_team} {spread_display} ({game.spread_source or 'Unknown'})")

                # Recommendations - group by bet_on team
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
# Page: ATS Rating Trajectory
# =============================================================================

def page_ats_trajectory():
    """Visualize how team ATS ratings evolve over time."""
    st.title("ATS Rating Trajectory")
    st.markdown("Track how team ATS ratings change throughout the season. Do TOP teams stay TOP?")

    # Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        page_sport = st.selectbox("League", ["NBA", "NFL", "NCAAM"], key="trajectory_sport")

    # Query historical ratings
    from sqlalchemy import text

    query = text("""
        SELECT snapshot_date, team, ats_rating, ats_rank
        FROM historical_ratings
        WHERE sport = :sport
        ORDER BY snapshot_date, team
    """)

    try:
        result = conn.execute(query, {'sport': page_sport})
        rows = result.fetchall()
    except Exception as e:
        st.error(f"Error loading historical ratings: {e}")
        return

    if not rows:
        st.warning(f"No historical ratings found for {page_sport}")
        return

    # Build DataFrame
    df = pd.DataFrame(rows, columns=['snapshot_date', 'team', 'ats_rating', 'ats_rank'])
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date']).dt.date

    # Get list of teams
    teams = sorted(df['team'].unique())

    st.markdown("---")

    # Team selector
    col1, col2 = st.columns([2, 2])
    with col1:
        view_mode = st.radio("View Mode", ["Selected Teams", "All Teams (by current tier)"], horizontal=True)

    if view_mode == "Selected Teams":
        with col2:
            selected_teams = st.multiselect(
                "Select Teams to Compare",
                teams,
                default=teams[:5] if len(teams) >= 5 else teams,
                key="trajectory_teams"
            )

        if not selected_teams:
            st.info("Select at least one team to view their rating trajectory.")
            return

        plot_df = df[df['team'].isin(selected_teams)]

        # Line chart
        fig = go.Figure()

        for team in selected_teams:
            team_data = plot_df[plot_df['team'] == team].sort_values('snapshot_date')
            fig.add_trace(go.Scatter(
                x=team_data['snapshot_date'],
                y=team_data['ats_rating'],
                mode='lines',
                name=team,
                hovertemplate=f"{team}<br>Date: %{{x}}<br>ATS Rating: %{{y:.3f}}<extra></extra>"
            ))

        # Add tier boundary lines
        fig.add_hline(y=0.70, line_dash="dash", line_color="green", annotation_text="TOP (0.70)")
        fig.add_hline(y=0.55, line_dash="dash", line_color="blue", annotation_text="HIGH (0.55)")
        fig.add_hline(y=0.45, line_dash="dash", line_color="gray", annotation_text="MID (0.45)")
        fig.add_hline(y=0.30, line_dash="dash", line_color="orange", annotation_text="LOW (0.30)")

        fig.update_layout(
            title=f"{page_sport} ATS Rating Over Time",
            xaxis_title="Date",
            yaxis_title="ATS Rating",
            yaxis=dict(range=[0, 1]),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )

        st.plotly_chart(fig, width='stretch')

    else:
        # Show all teams grouped by current tier
        latest_date = df['snapshot_date'].max()
        latest_ratings = df[df['snapshot_date'] == latest_date].copy()
        latest_ratings['tier'] = latest_ratings['ats_rating'].apply(get_tier)

        selected_tier = st.selectbox("Filter by Current Tier", ['TOP', 'HIGH', 'MID', 'LOW', 'BOTTOM'], key="trajectory_tier")

        tier_teams = latest_ratings[latest_ratings['tier'] == selected_tier]['team'].tolist()

        if not tier_teams:
            st.info(f"No teams currently in {selected_tier} tier")
            return

        st.caption(f"{len(tier_teams)} teams currently in {selected_tier} tier")

        plot_df = df[df['team'].isin(tier_teams)]

        fig = go.Figure()

        for team in tier_teams:
            team_data = plot_df[plot_df['team'] == team].sort_values('snapshot_date')
            fig.add_trace(go.Scatter(
                x=team_data['snapshot_date'],
                y=team_data['ats_rating'],
                mode='lines',
                name=team,
                opacity=0.7,
                hovertemplate=f"{team}<br>Date: %{{x}}<br>ATS Rating: %{{y:.3f}}<extra></extra>"
            ))

        # Add tier boundary lines
        fig.add_hline(y=0.70, line_dash="dash", line_color="green", annotation_text="TOP")
        fig.add_hline(y=0.55, line_dash="dash", line_color="blue", annotation_text="HIGH")
        fig.add_hline(y=0.45, line_dash="dash", line_color="gray", annotation_text="MID")
        fig.add_hline(y=0.30, line_dash="dash", line_color="orange", annotation_text="LOW")

        fig.update_layout(
            title=f"{page_sport} ATS Rating Trajectory - Current {selected_tier} Tier Teams",
            xaxis_title="Date",
            yaxis_title="ATS Rating",
            yaxis=dict(range=[0, 1]),
            height=500,
            showlegend=True if len(tier_teams) <= 15 else False
        )

        st.plotly_chart(fig, width='stretch')

    # Summary stats
    st.subheader("Rating Stability Analysis")

    # Calculate volatility (std dev) for each team
    volatility = df.groupby('team')['ats_rating'].agg(['std', 'mean', 'min', 'max']).reset_index()
    volatility.columns = ['Team', 'Volatility (Std)', 'Mean Rating', 'Min Rating', 'Max Rating']
    volatility['Range'] = volatility['Max Rating'] - volatility['Min Rating']
    volatility = volatility.sort_values('Volatility (Std)', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most Volatile Teams** (highest std dev)")
        top_volatile = volatility.head(10).copy()
        top_volatile['Volatility (Std)'] = top_volatile['Volatility (Std)'].apply(lambda x: f"{x:.3f}")
        top_volatile['Mean Rating'] = top_volatile['Mean Rating'].apply(lambda x: f"{x:.3f}")
        top_volatile['Range'] = top_volatile['Range'].apply(lambda x: f"{x:.3f}")
        st.dataframe(top_volatile[['Team', 'Volatility (Std)', 'Mean Rating', 'Range']], hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Most Stable Teams** (lowest std dev)")
        bottom_volatile = volatility.tail(10).sort_values('Volatility (Std)').copy()
        bottom_volatile['Volatility (Std)'] = bottom_volatile['Volatility (Std)'].apply(lambda x: f"{x:.3f}")
        bottom_volatile['Mean Rating'] = bottom_volatile['Mean Rating'].apply(lambda x: f"{x:.3f}")
        bottom_volatile['Range'] = bottom_volatile['Range'].apply(lambda x: f"{x:.3f}")
        st.dataframe(bottom_volatile[['Team', 'Volatility (Std)', 'Mean Rating', 'Range']], hide_index=True, use_container_width=True)


# =============================================================================
# Page: ATS Momentum (Slope Analysis)
# =============================================================================

def page_ats_momentum():
    """Track team ATS rating momentum - which teams are trending up or down."""
    st.title("📈 ATS Rating Momentum")
    st.caption("Track which teams are trending up or down against the spread")

    # Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        page_sport = st.selectbox("League", ["NBA", "NFL", "NCAAM"], key="momentum_sport")

    # Query historical ratings
    from sqlalchemy import text

    query = text("""
        SELECT snapshot_date, team, ats_rating, ats_rank
        FROM historical_ratings
        WHERE sport = :sport
        ORDER BY team, snapshot_date
    """)

    try:
        result = conn.execute(query, {'sport': page_sport})
        rows = result.fetchall()
    except Exception as e:
        st.error(f"Error loading historical ratings: {e}")
        return

    if not rows:
        st.warning(f"No historical ratings found for {page_sport}")
        return

    # Build DataFrame
    df = pd.DataFrame(rows, columns=['snapshot_date', 'team', 'ats_rating', 'ats_rank'])

    # Get unique teams
    teams = sorted(df['team'].unique())

    # Calculate slopes for each team
    results = []
    for team in teams:
        team_data = df[df['team'] == team].sort_values('snapshot_date')
        if team_data.empty:
            continue

        current_rating = team_data['ats_rating'].iloc[-1]
        current_rank = int(team_data['ats_rank'].iloc[-1]) if pd.notna(team_data['ats_rank'].iloc[-1]) else None
        slopes = calculate_team_slopes(df, team)

        results.append({
            'Team': team,
            'ATS Rating': current_rating,
            'Rank': current_rank,
            '1W Slope': slopes['1w'],
            '2W Slope': slopes['2w'],
            '3W Slope': slopes['3w'],
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        st.warning("No data available to calculate momentum")
        return

    # Add direction indicators
    def direction_indicator(slope):
        if slope is None:
            return '—'
        if slope > 0.005:
            return '↑↑'  # Strong rise
        elif slope > 0.001:
            return '↑'   # Rising
        elif slope < -0.005:
            return '↓↓'  # Strong fall
        elif slope < -0.001:
            return '↓'   # Falling
        return '→'       # Flat

    results_df['1W Dir'] = results_df['1W Slope'].apply(direction_indicator)
    results_df['2W Dir'] = results_df['2W Slope'].apply(direction_indicator)
    results_df['3W Dir'] = results_df['3W Slope'].apply(direction_indicator)

    # Calculate Momentum Change (1W slope - 3W slope)
    # Positive = accelerating (recent trend stronger), Negative = decelerating
    def calc_momentum_change(row):
        if pd.isna(row['1W Slope']) or pd.isna(row['3W Slope']):
            return None
        return row['1W Slope'] - row['3W Slope']

    results_df['Momentum'] = results_df.apply(calc_momentum_change, axis=1)

    def momentum_indicator(val):
        if val is None:
            return '—'
        if val > 0.003:
            return '🚀'  # Strong acceleration
        elif val > 0.001:
            return '⬆️'   # Accelerating
        elif val < -0.003:
            return '🔻'  # Strong deceleration
        elif val < -0.001:
            return '⬇️'   # Decelerating
        return '➡️'       # Steady

    results_df['Accel'] = results_df['Momentum'].apply(momentum_indicator)

    st.markdown("---")

    # Display controls
    col1, col2 = st.columns([1, 1])
    with col1:
        sort_by = st.selectbox("Sort by", ['ATS Rating', '1W Slope', '2W Slope', '3W Slope', 'Momentum', 'Rank'])
    with col2:
        ascending = st.checkbox("Ascending", value=False)

    # Sort and display
    sorted_df = results_df.sort_values(sort_by, ascending=ascending, na_position='last')

    # Format for display
    display_df = sorted_df[['Team', 'ATS Rating', 'Rank',
                            '1W Slope', '1W Dir',
                            '2W Slope', '2W Dir',
                            '3W Slope', '3W Dir',
                            'Momentum', 'Accel']].copy()

    # Format numeric columns
    display_df['ATS Rating'] = display_df['ATS Rating'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else '—')
    display_df['1W Slope'] = display_df['1W Slope'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else '—')
    display_df['2W Slope'] = display_df['2W Slope'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else '—')
    display_df['3W Slope'] = display_df['3W Slope'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else '—')
    display_df['Momentum'] = display_df['Momentum'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else '—')

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Highlight sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Hottest Teams (Rising Fast)")
        hot = results_df.nlargest(5, '2W Slope')[['Team', 'ATS Rating', '2W Slope', '2W Dir']].copy()
        hot['ATS Rating'] = hot['ATS Rating'].apply(lambda x: f"{x:.3f}")
        hot['2W Slope'] = hot['2W Slope'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else '—')
        st.dataframe(hot, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("❄️ Coldest Teams (Falling Fast)")
        cold = results_df.nsmallest(5, '2W Slope')[['Team', 'ATS Rating', '2W Slope', '2W Dir']].copy()
        cold['ATS Rating'] = cold['ATS Rating'].apply(lambda x: f"{x:.3f}")
        cold['2W Slope'] = cold['2W Slope'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else '—')
        st.dataframe(cold, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Team trajectory mini-chart
    st.subheader("Team Trajectory")
    selected_team = st.selectbox("Select team to view trajectory", teams, key="momentum_team_select")
    team_history = df[df['team'] == selected_team].sort_values('snapshot_date')

    if not team_history.empty:
        team_history['snapshot_date'] = pd.to_datetime(team_history['snapshot_date'])

        fig = px.line(team_history, x='snapshot_date', y='ats_rating',
                      title=f"{selected_team} ATS Rating Over Time")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="ATS Rating",
            height=400
        )
        # Add tier reference lines
        fig.add_hline(y=0.70, line_dash="dot", line_color="green", annotation_text="TOP (0.70)")
        fig.add_hline(y=0.55, line_dash="dot", line_color="lightgreen", annotation_text="HIGH (0.55)")
        fig.add_hline(y=0.45, line_dash="dot", line_color="gray", annotation_text="MID (0.45)")
        fig.add_hline(y=0.30, line_dash="dot", line_color="orange", annotation_text="LOW (0.30)")

        st.plotly_chart(fig, use_container_width=True)

    # Legend
    with st.expander("Understanding Slope & Momentum Values"):
        st.markdown("""
        **Slope** measures the daily rate of change in ATS rating over the specified window:

        | Symbol | Meaning | Slope Value |
        |--------|---------|-------------|
        | ↑↑ | Strong Rise | > +0.005/day |
        | ↑ | Rising | +0.001 to +0.005/day |
        | → | Flat | -0.001 to +0.001/day |
        | ↓ | Falling | -0.005 to -0.001/day |
        | ↓↓ | Strong Fall | < -0.005/day |

        ---

        **Momentum** (1W Slope - 3W Slope) shows if the trend is accelerating or decelerating:

        | Symbol | Meaning | Value |
        |--------|---------|-------|
        | 🚀 | Strong Acceleration | > +0.003 |
        | ⬆️ | Accelerating | +0.001 to +0.003 |
        | ➡️ | Steady | -0.001 to +0.001 |
        | ⬇️ | Decelerating | -0.003 to -0.001 |
        | 🔻 | Strong Deceleration | < -0.003 |

        ---

        **Strategy Implications**:
        - **Rising + Accelerating** (↑ + 🚀): Strong buy signal - team improving and getting better faster
        - **Rising + Decelerating** (↑ + 🔻): Trend may be peaking - watch for reversal
        - **Falling + Accelerating** (↓ + 🚀): Potential bottom - decline slowing down
        - **Falling + Decelerating** (↓ + 🔻): Avoid - team declining and getting worse faster
        """)




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
# Page: Current ATS Streaks
# =============================================================================

def page_current_streaks():
    """Display current ATS streaks for all teams in the selected sport."""
    st.title("🔥 Current ATS Streaks")
    st.caption(f"Track which {selected_sport} teams are currently hot or cold against the spread")

    streaks = get_current_streaks(conn, selected_sport)
    if not streaks:
        st.warning(f"No streak data available for {selected_sport}")
        return

    # Get ratings as dict for lookup
    rankings = {r.team: r for r in (get_team_rankings(conn, selected_sport) or [])}

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

    sort_by = st.selectbox("Sort by", ["Length", "Team", "Win Rtg", "ATS Rtg"])
    cols = ['Team', 'Length', 'Streak', 'Win Rtg', 'ATS Rtg']

    # Hot streaks
    st.subheader("🔥 Hot Streaks (2+ ATS Wins)")
    hot = df[(df['Type'] == 'WIN') & (df['Length'] >= 2)].sort_values('Length', ascending=False)
    if not hot.empty:
        st.dataframe(hot[cols], use_container_width=True, hide_index=True)

    # Cold streaks
    st.subheader("❄️ Cold Streaks (2+ ATS Losses)")
    cold = df[(df['Type'] == 'LOSS') & (df['Length'] >= 2)].sort_values('Length', ascending=False)
    if not cold.empty:
        st.dataframe(cold[cols], use_container_width=True, hide_index=True)

    # All teams
    st.subheader("📊 All Teams")
    df_sorted = df.sort_values(sort_by, ascending=(sort_by == "Team"), na_position='last')
    st.dataframe(df_sorted[cols], use_container_width=True, hide_index=True)


# =============================================================================
# Main Router
# =============================================================================

if page == "🎲 Today's Picks":
    page_todays_picks()
elif page == "🏆 Power Rankings":
    page_power_rankings()
elif page == "📉 ATS Rating Trajectory":
    page_ats_trajectory()
elif page == "📈 ATS Momentum":
    page_ats_momentum()
elif page == "🔥 Current Streaks":
    page_current_streaks()
elif page == "Macro Trends":
    page_macro_trends()
elif page == "Micro (Team) Analysis":
    page_micro_analysis()
elif page == "Streak Analysis":
    page_streak_analysis()
elif page == "Verification":
    page_verification()


# =============================================================================
# Footer
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.caption("Sports Betting Analytics v0.1")
