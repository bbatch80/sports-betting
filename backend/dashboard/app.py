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
    ["🎯 Opportunities", "Macro Trends", "Micro (Team) Analysis", "Streak Analysis", "Exploration", "Verification"]
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
    tab1, tab2 = st.tabs(["📋 Current Opportunities", "🔍 All Detected Patterns"])

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

if page == "🎯 Opportunities":
    page_opportunities()
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
