# Sports Betting Strategies Reference

This document provides comprehensive documentation for all betting strategies implemented in the sports-betting analytics system. Each strategy includes its logic, calculation methodology, win/loss evaluation, and configuration parameters.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Strategies](#strategies)
   - [Home Focus](#1-home-focus)
   - [Away Focus](#2-away-focus)
   - [Standard Spread Coverage](#3-standard-spread-coverage)
   - [Elite Team (Win %)](#4-elite-team-win-)
   - [Elite Team (Coverage %)](#5-elite-team-coverage-)
   - [Hot vs Cold (3 Games)](#6-hot-vs-cold-3-games)
   - [Hot vs Cold (5 Games)](#7-hot-vs-cold-5-games)
   - [Hot vs Cold (7 Games)](#8-hot-vs-cold-7-games)
   - [Opponent Perfect Form (Regression)](#9-opponent-perfect-form-regression)
   - [Common Opponent](#10-common-opponent-ncaam-only)
4. [Win/Loss Calculation](#winloss-calculation)
5. [Performance Tracking](#performance-tracking)
6. [Adding New Strategies](#adding-new-strategies)

---

## Overview

The system tracks betting opportunities across three sports:
- **NFL** (Pro Football)
- **NBA** (Pro Basketball)
- **NCAAM** (College Basketball)

Each strategy identifies games where historical data suggests a betting edge. Predictions are generated daily and matched against actual results to track performance over time.

---

## Core Concepts

### Spread Result Difference

The fundamental calculation used throughout the system:

```
spread_result_difference = (home_score - away_score) + closing_spread
```

- **Positive value**: Home team covered the spread
- **Negative value**: Away team covered the spread
- **Zero**: Push (tie against the spread)

### Handicap Points

Many strategies apply a fixed **11-point handicap** across all sports to find stronger edges:

| Strategy Type | Handicap |
|--------------|----------|
| Home Focus | 11 points |
| Away Focus | 11 points |
| Elite Team variants | 11 points |
| Hot vs Cold variants | 11 points |
| Regression | 11 points |
| Standard Spread Coverage | 0 points |
| Common Opponent | 0 points |

**Example**: If a team has 60% coverage at the 0-point handicap, they might have 75% coverage at an 11-point handicap (since the extra points make it easier to "cover").

### Coverage Percentage

The percentage of games where a team covered the spread:

```python
cover_pct = (games_covered / total_games) * 100
```

### Form-Based Strategies

Some strategies use recent form to identify momentum. Different lookback windows are available:

**Hot vs Cold (3 Games):**
- **Hot**: 67%+ coverage (≥2 of 3 games covered)
- **Cold**: 0% coverage (0 of 3 games covered)

**Hot vs Cold (5 Games):**
- **Hot**: 60%+ coverage (≥3 of 5 games covered)
- **Cold**: 40% or less coverage (≤2 of 5 covered)

**Hot vs Cold (7 Games):**
- **Hot**: 57%+ coverage (≥4 of 7 games covered)
- **Cold**: 43% or less coverage (≤3 of 7 covered)

**Perfect Form (Regression):**
- **Perfect**: 100% coverage in last 5 games (5/5)

---

## Strategies

### 1. Home Focus

**Concept**: Bet on the home team when their 11-point handicap-adjusted coverage percentage exceeds the away team's.

**Logic**:
1. Calculate each team's coverage % at the fixed 11-point handicap
2. If home team's handicap coverage % > away team's handicap coverage %, bet on home team
3. Sort opportunities by largest handicap percentage difference

**Prediction Generation** (`generate_predictions/lambda_function.py`):
```python
HOME_AWAY_HANDICAP = 11  # Fixed 11-point handicap for all sports

# Home team must have better handicap coverage than away team
home_focus_opportunities = games[
    games['home_cover_pct_handicap'] > games['away_cover_pct_handicap']
]
```

**Win Evaluation**: Home team must cover the spread after applying the 11-point handicap.
```
WIN if: spread_result_difference > 11
LOSS if: spread_result_difference < 11
PUSH if: spread_result_difference == 11
```

**Handicap**: 11 points (all sports)

**Example**:
- Lakers (home) have 65% handicap coverage at 11 points
- Celtics (away) have 55% handicap coverage at 11 points
- **Bet**: Lakers to cover
- **To Win**: Lakers must cover the spread by more than 11 points relative to the closing line

---

### 2. Away Focus

**Concept**: Bet on the away team when their 11-point handicap-adjusted coverage percentage exceeds the home team's.

**Logic**:
1. Calculate each team's coverage % at the fixed 11-point handicap
2. If away team's handicap coverage % > home team's handicap coverage %, bet on away team
3. Sort opportunities by largest handicap percentage difference

**Prediction Generation**:
```python
HOME_AWAY_HANDICAP = 11  # Fixed 11-point handicap for all sports

# Away team must have better handicap coverage than home team
away_focus_opportunities = games[
    games['away_cover_pct_handicap'] > games['home_cover_pct_handicap']
]
```

**Win Evaluation**: Away team must cover the spread after applying the 11-point handicap.
```
WIN if: spread_result_difference < -11
LOSS if: spread_result_difference > -11
PUSH if: spread_result_difference == -11
```

**Handicap**: 11 points (all sports)

---

### 3. Standard Spread Coverage

**Concept**: Bet on the team with significantly better season-long spread coverage (no handicap).

**Logic**:
1. Calculate each team's overall spread coverage percentage (0-point handicap)
2. Find games where one team has ≥10% better coverage than the other
3. Bet on the team with better coverage
4. Sort by largest coverage difference

**Prediction Generation**:
```python
# Calculate absolute difference in coverage percentages
coverage_difference = abs(home_cover_pct - away_cover_pct)

# Filter for ≥10% difference
coverage_based_opportunities = games[coverage_difference > 10]
```

**Win Evaluation**: The "better team" must cover the standard spread (0-point handicap).
```
If betting HOME:
  WIN if: spread_result_difference > 0
  LOSS if: spread_result_difference < 0

If betting AWAY:
  WIN if: spread_result_difference < 0
  LOSS if: spread_result_difference > 0
```

**Handicap**: 0 points (straight spread)

**Additional Data Tracked**:
- Average cover margin (how much teams cover by when they win)
- Average failure margin (how much teams lose by when they don't cover)
- Last 5 games coverage %
- Trend indicator (improving/declining/stable)

---

### 4. Elite Team (Win %)

**Strategy Key**: `elite_team_winpct`

**Concept**: Bet on top-tier teams (ranked by win percentage) that are also in good recent form.

**Logic**:
1. Calculate team standings by **win percentage**
2. Classify teams into tiers:
   - **Elite**: Top 25% by win percentage
   - **Mid**: Middle 50%
   - **Bottom**: Bottom 25%
3. Calculate recent form (average spread performance over last 5 games)
4. **Good Form**: Last 5 games average spread performance > 3 points
5. Find games where an Elite team in good form is playing
6. Bet on that Elite team with 11-point handicap

**Prediction Generation**:
```python
# Configuration
ELITE_PERCENTILE = 0.75  # Top 25%
GOOD_FORM_THRESHOLD = 3  # avg spread > 3 in last 5
ELITE_HANDICAP_POINTS = 11  # Fixed 11-point handicap

# Elite teams by WIN percentage in good form
elite_good_form = teams[
    (teams['tier'] == 'Elite') &
    (teams['last_5_avg_spread'] > GOOD_FORM_THRESHOLD)
]
```

**Win Evaluation**: Elite team must cover the spread with 11-point handicap.
```
If Elite team is HOME:
  WIN if: spread_result_difference > 11

If Elite team is AWAY:
  WIN if: spread_result_difference < -11
```

**Handicap**: 11 points

**Data Provided**:
- Elite team's win percentage
- Elite team's point differential
- Elite team's last 5 game spread average
- Opponent's tier and win percentage

---

### 5. Elite Team (Coverage %)

**Strategy Key**: `elite_team_coverage`

**Concept**: Bet on top-tier teams (ranked by spread coverage percentage) that are also in good recent form.

**Logic**:
1. Calculate team standings by **spread coverage percentage** (not win percentage)
2. Classify teams into tiers:
   - **Elite**: Top 25% by spread coverage %
   - **Mid**: Middle 50%
   - **Bottom**: Bottom 25%
3. Calculate recent form (average spread performance over last 5 games)
4. **Good Form**: Last 5 games average spread performance > 3 points
5. Find games where an Elite team in good form is playing
6. Bet on that Elite team with 11-point handicap

**Prediction Generation**:
```python
# Configuration
ELITE_PERCENTILE = 0.75  # Top 25%
GOOD_FORM_THRESHOLD = 3  # avg spread > 3 in last 5
ELITE_HANDICAP_POINTS = 11  # Fixed 11-point handicap

# Elite teams by SPREAD COVERAGE percentage in good form
elite_good_form = teams[
    (teams['coverage_tier'] == 'Elite') &
    (teams['last_5_avg_spread'] > GOOD_FORM_THRESHOLD)
]
```

**Win Evaluation**: Elite team must cover the spread with 11-point handicap.
```
If Elite team is HOME:
  WIN if: spread_result_difference > 11

If Elite team is AWAY:
  WIN if: spread_result_difference < -11
```

**Handicap**: 11 points

**Data Provided**:
- Elite team's spread coverage percentage
- Elite team's point differential
- Elite team's last 5 game spread average
- Opponent's tier and coverage percentage

**Key Difference from Win % Variant**: Teams that cover spreads consistently may not always have the best win records (underdogs can cover without winning). This variant focuses on ATS (against the spread) performance.

---

### 6. Hot vs Cold (3 Games)

**Strategy Key**: `hot_vs_cold_3`

**Concept**: When a "hot" team (strong recent 3-game form) plays a "cold" team (no recent coverage), bet on the hot team with an 11-point handicap cushion.

**Logic**:
1. Calculate last **3 games** spread coverage for each team
2. **Hot Team**: ≥67% coverage (covered ≥2 of last 3)
3. **Cold Team**: 0% coverage (covered 0 of last 3)
4. Find games with Hot vs Cold matchup
5. Bet on the Hot team with 11-point handicap

**Prediction Generation**:
```python
# Configuration for 3-game lookback
HOT_COLD_CONFIG = {
    3: {'hot_min': 2, 'cold_max': 0}  # Hot: ≥2/3, Cold: 0/3
}

# Hot vs Cold matchup (3 games)
if home_form['covers_last_3'] >= 2 and away_form['covers_last_3'] == 0:
    bet_on = home_team
elif away_form['covers_last_3'] >= 2 and home_form['covers_last_3'] == 0:
    bet_on = away_team
```

**Win Evaluation**: Hot team must cover with 11-point handicap.
```
If Hot team is HOME:
  WIN if: spread_result_difference > 11

If Hot team is AWAY:
  WIN if: spread_result_difference < -11
```

**Handicap**: 11 points

**Streak Tracking**: Each prediction includes the current win/loss streak for both teams' spread coverage.

---

### 7. Hot vs Cold (5 Games)

**Strategy Key**: `hot_vs_cold_5`

**Concept**: When a "hot" team (strong recent 5-game form) plays a "cold" team (weak recent form), bet on the hot team with an 11-point handicap cushion.

**Logic**:
1. Calculate last **5 games** spread coverage for each team
2. **Hot Team**: ≥60% coverage (covered ≥3 of last 5)
3. **Cold Team**: ≤40% coverage (covered ≤2 of last 5)
4. Find games with Hot vs Cold matchup
5. Bet on the Hot team with 11-point handicap

**Prediction Generation**:
```python
# Configuration for 5-game lookback
HOT_COLD_CONFIG = {
    5: {'hot_min': 3, 'cold_max': 2}  # Hot: ≥3/5, Cold: ≤2/5
}

# Hot vs Cold matchup (5 games)
if home_form['covers_last_5'] >= 3 and away_form['covers_last_5'] <= 2:
    bet_on = home_team
elif away_form['covers_last_5'] >= 3 and home_form['covers_last_5'] <= 2:
    bet_on = away_team
```

**Win Evaluation**: Hot team must cover with 11-point handicap.
```
If Hot team is HOME:
  WIN if: spread_result_difference > 11

If Hot team is AWAY:
  WIN if: spread_result_difference < -11
```

**Handicap**: 11 points

**Streak Tracking**: Each prediction includes the current win/loss streak for both teams' spread coverage.

---

### 8. Hot vs Cold (7 Games)

**Strategy Key**: `hot_vs_cold_7`

**Concept**: When a "hot" team (strong recent 7-game form) plays a "cold" team (weak recent form), bet on the hot team with an 11-point handicap cushion.

**Logic**:
1. Calculate last **7 games** spread coverage for each team
2. **Hot Team**: ≥57% coverage (covered ≥4 of last 7)
3. **Cold Team**: ≤43% coverage (covered ≤3 of last 7)
4. Find games with Hot vs Cold matchup
5. Bet on the Hot team with 11-point handicap

**Prediction Generation**:
```python
# Configuration for 7-game lookback
HOT_COLD_CONFIG = {
    7: {'hot_min': 4, 'cold_max': 3}  # Hot: ≥4/7, Cold: ≤3/7
}

# Hot vs Cold matchup (7 games)
if home_form['covers_last_7'] >= 4 and away_form['covers_last_7'] <= 3:
    bet_on = home_team
elif away_form['covers_last_7'] >= 4 and home_form['covers_last_7'] <= 3:
    bet_on = away_team
```

**Win Evaluation**: Hot team must cover with 11-point handicap.
```
If Hot team is HOME:
  WIN if: spread_result_difference > 11

If Hot team is AWAY:
  WIN if: spread_result_difference < -11
```

**Handicap**: 11 points

**Streak Tracking**: Each prediction includes the current win/loss streak for both teams' spread coverage.

**Rationale for Hot vs Cold Variants**: Different lookback windows capture different momentum signals:
- **3 games**: Most responsive to recent changes, but smaller sample size
- **5 games**: Balanced sample size, standard lookback
- **7 games**: More stable trends, less noise

The 11-point handicap provides a buffer because:
1. Hot teams may be overvalued by the market
2. Cold teams may be undervalued
3. The large handicap filters for only the most convincing opportunities

---

### 9. Opponent Perfect Form (Regression)

**Concept**: Bet against teams with perfect 5/5 spread coverage, expecting regression to the mean.

**Logic**:
1. Calculate last 5 games spread coverage for each team
2. **Perfect Form**: 100% coverage (5/5 games covered)
3. Find games where one team has perfect form
4. Bet on the **opponent** of the perfect form team (with 11-point handicap)

**Prediction Generation**:
```python
# Perfect form = 5/5 covered
is_perfect = (coverage_pct == 1.0) and (games_count >= 5)

# Bet AGAINST the perfect form team
if home_form['is_perfect']:
    bet_on = away_team  # Opponent of perfect home team
elif away_form['is_perfect']:
    bet_on = home_team  # Opponent of perfect away team
```

**Win Evaluation**: The opponent must cover with 11-point handicap.
```
If Perfect Form team is HOME (we bet AWAY):
  WIN if: spread_result_difference < -11
  (Away team covers by >11 points OR home team fails to cover by 11+)

If Perfect Form team is AWAY (we bet HOME):
  WIN if: spread_result_difference > 11
  (Home team covers by >11 points)
```

**Handicap**: 11 points

**Rationale**:
- Teams rarely sustain 100% spread coverage
- After a perfect streak, regression to mean is expected
- The 11-point handicap accounts for the team still being good, just not perfect

---

### 10. Common Opponent (NCAAM Only)

**Concept**: Compare how two teams performed against shared opponents to project the matchup.

**Logic**:
1. Find all opponents both teams have played
2. For each common opponent, compare:
   - Win percentage vs that opponent
   - Spread coverage vs that opponent
   - Average margin of victory vs that opponent
3. Adjust for home/away situations (5-point home advantage)
4. Project winner and margin based on aggregate performance

**Prediction Generation**:
```python
# Find common opponents
common = opponents_a.intersection(opponents_b)

# For each common opponent, compare performance
for opponent in common:
    team_a_vs_opp = get_performance(team_a, opponent)
    team_b_vs_opp = get_performance(team_b, opponent)

    # Calculate differentials
    win_pct_diff = team_a_win_pct - team_b_win_pct
    cover_pct_diff = team_a_cover_pct - team_b_cover_pct
    margin_diff = team_a_avg_margin - team_b_avg_margin

    # Adjust margins for home/away context
    adjusted_margin = margin + (home_adjustment * 5)
```

**Win Evaluation**: Projected winner must cover the standard spread.

**Handicap**: 0 points

**Confidence Levels**:
- **High**: 5+ common opponents
- **Medium**: 3-4 common opponents
- **Low**: 1-2 common opponents

**Note**: Only available for NCAAM due to the large number of teams and interconnected schedules.

---

## Win/Loss Calculation

### The Core Formula

The `calculate_bet_result` function in `evaluate_strategy_results/lambda_function.py`:

```python
def calculate_bet_result(bet_on_position, handicap_points, game_result):
    spread_result_diff = game_result['spread_result_difference']

    if bet_on_position == 'home':
        # Home team bet: need spread_result_diff > handicap
        adjusted_result = spread_result_diff - handicap_points
        if adjusted_result > 0:
            return 'win'
        elif adjusted_result < 0:
            return 'loss'
        else:
            return 'push'
    else:  # away
        # Away team bet: need spread_result_diff < -handicap
        adjusted_result = spread_result_diff + handicap_points
        if adjusted_result < 0:
            return 'win'
        elif adjusted_result > 0:
            return 'loss'
        else:
            return 'push'
```

### Examples

**Example 1: Home Focus (NBA, 9-point handicap)**
- Bet: Lakers (home) to cover
- Final: Lakers 110, Celtics 100
- Closing spread: -5.5 (Lakers favored)
- `spread_result_diff = (110 - 100) + (-5.5) = 4.5`
- `adjusted = 4.5 - 9 = -4.5`
- **Result: LOSS** (Lakers covered the spread but not by 9+ points)

**Example 2: Hot vs Cold (11-point handicap)**
- Bet: Hot team (away) to cover
- Final: Home 95, Away 105
- Closing spread: +3.5 (Away was underdog)
- `spread_result_diff = (95 - 105) + (3.5) = -6.5`
- `adjusted = -6.5 + 11 = 4.5`
- **Result: LOSS** (Away covered but not by 11+ points adjusted)

**Example 3: Coverage Based (0-point handicap)**
- Bet: Better coverage team (home) to cover
- Final: Home 28, Away 21
- Closing spread: -6.5 (Home favored)
- `spread_result_diff = (28 - 21) + (-6.5) = 0.5`
- **Result: WIN** (Home covered by 0.5 points)

---

## Performance Tracking

### Daily Flow

```
6:30 AM EST: generate_predictions Lambda runs
             -> Creates predictions/{sport}_{date}.json

3:00 AM EST: evaluate_strategy_results Lambda runs (next day)
             -> Reads yesterday's predictions
             -> Matches to actual game results
             -> Calculates wins/losses
             -> Updates strategy_tracking/performance/{sport}_strategy_performance.json
```

### Performance Data Structure

```json
{
  "sport": "nba",
  "last_updated": "2025-01-06T08:00:00Z",
  "strategies": {
    "home_focus": {
      "total_predictions": 145,
      "total_wins": 82,
      "total_losses": 63,
      "total_pushes": 0,
      "current_win_rate": 0.566,
      "current_streak": 3,
      "streak_type": "win",
      "daily_cumulative": [
        {
          "date": "2024-10-22",
          "day_predictions": 2,
          "day_wins": 1,
          "cumulative_wins": 1,
          "cumulative_total": 2,
          "cumulative_rate": 0.50
        }
      ]
    }
  }
}
```

### ROI Calculation

Assuming standard -110 odds (bet $110 to win $100):

```python
units_won = (wins * 1.0) - (losses * 1.1)
roi = (units_won / total_bets) * 100
```

---

## Adding New Strategies

### Checklist for New Strategies

1. **Define the Strategy**
   - What edge does it exploit?
   - What data does it need?
   - What handicap should be applied?

2. **Update generate_predictions Lambda**
   - Add generation logic in `generate_predictions_for_sport()`
   - Add to the `strategies` dictionary in the response
   - Include summary statistics

3. **Update evaluate_strategy_results Lambda**
   - Add strategy config to `STRATEGY_CONFIG` dict:
     ```python
     STRATEGY_CONFIG = {
         'new_strategy': {
             'handicap': 0,  # or specific value, or 'variable'
             'bet_on_field': 'field_name'  # field containing bet target
         }
     }
     ```

4. **Update Frontend Constants** (`frontend/src/constants/api.js`)
   - Add to `STRATEGIES` enum
   - Add to `STRATEGY_INFO` with name, description, icon

5. **Update Documentation** (this file)
   - Add strategy section with full explanation
   - Include examples

6. **Run Backfill** (if applicable)
   - If strategy uses historical data, run backfill script to populate past performance

### Strategy Configuration Reference

| Strategy | Handicap | bet_on_field | Notes |
|----------|----------|--------------|-------|
| `home_focus` | 11 | None | Implicitly bet on home |
| `away_focus` | 11 | None | Implicitly bet on away |
| `coverage_based` | 0 | `better_team` | 'home' or 'away' |
| `elite_team_winpct` | 11 | `elite_team` | Team name (ranked by Win %) |
| `elite_team_coverage` | 11 | `elite_team` | Team name (ranked by Coverage %) |
| `hot_vs_cold_3` | 11 | `bet_on` | Team name (3-game lookback) |
| `hot_vs_cold_5` | 11 | `bet_on` | Team name (5-game lookback) |
| `hot_vs_cold_7` | 11 | `bet_on` | Team name (7-game lookback) |
| `opponent_perfect_form` | 11 | `bet_on` | Team name |
| `common_opponent` | 0 | None | Uses projection winner |

---

## Quick Reference

| Strategy | Handicap | Trigger Condition | Bet On |
|----------|----------|-------------------|--------|
| Home Focus | 11 | Home handicap% > Away handicap% | Home |
| Away Focus | 11 | Away handicap% > Home handicap% | Away |
| Standard Spread Coverage | 0 | Coverage difference ≥ 10% | Better team |
| Elite Team (Win %) | 11 | Top 25% by win% + last 5 avg > 3 | Elite team |
| Elite Team (Coverage %) | 11 | Top 25% by coverage% + last 5 avg > 3 | Elite team |
| Hot vs Cold (3 Games) | 11 | Hot (≥2/3) vs Cold (0/3) | Hot team |
| Hot vs Cold (5 Games) | 11 | Hot (≥3/5) vs Cold (≤2/5) | Hot team |
| Hot vs Cold (7 Games) | 11 | Hot (≥4/7) vs Cold (≤3/7) | Hot team |
| Regression | 11 | Opponent has 5/5 perfect | Opponent |
| Common Opponent | 0 | Shared opponents exist (NCAAM) | Projected winner |

---

## Files Reference

| File | Purpose |
|------|---------|
| `backend/lambda_functions/generate_predictions/lambda_function.py` | Strategy prediction generation |
| `backend/lambda_functions/evaluate_strategy_results/lambda_function.py` | Win/loss evaluation |
| `backend/lambda_functions/predictions_api/lambda_function.py` | API endpoints |
| `backend/scripts/backfill_strategy_results.py` | Historical data backfill |
| `frontend/src/constants/api.js` | Frontend strategy definitions |
| `frontend/src/screens/StrategyPerformanceScreen.js` | Performance visualization |

---

*Last Updated: January 2026*
