"""
Convergent Matchup Analysis — Visualizations
Generates baseline coverage rates + convergent streak edge charts for NBA, NFL, NCAAM.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv('/Users/robertbatchelor/Projects/sports-betting/backend/.env')
sys.path.insert(0, '/Users/robertbatchelor/Projects/sports-betting/backend')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from src.db import AnalyticsRepository

OUT = '/Users/robertbatchelor/Projects/sports-betting/backend/notebooks/visuals'
repo = AnalyticsRepository()

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'over': '#2ecc71',
    'under': '#e74c3c',
    'ride': '#3498db',
    'fade': '#e67e22',
    'neutral': '#95a5a6',
}


def compute_all(sport):
    """Compute baselines and convergent matchup data for a sport."""
    df = repo.get_games(sport=sport)
    tt_df = df.dropna(subset=['home_team_total_result', 'away_team_total_result']).sort_values('game_date').reset_index(drop=True)

    # --- Baselines ---
    all_tt = np.concatenate([tt_df['home_team_total_result'].values, tt_df['away_team_total_result'].values])
    all_gt = tt_df.dropna(subset=['total_result'])['total_result'].values

    tt_bl_over = {h: (all_tt > -h).mean() * 100 for h in range(11)}
    tt_bl_under = {h: (all_tt < h).mean() * 100 for h in range(11)}
    gt_bl_over = {h: (all_gt > -h).mean() * 100 for h in range(21)}
    gt_bl_under = {h: (all_gt < h).mean() * 100 for h in range(21)}

    # --- Build per-team streak history ---
    records = []
    for _, row in tt_df.iterrows():
        gd = row['game_date']
        for side, team_col, margin_col in [('home', 'home_team', 'home_team_total_result'),
                                            ('away', 'away_team', 'away_team_total_result')]:
            m = row[margin_col]
            ou = 'OVER' if m > 0 else ('UNDER' if m < 0 else 'PUSH')
            records.append({
                'game_date': gd, 'team': row[team_col], 'tt_ou': ou, 'tt_margin': m,
                'home_team': row['home_team'], 'away_team': row['away_team'],
                'gt_margin': row.get('total_result', np.nan), 'side': side
            })

    rec_df = pd.DataFrame(records).sort_values(['team', 'game_date']).reset_index(drop=True)

    team_streaks = {}
    for team, grp in rec_df.groupby('team'):
        history = []
        for _, r in grp.iterrows():
            streak_len, streak_dir = 0, None
            if len(history) > 0 and history[-1] in ('OVER', 'UNDER'):
                streak_dir = history[-1]
                for past in reversed(history):
                    if past == streak_dir:
                        streak_len += 1
                    else:
                        break
            team_streaks[(team, str(r['game_date']), r['home_team'], r['away_team'])] = (streak_dir, streak_len)
            history.append(r['tt_ou'])

    # --- Pair games ---
    pairs = []
    for _, row in tt_df.iterrows():
        gd = str(row['game_date'])
        ht, at = row['home_team'], row['away_team']
        h_info = team_streaks.get((ht, gd, ht, at), (None, 0))
        a_info = team_streaks.get((at, gd, ht, at), (None, 0))
        pairs.append({
            'h_dir': h_info[0], 'h_len': h_info[1],
            'a_dir': a_info[0], 'a_len': a_info[1],
            'h_tt_margin': row['home_team_total_result'],
            'a_tt_margin': row['away_team_total_result'],
            'gt_margin': row.get('total_result', np.nan)
        })

    pairs_df = pd.DataFrame(pairs)

    return {
        'n_games': len(tt_df), 'n_tt': len(all_tt), 'n_gt': len(all_gt),
        'tt_bl_over': tt_bl_over, 'tt_bl_under': tt_bl_under,
        'gt_bl_over': gt_bl_over, 'gt_bl_under': gt_bl_under,
        'pairs_df': pairs_df,
    }


def get_convergent_edges(data, combo_dir, min_streak):
    """Compute RIDE and FADE edges for convergent matchups."""
    pdf = data['pairs_df']
    mask = (
        (pdf['h_dir'] == combo_dir) & (pdf['h_len'] >= min_streak) &
        (pdf['a_dir'] == combo_dir) & (pdf['a_len'] >= min_streak)
    )
    sub = pdf[mask]
    if len(sub) < 5:
        return None

    tt_m = np.concatenate([sub['h_tt_margin'].values, sub['a_tt_margin'].values])
    gt_m = sub['gt_margin'].dropna().values

    tt_ride, tt_fade, gt_ride, gt_fade = {}, {}, {}, {}
    for h in range(11):
        if combo_dir == 'OVER':
            tt_ride[h] = (tt_m > -h).mean() * 100 - data['tt_bl_over'][h]
            tt_fade[h] = (tt_m < h).mean() * 100 - data['tt_bl_under'][h]
        else:
            tt_ride[h] = (tt_m < h).mean() * 100 - data['tt_bl_under'][h]
            tt_fade[h] = (tt_m > -h).mean() * 100 - data['tt_bl_over'][h]

    for h in range(21):
        if len(gt_m) < 5:
            gt_ride[h] = np.nan
            gt_fade[h] = np.nan
        elif combo_dir == 'OVER':
            gt_ride[h] = (gt_m > -h).mean() * 100 - data['gt_bl_over'][h]
            gt_fade[h] = (gt_m < h).mean() * 100 - data['gt_bl_under'][h]
        else:
            gt_ride[h] = (gt_m < h).mean() * 100 - data['gt_bl_under'][h]
            gt_fade[h] = (gt_m > -h).mean() * 100 - data['gt_bl_over'][h]

    return {
        'n_games': len(sub), 'n_tt': len(tt_m), 'n_gt': len(gt_m),
        'tt_ride': tt_ride, 'tt_fade': tt_fade,
        'gt_ride': gt_ride, 'gt_fade': gt_fade,
    }


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Baseline Coverage Rates (all 3 sports, TT + GT)
# ═══════════════════════════════════════════════════════════════
def plot_baselines(all_data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Baseline Coverage Rates by Handicap', fontsize=18, fontweight='bold', y=0.98)

    for col, sport in enumerate(['NBA', 'NFL', 'NCAAM']):
        d = all_data[sport]

        # TT row
        ax = axes[0, col]
        hs = list(range(11))
        ax.plot(hs, [d['tt_bl_over'][h] for h in hs], color=COLORS['over'], marker='o', ms=5, lw=2, label='OVER covers')
        ax.plot(hs, [d['tt_bl_under'][h] for h in hs], color=COLORS['under'], marker='s', ms=5, lw=2, label='UNDER covers')
        ax.axhline(50, color='grey', ls='--', alpha=0.5)
        ax.set_title(f'{sport} — Team Totals', fontsize=13, fontweight='bold')
        ax.set_xlabel('Handicap (pts)')
        ax.set_ylabel('Coverage %')
        ax.set_ylim(30, 100)
        ax.set_xticks(hs)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax.legend(fontsize=9)
        ax.text(0.02, 0.02, f'n={d["n_tt"]:,} obs', transform=ax.transAxes, fontsize=8, color='grey')

        # GT row
        ax = axes[1, col]
        hs = list(range(21))
        ax.plot(hs, [d['gt_bl_over'][h] for h in hs], color=COLORS['over'], marker='o', ms=4, lw=2, label='OVER covers')
        ax.plot(hs, [d['gt_bl_under'][h] for h in hs], color=COLORS['under'], marker='s', ms=4, lw=2, label='UNDER covers')
        ax.axhline(50, color='grey', ls='--', alpha=0.5)
        ax.set_title(f'{sport} — Game Totals', fontsize=13, fontweight='bold')
        ax.set_xlabel('Handicap (pts)')
        ax.set_ylabel('Coverage %')
        ax.set_ylim(30, 100)
        ax.set_xticks(range(0, 21, 2))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax.legend(fontsize=9)
        ax.text(0.02, 0.02, f'n={d["n_gt"]:,} obs', transform=ax.transAxes, fontsize=8, color='grey')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/01_baseline_coverage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved 01_baseline_coverage.png')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Convergent Edge Charts — one per sport
# Shows RIDE vs FADE edge (pp above/below baseline) for each
# convergent scenario, at every handicap level.
# ═══════════════════════════════════════════════════════════════
def plot_convergent_edges(all_data):
    for sport in ['NBA', 'NFL', 'NCAAM']:
        d = all_data[sport]

        # Define scenarios to plot
        scenarios = []
        for combo_dir in ['OVER', 'UNDER']:
            for min_s in [1, 2, 3]:
                edges = get_convergent_edges(d, combo_dir, min_s)
                if edges is not None:
                    scenarios.append((combo_dir, min_s, edges))

        if not scenarios:
            continue

        n_rows = len(scenarios)
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'{sport} — Convergent Matchup Edges vs Baseline', fontsize=16, fontweight='bold', y=1.01)

        for i, (combo_dir, min_s, edges) in enumerate(scenarios):
            ride_dir = combo_dir
            fade_dir = 'UNDER' if combo_dir == 'OVER' else 'OVER'

            # TT edge chart
            ax = axes[i, 0]
            hs = list(range(11))
            ride_vals = [edges['tt_ride'][h] for h in hs]
            fade_vals = [edges['tt_fade'][h] for h in hs]
            ax.bar([h - 0.15 for h in hs], ride_vals, 0.3, color=COLORS['ride'], alpha=0.85, label=f'RIDE (bet {ride_dir})')
            ax.bar([h + 0.15 for h in hs], fade_vals, 0.3, color=COLORS['fade'], alpha=0.85, label=f'FADE (bet {fade_dir})')
            ax.axhline(0, color='black', lw=1)
            ax.set_title(f'Both {combo_dir} {min_s}+ — Team Totals  (n={edges["n_tt"]} obs)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Handicap (pts)')
            ax.set_ylabel('Edge vs Baseline (pp)')
            ax.set_xticks(hs)
            ax.set_ylim(-max(abs(min(ride_vals + fade_vals)) + 3, abs(max(ride_vals + fade_vals)) + 3),
                         max(abs(min(ride_vals + fade_vals)) + 3, abs(max(ride_vals + fade_vals)) + 3))
            ax.legend(fontsize=9)
            # Shade positive area
            ax.axhspan(0, ax.get_ylim()[1], alpha=0.03, color='green')
            ax.axhspan(ax.get_ylim()[0], 0, alpha=0.03, color='red')

            # GT edge chart
            ax = axes[i, 1]
            hs = list(range(21))
            ride_vals = [edges['gt_ride'].get(h, 0) for h in hs]
            fade_vals = [edges['gt_fade'].get(h, 0) for h in hs]
            if any(np.isnan(v) for v in ride_vals):
                ax.text(0.5, 0.5, f'Insufficient GT data (n={edges["n_gt"]})',
                        transform=ax.transAxes, ha='center', fontsize=11, color='grey')
                ax.set_title(f'Both {combo_dir} {min_s}+ — Game Totals', fontsize=11, fontweight='bold')
                continue

            ax.bar([h - 0.3 for h in hs], ride_vals, 0.6, color=COLORS['ride'], alpha=0.85, label=f'RIDE (bet {ride_dir})')
            ax.bar([h + 0.3 for h in hs], fade_vals, 0.6, color=COLORS['fade'], alpha=0.85, label=f'FADE (bet {fade_dir})')
            ax.axhline(0, color='black', lw=1)
            ax.set_title(f'Both {combo_dir} {min_s}+ — Game Totals  (n={edges["n_gt"]} obs)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Handicap (pts)')
            ax.set_ylabel('Edge vs Baseline (pp)')
            ax.set_xticks(range(0, 21, 2))
            y_abs = max(abs(min(ride_vals + fade_vals)) + 3, abs(max(ride_vals + fade_vals)) + 3)
            ax.set_ylim(-y_abs, y_abs)
            ax.legend(fontsize=9)
            ax.axhspan(0, ax.get_ylim()[1], alpha=0.03, color='green')
            ax.axhspan(ax.get_ylim()[0], 0, alpha=0.03, color='red')

        fig.tight_layout()
        fig.savefig(f'{OUT}/02_convergent_edges_{sport.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved 02_convergent_edges_{sport.lower()}.png')


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Summary heatmap — best edges per scenario/sport
# ═══════════════════════════════════════════════════════════════
def plot_summary(all_data):
    """A compact summary showing the best edge direction and magnitude for each scenario."""
    scenarios = []
    for combo_dir in ['OVER', 'UNDER']:
        for min_s in [1, 2, 3]:
            scenarios.append((combo_dir, min_s))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Convergent Matchup — Best Edge at H=0 by Scenario', fontsize=16, fontweight='bold', y=1.02)

    for col, sport in enumerate(['NBA', 'NFL', 'NCAAM']):
        ax = axes[col]
        d = all_data[sport]

        labels, ride_edges, fade_edges, ns = [], [], [], []
        for combo_dir, min_s in scenarios:
            edges = get_convergent_edges(d, combo_dir, min_s)
            if edges is None:
                labels.append(f'Both {combo_dir}\n{min_s}+')
                ride_edges.append(0)
                fade_edges.append(0)
                ns.append(0)
            else:
                labels.append(f'Both {combo_dir}\n{min_s}+')
                ride_edges.append(edges['tt_ride'][0])
                fade_edges.append(edges['tt_fade'][0])
                ns.append(edges['n_games'])

        x = np.arange(len(labels))
        w = 0.35
        bars_r = ax.bar(x - w/2, ride_edges, w, color=COLORS['ride'], alpha=0.85, label='RIDE edge')
        bars_f = ax.bar(x + w/2, fade_edges, w, color=COLORS['fade'], alpha=0.85, label='FADE edge')

        # Annotate with sample sizes
        for j, n in enumerate(ns):
            if n > 0:
                ax.text(j, ax.get_ylim()[0] + 0.5, f'n={n}', ha='center', fontsize=8, color='grey')

        ax.axhline(0, color='black', lw=1)
        ax.set_title(f'{sport}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Edge vs Baseline at H=0 (pp)')
        ax.legend(fontsize=9)
        y_max = max(abs(min(ride_edges + fade_edges)) + 5, abs(max(ride_edges + fade_edges)) + 5)
        ax.set_ylim(-y_max, y_max)
        ax.axhspan(0, y_max, alpha=0.03, color='green')
        ax.axhspan(-y_max, 0, alpha=0.03, color='red')

    fig.tight_layout()
    fig.savefig(f'{OUT}/03_summary_edges.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved 03_summary_edges.png')


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Loading data...')
    all_data = {}
    for sport in ['NBA', 'NFL', 'NCAAM']:
        all_data[sport] = compute_all(sport)
        print(f'  {sport}: {all_data[sport]["n_games"]} games')

    print('\nGenerating visuals...')
    plot_baselines(all_data)
    plot_convergent_edges(all_data)
    plot_summary(all_data)
    print('\nDone! Visuals saved to:', OUT)
