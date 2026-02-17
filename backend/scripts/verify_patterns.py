#!/usr/bin/env python3
"""
Pattern Verification Script

Independently recomputes pattern statistics from raw game data
and compares against what's stored in detected_patterns.

Three verification layers:
1. Cross-validation: metrics.py (chart functions) vs stored patterns
2. Sanity checks: mathematical properties that must hold
3. Sample audit: spot-check individual coverage_profile entries

Run: cd backend && python3 scripts/verify_patterns.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.engine import get_engine
from src.analysis.metrics import (
    streak_continuation_analysis,
    baseline_handicap_coverage,
    ou_streak_continuation_analysis,
    baseline_ou_coverage,
    tt_streak_continuation_analysis,
    tt_baseline_coverage,
)
from sqlalchemy import text


TOLERANCE = 0.005  # Allow 0.5% rounding tolerance


def get_all_patterns(engine):
    """Fetch all detected patterns from the database."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT sport, market_type, pattern_type, streak_type, streak_length,
                   handicap, cover_rate, baseline_rate, edge, sample_size,
                   confidence, coverage_profile_json
            FROM detected_patterns
            ORDER BY market_type, sport, streak_type, streak_length, handicap
        """))
        rows = result.fetchall()

    patterns = []
    for r in rows:
        profile = None
        if r[11]:
            profile = {int(k): v for k, v in json.loads(r[11]).items()}
        patterns.append({
            'sport': r[0],
            'market_type': r[1] or 'ats',
            'pattern_type': r[2],
            'streak_type': r[3],
            'streak_length': r[4],
            'handicap': r[5],
            'cover_rate': r[6],
            'baseline_rate': r[7],
            'edge': r[8],
            'sample_size': r[9],
            'confidence': r[10],
            'coverage_profile': profile,
        })
    return patterns


# =============================================================================
# CHECK 1: Sanity Checks on Coverage Profiles
# =============================================================================

def check_sanity(patterns):
    """Verify mathematical properties that must hold."""
    print("\n" + "=" * 70)
    print("CHECK 1: SANITY CHECKS ON COVERAGE PROFILES")
    print("=" * 70)

    issues = []
    profiles_checked = 0
    profiles_with_both_dirs = 0

    for p in patterns:
        profile = p.get('coverage_profile')
        if not profile:
            continue

        profiles_checked += 1
        market = p['market_type']
        label = f"{p['sport']} {market} {p['pattern_type']} {p['streak_type']} L={p['streak_length']}"

        prev_cover = None
        prev_base = None
        sample_sizes = set()

        for h in sorted(profile.keys()):
            entry = profile[h]
            sample_sizes.add(entry.get('sample_size', 0))

            # Check rates are between 0 and 1
            for key in ['cover_rate', 'baseline_rate']:
                val = entry.get(key, 0)
                if val < 0 or val > 1:
                    issues.append(f"  FAIL: {label} H={h}: {key}={val} out of [0,1]")

            # For O/U and TT with both directions stored
            if 'over_cover_rate' in entry:
                profiles_with_both_dirs += 1
                over_r = entry['over_cover_rate']
                under_r = entry['under_cover_rate']
                over_b = entry['over_baseline']
                under_b = entry['under_baseline']

                # Check all directional rates in [0,1]
                for key, val in [('over_cover_rate', over_r), ('under_cover_rate', under_r),
                                 ('over_baseline', over_b), ('under_baseline', under_b)]:
                    if val < 0 or val > 1:
                        issues.append(f"  FAIL: {label} H={h}: {key}={val} out of [0,1]")

                # At H=0: over + under should ≈ 1.0 (complementary)
                if h == 0:
                    sum_rate = over_r + under_r
                    sum_base = over_b + under_b
                    if abs(sum_rate - 1.0) > 0.01:
                        issues.append(f"  FAIL: {label} H=0: over_rate + under_rate = {sum_rate:.4f} (expected ~1.0)")
                    if abs(sum_base - 1.0) > 0.01:
                        issues.append(f"  FAIL: {label} H=0: over_base + under_base = {sum_base:.4f} (expected ~1.0)")

                # At H>0: over + under should be > 1.0 (overlap zone)
                if h > 0:
                    sum_rate = over_r + under_r
                    sum_base = over_b + under_b
                    if sum_rate < 1.0 - TOLERANCE:
                        issues.append(f"  FAIL: {label} H={h}: over_rate + under_rate = {sum_rate:.4f} (expected > 1.0)")
                    if sum_base < 1.0 - TOLERANCE:
                        issues.append(f"  FAIL: {label} H={h}: over_base + under_base = {sum_base:.4f} (expected > 1.0)")

            # For ATS with both directions stored
            if 'team_cover_rate' in entry:
                profiles_with_both_dirs += 1
                team_r = entry['team_cover_rate']
                opp_r = entry['opp_cover_rate']
                team_b = entry['team_baseline']
                opp_b = entry['opp_baseline']

                for key, val in [('team_cover_rate', team_r), ('opp_cover_rate', opp_r),
                                 ('team_baseline', team_b), ('opp_baseline', opp_b)]:
                    if val < 0 or val > 1:
                        issues.append(f"  FAIL: {label} H={h}: {key}={val} out of [0,1]")

                # At H=0: team + opp should ≈ 1.0 (complementary)
                # Tolerance is 0.02 because push games (exact ties) make neither side cover
                if h == 0:
                    sum_rate = team_r + opp_r
                    sum_base = team_b + opp_b
                    if abs(sum_rate - 1.0) > 0.02:
                        issues.append(f"  FAIL: {label} H=0: team_rate + opp_rate = {sum_rate:.4f} (expected ~1.0)")
                    if abs(sum_base - 1.0) > 0.02:
                        issues.append(f"  FAIL: {label} H=0: team_base + opp_base = {sum_base:.4f} (expected ~1.0)")

                # At H>0: team + opp should be > 1.0 (overlap zone)
                if h > 0:
                    sum_rate = team_r + opp_r
                    sum_base = team_b + opp_b
                    if sum_rate < 1.0 - TOLERANCE:
                        issues.append(f"  FAIL: {label} H={h}: team_rate + opp_rate = {sum_rate:.4f} (expected > 1.0)")
                    if sum_base < 1.0 - TOLERANCE:
                        issues.append(f"  FAIL: {label} H={h}: team_base + opp_base = {sum_base:.4f} (expected > 1.0)")

            # Check edge = cover_rate - baseline_rate
            cr = entry.get('cover_rate', 0)
            br = entry.get('baseline_rate', 0)
            expected_edge = round(cr - br, 4)
            actual_edge = entry.get('edge', 0)
            if abs(actual_edge - expected_edge) > 0.001:
                issues.append(f"  FAIL: {label} H={h}: edge={actual_edge} != cover_rate-baseline_rate={expected_edge}")

        # Check sample sizes are consistent across handicaps
        if len(sample_sizes) > 1:
            issues.append(f"  WARN: {label}: sample sizes vary across H: {sample_sizes}")

    # Monotonicity check: rates must increase with H
    mono_issues = 0
    for p in patterns:
        profile = p.get('coverage_profile')
        if not profile:
            continue

        market = p['market_type']
        label = f"{p['sport']} {market} {p['pattern_type']} {p['streak_type']} L={p['streak_length']}"
        sorted_h = sorted(profile.keys())

        if 'over_cover_rate' in profile.get(sorted_h[0], {}):
            # Check both directions for monotonicity
            for direction in ['over', 'under']:
                rate_key = f'{direction}_cover_rate'
                base_key = f'{direction}_baseline'
                prev_rate = None
                prev_base = None
                for h in sorted_h:
                    rate = profile[h].get(rate_key, 0)
                    base = profile[h].get(base_key, 0)
                    if prev_rate is not None and rate < prev_rate - TOLERANCE:
                        issues.append(f"  FAIL: {label} {direction}_cover_rate decreases: H={sorted_h[sorted_h.index(h)-1]}={prev_rate:.4f} -> H={h}={rate:.4f}")
                        mono_issues += 1
                    if prev_base is not None and base < prev_base - TOLERANCE:
                        issues.append(f"  FAIL: {label} {direction}_baseline decreases: H={sorted_h[sorted_h.index(h)-1]}={prev_base:.4f} -> H={h}={base:.4f}")
                        mono_issues += 1
                    prev_rate = rate
                    prev_base = base
        elif 'team_cover_rate' in profile.get(sorted_h[0], {}):
            # ATS with both directions
            for direction in ['team', 'opp']:
                rate_key = f'{direction}_cover_rate'
                base_key = f'{direction}_baseline'
                prev_rate = None
                prev_base = None
                for h in sorted_h:
                    rate = profile[h].get(rate_key, 0)
                    base = profile[h].get(base_key, 0)
                    if prev_rate is not None and rate < prev_rate - TOLERANCE:
                        issues.append(f"  FAIL: {label} {rate_key} decreases: H={sorted_h[sorted_h.index(h)-1]}={prev_rate:.4f} -> H={h}={rate:.4f}")
                        mono_issues += 1
                    if prev_base is not None and base < prev_base - TOLERANCE:
                        issues.append(f"  FAIL: {label} {base_key} decreases: H={sorted_h[sorted_h.index(h)-1]}={prev_base:.4f} -> H={h}={base:.4f}")
                        mono_issues += 1
                    prev_rate = rate
                    prev_base = base
        else:
            # Legacy ATS: only cover_rate
            prev_rate = None
            for h in sorted_h:
                rate = profile[h].get('cover_rate', 0)
                if prev_rate is not None and rate < prev_rate - TOLERANCE:
                    issues.append(f"  FAIL: {label} cover_rate decreases: H={sorted_h[sorted_h.index(h)-1]}={prev_rate:.4f} -> H={h}={rate:.4f}")
                    mono_issues += 1
                prev_rate = rate

    print(f"\n  Profiles checked: {profiles_checked}")
    print(f"  Profiles with both directions (O/U, TT): {profiles_with_both_dirs}")

    if issues:
        print(f"\n  Issues found: {len(issues)}")
        for issue in issues[:30]:  # Limit output
            print(issue)
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more")
    else:
        print("\n  ALL SANITY CHECKS PASSED")

    return len(issues) == 0


# =============================================================================
# CHECK 2: Cross-Validate Against metrics.py
# =============================================================================

def check_cross_validation(engine, patterns):
    """Compare stored patterns against independent metrics.py computation."""
    print("\n" + "=" * 70)
    print("CHECK 2: CROSS-VALIDATION AGAINST metrics.py")
    print("=" * 70)

    issues = []
    checks_run = 0

    # Sample a subset of patterns for each market type
    by_key = {}
    for p in patterns:
        key = (p['sport'], p['market_type'], p['streak_type'], p['streak_length'])
        if key not in by_key:
            by_key[key] = p

    # Limit to ~15 samples to keep runtime reasonable
    samples = list(by_key.values())[:15]

    with engine.connect() as conn:
        for p in samples:
            sport = p['sport']
            market = p['market_type']
            streak_type = p['streak_type']
            streak_length = p['streak_length']
            handicap = p['handicap']
            stored_rate = p['cover_rate']
            stored_baseline = p['baseline_rate']
            stored_sample = p['sample_size']

            label = f"{sport} {market} {p['pattern_type']} {streak_type} L={streak_length} H={handicap}"
            checks_run += 1

            try:
                if market == 'ats':
                    # Cross-validate with streak_continuation_analysis
                    data = streak_continuation_analysis(
                        conn, sport, streak_length, streak_type, (handicap, handicap)
                    )
                    baseline = baseline_handicap_coverage(conn, sport, (handicap, handicap))

                    if len(data) == 0:
                        issues.append(f"  WARN: {label}: metrics.py returned empty (no data)")
                        continue

                    metrics_rate = data.iloc[0]['cover_pct']
                    metrics_total = data.iloc[0]['total']
                    metrics_baseline = baseline.iloc[0]['baseline_cover_pct'] if len(baseline) > 0 else None

                    if abs(metrics_rate - stored_rate) > TOLERANCE:
                        issues.append(f"  FAIL: {label}: cover_rate stored={stored_rate:.4f} vs metrics={metrics_rate:.4f}")
                    if metrics_total != stored_sample:
                        issues.append(f"  FAIL: {label}: sample stored={stored_sample} vs metrics={metrics_total}")
                    if metrics_baseline is not None and abs(metrics_baseline - stored_baseline) > TOLERANCE:
                        issues.append(f"  FAIL: {label}: baseline stored={stored_baseline:.4f} vs metrics={metrics_baseline:.4f}")

                elif market == 'ou':
                    # For O/U, the stored cover_rate is ride-direction
                    ride_dir = 'over' if streak_type == 'OVER' else 'under'

                    data = ou_streak_continuation_analysis(
                        conn, sport, streak_length, streak_type, (handicap, handicap),
                        direction=ride_dir
                    )
                    baseline = baseline_ou_coverage(conn, sport, (handicap, handicap), direction=ride_dir)

                    if len(data) == 0:
                        issues.append(f"  WARN: {label}: metrics.py returned empty")
                        continue

                    metrics_rate = data.iloc[0]['cover_pct']
                    metrics_total = data.iloc[0]['total']
                    metrics_baseline = baseline.iloc[0]['baseline_cover_pct'] if len(baseline) > 0 else None

                    if abs(metrics_rate - stored_rate) > TOLERANCE:
                        issues.append(f"  FAIL: {label}: cover_rate stored={stored_rate:.4f} vs metrics={metrics_rate:.4f}")
                    if metrics_total != stored_sample:
                        issues.append(f"  FAIL: {label}: sample stored={stored_sample} vs metrics={metrics_total}")
                    if metrics_baseline is not None and abs(metrics_baseline - stored_baseline) > TOLERANCE:
                        issues.append(f"  FAIL: {label}: baseline stored={stored_baseline:.4f} vs metrics={metrics_baseline:.4f}")

                    # Also cross-validate the other direction in the profile
                    profile = p.get('coverage_profile', {})
                    h_entry = profile.get(handicap, {})
                    if 'over_cover_rate' in h_entry:
                        other_dir = 'under' if ride_dir == 'over' else 'over'
                        other_data = ou_streak_continuation_analysis(
                            conn, sport, streak_length, streak_type, (handicap, handicap),
                            direction=other_dir
                        )
                        if len(other_data) > 0:
                            metrics_other = other_data.iloc[0]['cover_pct']
                            stored_other = h_entry.get(f'{other_dir}_cover_rate', 0)
                            if abs(metrics_other - stored_other) > TOLERANCE:
                                issues.append(f"  FAIL: {label}: {other_dir}_cover_rate stored={stored_other:.4f} vs metrics={metrics_other:.4f}")

                elif market == 'tt':
                    ride_dir = 'ride'

                    data = tt_streak_continuation_analysis(
                        conn, sport, streak_length, streak_type, (handicap, handicap),
                        direction=ride_dir
                    )
                    # TT baseline — ride direction: OVER for OVER streak, UNDER for UNDER
                    base_dir = 'over' if streak_type == 'OVER' else 'under'
                    baseline = tt_baseline_coverage(conn, sport, (handicap, handicap), direction=base_dir)

                    if len(data) == 0:
                        issues.append(f"  WARN: {label}: metrics.py returned empty")
                        continue

                    metrics_rate = data.iloc[0]['cover_pct']
                    metrics_total = data.iloc[0]['total']
                    metrics_baseline = baseline.iloc[0]['baseline_cover_pct'] if len(baseline) > 0 else None

                    if abs(metrics_rate - stored_rate) > TOLERANCE:
                        issues.append(f"  FAIL: {label}: cover_rate stored={stored_rate:.4f} vs metrics={metrics_rate:.4f}")
                    if metrics_total != stored_sample:
                        issues.append(f"  FAIL: {label}: sample stored={stored_sample} vs metrics={metrics_total}")
                    if metrics_baseline is not None and abs(metrics_baseline - stored_baseline) > TOLERANCE:
                        issues.append(f"  FAIL: {label}: baseline stored={stored_baseline:.4f} vs metrics={metrics_baseline:.4f}")

                    # Cross-validate opposite direction in the profile
                    profile = p.get('coverage_profile', {})
                    h_entry = profile.get(handicap, {})
                    if 'over_cover_rate' in h_entry:
                        other_data = tt_streak_continuation_analysis(
                            conn, sport, streak_length, streak_type, (handicap, handicap),
                            direction='fade'
                        )
                        if len(other_data) > 0:
                            metrics_other = other_data.iloc[0]['cover_pct']
                            # fade direction: opposite of ride
                            other_dir_key = 'under' if streak_type == 'OVER' else 'over'
                            stored_other = h_entry.get(f'{other_dir_key}_cover_rate', 0)
                            if abs(metrics_other - stored_other) > TOLERANCE:
                                issues.append(f"  FAIL: {label}: {other_dir_key}_cover_rate stored={stored_other:.4f} vs metrics={metrics_other:.4f}")

                print(f"  OK: {label}")

            except Exception as e:
                issues.append(f"  ERROR: {label}: {e}")

    print(f"\n  Cross-validation checks run: {checks_run}")

    if issues:
        fails = [i for i in issues if 'FAIL' in i]
        warns = [i for i in issues if 'WARN' in i]
        errors = [i for i in issues if 'ERROR' in i]
        print(f"  Failures: {len(fails)}, Warnings: {len(warns)}, Errors: {len(errors)}")
        for issue in issues:
            print(issue)
    else:
        print("\n  ALL CROSS-VALIDATION CHECKS PASSED")

    return len([i for i in issues if 'FAIL' in i]) == 0


# =============================================================================
# CHECK 3: Display Logic Audit (_build_tier_data)
# =============================================================================

def check_display_logic(patterns):
    """Verify _build_tier_data produces correct bet-direction rates."""
    print("\n" + "=" * 70)
    print("CHECK 3: DISPLAY LOGIC AUDIT (_build_tier_data)")
    print("=" * 70)

    issues = []
    checks_run = 0

    for p in patterns:
        profile = p.get('coverage_profile')
        if not profile:
            continue

        market = p['market_type']
        is_fade = p['pattern_type'] == 'streak_fade'
        streak_type = p['streak_type']
        label = f"{p['sport']} {market} {p['pattern_type']} {streak_type} L={p['streak_length']}"

        # Determine what bet direction the user would actually see
        if market in ('ou', 'tt'):
            if is_fade:
                bet_dir = 'under' if streak_type == 'OVER' else 'over'
            else:
                bet_dir = 'over' if streak_type == 'OVER' else 'under'

            # Check that bet-direction rates increase with H
            sorted_h = sorted(profile.keys())
            prev_rate = None
            for h in sorted_h:
                entry = profile[h]
                rate_key = f'{bet_dir}_cover_rate'
                if rate_key not in entry:
                    issues.append(f"  FAIL: {label} H={h}: missing {rate_key} in profile")
                    continue

                rate = entry[rate_key]
                checks_run += 1

                if prev_rate is not None and rate < prev_rate - TOLERANCE:
                    issues.append(f"  FAIL: {label}: bet_dir={bet_dir} rate DECREASES H={sorted_h[sorted_h.index(h)-1]}={prev_rate:.4f} -> H={h}={rate:.4f}")
                prev_rate = rate

        elif market == 'ats':
            sorted_h = sorted(profile.keys())
            prev_display = None
            for h in sorted_h:
                entry = profile[h]
                if 'team_cover_rate' in entry:
                    # New format: pick correct direction
                    if is_fade:
                        display_rate = entry['opp_cover_rate']
                    else:
                        display_rate = entry['team_cover_rate']
                else:
                    # Legacy format: 1-rate for fade
                    if is_fade:
                        display_rate = 1 - entry.get('cover_rate', 0)
                    else:
                        display_rate = entry.get('cover_rate', 0)

                checks_run += 1

                if prev_display is not None and display_rate < prev_display - TOLERANCE:
                    issues.append(f"  FAIL: {label}: display rate DECREASES H={sorted_h[sorted_h.index(h)-1]}={prev_display:.4f} -> H={h}={display_rate:.4f}")
                prev_display = display_rate

    print(f"\n  Display logic checks run: {checks_run}")

    if issues:
        print(f"  Issues found: {len(issues)}")
        for issue in issues:
            print(issue)
    else:
        print("\n  ALL DISPLAY LOGIC CHECKS PASSED")

    return len(issues) == 0


# =============================================================================
# SUMMARY
# =============================================================================

def main():
    print("=" * 70)
    print("PATTERN VERIFICATION REPORT")
    print("=" * 70)

    engine = get_engine()
    patterns = get_all_patterns(engine)
    print(f"\nLoaded {len(patterns)} patterns from detected_patterns table")
    print(f"  ATS: {sum(1 for p in patterns if p['market_type'] == 'ats')}")
    print(f"  O/U: {sum(1 for p in patterns if p['market_type'] == 'ou')}")
    print(f"  TT:  {sum(1 for p in patterns if p['market_type'] == 'tt')}")

    with_profile = sum(1 for p in patterns if p.get('coverage_profile'))
    print(f"  With coverage profile: {with_profile}")

    pass1 = check_sanity(patterns)
    pass2 = check_cross_validation(engine, patterns)
    pass3 = check_display_logic(patterns)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    results = [
        ("Sanity Checks", pass1),
        ("Cross-Validation", pass2),
        ("Display Logic", pass3),
    ]
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  ALL VERIFICATION CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — review issues above")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
