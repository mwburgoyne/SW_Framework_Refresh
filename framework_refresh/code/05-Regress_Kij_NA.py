#!/usr/bin/env python3
"""
kij_NA Regression for All S&W Gases
=====================================

For each gas: load pointwise kij_NA data, evaluate S&W original kij_NA against
y_H2O experimental data, find optimal constant kij_NA by grid search on MARE(y_H2O),
and determine whether S&W values need updating.

Optimization target: MARE on y_H2O (vapor water content) — the physical observable.

Special handling:
  - H2S: also evaluates S&W Eq 17 (T-dependent) and proposed inverse-quadratic
  - CO2: also evaluates Yan et al. 2011 value (0.18756)
  - H2: uses Paper 1 methodology exactly (P >= 50 bar, T = 50-150°C)
  - nC4H10: only 1 data point — report only, no optimization

Output: data/kij_na_all_gases_report.txt

Usage:
    cd framework_refresh/code
    python regress_kij_na.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
from datetime import datetime

from _lib_vle_engine import SWBinaryVLE, COMPONENTS, KIJ_NA, kij_na_h2s_sw_eq17, get_kij_na

# =============================================================================
# Configuration
# =============================================================================

GASES = ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8', 'nC4H10']

# S&W reference values (constant kij_NA for all except H2S)
SW_KIJ_NA = {
    'CO2': 0.1896,
    'H2S': 'T-dep',  # S&W Eq 17: 0.19031 - 0.05965*Tr
    'CH4': 0.4850,
    'N2': 0.4778,
    'H2': 0.468,
    'C2H6': 0.4920,
    'C3H8': 0.5070,
    'nC4H10': 0.5080,
}

# Grid search ranges (gas-specific)
SEARCH_RANGES = {
    'CO2': (0.0, 0.5),
    'H2S': (-0.3, 0.5),
    'CH4': (0.2, 0.7),
    'N2': (0.2, 0.7),
    'H2': (0.30, 0.70),   # Matches Paper 1 exactly
    'C2H6': (0.2, 0.7),
    'C3H8': (0.3, 0.7),
    'nC4H10': None,        # Skip (1 point)
}

# Default P filter (bar)
DEFAULT_P_MIN = 20

# Per-gas P filter overrides
P_MIN_OVERRIDE = {
    'C3H8': 10,    # All C3H8 data < 44 bar; keep at P >= 10
    'H2': 50,      # Paper 1 methodology
}

# Per-gas T filter overrides (K) — only H2 has T filter from Paper 1
T_FILTER = {
    'H2': (323.15, 423.15),  # 50-150°C (UHS-relevant)
}

# Bound-hitting thresholds — exclude kij_NA near known search bounds
BOUND_THRESHOLDS = [-1.0, -0.5, 1.0, 1.5]
BOUND_TOLERANCE = 0.001

# P threshold for sensitivity analysis
P_SENSITIVITY = 50  # Also report at P >= 50 bar where data permits

# H2S proposed inverse-quadratic form (from plot_kij_per_gas.py)
def kij_na_h2s_invquad(T_K):
    """Proposed inverse-quadratic kij_NA for H2S."""
    return -1.5977 + 1541.16 / T_K - 334577.0 / T_K**2


# =============================================================================
# Core functions
# =============================================================================

def is_bound_hitting(kij_na):
    """Check if a kij_NA value is at a known search bound."""
    for bound in BOUND_THRESHOLDS:
        if abs(kij_na - bound) < BOUND_TOLERANCE:
            return True
    return False


def load_and_filter(gas, csv_path='../../shared/data/pointwise_kij_results.csv'):
    """
    Load CSV and filter for a given gas.

    Returns (df_filtered, filter_stats_dict)
    """
    df = pd.read_csv(csv_path)
    df_gas = df[df['Gas'] == gas].copy()

    # Must have converged kij_NA and y_H2O_exp
    mask = (df_gas['kij_NA'].notna()) & (df_gas['kij_NA_conv'] == True) & (df_gas['y_H2O_exp'].notna())
    df_work = df_gas[mask].copy()
    n_total = len(df_work)

    stats = {'n_total': n_total}

    # Bound-hitting filter
    bound_mask = df_work['kij_NA'].apply(is_bound_hitting)
    n_bound = bound_mask.sum()
    df_work = df_work[~bound_mask]
    stats['n_bound_excluded'] = n_bound

    # P filter
    p_min = P_MIN_OVERRIDE.get(gas, DEFAULT_P_MIN)
    n_low_p = (df_work['P_bar'] < p_min).sum()
    df_work = df_work[df_work['P_bar'] >= p_min]
    stats['p_min'] = p_min
    stats['n_low_p_excluded'] = n_low_p

    # T filter (only H2)
    if gas in T_FILTER:
        t_lo, t_hi = T_FILTER[gas]
        n_t_out = ((df_work['T_K'] < t_lo) | (df_work['T_K'] > t_hi)).sum()
        df_work = df_work[(df_work['T_K'] >= t_lo) & (df_work['T_K'] <= t_hi)]
        stats['n_t_excluded'] = n_t_out
        stats['t_filter'] = (t_lo, t_hi)

    stats['n_final'] = len(df_work)
    return df_work, stats


def calc_mare_y_h2o(df, gas, kij_na_value):
    """
    Calculate MARE on y_H2O at a constant kij_NA.
    Returns (mare_pct, n_valid, list_of_per_point_errors).
    """
    vle = SWBinaryVLE(gas, salinity_molal=0.0)
    rel_errors = []
    for _, row in df.iterrows():
        try:
            y_calc = vle.calc_water_content_with_kij(row['T_K'], row['P_bar'] * 1e5, kij_na_value)
            if y_calc and y_calc > 0 and row['y_H2O_exp'] > 0:
                rel_errors.append(abs(y_calc - row['y_H2O_exp']) / row['y_H2O_exp'] * 100)
        except Exception:
            pass
    mare = np.mean(rel_errors) if rel_errors else np.nan
    return mare, len(rel_errors), rel_errors


def calc_mare_y_h2o_func(df, gas, kij_na_func):
    """
    Calculate MARE on y_H2O using a T-dependent kij_NA function.
    kij_na_func takes T_K and returns kij_NA.
    Returns (mare_pct, n_valid).
    """
    vle = SWBinaryVLE(gas, salinity_molal=0.0)
    rel_errors = []
    for _, row in df.iterrows():
        try:
            kij = kij_na_func(row['T_K'])
            y_calc = vle.calc_water_content_with_kij(row['T_K'], row['P_bar'] * 1e5, kij)
            if y_calc and y_calc > 0 and row['y_H2O_exp'] > 0:
                rel_errors.append(abs(y_calc - row['y_H2O_exp']) / row['y_H2O_exp'] * 100)
        except Exception:
            pass
    mare = np.mean(rel_errors) if rel_errors else np.nan
    return mare, len(rel_errors)


def optimize_constant(df, gas, search_range, coarse_step=0.02):
    """
    Two-stage grid search for optimal constant kij_NA minimizing MARE(y_H2O).
    Returns (optimal_kij_na, optimal_mare, coarse_results_list).
    """
    lo, hi = search_range
    vle = SWBinaryVLE(gas, salinity_molal=0.0)

    # Coarse search
    coarse_results = []
    best_kij = (lo + hi) / 2
    best_mare = float('inf')

    for kij in np.arange(lo, hi + coarse_step / 2, coarse_step):
        mare, n_valid, _ = calc_mare_y_h2o(df, gas, kij)
        coarse_results.append((kij, mare))
        if not np.isnan(mare) and mare < best_mare:
            best_mare = mare
            best_kij = kij

    # Fine search around coarse optimum
    fine_step = 0.001
    for kij in np.arange(best_kij - 0.02, best_kij + 0.02 + fine_step / 2, fine_step):
        if kij < lo or kij > hi:
            continue
        mare, n_valid, _ = calc_mare_y_h2o(df, gas, kij)
        if not np.isnan(mare) and mare < best_mare:
            best_mare = mare
            best_kij = kij

    return round(best_kij, 3), best_mare, coarse_results


def compute_kij_na_mae(df, value_or_func):
    """
    Secondary diagnostic: MAE of pointwise kij_NA values vs a reference.
    value_or_func: float (constant) or callable(T_K) -> float.
    """
    if callable(value_or_func):
        ref = df['T_K'].apply(value_or_func)
    else:
        ref = value_or_func
    return (df['kij_NA'] - ref).abs().mean()


# =============================================================================
# Per-gas analysis
# =============================================================================

def analyze_gas(gas, df, write):
    """Full per-gas analysis. Returns dict with summary results."""
    write(f"\n{'='*80}")
    write(f"  {gas}")
    write(f"{'='*80}")

    result = {'gas': gas}

    if len(df) == 0:
        write(f"  No converged kij_NA data with y_H2O_exp.")
        result['n'] = 0
        return result

    # --- Data summary ---
    T_K = df['T_K'].values
    T_C = T_K - 273.15
    P_bar = df['P_bar'].values
    kij_na = df['kij_NA'].values
    sources = df['Source'].unique()

    write(f"\n  Data points: {len(df)}")
    write(f"  T range: {T_K.min():.1f} - {T_K.max():.1f} K ({T_C.min():.0f} - {T_C.max():.0f} C)")
    write(f"  P range: {P_bar.min():.0f} - {P_bar.max():.0f} bar")
    write(f"  kij_NA: mean = {np.mean(kij_na):.4f}, std = {np.std(kij_na):.4f}, "
          f"range = [{kij_na.min():.4f}, {kij_na.max():.4f}]")
    write(f"  Sources ({len(sources)}):")
    for src in sorted(sources):
        n_s = (df['Source'] == src).sum()
        write(f"    {src}: {n_s} pts")

    result['n'] = len(df)
    result['T_range'] = (T_K.min(), T_K.max())
    result['P_range'] = (P_bar.min(), P_bar.max())
    result['kij_na_mean'] = np.mean(kij_na)
    result['kij_na_std'] = np.std(kij_na)

    # --- nC4H10: report only ---
    if gas == 'nC4H10':
        write(f"\n  Only {len(df)} data point(s) — insufficient for optimization.")
        row = df.iloc[0]
        write(f"  Single point: T={row['T_K']:.1f} K, P={row['P_bar']:.0f} bar, "
              f"kij_NA={row['kij_NA']:.4f}, y_H2O_exp={row['y_H2O_exp']:.6f}")

        # Evaluate S&W value at this point
        sw_val = SW_KIJ_NA[gas]
        mare_sw, _, _ = calc_mare_y_h2o(df, gas, sw_val)
        write(f"  S&W kij_NA = {sw_val:.4f}, MARE(y_H2O) = {mare_sw:.2f}%")

        result['sw_kij_na'] = sw_val
        result['sw_mare'] = mare_sw
        result['recommendation'] = 'insufficient data'
        return result

    # --- S&W evaluation ---
    write(f"\n  --- S&W Evaluation ---")

    if gas == 'H2S':
        # T-dependent S&W Eq 17
        mare_sw, n_sw = calc_mare_y_h2o_func(df, gas, kij_na_h2s_sw_eq17)
        kij_na_mae_sw = compute_kij_na_mae(df, kij_na_h2s_sw_eq17)
        sw_label = f"Eq 17: 0.19031 - 0.05965*Tr (Tc={COMPONENTS['H2S'].Tc} K)"
        write(f"  S&W form: {sw_label}")
        write(f"  S&W MARE(y_H2O) = {mare_sw:.2f}% (n={n_sw})")
        write(f"  S&W MAE(kij_NA) = {kij_na_mae_sw:.4f}")
        result['sw_kij_na'] = 'Eq 17'
        result['sw_mare'] = mare_sw
    else:
        sw_val = SW_KIJ_NA[gas]
        if gas == 'H2':
            sw_val = 0.468  # This work value, matches Paper 1
        mare_sw, n_sw, _ = calc_mare_y_h2o(df, gas, sw_val)
        kij_na_mae_sw = compute_kij_na_mae(df, sw_val)
        write(f"  S&W kij_NA = {sw_val:.4f}")
        write(f"  S&W MARE(y_H2O) = {mare_sw:.2f}% (n={n_sw})")
        write(f"  S&W MAE(kij_NA) = {kij_na_mae_sw:.4f}")
        result['sw_kij_na'] = sw_val
        result['sw_mare'] = mare_sw

    # --- CO2: Yan et al. 2011 ---
    if gas == 'CO2':
        yan_val = 0.18756
        mare_yan, n_yan, _ = calc_mare_y_h2o(df, gas, yan_val)
        write(f"\n  --- Yan et al. 2011 ---")
        write(f"  kij_NA = {yan_val}")
        write(f"  MARE(y_H2O) = {mare_yan:.2f}% (n={n_yan})")
        result['yan_mare'] = mare_yan

    # --- H2S: additional T-dependent forms ---
    if gas == 'H2S':
        # Proposed inverse-quadratic
        mare_iq, n_iq = calc_mare_y_h2o_func(df, gas, kij_na_h2s_invquad)
        kij_na_mae_iq = compute_kij_na_mae(df, kij_na_h2s_invquad)
        write(f"\n  --- Proposed Inverse-Quadratic ---")
        write(f"  Form: kij_NA = -1.5977 + 1541.16/T - 334577/T^2")
        write(f"  MARE(y_H2O) = {mare_iq:.2f}% (n={n_iq})")
        write(f"  MAE(kij_NA) = {kij_na_mae_iq:.4f}")
        result['invquad_mare'] = mare_iq

    # --- Grid search optimization ---
    search_range = SEARCH_RANGES.get(gas)
    if search_range is None:
        result['recommendation'] = 'insufficient data'
        return result

    write(f"\n  --- Grid Search Optimization ---")
    write(f"  Search range: [{search_range[0]:.2f}, {search_range[1]:.2f}]")

    kij_opt, mare_opt, coarse_results = optimize_constant(df, gas, search_range)
    kij_na_mae_opt = compute_kij_na_mae(df, kij_opt)

    write(f"\n  Coarse scan (step=0.02):")
    write(f"  {'kij_NA':>10}  {'MARE(y_H2O)':>14}")
    write(f"  {'-'*28}")
    for kij, mare in coarse_results:
        marker = ""
        if gas != 'H2S':
            sw_ref = SW_KIJ_NA.get(gas, None)
            if sw_ref is not None and isinstance(sw_ref, (int, float)) and abs(kij - sw_ref) < 0.01:
                marker = " <-- S&W"
        write(f"  {kij:10.2f}  {mare:13.2f}%{marker}")

    write(f"\n  Optimal kij_NA = {kij_opt:.3f}")
    write(f"  Optimal MARE(y_H2O) = {mare_opt:.2f}%")
    write(f"  Optimal MAE(kij_NA) = {kij_na_mae_opt:.4f}")

    result['opt_kij_na'] = kij_opt
    result['opt_mare'] = mare_opt

    # --- Improvement assessment ---
    if gas == 'H2S':
        improvement = mare_sw - mare_opt
        write(f"\n  Improvement vs S&W Eq 17: {improvement:+.2f} pp")
    else:
        improvement = mare_sw - mare_opt
        write(f"\n  Improvement vs S&W constant: {improvement:+.2f} pp")

    result['improvement_pp'] = improvement

    # --- Recommendation ---
    if improvement > 5:
        rec = f"UPDATE to {kij_opt:.3f} (>{improvement:.1f} pp improvement)"
    elif improvement > 2:
        rec = f"MARGINAL ({improvement:.1f} pp) — consider {kij_opt:.3f}"
    else:
        rec = f"KEEP S&W ({improvement:.1f} pp improvement, not significant)"
    write(f"\n  Recommendation: {rec}")
    result['recommendation'] = rec

    # --- P >= 50 bar sensitivity ---
    if gas not in ('H2', 'nC4H10', 'C3H8'):  # H2 already at P>=50; C3H8 has no data >=50
        df_high_p = df[df['P_bar'] >= P_SENSITIVITY]
        if len(df_high_p) >= 5:
            write(f"\n  --- Sensitivity: P >= {P_SENSITIVITY} bar (n={len(df_high_p)}) ---")
            if gas == 'H2S':
                mare_sw_hp, _ = calc_mare_y_h2o_func(df_high_p, gas, kij_na_h2s)
            else:
                mare_sw_hp, _, _ = calc_mare_y_h2o(df_high_p, gas, SW_KIJ_NA[gas])
            kij_opt_hp, mare_opt_hp, _ = optimize_constant(df_high_p, gas, search_range)
            write(f"  S&W MARE = {mare_sw_hp:.2f}%, Optimal kij_NA = {kij_opt_hp:.3f} (MARE = {mare_opt_hp:.2f}%)")
            result['hp_sw_mare'] = mare_sw_hp
            result['hp_opt_kij'] = kij_opt_hp
            result['hp_opt_mare'] = mare_opt_hp

    return result


# =============================================================================
# Main
# =============================================================================

def run_all(report_path='../../shared/data/kij_na_all_gases_report.txt'):
    """Run kij_NA regression for all gases and generate report."""

    lines = []
    def write(text=""):
        print(text)
        lines.append(text)

    write("=" * 80)
    write("kij_NA REGRESSION REPORT — ALL S&W GASES")
    write("=" * 80)
    write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write(f"Optimization target: MARE on y_H2O (vapor water content)")
    write(f"Default P filter: >= {DEFAULT_P_MIN} bar (gas-specific overrides apply)")
    write()

    # --- Load and filter data for all gases ---
    all_data = {}
    all_stats = {}

    write("-" * 80)
    write("DATA LOADING AND FILTERING")
    write("-" * 80)

    for gas in GASES:
        df, fstats = load_and_filter(gas)
        all_data[gas] = df
        all_stats[gas] = fstats

        filters = [f"total conv={fstats['n_total']}"]
        if fstats['n_bound_excluded'] > 0:
            filters.append(f"bound-hit={fstats['n_bound_excluded']}")
        filters.append(f"P<{fstats['p_min']}bar={fstats['n_low_p_excluded']}")
        if 'n_t_excluded' in fstats:
            t_lo, t_hi = fstats['t_filter']
            filters.append(f"T outside {t_lo-273.15:.0f}-{t_hi-273.15:.0f}C={fstats['n_t_excluded']}")
        filters.append(f"final={fstats['n_final']}")

        write(f"  {gas:8s}: {', '.join(filters)}")

    write()

    # --- Per-gas analysis ---
    results = []
    for gas in GASES:
        result = analyze_gas(gas, all_data[gas], write)
        results.append(result)

    # --- Summary table ---
    write(f"\n\n{'='*80}")
    write("SUMMARY TABLE")
    write(f"{'='*80}")
    write()
    write(f"{'Gas':>8} {'n':>5} {'S&W kij_NA':>14} {'S&W MARE%':>12} "
          f"{'Opt kij_NA':>12} {'Opt MARE%':>12} {'Impr pp':>10}  Recommendation")
    write("-" * 110)

    for r in results:
        gas = r['gas']
        n = r.get('n', 0)
        if n == 0:
            write(f"{gas:>8} {0:>5}  {'—':>14} {'—':>12} {'—':>12} {'—':>12} {'—':>10}  No data")
            continue

        # S&W value string
        sw_str = r.get('sw_kij_na', '—')
        if isinstance(sw_str, float):
            sw_str = f"{sw_str:.4f}"
        elif sw_str == 'Eq 17':
            sw_str = "Eq 17 (T-dep)"

        sw_mare = r.get('sw_mare', np.nan)
        sw_mare_str = f"{sw_mare:.2f}" if not np.isnan(sw_mare) else "—"

        opt_kij = r.get('opt_kij_na', None)
        opt_mare = r.get('opt_mare', None)
        impr = r.get('improvement_pp', None)

        opt_str = f"{opt_kij:.3f}" if opt_kij is not None else "—"
        opt_mare_str = f"{opt_mare:.2f}" if opt_mare is not None else "—"
        impr_str = f"{impr:+.1f}" if impr is not None else "—"
        rec = r.get('recommendation', '—')

        write(f"{gas:>8} {n:>5}  {sw_str:>14} {sw_mare_str:>12} "
              f"{opt_str:>12} {opt_mare_str:>12} {impr_str:>10}  {rec}")

    write()

    # --- H2S special comparison ---
    h2s_result = next((r for r in results if r['gas'] == 'H2S'), None)
    if h2s_result and h2s_result.get('n', 0) > 0:
        write("-" * 80)
        write("H2S FORM COMPARISON")
        write("-" * 80)
        write(f"  S&W Eq 17 (linear in Tr):      MARE = {h2s_result.get('sw_mare', np.nan):.2f}%")
        write(f"  Inverse-quadratic (A+B/T+C/T2): MARE = {h2s_result.get('invquad_mare', np.nan):.2f}%")
        write(f"  Optimal constant:               MARE = {h2s_result.get('opt_mare', np.nan):.2f}% "
              f"(kij_NA = {h2s_result.get('opt_kij_na', '?')})")
        write()

    # --- CO2 comparison ---
    co2_result = next((r for r in results if r['gas'] == 'CO2'), None)
    if co2_result and co2_result.get('n', 0) > 0:
        write("-" * 80)
        write("CO2 VALUE COMPARISON")
        write("-" * 80)
        write(f"  S&W 1992 (0.1896):   MARE = {co2_result.get('sw_mare', np.nan):.2f}%")
        write(f"  Yan et al. 2011 (0.18756): MARE = {co2_result.get('yan_mare', np.nan):.2f}%")
        write(f"  Optimal constant:    MARE = {co2_result.get('opt_mare', np.nan):.2f}% "
              f"(kij_NA = {co2_result.get('opt_kij_na', '?')})")
        write()

    # --- High-P sensitivity summary ---
    hp_gases = [r for r in results if 'hp_opt_kij' in r]
    if hp_gases:
        write("-" * 80)
        write(f"P >= {P_SENSITIVITY} BAR SENSITIVITY")
        write("-" * 80)
        write(f"{'Gas':>8} {'n(P>=50)':>10} {'S&W MARE%':>12} {'Opt kij_NA':>12} {'Opt MARE%':>12}")
        write("-" * 60)
        for r in hp_gases:
            gas = r['gas']
            # Count points at P >= 50
            n_hp = len(all_data[gas][all_data[gas]['P_bar'] >= P_SENSITIVITY])
            write(f"{gas:>8} {n_hp:>10} {r['hp_sw_mare']:>12.2f} "
                  f"{r['hp_opt_kij']:>12.3f} {r['hp_opt_mare']:>12.2f}")
        write()

    # --- Conclusions ---
    write("=" * 80)
    write("CONCLUSIONS")
    write("=" * 80)
    write()
    for r in results:
        gas = r['gas']
        rec = r.get('recommendation', 'no data')
        write(f"  {gas:8s}: {rec}")
    write()

    # Save report
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    write(f"Report saved to: {report_path}")


if __name__ == '__main__':
    run_all()
