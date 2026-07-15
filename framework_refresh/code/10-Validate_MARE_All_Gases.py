#!/usr/bin/env python3
"""
================================================================================
PAPER 2 — SOLUBILITY MARE VALIDATION (ALL GASES, ALL FRAMEWORKS)
================================================================================

Computes Mean Absolute Relative Error (MARE) for:
  - Track 1 (proposed): MC-3 alpha + proposed kij + Sechenov (freshwater + brine)
  - Track 2 (dropin):   S&W alpha + dropin kij + embedded delta (freshwater + brine)
  - S&W original:       S&W alpha + S&W kij (freshwater + brine)
  - Spycher & Pruess 2010 (CO2 only): Modified SRK (freshwater + brine)

Freshwater source: pointwise_kij_results.csv (converged kij_AQ points)
Brine source:      solubility_points.xlsx (Sal_m > 0)

Applies same filters as figure generation:
  - T_MIN = 0°C, T_MAX = 200°C
  - Source exclusions (Barrett 1988 for H2S, Prutton & Savage 1945 for CO2,
    Blount 1982 and McGee 1981 for CH4)

Output: ../../shared/data/mare_all_gases_report.txt

Dependencies: numpy, pandas, openpyxl, pyrestoolbox (optional, for S&P CO2)
Shared modules: _lib_vle_engine.py (from parent code/ dir)
================================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
from _lib_vle_engine import (COMPONENTS, SWBinaryVLE, get_kij_aq, get_sechenov_ks,
                             calc_embedded_delta_kij,
                             EMBEDDED_SALINITY_PARAMS, EMBEDDED_SALINITY_PARAMS_DROPIN,
                             kij_aq_co2_proposed, kij_aq_h2s_proposed,
                             kij_aq_ch4, kij_aq_n2_proposed,
                             kij_aq_h2_proposed, kij_aq_c2h6_proposed,
                             kij_aq_c3h8_proposed)

# Try importing Spycher & Pruess from pyrestoolbox
try:
    from pyrestoolbox import brine as prt_brine
    HAS_PRT = True
except ImportError:
    HAS_PRT = False

# ── Paths ──
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
KIJ_CSV = os.path.join(DATA_DIR, 'pointwise_kij_results.csv')
XLSX_PATH = os.path.join(DATA_DIR, 'solubility_points.xlsx')
OUTPUT_PATH = os.path.join(DATA_DIR, 'mare_all_gases_report.txt')

# ── Constants ──
T_MIN_K = 273.15  # 0°C
T_MAX_K = 473.15  # 200°C

# ── Source exclusions — consistent with 12-Generate_Figures.py ──
EXCLUDE_SOURCES = {
    'CH4': {'Blount 1982', 'McGee 1981'},
    'CO2': {'Prutton & Savage 1945'},
    'H2S': {'Barrett 1988'},
}

# ── Gas list ──
GASES = ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8']
NO_SW_ORIGINAL = {'H2'}

# ── Column maps for xlsx (wide format) ──
X_COL = {'CO2': 'x_CO2', 'H2S': 'x_H2S', 'N2': 'x_N2', 'H2': 'x_H2',
          'CH4': 'x_CH4', 'C2H6': 'x_C2H6', 'C3H8': 'x_C3H8'}
Z_COL = {'CO2': 'z_CO2', 'H2S': 'z_H2S', 'N2': 'z_N2', 'H2': 'z_H2',
          'CH4': 'z_CH4', 'C2H6': 'z_C2H6', 'C3H8': 'z_C3H8'}


def calc_mare(x_pred, x_exp):
    """Mean absolute relative error in percent."""
    valid = (x_exp > 0) & np.isfinite(x_pred) & (x_pred > 0)
    if valid.sum() == 0:
        return np.nan, 0
    return np.mean(np.abs(x_pred[valid] - x_exp[valid]) / x_exp[valid]) * 100, valid.sum()


def spycher_pruess_co2(T_K, P_bar, salinity_molal):
    """CO2 solubility via Spycher & Pruess 2010 (pyrestoolbox)."""
    if not HAS_PRT:
        return np.nan
    T_C = T_K - 273.15
    if salinity_molal > 0:
        ppm = salinity_molal * 58.44 / (1000 + salinity_molal * 58.44) * 1e6
    else:
        ppm = 0
    try:
        result = prt_brine.CO2_Brine_Mixture(pres=P_bar, temp=T_C, ppm=ppm, metric=True)
        x = result.x[0]
        if x is not None and np.isfinite(x) and x > 0:
            return x
    except Exception:
        pass
    return np.nan


def calc_freshwater_mare(gas, kij_df):
    """Compute freshwater MARE for all frameworks + S&P for CO2."""
    fw = kij_df[(kij_df['Gas'] == gas) & (kij_df['kij_AQ_conv'] == True)].copy()
    fw = fw[(fw['T_K'] >= T_MIN_K) & (fw['T_K'] <= T_MAX_K)].copy()

    excl = EXCLUDE_SOURCES.get(gas, set())
    if excl and 'Source' in fw.columns:
        fw = fw[~fw['Source'].isin(excl)].copy()
    if gas == 'H2':
        fw = fw[fw['Source'] != 'Gillespie 1980'].copy()

    if len(fw) == 0:
        return None

    x_exp = fw['x_gas_exp'].values
    T_K_arr = fw['T_K'].values
    P_bar_arr = fw['P_bar'].values

    vle_prop = SWBinaryVLE(gas, 0.0, framework='proposed')
    vle_drop = SWBinaryVLE(gas, 0.0, framework='dropin')
    has_sw = gas not in NO_SW_ORIGINAL
    vle_sw = SWBinaryVLE(gas, 0.0, framework='sw_original') if has_sw else None

    x_prop = np.full_like(x_exp, np.nan)
    x_drop = np.full_like(x_exp, np.nan)
    x_sw = np.full_like(x_exp, np.nan)
    x_sp = np.full_like(x_exp, np.nan)

    for i in range(len(fw)):
        T, P_Pa = T_K_arr[i], P_bar_arr[i] * 1e5

        # Track 1 (proposed)
        try:
            x = vle_prop.calc_gas_solubility(T, P_Pa)
            if x is not None and np.isfinite(x) and x > 0:
                x_prop[i] = x
        except Exception:
            pass

        # Track 2 (dropin)
        try:
            x = vle_drop.calc_gas_solubility(T, P_Pa)
            if x is not None and np.isfinite(x) and x > 0:
                x_drop[i] = x
        except Exception:
            pass

        # S&W original
        if has_sw:
            try:
                x = vle_sw.calc_gas_solubility(T, P_Pa)
                if x is not None and np.isfinite(x) and x > 0:
                    x_sw[i] = x
            except Exception:
                pass

        # Spycher & Pruess (CO2 only)
        if gas == 'CO2' and HAS_PRT:
            x_sp[i] = spycher_pruess_co2(T, P_bar_arr[i], 0.0)

    mare_prop, n_prop = calc_mare(x_prop, x_exp)
    mare_drop, n_drop = calc_mare(x_drop, x_exp)
    mare_sw, n_sw = calc_mare(x_sw, x_exp) if has_sw else (np.nan, 0)
    mare_sp, n_sp = calc_mare(x_sp, x_exp) if gas == 'CO2' and HAS_PRT else (np.nan, 0)

    return {
        'n': len(fw), 'n_prop': n_prop, 'n_drop': n_drop, 'n_sw': n_sw, 'n_sp': n_sp,
        'mare_prop': mare_prop, 'mare_drop': mare_drop,
        'mare_sw': mare_sw, 'mare_sp': mare_sp,
        'x_exp': x_exp, 'x_prop': x_prop, 'x_drop': x_drop, 'x_sw': x_sw, 'x_sp': x_sp,
        'T_K': T_K_arr, 'P_bar': P_bar_arr,
    }


def calc_brine_mare(gas, xlsx_df):
    """Compute brine MARE for all frameworks + S&P for CO2."""
    x_col = X_COL.get(gas)
    z_col = Z_COL.get(gas)
    if x_col is None or z_col is None or z_col not in xlsx_df.columns:
        return None

    mask = ((xlsx_df[z_col] == 1.0) & xlsx_df[x_col].notna() & (xlsx_df[x_col] > 0)
            & (xlsx_df['Sal_m'] > 0)
            & (xlsx_df['T_K'] >= T_MIN_K) & (xlsx_df['T_K'] <= T_MAX_K))
    br = xlsx_df[mask].copy()

    excl = EXCLUDE_SOURCES.get(gas, set())
    if excl and 'Source' in br.columns:
        br = br[~br['Source'].isin(excl)].copy()

    if len(br) < 5:
        return None

    x_exp = br[x_col].values
    T_K_arr = br['T_K'].values
    P_bar_arr = br['P_bar'].values
    Sal_arr = br['Sal_m'].values
    has_sw = gas not in NO_SW_ORIGINAL

    x_prop = np.full_like(x_exp, np.nan)
    x_drop = np.full_like(x_exp, np.nan)
    x_sw = np.full_like(x_exp, np.nan)
    x_sp = np.full_like(x_exp, np.nan)

    for i in range(len(br)):
        T, P_Pa = T_K_arr[i], P_bar_arr[i] * 1e5
        m = Sal_arr[i]

        # Track 1 (proposed) — Sechenov
        try:
            vle = SWBinaryVLE(gas, m, framework='proposed')
            x = vle.calc_gas_solubility(T, P_Pa)
            if x is not None and np.isfinite(x) and x > 0:
                x_prop[i] = x
        except Exception:
            pass

        # Track 2 (dropin) — embedded delta
        try:
            vle = SWBinaryVLE(gas, m, framework='dropin')
            x = vle.calc_gas_solubility(T, P_Pa)
            if x is not None and np.isfinite(x) and x > 0:
                x_drop[i] = x
        except Exception:
            pass

        # S&W original
        if has_sw:
            try:
                vle = SWBinaryVLE(gas, m, framework='sw_original')
                x = vle.calc_gas_solubility(T, P_Pa)
                if x is not None and np.isfinite(x) and x > 0:
                    x_sw[i] = x
            except Exception:
                pass

        # Spycher & Pruess (CO2 only)
        if gas == 'CO2' and HAS_PRT:
            x_sp[i] = spycher_pruess_co2(T, P_bar_arr[i], m)

    mare_prop, n_prop = calc_mare(x_prop, x_exp)
    mare_drop, n_drop = calc_mare(x_drop, x_exp)
    mare_sw, n_sw = calc_mare(x_sw, x_exp) if has_sw else (np.nan, 0)
    mare_sp, n_sp = calc_mare(x_sp, x_exp) if gas == 'CO2' and HAS_PRT else (np.nan, 0)

    return {
        'n': len(br), 'n_prop': n_prop, 'n_drop': n_drop, 'n_sw': n_sw, 'n_sp': n_sp,
        'mare_prop': mare_prop, 'mare_drop': mare_drop,
        'mare_sw': mare_sw, 'mare_sp': mare_sp,
        'x_exp': x_exp, 'x_prop': x_prop, 'x_drop': x_drop, 'x_sw': x_sw, 'x_sp': x_sp,
        'T_K': T_K_arr, 'P_bar': P_bar_arr, 'Sal_m': Sal_arr,
    }


def analyze_co2_sp_breakdown(fw_res, br_res):
    """Breakdown of CO2 proposed vs S&P errors by T and P bins."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("CO2: PROPOSED vs SPYCHER & PRUESS 2010 — BREAKDOWN BY CONDITIONS")
    lines.append("=" * 78)

    for label, res in [("FRESHWATER", fw_res), ("BRINE", br_res)]:
        if res is None:
            continue
        lines.append(f"\n  {label}")
        lines.append(f"  {'Condition':<30s} {'n':>5} {'MARE_prop%':>11} {'MARE_sp%':>10} {'MARE_sw%':>10}")
        lines.append("  " + "-" * 70)

        x_exp = res['x_exp']
        x_prop = res['x_prop']
        x_sp = res['x_sp']
        x_sw = res['x_sw']
        T_K = res['T_K']
        P_bar = res['P_bar']

        # Temperature bins
        T_bins = [(273.15, 323.15, "T < 50C"), (323.15, 373.15, "50C <= T < 100C"),
                  (373.15, 423.15, "100C <= T < 150C"), (423.15, 473.15, "150C <= T <= 200C")]
        for T_lo, T_hi, desc in T_bins:
            mask = (T_K >= T_lo) & (T_K < T_hi)
            if mask.sum() == 0:
                continue
            m_prop, _ = calc_mare(x_prop[mask], x_exp[mask])
            m_sp, _ = calc_mare(x_sp[mask], x_exp[mask])
            m_sw, _ = calc_mare(x_sw[mask], x_exp[mask])
            sp_str = f"{m_sp:.1f}" if not np.isnan(m_sp) else "---"
            pr_str = f"{m_prop:.1f}" if not np.isnan(m_prop) else "---"
            sw_str = f"{m_sw:.1f}" if not np.isnan(m_sw) else "---"
            lines.append(f"  {desc:<30s} {mask.sum():>5} {pr_str:>11} {sp_str:>10} {sw_str:>10}")

        # Pressure bins
        P_bins = [(0, 50, "P < 50 bar"), (50, 200, "50 <= P < 200 bar"),
                  (200, 500, "200 <= P < 500 bar"), (500, 2000, "P >= 500 bar")]
        lines.append("")
        for P_lo, P_hi, desc in P_bins:
            mask = (P_bar >= P_lo) & (P_bar < P_hi)
            if mask.sum() == 0:
                continue
            m_prop, _ = calc_mare(x_prop[mask], x_exp[mask])
            m_sp, _ = calc_mare(x_sp[mask], x_exp[mask])
            m_sw, _ = calc_mare(x_sw[mask], x_exp[mask])
            sp_str = f"{m_sp:.1f}" if not np.isnan(m_sp) else "---"
            pr_str = f"{m_prop:.1f}" if not np.isnan(m_prop) else "---"
            sw_str = f"{m_sw:.1f}" if not np.isnan(m_sw) else "---"
            lines.append(f"  {desc:<30s} {mask.sum():>5} {pr_str:>11} {sp_str:>10} {sw_str:>10}")

        # x_gas magnitude bins
        x_bins = [(0, 0.003, "x < 0.003"), (0.003, 0.01, "0.003 <= x < 0.01"),
                  (0.01, 0.03, "0.01 <= x < 0.03"), (0.03, 1, "x >= 0.03")]
        lines.append("")
        for x_lo, x_hi, desc in x_bins:
            mask = (x_exp >= x_lo) & (x_exp < x_hi)
            if mask.sum() == 0:
                continue
            m_prop, _ = calc_mare(x_prop[mask], x_exp[mask])
            m_sp, _ = calc_mare(x_sp[mask], x_exp[mask])
            m_sw, _ = calc_mare(x_sw[mask], x_exp[mask])
            sp_str = f"{m_sp:.1f}" if not np.isnan(m_sp) else "---"
            pr_str = f"{m_prop:.1f}" if not np.isnan(m_prop) else "---"
            sw_str = f"{m_sw:.1f}" if not np.isnan(m_sw) else "---"
            lines.append(f"  {desc:<30s} {mask.sum():>5} {pr_str:>11} {sp_str:>10} {sw_str:>10}")

    return lines


def main():
    print("Loading data...")
    kij_df = pd.read_csv(KIJ_CSV)
    if 'Source' in kij_df.columns:
        kij_df['Source'] = kij_df['Source'].str.replace('Chahab', 'Chabab', regex=False)

    xlsx_df = pd.read_excel(XLSX_PATH, sheet_name='QCd Data', engine='openpyxl')

    lines = []
    lines.append("=" * 90)
    lines.append("SOLUBILITY MARE — ALL GASES, ALL FRAMEWORKS")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Filters: {T_MIN_K - 273.15:.0f}°C <= T <= {T_MAX_K - 273.15:.0f}°C, source exclusions applied")
    lines.append(f"Source exclusions: {EXCLUDE_SOURCES}")
    if HAS_PRT:
        lines.append("Spycher & Pruess 2010 (CO2 only): via pyrestoolbox")
    lines.append("")

    # ─── FRESHWATER TABLE ───
    lines.append("-" * 90)
    lines.append("FRESHWATER (from pointwise_kij_results.csv)")
    lines.append("-" * 90)
    hdr = f"{'Gas':<8} {'n':>5}  {'Track1%':>8}  {'Track2%':>8}  {'S&W%':>8}  {'S&P%':>8}"
    lines.append(hdr)
    lines.append("-" * 55)

    fw_results = {}
    for gas in GASES:
        print(f"  {gas} freshwater...")
        res = calc_freshwater_mare(gas, kij_df)
        if res is None:
            lines.append(f"{gas:<8} — no data")
            continue
        fw_results[gas] = res

        def fmt(v):
            return f"{v:.1f}" if not np.isnan(v) else "---"

        lines.append(f"{gas:<8} {res['n']:>5}  {fmt(res['mare_prop']):>8}  "
                      f"{fmt(res['mare_drop']):>8}  {fmt(res['mare_sw']):>8}  "
                      f"{fmt(res['mare_sp']):>8}")

    # ─── BRINE TABLE ───
    lines.append("")
    lines.append("-" * 90)
    lines.append("BRINE (from solubility_points.xlsx, Sal_m > 0)")
    lines.append("-" * 90)
    hdr_br = (f"{'Gas':<8} {'n':>5}  {'Track1%':>8}  {'Track2%':>8}  "
              f"{'S&W%':>8}  {'S&P%':>8}  {'Sal range':>12}")
    lines.append(hdr_br)
    lines.append("-" * 65)

    br_results = {}
    for gas in GASES:
        print(f"  {gas} brine...")
        res = calc_brine_mare(gas, xlsx_df)
        if res is None:
            lines.append(f"{gas:<8} — insufficient brine data (<5 pts)")
            continue
        br_results[gas] = res

        def fmt(v):
            return f"{v:.1f}" if not np.isnan(v) else "---"

        sal_range = f"{res['Sal_m'].min():.1f}-{res['Sal_m'].max():.1f}m"
        lines.append(f"{gas:<8} {res['n']:>5}  {fmt(res['mare_prop']):>8}  "
                      f"{fmt(res['mare_drop']):>8}  {fmt(res['mare_sw']):>8}  "
                      f"{fmt(res['mare_sp']):>8}  {sal_range:>12}")

    # ─── NOTES ───
    lines.append("")
    lines.append("-" * 90)
    lines.append("Column legend:")
    lines.append("  Track1:  Proposed framework (MC-3 alpha + proposed kij + Sechenov for brine)")
    lines.append("  Track2:  Drop-in framework (S&W alpha + dropin kij + embedded delta for brine)")
    lines.append("  S&W:     S&W original framework (S&W alpha + S&W kij)")
    lines.append("  S&P:     Spycher & Pruess 2010 modified SRK (CO2 only)")
    lines.append("  MARE = (100/n) * sum(|x_pred - x_exp| / x_exp)")

    # ─── CO2 S&P BREAKDOWN ───
    if 'CO2' in fw_results and HAS_PRT:
        sp_lines = analyze_co2_sp_breakdown(fw_results.get('CO2'), br_results.get('CO2'))
        lines.extend(sp_lines)

    # ─── H2S CONDITIONED MARE ───
    if 'H2S' in fw_results:
        lines.append("")
        lines.append("=" * 78)
        lines.append("H2S conditioned MARE (low-x sensitivity analysis):")
        res = fw_results['H2S']
        x_exp = res['x_exp']
        x_prop = res['x_prop']
        for threshold in [0.003, 0.01]:
            mask = (x_exp >= threshold) & np.isfinite(x_prop) & (x_prop > 0)
            if mask.sum() > 0:
                mare_cond = np.mean(np.abs(x_prop[mask] - x_exp[mask]) / x_exp[mask]) * 100
                lines.append(f"  x >= {threshold}: MARE = {mare_cond:.1f}% (n={mask.sum()})")

    report = '\n'.join(lines)
    print()
    print(report)

    with open(OUTPUT_PATH, 'w') as f:
        f.write(report + '\n')
    print(f"\nReport saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
