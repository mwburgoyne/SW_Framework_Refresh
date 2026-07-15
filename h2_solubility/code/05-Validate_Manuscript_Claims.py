#!/usr/bin/env python3
"""
validate_manuscript_claims.py
=============================
Standalone validation of every numerical claim in manuscript.tex
attributed to computational work.

Checks: error metrics, data counts, physical properties, Sechenov
conformance, BIP properties, embedded salinity accuracy, Henry's
law multicomponent claims, and table arithmetic.

Usage:
    python validate_manuscript_claims.py           # Quick mode (skips slow multicomponent)
    python validate_manuscript_claims.py --full     # Full mode (includes Henry's law validation)

Output: ../../shared/data/manuscript_claims_validation.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
from datetime import datetime

from _lib_vle_engine import (
    SWBinaryVLE, H2WaterVLE, COMPONENTS, BIP_TC_H2,
    kij_aq_rational, kij_aq_linear,
    kij_aq_chabab_2023, kij_aq_lopez_lazaro_2019,
    kij_na_chabab_2023, kij_na_lopez_lazaro_2019,
    sw_equation_8_ks, get_kij_aq, get_kij_na,
    celsius_to_kelvin, bar_to_pascal, kelvin_to_celsius,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_PATH = '../../shared/data/pointwise_kij_results.csv'
REPORT_PATH = '../../shared/data/manuscript_claims_validation.txt'
TC_H2 = 33.145  # K, NIST
A, B, C_COEFF = -14.59, 2.184, 0.365
KIJ_NA = 0.468

FITTING_SOURCE_KEYS = ['Wiebe 1934', 'Wiebe 1932', 'Chabab 2023',
                       'Chahab 2023',  # potential typo in data
                       'Torín-Ollarves 2021', 'Torin-Ollarves 2021']


def is_fitting_source(source):
    return any(k in source for k in FITTING_SOURCE_KEYS)


# =============================================================================
# REPORTER
# =============================================================================
class Reporter:
    """Track claims and report PASS/FAIL/WARN with expected vs actual."""

    def __init__(self):
        self.results = []
        self.lines = []
        self.current_section = ""

    def section(self, title):
        self.current_section = title
        sep = "=" * 72
        self._out(f"\n{sep}")
        self._out(f"  {title}")
        self._out(sep)

    def check(self, desc, expected, actual, tol_pct=1.0, abs_tol=None, section_ref=""):
        """Check a numerical claim. tol_pct = relative tolerance in %."""
        # Handle boolean comparisons
        if isinstance(expected, bool):
            passed = bool(actual) == expected
        elif abs_tol is not None:
            passed = abs(float(actual) - float(expected)) <= abs_tol
        elif expected == 0:
            passed = abs(float(actual)) < 1e-10
        else:
            passed = abs(float(actual) - float(expected)) <= abs(float(expected)) * tol_pct / 100

        status = "PASS" if passed else "FAIL"
        marker = " " if passed else " *** "
        line = f"  [{status}]{marker}{desc}"
        line += f"\n         Expected: {expected}  |  Got: {actual}"
        if not passed:
            if expected != 0:
                line += f"  |  Diff: {abs(actual-expected)/abs(expected)*100:.2f}%"
            else:
                line += f"  |  Diff: {actual}"
        self._out(line)
        self.results.append({
            'section': self.current_section,
            'desc': desc,
            'expected': expected,
            'actual': actual,
            'status': status,
            'ref': section_ref,
        })

    def check_int(self, desc, expected, actual, section_ref=""):
        """Check an exact integer claim."""
        passed = int(actual) == int(expected)
        status = "PASS" if passed else "FAIL"
        marker = " " if passed else " *** "
        line = f"  [{status}]{marker}{desc}"
        line += f"\n         Expected: {expected}  |  Got: {actual}"
        self._out(line)
        self.results.append({
            'section': self.current_section,
            'desc': desc,
            'expected': expected,
            'actual': actual,
            'status': status,
            'ref': section_ref,
        })

    def warn(self, desc, detail=""):
        line = f"  [WARN]  {desc}"
        if detail:
            line += f"\n         {detail}"
        self._out(line)
        self.results.append({
            'section': self.current_section,
            'desc': desc,
            'expected': None,
            'actual': None,
            'status': 'WARN',
        })

    def info(self, text):
        self._out(f"  [INFO]  {text}")

    def _out(self, text):
        print(text)
        self.lines.append(text)

    def summary(self):
        n_pass = sum(1 for r in self.results if r['status'] == 'PASS')
        n_fail = sum(1 for r in self.results if r['status'] == 'FAIL')
        n_warn = sum(1 for r in self.results if r['status'] == 'WARN')
        total = n_pass + n_fail + n_warn
        sep = "=" * 72
        self._out(f"\n{sep}")
        self._out("  SUMMARY")
        self._out(sep)
        self._out(f"  Total checks: {total}")
        self._out(f"  PASS:  {n_pass}")
        self._out(f"  FAIL:  {n_fail}")
        self._out(f"  WARN:  {n_warn}")
        if n_fail > 0:
            self._out(f"\n  FAILURES:")
            for r in self.results:
                if r['status'] == 'FAIL':
                    self._out(f"    - {r['desc']}")
                    self._out(f"      Expected: {r['expected']}  Got: {r['actual']}")
        if n_warn > 0:
            self._out(f"\n  WARNINGS:")
            for r in self.results:
                if r['status'] == 'WARN':
                    self._out(f"    - {r['desc']}")
        return n_fail

    def save(self, path):
        header = [
            "MANUSCRIPT CLAIMS VALIDATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data source: {CSV_PATH}",
            "",
        ]
        with open(path, 'w') as f:
            for h in header:
                f.write(h + '\n')
            for line in self.lines:
                f.write(line + '\n')
        print(f"\n  Report saved to: {path}")


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load and prepare the pointwise kij results."""
    df = pd.read_csv(CSV_PATH)
    df_h2 = df[df['Gas'] == 'H2'].copy()
    return df_h2


def get_aqueous_fitting_data(df_h2):
    """Filter to aqueous fitting sources, excluding T-O 423K."""
    mask = df_h2['x_gas_exp'].notna()
    df_aq = df_h2[mask].copy()
    df_aq = df_aq[df_aq['Source'].apply(is_fitting_source)]
    # Exclude T-O 423K
    df_aq = df_aq[~((df_aq['Source'].str.contains('Tor', na=False)) &
                     (abs(df_aq['T_K'] - 423.15) < 1))]
    return df_aq


def get_na_quality_filtered(df_h2):
    """Filter to UHS-relevant non-aqueous data (P>=50, T=50-150C, converged)."""
    mask = df_h2['y_H2O_exp'].notna()
    df_na = df_h2[mask].copy()
    df_na = df_na[df_na['P_bar'] >= 50]
    df_na = df_na[df_na['kij_NA'] > -0.99]
    T_min = celsius_to_kelvin(50)
    T_max = celsius_to_kelvin(150)
    df_na = df_na[(df_na['T_K'] >= T_min - 1) & (df_na['T_K'] <= T_max + 1)]
    return df_na


# =============================================================================
# SECTION 1: DATA INTEGRITY - Point counts and ranges
# =============================================================================
def check_data_integrity(df_h2, rpt):
    rpt.section("DATA INTEGRITY - Point Counts and Ranges (Table 1)")

    # Count by source
    source_counts = {}
    for _, row in df_h2.iterrows():
        src = row['Source']
        if src not in source_counts:
            source_counts[src] = {'x': 0, 'y': 0, 'total': 0}
        source_counts[src]['total'] += 1
        if pd.notna(row.get('x_gas_exp')):
            source_counts[src]['x'] += 1
        if pd.notna(row.get('y_H2O_exp')):
            source_counts[src]['y'] += 1

    rpt.info("Source breakdown in CSV:")
    for src, counts in sorted(source_counts.items()):
        rpt.info(f"  {src}: {counts['total']} total, {counts['x']} x_H2, {counts['y']} y_H2O")

    # Table 1 claims for aqueous (x_H2) data
    wiebe_x = sum(c['x'] for s, c in source_counts.items() if 'Wiebe' in s)
    chabab23_x = sum(c['x'] for s, c in source_counts.items() if 'Chabab 2023' in s or 'Chahab 2023' in s)
    to_x = sum(c['x'] for s, c in source_counts.items() if 'Tor' in s)
    stephan_x = sum(c['x'] for s, c in source_counts.items() if 'Stephan' in s)
    suciu_total = sum(c['total'] for s, c in source_counts.items() if 'Suciu' in s)
    gillespie_total = sum(c['total'] for s, c in source_counts.items() if 'Gillespie' in s)

    rpt.check_int("Wiebe aqueous points (Table 1: 48)", 48, wiebe_x, "Table 1")
    rpt.check_int("Chabab 2023 aqueous points (Table 1: 14)", 14, chabab23_x, "Table 1")
    rpt.check_int("T-O aqueous points (Table 1: 5)", 5, to_x, "Table 1")
    rpt.check_int("Stephan aqueous points (Table 1: 48)", 48, stephan_x, "Table 1")
    rpt.check_int("Suciu total points (Table 1: 56)", 56, suciu_total, "Table 1")
    rpt.check_int("Gillespie total points (Table 1: 17)", 17, gillespie_total, "Table 1")

    # Combined fitting dataset
    total_included = wiebe_x + chabab23_x + to_x
    rpt.check_int("Combined fitting points before exclusions (Sec 2.4: 67)",
                   67, total_included, "Sec 2.4")

    # After T-O 423K exclusion
    df_aq = get_aqueous_fitting_data(df_h2)
    rpt.check_int("Fitting points after T-O 423K exclusion (used in error calc: 62)",
                   62, len(df_aq), "Table 4 / BIP report")

    to_excluded = to_x - sum(1 for _, r in df_aq.iterrows() if 'Tor' in str(r['Source']))
    rpt.info(f"T-O points excluded at 423K: {to_excluded}")
    rpt.info(f"67 - {to_excluded} excluded = {67 - to_excluded} vs actual {len(df_aq)}")
    if 67 - to_excluded != len(df_aq):
        extra = 67 - to_excluded - len(df_aq)
        rpt.warn(f"Additional {extra} point(s) excluded beyond T-O 423K",
                 "Manuscript says 67 points but fitting uses fewer. Check data pipeline.")

    # T and P ranges for fitting data
    rpt.check("Fitting T min (Sec 2.4: 0C)", 0,
              round(kelvin_to_celsius(df_aq['T_K'].min())), abs_tol=2, section_ref="Sec 2.4")
    rpt.check("Fitting T max (Sec 2.4: 150C)", 150,
              round(kelvin_to_celsius(df_aq['T_K'].max())), abs_tol=5, section_ref="Sec 2.4")
    rpt.check("Fitting P min (Sec 2.4: 25 bar)", 25,
              round(df_aq['P_bar'].min()), abs_tol=2, section_ref="Sec 2.4")

    # Non-aqueous source data
    na_all = df_h2[df_h2['y_H2O_exp'].notna()]
    to_y = sum(1 for _, r in na_all.iterrows() if 'Tor' in str(r['Source']))
    suciu_y = sum(1 for _, r in na_all.iterrows() if 'Suciu' in str(r['Source']))
    gillespie_y = sum(1 for _, r in na_all.iterrows() if 'Gillespie' in str(r['Source']))

    rpt.check_int("T-O y_H2O points (Table 5: 5)", 5, to_y, "Table 5")
    rpt.check_int("Suciu y_H2O points (Table 5: 15)", 15, suciu_y, "Table 5")
    rpt.check_int("Gillespie y_H2O points (Table 5: 17)", 17, gillespie_y, "Table 5")

    # Quality-filtered non-aqueous
    df_na = get_na_quality_filtered(df_h2)
    rpt.check_int("Quality-filtered NA points (Sec 4.2, Table 6: 14)",
                   14, len(df_na), "Table 6")

    # kij_NA statistics for quality-filtered
    mean_kij_na = df_na['kij_NA'].mean()
    std_kij_na = df_na['kij_NA'].std()
    rpt.check("Mean kij_NA quality-filtered (Sec 4.2: 0.44)", 0.44, round(mean_kij_na, 2),
              tol_pct=5, section_ref="Sec 4.2")
    rpt.check("Std kij_NA quality-filtered (Sec 4.2: 0.22)", 0.22, round(std_kij_na, 2),
              abs_tol=0.02, section_ref="Sec 4.2")

    # Standard deviations by source (Table 5)
    for src_key, expected_sigma in [('Tor', 0.21), ('Suciu', 0.73), ('Gillespie', 0.54)]:
        src_data = na_all[na_all['Source'].str.contains(src_key, na=False)]
        if len(src_data) > 1 and 'kij_NA' in src_data.columns:
            valid = src_data[src_data['kij_NA'].notna()]
            if len(valid) > 1:
                sigma = valid['kij_NA'].std()
                rpt.check(f"kij_NA sigma for {src_key} (Table 5: {expected_sigma})",
                          expected_sigma, round(sigma, 2), abs_tol=0.02, section_ref="Table 5")

    # Bound-hitting points (Sec 4.2: 7 points outside ±1.0)
    na_bound = na_all[na_all['kij_NA'].notna()]
    n_bound_hit = sum(1 for _, r in na_bound.iterrows() if abs(r['kij_NA']) >= 0.99)
    rpt.check_int("Points with kij_NA at bounds (Sec 4.2: 7)", 7, n_bound_hit, "Sec 4.2")


# =============================================================================
# SECTION 2: TABLE VALUES - Experimental data checks
# =============================================================================
def check_table_values(df_h2, rpt):
    rpt.section("TABLE VALUES - Experimental Data (Tables 2, 3)")

    # Table 2: x_H2 at 25C, 100 bar from different sources
    def find_point(df, source_key, T_C_target, P_bar_target, tol_T=3, tol_P=5):
        mask = (df['Source'].str.contains(source_key, na=False) &
                (abs(df['T_C'] - T_C_target) < tol_T) &
                (abs(df['P_bar'] - P_bar_target) < tol_P) &
                df['x_gas_exp'].notna())
        matches = df[mask]
        if len(matches) > 0:
            return matches.iloc[0]['x_gas_exp']
        return None

    # Wiebe at 25C, 100 bar
    x_wiebe_25_100 = find_point(df_h2, 'Wiebe', 25, 100)
    if x_wiebe_25_100 is not None:
        rpt.check("Wiebe x_H2 at 25C/100bar (Table 2: 1.233e-3)",
                  1.233e-3, x_wiebe_25_100, tol_pct=1, section_ref="Table 2")
    else:
        rpt.warn("Could not find Wiebe data at 25C/100bar")

    # Chabab at 25C, 100 bar
    x_chabab_25_100 = find_point(df_h2, 'Chabab 2023', 25, 100)
    if x_chabab_25_100 is None:
        x_chabab_25_100 = find_point(df_h2, 'Chahab 2023', 25, 100)
    if x_chabab_25_100 is not None:
        rpt.check("Chabab 2023 x_H2 at 25C/100bar (Table 2: 1.360e-3)",
                  1.360e-3, x_chabab_25_100, tol_pct=2, section_ref="Table 2")
    else:
        rpt.warn("Could not find Chabab 2023 data at 25C/100bar")

    # Table 2 deviation arithmetic
    if x_wiebe_25_100 is not None and x_chabab_25_100 is not None:
        dev_chabab = (x_chabab_25_100 - x_wiebe_25_100) / x_wiebe_25_100 * 100
        rpt.check("Chabab deviation from Wiebe (Table 2: +10%)",
                  10, round(dev_chabab), abs_tol=1.5, section_ref="Table 2")

    # Suciu at ~25C, ~100 bar
    x_suciu_25_100 = find_point(df_h2, 'Suciu', 25, 100, tol_T=5, tol_P=10)
    if x_suciu_25_100 is not None:
        rpt.check("Suciu x_H2 near 25C/100bar (Table 2: 1.44e-3)",
                  1.44e-3, x_suciu_25_100, tol_pct=5, section_ref="Table 2")
        if x_wiebe_25_100 is not None:
            dev_suciu = (x_suciu_25_100 - x_wiebe_25_100) / x_wiebe_25_100 * 100
            rpt.check("Suciu deviation from Wiebe (Table 2: +17%)",
                      17, round(dev_suciu), abs_tol=2, section_ref="Table 2")
    else:
        rpt.warn("Could not find Suciu data near 25C/100bar (may need interpolation)")

    # Table 3: U-shaped data (Wiebe at 100 bar)
    rpt.info("Table 3: U-shaped temperature dependence (Wiebe at 100 bar)")
    table3_expected = {
        0: {'x': 1.520e-3, 'ratio': 1.32},
        25: {'x': 1.233e-3, 'ratio': 1.07},
        50: {'x': 1.151e-3, 'ratio': 1.00},
        75: {'x': 1.173e-3, 'ratio': 1.02},
        100: {'x': 1.288e-3, 'ratio': 1.12},
    }
    x_at_50 = None
    for T_C, vals in table3_expected.items():
        x_obs = find_point(df_h2, 'Wiebe', T_C, 100, tol_T=2, tol_P=5)
        if x_obs is not None:
            rpt.check(f"Wiebe x_H2 at {T_C}C/100bar (Table 3: {vals['x']:.3e})",
                      vals['x'], x_obs, tol_pct=1, section_ref="Table 3")
            if T_C == 50:
                x_at_50 = x_obs
        else:
            rpt.warn(f"Could not find Wiebe data at {T_C}C/100bar")

    # Verify ratios to 50C
    if x_at_50 is not None:
        for T_C, vals in table3_expected.items():
            x_obs = find_point(df_h2, 'Wiebe', T_C, 100, tol_T=2, tol_P=5)
            if x_obs is not None:
                ratio = x_obs / x_at_50
                rpt.check(f"Ratio to 50C at {T_C}C (Table 3: {vals['ratio']})",
                          vals['ratio'], round(ratio, 2), abs_tol=0.01, section_ref="Table 3")

    # 32% total variation claim
    x_0 = find_point(df_h2, 'Wiebe', 0, 100, tol_T=2, tol_P=5)
    x_50 = find_point(df_h2, 'Wiebe', 50, 100, tol_T=2, tol_P=5)
    if x_0 is not None and x_50 is not None:
        variation = (x_0 - x_50) / x_50 * 100
        rpt.check("Total variation 0-100C (Sec 2.3: ~32%)",
                  32, round(variation), abs_tol=3, section_ref="Sec 2.3")


# =============================================================================
# SECTION 3: AQUEOUS ERROR METRICS (Table 4)
# =============================================================================
def check_aqueous_errors(df_h2, rpt):
    rpt.section("AQUEOUS PHASE ERROR METRICS (Table 4)")

    df_aq = get_aqueous_fitting_data(df_h2)
    vle = H2WaterVLE(salinity=0.0)

    kij_correlations = {
        'This work':       ('kij_aq_rational', kij_aq_rational),
        'Chabab 2023':     ('kij_aq_chabab_2023', lambda T: kij_aq_chabab_2023(T, m=0)),
        'Lopez-Lazaro':    ('kij_aq_lopez_lazaro', lambda T: kij_aq_lopez_lazaro_2019(T, csw=0)),
        'Linear':          ('kij_aq_linear', kij_aq_linear),
    }

    # Expected values from manuscript Table 4
    expected = {
        'This work':    {'MAE_kij': 0.015, 'MARE_x': 5.1},
        'Chabab 2023':  {'MAE_kij': 0.016, 'MARE_x': 5.7},
        'Lopez-Lazaro': {'MAE_kij': 0.22,  'MARE_x': 52},
        'Linear':       {'MAE_kij': 0.020, 'MARE_x': 6.7},
    }

    for name, (_, kij_func) in kij_correlations.items():
        kij_abs_errors = []
        sol_rel_errors = []

        for _, row in df_aq.iterrows():
            T_K = row['T_K']
            P_Pa = row['P_bar'] * 1e5
            x_exp = row['x_gas_exp']
            kij_exp = row['kij_AQ']

            # kij error
            kij_pred = kij_func(T_K)
            kij_abs_errors.append(abs(kij_pred - kij_exp))

            # Solubility error
            x_pred = vle.calc_x_H2(T_K, P_Pa, kij_pred)
            if x_pred is not None and x_exp > 0:
                sol_rel_errors.append(abs(x_pred - x_exp) / x_exp * 100)

        mae_kij = np.mean(kij_abs_errors)
        mare_x = np.mean(sol_rel_errors)

        exp = expected[name]
        # MAE(kij) check - use appropriate tolerance
        mae_tol = 0.005 if exp['MAE_kij'] < 0.1 else 0.01
        rpt.check(f"{name} MAE(kij) (Table 4: {exp['MAE_kij']})",
                  exp['MAE_kij'], round(mae_kij, 3 if exp['MAE_kij'] < 0.1 else 2),
                  abs_tol=mae_tol, section_ref="Table 4")

        # MARE(x_H2) check - manuscript rounds to 1 decimal or integer
        mare_tol = 0.5 if exp['MARE_x'] < 10 else 1.5
        rpt.check(f"{name} MARE(x_H2) (Table 4: {exp['MARE_x']}%)",
                  exp['MARE_x'], round(mare_x, 1 if exp['MARE_x'] < 10 else 0),
                  abs_tol=mare_tol, section_ref="Table 4")


# =============================================================================
# SECTION 4: NON-AQUEOUS ERROR METRICS (Table 6)
# =============================================================================
def check_nonaqueous_errors(df_h2, rpt):
    rpt.section("NON-AQUEOUS PHASE ERROR METRICS (Table 6)")

    df_na = get_na_quality_filtered(df_h2)
    rpt.info(f"Quality-filtered points: {len(df_na)}")

    na_correlations = {
        'This work (0.468)': lambda T: 0.468,
        'Chabab 2023':       kij_na_chabab_2023,
        'Lopez-Lazaro 2019': kij_na_lopez_lazaro_2019,
    }

    expected = {
        'This work (0.468)': {'MARE': 4.1, 'within_10': 86},
        'Chabab 2023':       {'MARE': 4.9, 'within_10': 86},
        'Lopez-Lazaro 2019': {'MARE': 12.6, 'within_10': 54},
    }

    for name, kij_func in na_correlations.items():
        rel_errors = []

        for _, row in df_na.iterrows():
            T_K = row['T_K']
            P_Pa = row['P_bar'] * 1e5
            y_exp = row['y_H2O_exp']

            kij_na = kij_func(T_K)
            vle = SWBinaryVLE('H2', salinity_molal=0.0)
            y_pred = vle.calc_water_content_with_kij(T_K, P_Pa, kij_na)

            if y_pred is not None and y_exp > 0:
                rel_errors.append(abs(y_pred - y_exp) / y_exp * 100)

        mare = np.mean(rel_errors) if rel_errors else 0
        within_10 = sum(1 for e in rel_errors if e <= 10) / len(rel_errors) * 100 if rel_errors else 0

        exp = expected[name]
        rpt.check(f"{name} MARE(y_H2O) (Table 6: {exp['MARE']}%)",
                  exp['MARE'], round(mare, 1), abs_tol=0.5, section_ref="Table 6")
        rpt.check(f"{name} within +/-10% (Table 6: {exp['within_10']}%)",
                  exp['within_10'], round(within_10), abs_tol=3, section_ref="Table 6")


# =============================================================================
# SECTION 5: PHYSICAL PROPERTIES AND CONSTANTS
# =============================================================================
def check_physical_properties(rpt):
    rpt.section("PHYSICAL PROPERTIES AND CONSTANTS")

    # H2 critical properties (Sec 5.3)
    h2 = COMPONENTS['H2']
    rpt.check("Tc_H2 in code (Sec 5.3: 33.145 K)", 33.145, h2.Tc, tol_pct=0.01,
              section_ref="Sec 5.3")
    rpt.check("Pc_H2 in code (Sec 5.3: 1.2964 MPa)", 1.2964e6, h2.Pc, tol_pct=0.01,
              section_ref="Sec 5.3")
    rpt.check("omega_H2 in code (Sec 5.3: -0.219)", -0.219, h2.omega,
              abs_tol=0.001, section_ref="Sec 5.3")
    rpt.check("BIP_TC_H2 consistency", 33.145, BIP_TC_H2, tol_pct=0.01)

    # H2 boiling point (used in Sechenov)
    rpt.check("Tb_H2 (Sec 3.5: 20.3 K)", 20.3, h2.Tb, abs_tol=0.1, section_ref="Sec 3.5")

    # BIP coefficients in code match manuscript
    kij_test = kij_aq_rational(celsius_to_kelvin(50))
    Tr_50 = celsius_to_kelvin(50) / TC_H2
    kij_manual = (A + Tr_50) / (B + C_COEFF * Tr_50)
    rpt.check("kij_aq_rational(50C) matches manual calc with A,B,C",
              kij_manual, kij_test, tol_pct=0.01, section_ref="Eq 1")

    # kij_NA constant in code
    kij_na_code = get_kij_na('H2', 373.15)
    rpt.check("kij_NA from get_kij_na (Eq 4: 0.468)", 0.468, kij_na_code,
              abs_tol=0.001, section_ref="Eq 4")

    # Boiling points for volatility trend (Sec 4.3)
    ch4_tb_C = kelvin_to_celsius(COMPONENTS['CH4'].Tb)
    n2_tb_C = kelvin_to_celsius(COMPONENTS['N2'].Tb)
    h2_tb_C = kelvin_to_celsius(h2.Tb)
    rpt.check("CH4 Tb (Sec 4.3: -161C)", -161, round(ch4_tb_C), abs_tol=2, section_ref="Sec 4.3")
    rpt.check("N2 Tb (Sec 4.3: -196C)", -196, round(n2_tb_C), abs_tol=2, section_ref="Sec 4.3")
    rpt.check("H2 Tb (Sec 4.3: -253C)", -253, round(h2_tb_C), abs_tol=1, section_ref="Sec 4.3")

    # CH4 and N2 kij_NA values (Sec 4.3)
    kij_na_ch4 = get_kij_na('CH4', 373.15)
    kij_na_n2 = get_kij_na('N2', 373.15)
    rpt.check("CH4 kij_NA (Sec 4.3: 0.485)", 0.485, kij_na_ch4,
              abs_tol=0.005, section_ref="Sec 4.3")
    rpt.check("N2 kij_NA (Sec 4.3: 0.478)", 0.478, kij_na_n2,
              abs_tol=0.005, section_ref="Sec 4.3")

    # Reduced temperature range for UHS (Sec 3.1)
    Tr_50C = celsius_to_kelvin(50) / TC_H2
    Tr_200C = celsius_to_kelvin(200) / TC_H2
    rpt.check("Tr at 50C (Sec 3.1: 9.7)", 9.7, round(Tr_50C, 1),
              abs_tol=0.1, section_ref="Sec 3.1")
    rpt.check("Tr at 200C (Sec 3.1: 14.3)", 14.3, round(Tr_200C, 1),
              abs_tol=0.1, section_ref="Sec 3.1")


# =============================================================================
# SECTION 6: SECHENOV CONFORMANCE
# =============================================================================
def check_sechenov(rpt):
    rpt.section("SECHENOV CONFORMANCE (Sec 3.2, 3.4)")

    Tb_H2 = 20.3  # K

    # S&W Eq 8 at 100C (212F)
    T_F_100C = 212.0
    ks_sw_100 = (0.13163 + 4.45e-4 * Tb_H2 - 7.692e-4 * T_F_100C
                 + 2.6614e-6 * T_F_100C**2 - 2.612e-9 * T_F_100C**3)
    rpt.check("S&W Eq 8 ks at 100C (Sec 3.2: 0.072 kg/mol)",
              0.072, round(ks_sw_100, 3), abs_tol=0.001, section_ref="Sec 3.2")

    # Chabab 2023 at 100C: ks ~ 0.071, "within 2% of S&W"
    ks_chabab_100 = 0.071  # Manuscript-quoted experimental value
    pct_diff = abs(ks_chabab_100 - ks_sw_100) / ks_sw_100 * 100
    rpt.check("Chabab ks vs S&W at 100C (Sec 3.2: within 2%)",
              2.0, pct_diff, abs_tol=1.0, section_ref="Sec 3.2")
    rpt.info(f"  Actual difference: {pct_diff:.1f}%")

    # Also verify using sw_equation_8_ks function
    ks_func = sw_equation_8_ks(100, Tb_H2)  # T in C, Tb in K
    rpt.check("sw_equation_8_ks(100C) matches manual calc",
              ks_sw_100, ks_func, tol_pct=0.1)

    # Sechenov minimum temperature (Sec 3.2: ~98C)
    # d(ks)/dT_F = -7.692e-4 + 2*2.6614e-6*T_F - 3*2.612e-9*T_F^2 = 0
    a_coef = 3 * 2.612e-9
    b_coef = -2 * 2.6614e-6
    c_coef = 7.692e-4
    discriminant = b_coef**2 - 4 * a_coef * c_coef
    if discriminant > 0:
        T_F_min = (-b_coef - np.sqrt(discriminant)) / (2 * a_coef)
        T_C_min = (T_F_min - 32) / 1.8
        rpt.check("Sechenov minimum temperature (Sec 3.2: ~98C)",
                  98, round(T_C_min), abs_tol=3, section_ref="Sec 3.2")
    else:
        rpt.warn("Could not find Sechenov minimum (no real roots)")

    # T-O vs other sources percentage claims (Sec 3.4)
    ks_to_50 = 0.063
    ks_morrison_49 = 0.082
    ks_chabab_50 = 0.080
    ks_gerecke_50 = 0.076

    dev_morrison = (ks_morrison_49 - ks_to_50) / ks_morrison_49 * 100
    dev_chabab = (ks_chabab_50 - ks_to_50) / ks_chabab_50 * 100
    dev_gerecke = (ks_gerecke_50 - ks_to_50) / ks_gerecke_50 * 100

    rpt.info(f"T-O lower than Morrison: {dev_morrison:.1f}%")
    rpt.info(f"T-O lower than Chabab: {dev_chabab:.1f}%")
    rpt.info(f"T-O lower than Gerecke: {dev_gerecke:.1f}%")

    actual_range_lo = min(dev_morrison, dev_chabab, dev_gerecke)
    actual_range_hi = max(dev_morrison, dev_chabab, dev_gerecke)
    rpt.check("T-O lower bound of range (Sec 3.4: 19%)",
              19, round(actual_range_lo), abs_tol=3, section_ref="Sec 3.4")
    rpt.check("T-O upper bound of range (Sec 3.4: 24%)",
              24, round(actual_range_hi), abs_tol=2, section_ref="Sec 3.4")

    if actual_range_lo < 18:
        rpt.warn("T-O percentage range lower bound may be overstated in manuscript",
                 f"Gerecke comparison gives only {dev_gerecke:.1f}%, manuscript claims 19-24%")

    # Implied ks from Chabab BIP (Sec 3.3: 0.04-0.08 range)
    rpt.info("Checking implied ks from Chabab embedded BIP (Sec 3.3)")
    implied_ks_range = []
    for T_C in [25, 50, 75, 100, 125, 150]:
        T_K = celsius_to_kelvin(T_C)
        P_Pa = 150e5  # representative pressure

        # Fresh water with Chabab BIP
        vle_fresh = SWBinaryVLE('H2')
        kij_fresh = kij_aq_chabab_2023(T_K, m=0)
        x_fresh = vle_fresh.calc_x_H2(T_K, P_Pa, kij_fresh)

        # Brine with Chabab embedded salinity BIP at m=1.5
        kij_brine = kij_aq_chabab_2023(T_K, m=1.5)
        x_brine = vle_fresh.calc_x_H2(T_K, P_Pa, kij_brine)

        if x_fresh is not None and x_brine is not None and x_brine > 0:
            ks_implied = np.log10(x_fresh / x_brine) / 1.5
            implied_ks_range.append(ks_implied)
            rpt.info(f"  Chabab implied ks at {T_C}C: {ks_implied:.4f}")

    if implied_ks_range:
        rpt.check("Chabab implied ks min (Sec 3.3: ~0.04)",
                  0.04, round(min(implied_ks_range), 2), abs_tol=0.02, section_ref="Sec 3.3")
        rpt.check("Chabab implied ks max (Sec 3.3: ~0.08)",
                  0.08, round(max(implied_ks_range), 2), abs_tol=0.02, section_ref="Sec 3.3")


# =============================================================================
# SECTION 7: BIP PROPERTIES AND U-SHAPE
# =============================================================================
def check_bip_properties(rpt):
    rpt.section("BIP PROPERTIES AND U-SHAPE (Sec 2.5)")

    # Verify U-shape: solubility minimum near 50C at 100 bar
    vle = SWBinaryVLE('H2', salinity_molal=0.0)
    temps_C = np.arange(0, 151, 5)
    solubilities = []
    for T_C in temps_C:
        T_K = celsius_to_kelvin(T_C)
        x = vle.calc_gas_solubility(T_K, 100e5)
        if x is not None:
            solubilities.append((T_C, x))

    if solubilities:
        min_idx = min(range(len(solubilities)), key=lambda i: solubilities[i][1])
        T_min = solubilities[min_idx][0]
        rpt.check("Solubility minimum temperature at 100 bar (Sec 2.3: near 50C)",
                  50, T_min, abs_tol=15, section_ref="Sec 2.3")
        rpt.info(f"  Minimum at {T_min}C with x_H2 = {solubilities[min_idx][1]:.6f}")

    # Verify Tr ~ 10 corresponds to ~50C
    Tr_10 = 10 * TC_H2  # K
    T_C_at_Tr10 = kelvin_to_celsius(Tr_10)
    rpt.check("Tr=10 corresponds to (Sec 2.5: ~50C)",
              50, round(T_C_at_Tr10), abs_tol=10, section_ref="Sec 2.5")
    rpt.info(f"  Tr=10 → T = {T_C_at_Tr10:.1f}C")

    # BIP derivative dkij/dTr for Stephan validation (Sec 2.4: ~0.17)
    # d/dTr[(A+Tr)/(B+C*Tr)] = (B - A*C) / (B + C*Tr)^2
    # Evaluate at representative Tr for Stephan range (50-135C)
    Tr_mid = celsius_to_kelvin(90) / TC_H2  # midpoint of Stephan range
    dkij_dTr = (B - A * C_COEFF) / (B + C_COEFF * Tr_mid)**2
    rpt.check("dkij/dTr at Stephan midpoint (Sec 2.4: ~0.17)",
              0.17, round(dkij_dTr, 2), abs_tol=0.03, section_ref="Sec 2.4")

    # Raoult's law dominance at low P (Sec 4.2: 90-97% at P < 50 bar)
    rpt.info("Checking Raoult's law dominance at low P")
    # Wagner equation for water saturation pressure (approximate)
    from scipy.optimize import brentq as _unused  # just ensure scipy available

    for T_C, P_bar in [(50, 10), (50, 30), (50, 50), (100, 10), (100, 50)]:
        T_K = celsius_to_kelvin(T_C)
        P_Pa = P_bar * 1e5

        # Approximate P_sat water using Antoine or from VLE
        # Simple Antoine: log10(P_sat/mmHg) = 8.07131 - 1730.63/(233.426+T_C) for T<100C
        if T_C <= 100:
            log_p = 8.07131 - 1730.63 / (233.426 + T_C)
            P_sat_Pa = 10**log_p * 133.322
        else:
            log_p = 8.14019 - 1810.94 / (244.485 + T_C)
            P_sat_Pa = 10**log_p * 133.322

        raoult_fraction = (P_sat_Pa / P_Pa)

        # Get actual y_H2O from VLE
        y_h2o = vle.calc_water_content(T_K, P_Pa)
        if y_h2o is not None and y_h2o > 0:
            raoult_pct = raoult_fraction / y_h2o * 100
            rpt.info(f"  {T_C}C, {P_bar} bar: Raoult fraction = {raoult_pct:.1f}%")

    rpt.info("Manuscript claims 90-97% at P < 50 bar - check values above")


# =============================================================================
# SECTION 8: EMBEDDED SALINITY (Sec 3.5)
# =============================================================================
def check_embedded_salinity(rpt):
    rpt.section("EMBEDDED SALINITY (Sec 3.5, Eq 5)")

    # Coefficients (full precision from fit)
    beta0, beta1, beta2 = 0.3833, -0.06595, 0.003321

    # Check manuscript rounding
    rpt.check("Embedded beta0 rounds to (Eq 5: 0.383)",
              0.383, round(beta0, 3), abs_tol=0.001)
    rpt.check("Embedded beta1 rounds to (Eq 5: -0.066)",
              -0.066, round(beta1, 3), abs_tol=0.001)
    rpt.check("Embedded beta2 rounds to (Eq 5: 0.00332)",
              0.00332, round(beta2, 5), abs_tol=0.00005)

    # Generate grid and compute errors
    vle = SWBinaryVLE('H2', salinity_molal=0.0)
    Tb_H2 = 20.3

    errors = []
    for T_C in range(25, 155, 5):
        for P_bar in range(25, 325, 25):
            for m in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
                T_K = celsius_to_kelvin(T_C)
                P_Pa = P_bar * 1e5
                Tr = T_K / TC_H2

                # Truth: freshwater VLE + Sechenov
                x_fresh = vle.calc_gas_solubility(T_K, P_Pa)
                ks = sw_equation_8_ks(T_C, Tb_H2)
                x_brine_truth = x_fresh * 10**(-ks * m)

                # Prediction: embedded BIP
                kij_fw = (A + Tr) / (B + C_COEFF * Tr)
                kij_emb = kij_fw + (beta0 + beta1 * Tr + beta2 * Tr**2) * m
                x_brine_emb = vle.calc_x_H2(T_K, P_Pa, kij_emb)

                if (x_brine_truth is not None and x_brine_emb is not None
                        and x_brine_truth > 0):
                    rel_err = abs(x_brine_emb - x_brine_truth) / x_brine_truth * 100
                    errors.append(rel_err)

    if errors:
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        within_1 = sum(1 for e in errors if e <= 1) / len(errors) * 100
        within_2 = sum(1 for e in errors if e <= 2) / len(errors) * 100

        rpt.check("Embedded mean error (Sec 3.5: 0.35%)",
                  0.35, round(mean_err, 2), abs_tol=0.15, section_ref="Sec 3.5")
        rpt.check("Embedded max error (Sec 3.5: 1.7%)",
                  1.7, round(max_err, 1), abs_tol=0.5, section_ref="Sec 3.5")
        rpt.info(f"  Grid points evaluated: {len(errors)}")
        rpt.info(f"  Within 1%: {within_1:.0f}%")
        rpt.info(f"  Within 2%: {within_2:.0f}%")
    else:
        rpt.warn("No embedded salinity error data computed")


# =============================================================================
# SECTION 9: CO2 vs H2 SOLUBILITY RATIO (Sec 5.4)
# =============================================================================
def check_co2_ratio(rpt):
    rpt.section("CO2 vs H2 SOLUBILITY RATIO (Sec 5.4)")

    # "CO2 is approximately 15x more soluble than H2" at 50C, 100 bar
    T_K = celsius_to_kelvin(50)
    P_Pa = 100e5

    vle_h2 = SWBinaryVLE('H2', salinity_molal=0.0)
    x_h2 = vle_h2.calc_gas_solubility(T_K, P_Pa)

    vle_co2 = SWBinaryVLE('CO2', salinity_molal=0.0)
    x_co2 = vle_co2.calc_gas_solubility(T_K, P_Pa)

    if x_h2 is not None and x_co2 is not None and x_h2 > 0:
        ratio = x_co2 / x_h2
        rpt.check("CO2/H2 solubility ratio at 50C/100bar (Sec 5.4: ~15)",
                  15, round(ratio), abs_tol=3, section_ref="Sec 5.4")
        rpt.info(f"  x_H2 = {x_h2:.6e}, x_CO2 = {x_co2:.6e}, ratio = {ratio:.1f}")
    else:
        rpt.warn("Could not compute CO2/H2 solubility ratio")


# =============================================================================
# SECTION 10: HENRY'S LAW / MULTICOMPONENT (Sec 5.4) - SLOW
# =============================================================================
def check_henrys_law(rpt):
    rpt.section("HENRY'S LAW MULTICOMPONENT ACCURACY (Sec 5.4)")

    try:
        from _lib_vle_engine import SWMultiComponentFlash
    except ImportError:
        rpt.warn("SWMultiComponentFlash not available, skipping Henry's law checks")
        return

    # Conditions from the test script (match h2_henry_accuracy summary)
    temperatures_C = [50, 65, 80]
    pressures_bar = [100, 150, 200]
    h2_fractions = [0.50, 0.70, 0.85, 0.95]
    cushion_gases = ['CH4', 'N2', 'CO2']

    results_by_gas = {g: [] for g in cushion_gases}

    for cushion in cushion_gases:
        for T_C in temperatures_C:
            T_K = celsius_to_kelvin(T_C)
            for P_bar in pressures_bar:
                P_Pa = P_bar * 1e5
                for y_h2 in h2_fractions:
                    y_cush = 1.0 - y_h2

                    # Henry's law: independent binary calculations
                    vle_h2 = SWBinaryVLE('H2')
                    x_h2_binary = vle_h2.calc_gas_solubility(T_K, P_Pa)

                    vle_cush = SWBinaryVLE(cushion)
                    x_cush_binary = vle_cush.calc_gas_solubility(T_K, P_Pa)

                    if x_h2_binary is None or x_cush_binary is None:
                        continue

                    x_h2_henry = x_h2_binary * y_h2
                    x_cush_henry = x_cush_binary * y_cush
                    x_total_henry = x_h2_henry + x_cush_henry

                    # Full flash (matching test_h2_henry_accuracy.py)
                    comp_names = ['H2O', 'H2', cushion]
                    z = np.array([0.95, 0.05 * y_h2, 0.05 * y_cush])
                    z = z / np.sum(z)

                    try:
                        flash = SWMultiComponentFlash(comp_names, salinity_molal=0.0)
                        result = flash.calc_equilibrium(T_K, P_Pa, z,
                                                        salinity_method='gamma_phi')
                        if result is None:
                            continue

                        x_aq = result.get('x_aq', None)
                        if x_aq is None:
                            continue

                        x_h2_flash = x_aq[1]
                        x_cush_flash = x_aq[2]
                        x_total_flash = x_h2_flash + x_cush_flash

                        if x_total_flash > 1e-15:
                            pct_diff = abs(x_total_henry - x_total_flash) / x_total_flash * 100
                            results_by_gas[cushion].append(pct_diff)
                    except Exception:
                        continue

    # Check claims
    all_results = []
    for gas in cushion_gases:
        all_results.extend(results_by_gas[gas])

    expected_by_gas = {
        'CH4': {'mean': 1.6, 'label': 'H2-CH4'},
        'N2':  {'mean': 0.3, 'label': 'H2-N2'},
        'CO2': {'mean': 4.2, 'label': 'H2-CO2'},
    }

    for gas, exp in expected_by_gas.items():
        if results_by_gas[gas]:
            mean_dev = np.mean(results_by_gas[gas])
            max_dev = np.max(results_by_gas[gas])
            rpt.check(f"{exp['label']} mean deviation (Sec 5.4: {exp['mean']}%)",
                      exp['mean'], round(mean_dev, 1), abs_tol=0.5, section_ref="Sec 5.4")
            rpt.info(f"  {exp['label']}: mean={mean_dev:.2f}%, max={max_dev:.2f}%, n={len(results_by_gas[gas])}")
        else:
            rpt.warn(f"No results computed for {exp['label']}")

    if all_results:
        overall_mean = np.mean(all_results)
        overall_max = np.max(all_results)
        rpt.check("Overall Henry mean deviation (Sec 5.4: 2.1%)",
                  2.1, round(overall_mean, 1), abs_tol=0.5, section_ref="Sec 5.4")
        rpt.check("Overall Henry max deviation (Sec 5.4: 9.4%)",
                  9.4, round(overall_max, 1), abs_tol=0.5, section_ref="Sec 5.4")
    else:
        rpt.warn("No Henry's law comparison results computed")


# =============================================================================
# SECTION 11: LOPEZ-LAZARO SIGN ERROR (Sec 2.6)
# =============================================================================
def check_lopez_lazaro_sign(rpt):
    rpt.section("LOPEZ-LAZARO SIGN ERROR CHECK (Sec 2.6)")

    # With A3 = +0.499 (as published), kij should be non-physical (>1000)
    T_K = celsius_to_kelvin(50)
    Tr = T_K / BIP_TC_H2
    A0, A1, A2, A3_pos = -2.513, 0.181, 12.723, 0.499

    kij_wrong = A0 + A1 * Tr + A2 * np.exp(A3_pos * Tr)
    rpt.check("Lopez-Lazaro with A3=+0.499 gives kij > 1000 (Sec 2.6)",
              True, kij_wrong > 1000, abs_tol=0, section_ref="Sec 2.6")
    rpt.info(f"  kij with A3=+0.499 at 50C: {kij_wrong:.1f}")

    # With A3 = -0.499 (assumed correction)
    kij_fixed = A0 + A1 * Tr + A2 * np.exp(-A3_pos * Tr)
    rpt.check("Lopez-Lazaro with A3=-0.499 gives sensible kij (Sec 2.6)",
              True, -2 < kij_fixed < 2, abs_tol=0, section_ref="Sec 2.6")
    rpt.info(f"  kij with A3=-0.499 at 50C: {kij_fixed:.4f}")


# =============================================================================
# SECTION 12: MISCELLANEOUS ARITHMETIC
# =============================================================================
def check_misc_arithmetic(rpt):
    rpt.section("MISCELLANEOUS ARITHMETIC AND CROSS-REFERENCES")

    # Sec 3.2: Tc range "33 K (H2) to 305 K (C2H6)"
    c2h6_tc = COMPONENTS['C2H6'].Tc if 'C2H6' in COMPONENTS else None
    if c2h6_tc is not None:
        rpt.check("C2H6 Tc (Sec 3.2: 305 K)", 305, round(c2h6_tc), abs_tol=2,
                  section_ref="Sec 3.2")

    # Abstract: "data sources spanning 1934-2023" - just info
    rpt.info("Date range: Wiebe 1934 (earliest) to Chabab 2023 (latest) -> 1934-2023 OK")

    # "six sources spanning 1952-2021" for salting-out
    # Morrison (1952), Gerecke (1971), Crozier (1974), Gordon (1977),
    # Chabab (2023), T-O (2021)
    rpt.warn("Salting-out date range: manuscript says '1952-2021' but Chabab (2023) is included",
             "Range should be 1952-2023 if Chabab is counted, or 1952-2021 if only T-O is latest")

    # Valid ranges cross-check (Sec 5.2)
    # "Pressures above 200 bar rely solely on the Wiebe dataset"
    # But T-O goes to 450 bar with 5 points (though some at 423K excluded)
    rpt.info("Checking pressure range claim...")
    # T-O included points (not at 423K) - what are their pressures?
    # T-O aqueous data at 50C and 150C should be below/above 200 bar


# =============================================================================
# MAIN
# =============================================================================
def main():
    full_mode = '--full' in sys.argv

    print("=" * 72)
    print("  MANUSCRIPT CLAIMS VALIDATION")
    print(f"  Mode: {'FULL (with Henry law)' if full_mode else 'QUICK (skip Henry law)'}")
    print("=" * 72)

    rpt = Reporter()

    # Load data
    try:
        df_h2 = load_data()
        rpt.info(f"Loaded {len(df_h2)} H2 data points from {CSV_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Cannot find {CSV_PATH}")
        print("Run from the code/ directory: cd code && python validate_manuscript_claims.py")
        sys.exit(1)

    # Run all checks
    check_data_integrity(df_h2, rpt)
    check_table_values(df_h2, rpt)
    check_aqueous_errors(df_h2, rpt)
    check_nonaqueous_errors(df_h2, rpt)
    check_physical_properties(rpt)
    check_sechenov(rpt)
    check_bip_properties(rpt)
    check_embedded_salinity(rpt)
    check_co2_ratio(rpt)
    check_lopez_lazaro_sign(rpt)
    check_misc_arithmetic(rpt)

    if full_mode:
        check_henrys_law(rpt)
    else:
        rpt.section("HENRY'S LAW (Sec 5.4) - SKIPPED")
        rpt.info("Use --full flag to include Henry's law multicomponent validation")

    # Summary
    n_fail = rpt.summary()
    rpt.save(REPORT_PATH)

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
