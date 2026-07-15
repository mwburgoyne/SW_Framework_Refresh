#!/usr/bin/env python3
"""
Fit Track 2 (drop-in) kij_AQ(T) correlations from S&W-alpha pointwise data.

Track 2 serves users with existing S&W implementations who cannot change
their alpha function. Pointwise kij values were regressed using S&W original
alpha (framework='sw_original') and are stored in pointwise_kij_results_sw_alpha.csv.

Same functional forms as Track 1 (proposed):
  CO2  : cubic in T(K)
  H2S  : exponential
  CH4  : rational (A + Tr)/(B + C*Tr)
  N2   : linear in T(K)
  H2   : rational
  C2H6 : rational
  C3H8 : rational

All fits use L1 (MAE-minimizing) regression via Nelder-Mead.

Usage:
    cd framework_refresh/code
    python 02a-Fit_Dropin_Kij.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from _lib_vle_engine import COMPONENTS

# =============================================================================
# Configuration
# =============================================================================
CSV_PATH = '../../shared/data/pointwise_kij_results_sw_alpha.csv'

# Same exclusions as Track 1 (01-Regress_Pointwise_Kij.py, 12-Generate_Figures.py)
EXCLUDE_SOURCES = {
    'CH4': {'Blount 1982', 'McGee 1981'},
    'CO2': {'Prutton & Savage 1945'},
    'H2S': {'Barrett 1988'},
}

T_MIN_K = 273.15  # 0 deg C
T_MAX_K = 473.15  # 200 deg C

# CH4 T limit (same as Track 1: only fit T <= 200C)
CH4_T_MAX_K = 473.15


# =============================================================================
# Functional Forms
# =============================================================================
def cubic_form(T_K, c0, c1, c2, c3):
    """kij = c0 + c1*T + c2*T^2 + c3*T^3"""
    return c0 + c1*T_K + c2*T_K**2 + c3*T_K**3

def exp_form(T_K, a, b, c):
    """kij = a/T + b*exp(c/T) + d"""
    # Parameterized as: a/T + b*exp(c/T) + d
    # But we need 4 params for the full exp form
    pass

def exp_form_4p(T_K, a, b, c, d):
    """kij = a/T + b*exp(c/T) + d"""
    return a/T_K + b*np.exp(c/T_K) + d

def rational_form(T_K, Tc, A, B, C):
    """kij = (A + Tr) / (B + C*Tr) where Tr = T/Tc"""
    Tr = T_K / Tc
    return (A + Tr) / (B + C * Tr)

def linear_form(T_K, a, b):
    """kij = a + b*T"""
    return a + b*T_K


# =============================================================================
# L1 Fitting Functions
# =============================================================================
def fit_l1_cubic(T, kij, x0=None):
    """Fit cubic form minimizing MAE."""
    if x0 is None:
        x0 = [-1.5, 1e-2, -2e-5, 2e-8]
    def obj(params):
        pred = cubic_form(T, *params)
        return np.mean(np.abs(pred - kij))
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 200000, 'xatol': 1e-14, 'fatol': 1e-14})
    return res.x, res.fun

def fit_l1_exp(T, kij, x0=None):
    """Fit exp form minimizing MAE."""
    if x0 is None:
        x0 = [-70.0, 1500.0, -4500.0, 0.2]
    def obj(params):
        a, b, c, d = params
        pred = a/T + b*np.exp(c/T) + d
        if not np.all(np.isfinite(pred)):
            return 1e10
        return np.mean(np.abs(pred - kij))
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 200000, 'xatol': 1e-14, 'fatol': 1e-14})
    return res.x, res.fun

def fit_l1_rational(T, kij, Tc, x0=None):
    """Fit rational form minimizing MAE."""
    Tr = T / Tc
    if x0 is None:
        x0 = [-5.0, 2.0, 0.5]
    def obj(params):
        A, B, C = params
        denom = B + C * Tr
        if np.any(np.abs(denom) < 1e-10):
            return 1e10
        pred = (A + Tr) / denom
        return np.mean(np.abs(pred - kij))
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 200000, 'xatol': 1e-14, 'fatol': 1e-14})
    return res.x, res.fun

def fit_l1_linear(T, kij, x0=None):
    """Fit linear form minimizing MAE."""
    if x0 is None:
        x0 = [-1.5, 3e-3]
    def obj(params):
        pred = linear_form(T, *params)
        return np.mean(np.abs(pred - kij))
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 200000, 'xatol': 1e-14, 'fatol': 1e-14})
    return res.x, res.fun


# =============================================================================
# Data Loading
# =============================================================================
def load_and_filter(csv_path, gas, T_max_K=None):
    """Load pointwise kij data for a gas, applying exclusions and T range filters."""
    df = pd.read_csv(csv_path)

    # Normalize source names
    if 'Source' in df.columns:
        df['Source'] = df['Source'].str.replace('Chahab', 'Chabab', regex=False)

    # Filter to gas, converged kij_AQ, freshwater
    mask = ((df['Gas'] == gas)
            & (df['kij_AQ'].notna())
            & (df['kij_AQ_conv'] == True)
            & (df['T_K'] >= T_MIN_K))

    if T_max_K is not None:
        mask = mask & (df['T_K'] <= T_max_K)
    else:
        mask = mask & (df['T_K'] <= T_MAX_K)

    data = df[mask].copy()

    # Apply source exclusions
    excl = EXCLUDE_SOURCES.get(gas, set())
    if excl and 'Source' in data.columns:
        data = data[~data['Source'].isin(excl)]

    return data


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("TRACK 2 (DROP-IN) kij_AQ CORRELATION FITTING")
    print("S&W original alpha, L1-optimal regression")
    print("=" * 70)

    if not os.path.exists(CSV_PATH):
        print(f"\nERROR: {CSV_PATH} not found.")
        print("Run 01-Regress_Pointwise_Kij.py with FRAMEWORK='sw_original' first.")
        sys.exit(1)

    # Gas-specific fitting configurations
    # Each entry: (gas, form_name, fit_function, extra_args, x0)
    GAS_CONFIGS = [
        ('CO2',  'cubic',    'cubic',    {},                    [-1.5, 1e-2, -2e-5, 2e-8]),
        ('H2S',  'exp',      'exp',      {},                    [-70.0, 1500.0, -4500.0, 0.2]),
        ('CH4',  'rational', 'rational', {'Tc': 190.60},        [-2.0, 1.7, 0.2]),
        ('N2',   'linear',   'linear',   {},                    [-1.5, 3e-3]),
        ('H2',   'rational', 'rational', {'Tc': 33.145},        [-14.0, 3.5, 0.2]),
        ('C2H6', 'rational', 'rational', {'Tc': 305.40},        [-1.3, 0.4, 1.3]),
        ('C3H8', 'rational', 'rational', {'Tc': 369.80},        [-1.1, 0.6, 1.3]),
    ]

    results = {}
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TRACK 2 (DROP-IN) kij_AQ CORRELATIONS")
    report_lines.append("Framework: S&W original alpha")
    report_lines.append("Regression: L1-optimal (MAE-minimizing)")
    report_lines.append("=" * 70)
    report_lines.append("")

    for gas, form_name, fit_type, extra_args, x0 in GAS_CONFIGS:
        T_max = CH4_T_MAX_K if gas == 'CH4' else T_MAX_K
        data = load_and_filter(CSV_PATH, gas, T_max_K=T_max)

        if len(data) < 5:
            print(f"\n{gas}: only {len(data)} points — skipping.")
            continue

        T = data['T_K'].values
        kij = data['kij_AQ'].values

        print(f"\n--- {gas} ({len(data)} points, {form_name}) ---")

        if fit_type == 'cubic':
            params, mae = fit_l1_cubic(T, kij, x0)
            c0, c1, c2, c3 = params
            kij_pred = cubic_form(T, c0, c1, c2, c3)
            param_str = (f"  kij = {c0:.4f} + {c1:.6e}*T + {c2:.6e}*T^2 + {c3:.6e}*T^3")
            code_str = (f"  return {c0:.4f} + {c1:.4e}*T_K + {c2:.4e}*T_K**2 + {c3:.4e}*T_K**3")
            results[gas] = {'form': 'cubic', 'params': params, 'mae': mae, 'n': len(data)}

        elif fit_type == 'exp':
            params, mae = fit_l1_exp(T, kij, x0)
            a, b, c, d = params
            kij_pred = a/T + b*np.exp(c/T) + d
            param_str = (f"  kij = {a:.4f}/T + {b:.4f}*exp({c:.2f}/T) + {d:.5f}")
            code_str = (f"  return {a:.4f}/T_K + {b:.4f}*np.exp({c:.2f}/T_K) + {d:.5f}")
            results[gas] = {'form': 'exp', 'params': params, 'mae': mae, 'n': len(data)}

        elif fit_type == 'rational':
            Tc = extra_args['Tc']
            params, mae = fit_l1_rational(T, kij, Tc, x0)
            A, B, C = params
            Tr = T / Tc
            kij_pred = (A + Tr) / (B + C * Tr)
            param_str = (f"  kij = ({A:.4f} + Tr) / ({B:.4f} + {C:.4f}*Tr)  "
                        f"[Tr = T/{Tc:.2f}]")
            code_str = (f"  Tr = T_K / {Tc:.2f}\n"
                       f"  return ({A:.4f} + Tr) / ({B:.4f} + {C:.4f} * Tr)")
            results[gas] = {'form': 'rational', 'params': params, 'Tc': Tc,
                          'mae': mae, 'n': len(data)}

        elif fit_type == 'linear':
            params, mae = fit_l1_linear(T, kij, x0)
            a, b = params
            kij_pred = linear_form(T, a, b)
            param_str = f"  kij = {a:.4f} + {b:.6e}*T"
            code_str = f"  return {a:.4f} + {b:.6e} * T_K"
            results[gas] = {'form': 'linear', 'params': params, 'mae': mae, 'n': len(data)}

        residuals = kij - kij_pred
        max_err = np.max(np.abs(residuals))

        print(f"  n = {len(data)}, MAE = {mae:.4f}, Max error = {max_err:.4f}")
        print(f"  T range: {T.min():.1f} - {T.max():.1f} K")
        print(param_str)

        report_lines.append(f"{gas} ({form_name}, n={len(data)}, T={T.min():.0f}-{T.max():.0f} K)")
        report_lines.append(f"  MAE = {mae:.4f}, Max error = {max_err:.4f}")
        report_lines.append(param_str)
        report_lines.append(f"  Python: {code_str}")
        report_lines.append("")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Gas':>6} {'Form':>10} {'n':>5} {'MAE':>8} {'Params'}")
    print("-" * 70)

    summary_lines = []
    summary_lines.append("")
    summary_lines.append("SUMMARY TABLE")
    summary_lines.append("-" * 70)
    summary_lines.append(f"{'Gas':>6} {'Form':>10} {'n':>5} {'MAE':>8}")
    summary_lines.append("-" * 70)

    for gas in ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8']:
        if gas not in results:
            continue
        r = results[gas]
        if r['form'] == 'cubic':
            p_str = f"c0={r['params'][0]:.4f}"
        elif r['form'] == 'exp':
            p_str = f"a={r['params'][0]:.1f}, b={r['params'][1]:.1f}"
        elif r['form'] == 'rational':
            p_str = f"A={r['params'][0]:.4f}, B={r['params'][1]:.4f}, C={r['params'][2]:.4f}"
        elif r['form'] == 'linear':
            p_str = f"a={r['params'][0]:.4f}, b={r['params'][1]:.6e}"
        else:
            p_str = ""
        print(f"{gas:>6} {r['form']:>10} {r['n']:>5} {r['mae']:>8.4f} {p_str}")
        summary_lines.append(f"{gas:>6} {r['form']:>10} {r['n']:>5} {r['mae']:>8.4f}")

    report_lines.extend(summary_lines)

    # Python code block for _lib_vle_engine.py
    print("\n" + "=" * 70)
    print("PYTHON CODE FOR _lib_vle_engine.py")
    print("=" * 70)

    code_lines = []
    code_lines.append("")
    code_lines.append("# --- Drop-in kij_AQ (Track 2, S&W original alpha) ---")

    for gas in ['CO2', 'H2S', 'N2', 'H2', 'CH4', 'C2H6', 'C3H8']:
        if gas not in results:
            continue
        r = results[gas]
        fn_name = f"kij_aq_{gas.lower()}_dropin"

        if r['form'] == 'cubic':
            c0, c1, c2, c3 = r['params']
            code_lines.append(f"def {fn_name}(T_K: float, salinity_molal: float = 0.0) -> float:")
            code_lines.append(f'    """{gas}: cubic in T(K), n={r["n"]}, MAE={r["mae"]:.4f} (S&W alpha)."""')
            code_lines.append(f"    return {c0:.4f} + {c1:.4e}*T_K + {c2:.4e}*T_K**2 + {c3:.4e}*T_K**3")
            code_lines.append("")

        elif r['form'] == 'exp':
            a, b, c, d = r['params']
            code_lines.append(f"def {fn_name}(T_K: float, salinity_molal: float = 0.0) -> float:")
            code_lines.append(f'    """{gas}: exp form, n={r["n"]}, MAE={r["mae"]:.4f} (S&W alpha)."""')
            code_lines.append(f"    return {a:.4f}/T_K + {b:.4f}*np.exp({c:.2f}/T_K) + {d:.5f}")
            code_lines.append("")

        elif r['form'] == 'rational':
            A, B, C = r['params']
            Tc = r['Tc']
            code_lines.append(f"def {fn_name}(T_K: float, salinity_molal: float = 0.0) -> float:")
            code_lines.append(f'    """{gas}: rational, n={r["n"]}, MAE={r["mae"]:.4f} (S&W alpha)."""')
            code_lines.append(f"    Tr = T_K / {Tc:.2f}")
            code_lines.append(f"    return ({A:.4f} + Tr) / ({B:.4f} + {C:.4f} * Tr)")
            code_lines.append("")

        elif r['form'] == 'linear':
            a, b = r['params']
            code_lines.append(f"def {fn_name}(T_K: float, salinity_molal: float = 0.0) -> float:")
            code_lines.append(f'    """{gas}: linear in T(K), n={r["n"]}, MAE={r["mae"]:.4f} (S&W alpha)."""')
            code_lines.append(f"    return {a:.4f} + {b:.6e} * T_K")
            code_lines.append("")

    # Dispatch dict
    code_lines.append("KIJ_AQ_DROPIN: Dict[str, Callable] = {")
    for gas in ['H2', 'CO2', 'N2', 'H2S', 'CH4', 'C2H6', 'C3H8']:
        if gas in results:
            code_lines.append(f"    '{gas}': kij_aq_{gas.lower()}_dropin,")
    # Heavy HCs use S&W Eqs 11-12 at cs=0
    for hc in ['iC4H10', 'nC4H10', 'iC5H12', 'nC5H12', 'nC6H14', 'nC7H16', 'nC8H18', 'nC10H22']:
        code_lines.append(f"    '{hc}': _kij_aq_hc_proposed('{hc}'),")
    code_lines.append("}")

    for line in code_lines:
        print(line)

    report_lines.append("")
    report_lines.append("PYTHON CODE (copy to _lib_vle_engine.py)")
    report_lines.extend(code_lines)

    # Save report
    report_path = '../../shared/data/dropin_kij_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")
