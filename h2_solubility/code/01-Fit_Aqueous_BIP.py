"""
kij_AQ Regression Report
========================
Fits rational form to point-regressed kij_AQ values and outputs text report.

Reports both:
- MAE/MARE against kij values (fit quality)
- MARE against x_H2 predictions (practical accuracy)

Filters: Fitting sources only (Wiebe, Chabab 2023, T-O excl. 423K)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from datetime import datetime

# Import validated VLE for x_H2 calculations
from _lib_vle_engine import H2WaterVLE, COMPONENTS

TC_H2 = COMPONENTS['H2'].Tc  # 33.145

FITTING_SOURCES = ['Wiebe 1934', 'Wiebe 1932', 'Chabab 2023', 'Chahab 2023',
                   'Torín-Ollarves 2021', 'Torin-Ollarves 2021']

def is_fitting_source(source):
    return any(fit in source for fit in FITTING_SOURCES)

def load_data(csv_path='../../shared/data/pointwise_kij_results.csv'):
    """Load and filter kij_AQ data."""
    df = pd.read_csv(csv_path)
    df_h2 = df[df['Gas'] == 'H2'].copy()

    # Filter for valid kij_AQ data
    df_aq = df_h2[df_h2['kij_AQ'].notna()].copy()

    # Filter for fitting sources
    df_aq = df_aq[df_aq['Source'].apply(is_fitting_source)]

    # Exclude T-O 423K
    n_423 = ((df_aq['Source'].str.contains('Tor')) & (abs(df_aq['T_K'] - 423.15) < 1)).sum()
    df_aq = df_aq[~((df_aq['Source'].str.contains('Tor')) & (abs(df_aq['T_K'] - 423.15) < 1))]

    return df_aq, n_423

def rational_form(Tr, A, B, C):
    """kij = (A + Tr) / (B + C*Tr)"""
    return (A + Tr) / (B + C * Tr)

def calc_xh2_mare(A, B, C, T_K_arr, P_Pa_arr, x_exp_arr, vle):
    """Calculate MARE on x_H2 predictions."""
    rel_errors = []
    for i in range(len(T_K_arr)):
        Tr = T_K_arr[i] / TC_H2
        kij = (A + Tr) / (B + C * Tr)
        try:
            x_pred = vle.calc_x_H2(T_K_arr[i], P_Pa_arr[i], kij)
            if x_pred is not None and x_pred > 0 and x_exp_arr[i] > 0:
                rel_errors.append(abs((x_pred - x_exp_arr[i]) / x_exp_arr[i]) * 100)
        except:
            pass
    return np.mean(rel_errors) if rel_errors else np.nan

def run_regression(report_path='../../shared/data/aqueous_bip_report.txt'):
    """Run regression and generate report."""

    lines = []
    def write(text=""):
        print(text)
        lines.append(text)

    write("=" * 80)
    write("kij_AQ REGRESSION REPORT")
    write("=" * 80)
    write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write()
    write("Correlation form: kij = (A + Tr) / (B + C*Tr)")
    write()

    # Load data
    df_aq, n_423 = load_data()

    write("DATA FILTERING")
    write("-" * 40)
    write(f"  Fitting sources: Wiebe, Chabab 2023, T-O")
    write(f"  Excluded T-O 423K: {n_423} points")
    write(f"  Remaining data points: {len(df_aq)}")
    write()

    # Extract arrays
    T_K = df_aq['T_K'].values
    Tr = T_K / TC_H2
    kij_aq = df_aq['kij_AQ'].values
    sources = df_aq['Source'].values
    P_bar = df_aq['P_bar'].values
    x_exp = df_aq['x_gas_exp'].values if 'x_gas_exp' in df_aq.columns else None

    # Initialize VLE for x_H2 calculations
    vle = H2WaterVLE(salinity=0.0)
    P_Pa = P_bar * 1e5

    # Statistics
    write("=" * 80)
    write("DATA STATISTICS")
    write("=" * 80)
    write()
    write(f"  T range: {T_K.min():.1f} - {T_K.max():.1f} K ({T_K.min()-273.15:.0f} - {T_K.max()-273.15:.0f} °C)")
    write(f"  Tr range: {Tr.min():.2f} - {Tr.max():.2f}")
    write(f"  P range: {P_bar.min():.0f} - {P_bar.max():.0f} bar")
    write(f"  kij_AQ range: {kij_aq.min():.4f} - {kij_aq.max():.4f}")
    write(f"  kij_AQ mean: {np.mean(kij_aq):.4f}")
    write(f"  kij_AQ std: {np.std(kij_aq):.4f}")
    write()

    # Current parameters (from vle_engine.py)
    A_curr, B_curr, C_curr = -14.59, 2.184, 0.365
    kij_current = rational_form(Tr, A_curr, B_curr, C_curr)
    mae_current = np.mean(np.abs(kij_aq - kij_current))
    mare_kij_current = np.mean(np.abs((kij_aq - kij_current) / kij_aq) * 100)
    mare_xh2_current = calc_xh2_mare(A_curr, B_curr, C_curr, T_K, P_Pa, x_exp, vle)

    write("=" * 80)
    write("CURRENT PARAMETERS")
    write("=" * 80)
    write()
    write(f"  A = {A_curr}")
    write(f"  B = {B_curr}")
    write(f"  C = {C_curr}")
    write()
    write(f"  MAE(kij) = {mae_current:.4f}")
    write(f"  MARE(kij) = {mare_kij_current:.1f}%")
    write(f"  MARE(x_H2) = {mare_xh2_current:.2f}%")
    write()

    # Fit to kij values using least squares
    write("=" * 80)
    write("REGRESSION: FIT TO kij VALUES")
    write("=" * 80)
    write()

    # Use differential evolution for robust fitting
    def objective_kij(params):
        A, B, C = params
        if B + C * Tr.min() <= 0.01 or B + C * Tr.max() <= 0.01:
            return 1e6
        kij_pred = rational_form(Tr, A, B, C)
        return np.mean((kij_aq - kij_pred)**2)  # MSE

    bounds = [(-25, -5), (0.5, 5.0), (0.1, 1.0)]
    result = differential_evolution(objective_kij, bounds, seed=42, maxiter=500, tol=1e-8)
    A_fit, B_fit, C_fit = result.x

    kij_fitted = rational_form(Tr, A_fit, B_fit, C_fit)
    mae_fitted = np.mean(np.abs(kij_aq - kij_fitted))
    mare_kij_fitted = np.mean(np.abs((kij_aq - kij_fitted) / kij_aq) * 100)
    r2 = 1 - np.sum((kij_aq - kij_fitted)**2) / np.sum((kij_aq - np.mean(kij_aq))**2)
    mare_xh2_fitted = calc_xh2_mare(A_fit, B_fit, C_fit, T_K, P_Pa, x_exp, vle)

    write(f"  Fitted parameters:")
    write(f"    A = {A_fit:.4f}")
    write(f"    B = {B_fit:.4f}")
    write(f"    C = {C_fit:.4f}")
    write()
    write(f"  MAE(kij) = {mae_fitted:.4f}")
    write(f"  MARE(kij) = {mare_kij_fitted:.1f}%")
    write(f"  R² = {r2:.4f}")
    write(f"  MARE(x_H2) = {mare_xh2_fitted:.2f}%")
    write()

    # Chabab 2023 comparison
    def kij_chabab(T_K):
        Tr = T_K / TC_H2
        D0, D1, D2, D3 = -2.11917, 0.14888, -13.01835, -0.43946
        return D0 + D1 * Tr + D2 * np.exp(D3 * Tr)

    kij_chabab_vals = np.array([kij_chabab(T) for T in T_K])
    mae_chabab = np.mean(np.abs(kij_aq - kij_chabab_vals))
    mare_kij_chabab = np.mean(np.abs((kij_aq - kij_chabab_vals) / kij_aq) * 100)

    # Calculate x_H2 MARE for Chabab
    rel_errors_chabab = []
    for i in range(len(T_K)):
        kij = kij_chabab(T_K[i])
        try:
            x_pred = vle.calc_x_H2(T_K[i], P_Pa[i], kij)
            if x_pred is not None and x_pred > 0 and x_exp[i] > 0:
                rel_errors_chabab.append(abs((x_pred - x_exp[i]) / x_exp[i]) * 100)
        except:
            pass
    mare_xh2_chabab = np.mean(rel_errors_chabab) if rel_errors_chabab else np.nan

    write("=" * 80)
    write("CHABAB 2023 COMPARISON")
    write("=" * 80)
    write()
    write(f"  Form: D0 + D1*Tr + D2*exp(D3*Tr)")
    write(f"  MAE(kij) = {mae_chabab:.4f}")
    write(f"  MARE(kij) = {mare_kij_chabab:.1f}%")
    write(f"  MARE(x_H2) = {mare_xh2_chabab:.2f}%")
    write()

    # Predicted kij at key temperatures
    write("=" * 80)
    write("PREDICTED kij_AQ AT KEY TEMPERATURES")
    write("=" * 80)
    write()
    write(f"{'T (°C)':<10} {'Tr':<10} {'Current':<12} {'Fitted':<12} {'Chabab':<12}")
    write("-" * 60)
    for T_C in [0, 25, 50, 75, 100, 125, 150]:
        T_K_val = T_C + 273.15
        Tr_val = T_K_val / TC_H2
        kij_curr = rational_form(Tr_val, A_curr, B_curr, C_curr)
        kij_fit = rational_form(Tr_val, A_fit, B_fit, C_fit)
        kij_chab = kij_chabab(T_K_val)
        write(f"{T_C:<10} {Tr_val:<10.2f} {kij_curr:<12.4f} {kij_fit:<12.4f} {kij_chab:<12.4f}")
    write()

    # Summary table
    write("=" * 80)
    write("SUMMARY")
    write("=" * 80)
    write()
    write(f"{'Method':<25} {'A':<10} {'B':<10} {'C':<10} {'MAE(kij)':<10} {'MARE(kij)':<12} {'MARE(x_H2)':<12}")
    write("-" * 95)
    write(f"{'Current':<25} {A_curr:<10.4f} {B_curr:<10.4f} {C_curr:<10.4f} {mae_current:<10.4f} {mare_kij_current:<12.1f}% {mare_xh2_current:<12.2f}%")
    write(f"{'Fitted (to kij)':<25} {A_fit:<10.4f} {B_fit:<10.4f} {C_fit:<10.4f} {mae_fitted:<10.4f} {mare_kij_fitted:<12.1f}% {mare_xh2_fitted:<12.2f}%")
    write(f"{'Chabab 2023':<25} {'-':<10} {'-':<10} {'-':<10} {mae_chabab:<10.4f} {mare_kij_chabab:<12.1f}% {mare_xh2_chabab:<12.2f}%")
    write()

    # Analysis
    write("=" * 80)
    write("ANALYSIS")
    write("=" * 80)
    write()

    kij_improvement = (mae_current - mae_fitted) / mae_current * 100
    xh2_change = mare_xh2_fitted - mare_xh2_current

    write(f"1. Fitting to kij values improves MAE(kij) by {kij_improvement:.1f}%")
    write(f"2. MARE(x_H2) {'improves' if xh2_change < 0 else 'worsens'} by {abs(xh2_change):.2f} percentage points")
    write(f"3. Current parameters achieve R² = {r2:.4f} against point-regressed values")
    write()

    if abs(kij_improvement) < 5 and abs(xh2_change) < 0.5:
        write("Conclusion: Current parameters are near-optimal - no update needed")
    elif xh2_change > 0.5:
        write("Conclusion: Fitting to kij worsens x_H2 predictions - keep current parameters")
    else:
        write("Conclusion: Consider updating parameters to fitted values")
    write()

    # Parameter changes
    write("=" * 80)
    write("PARAMETER CHANGES (Current → Fitted)")
    write("=" * 80)
    write()
    write(f"  A: {A_curr:.4f} → {A_fit:.4f} (Δ = {A_fit - A_curr:+.4f})")
    write(f"  B: {B_curr:.4f} → {B_fit:.4f} (Δ = {B_fit - B_curr:+.4f})")
    write(f"  C: {C_curr:.4f} → {C_fit:.4f} (Δ = {C_fit - C_curr:+.4f})")
    write()

    # Save report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    run_regression()
