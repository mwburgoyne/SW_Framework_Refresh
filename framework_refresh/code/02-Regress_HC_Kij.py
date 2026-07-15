#!/usr/bin/env python3
"""
Pointwise kij regression for C2H6, C3H8, nC4H10 from extracted_solubility_data.csv.

These gases are not in the main solubility_points.xlsx (only in the extracted CSV).
Results are appended to pointwise_kij_results.csv for consistent plotting.

Also fits the rational BIP form kij = (A + Tr) / (B + C*Tr) to CH4 and CO2.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, curve_fit

from _lib_vle_engine import (
    COMPONENTS, SWBinaryVLE,
    kij_aq_ch4, kij_aq_co2,
    kij_aq_hydrocarbon, get_kij_na,
)

CSV_EXTRACTED = '../../shared/data/extracted_solubility_data.csv'
CSV_RESULTS = '../../shared/data/pointwise_kij_results.csv'


# ── Regression functions (from regress_pointwise_kij.py) ─────────────────────

def regress_kij_aq(vle, T_K, P_Pa, x_gas_exp, kij_bounds=(-1.5, 1.0)):
    """Find kij_AQ that reproduces experimental x_gas."""
    def objective(kij):
        try:
            x_calc = vle._calc_x_with_kij(T_K, P_Pa, kij)
            if x_calc is None or x_calc <= 0 or x_calc >= 1:
                return 1e10
            return ((x_calc - x_gas_exp) / max(x_gas_exp, 1e-6))**2
        except:
            return 1e10
    try:
        result = minimize_scalar(objective, bounds=kij_bounds, method='bounded',
                                options={'xatol': 1e-5, 'maxiter': 100})
        if result.fun < 0.01:
            return result.x, True
        else:
            result2 = minimize_scalar(objective, bounds=(-2.0, 1.5), method='bounded',
                                     options={'xatol': 1e-5, 'maxiter': 100})
            if result2.fun < result.fun:
                return result2.x, result2.fun < 0.04
            return result.x, result.fun < 0.04
    except:
        return None, False


def regress_kij_na(vle, T_K, P_Pa, y_H2O_exp, kij_bounds=(-0.5, 1.0)):
    """Find kij_NA that reproduces experimental y_H2O."""
    Psat_H2O = np.exp(73.649 - 7258.2/T_K - 7.3037*np.log(T_K) + 4.1653e-6*T_K**2)
    y_raoult = Psat_H2O / P_Pa
    if y_H2O_exp < 0.3 * y_raoult:
        return 1.5, False

    def objective(kij):
        try:
            y_calc = vle.calc_water_content_with_kij(T_K, P_Pa, kij)
            if y_calc is None or y_calc <= 0 or y_calc >= 1:
                return 1e10
            return ((y_calc - y_H2O_exp) / max(y_H2O_exp, 1e-6))**2
        except:
            return 1e10
    try:
        result = minimize_scalar(objective, bounds=kij_bounds, method='bounded',
                                options={'xatol': 1e-5, 'maxiter': 100})
        if result.fun < 0.01:
            return result.x, True
        else:
            result2 = minimize_scalar(objective, bounds=(-1.0, 1.5), method='bounded',
                                     options={'xatol': 1e-5, 'maxiter': 100})
            if result2.fun < result.fun:
                return result2.x, result2.fun < 0.04
            return result.x, result.fun < 0.04
    except:
        return None, False


# ── Load and regress C2-C4 data ──────────────────────────────────────────────

def regress_extracted_gases():
    """Run pointwise kij regression for C2H6, C3H8, nC4H10."""
    df = pd.read_csv(CSV_EXTRACTED, comment='#')
    df = df[df['Gas'].notna()]

    # Convert numeric columns
    for col in ['T_K', 'P_bar', 'Sal_m', 'x_Gas', 'y_H2O']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    target_gases = ['C2H6', 'C3H8', 'nC4H10']
    results = []

    for gas in target_gases:
        gdata = df[df['Gas'] == gas].copy()
        gdata = gdata.dropna(subset=['T_K', 'P_bar'])

        if gas not in COMPONENTS:
            print(f"  {gas}: not in COMPONENTS dict — skipping.")
            continue

        print(f"\n  {gas}: {len(gdata)} points")
        vle = SWBinaryVLE(gas, salinity_molal=0.0)

        for _, row in gdata.iterrows():
            T_K = row['T_K']
            P_Pa = row['P_bar'] * 1e5
            x_gas = row['x_Gas'] if pd.notna(row['x_Gas']) else None
            y_H2O = row['y_H2O'] if pd.notna(row['y_H2O']) else None
            source = row['Source']

            kij_aq, kij_aq_conv = None, False
            kij_na, kij_na_conv = None, False

            if x_gas is not None and x_gas > 0:
                kij_aq, kij_aq_conv = regress_kij_aq(vle, T_K, P_Pa, x_gas)

            if y_H2O is not None and y_H2O > 0:
                kij_na, kij_na_conv = regress_kij_na(vle, T_K, P_Pa, y_H2O)

            if kij_aq is not None or kij_na is not None:
                results.append({
                    'Gas': gas,
                    'Source': source,
                    'T_K': T_K,
                    'T_C': T_K - 273.15,
                    'P_Pa': P_Pa,
                    'P_bar': row['P_bar'],
                    'x_gas_exp': x_gas if x_gas else '',
                    'y_H2O_exp': y_H2O if y_H2O else '',
                    'kij_AQ': kij_aq if kij_aq is not None else '',
                    'kij_AQ_conv': kij_aq_conv,
                    'kij_NA': kij_na if kij_na is not None else '',
                    'kij_NA_conv': kij_na_conv,
                })

        n_aq = sum(1 for r in results if r['Gas'] == gas and r['kij_AQ_conv'])
        n_na = sum(1 for r in results if r['Gas'] == gas and r['kij_NA_conv'])
        print(f"    kij_AQ converged: {n_aq}")
        print(f"    kij_NA converged: {n_na}")

    return pd.DataFrame(results)


# ── Fit rational form ────────────────────────────────────────────────────────

def rational_form(Tr, A, B, C):
    """kij = (A + Tr) / (B + C*Tr)"""
    return (A + Tr) / (B + C * Tr)


def fit_rational(df, gas):
    """Fit rational BIP form to a gas's kij_AQ data."""
    mask = (df['Gas'] == gas) & (df['kij_AQ_conv'] == True)
    # Handle mixed types from CSV
    if df['kij_AQ'].dtype == object:
        mask = mask & (df['kij_AQ'] != '')
    data = df[mask].copy()
    data['kij_AQ'] = pd.to_numeric(data['kij_AQ'], errors='coerce')
    data = data.dropna(subset=['kij_AQ'])

    if len(data) < 5:
        print(f"  {gas}: only {len(data)} points — skipping fit.")
        return None

    Tc = COMPONENTS[gas].Tc
    Tr = (data['T_K'] / Tc).values
    kij = data['kij_AQ'].values

    popt, pcov = curve_fit(rational_form, Tr, kij, p0=[-5.0, 2.0, 0.5], maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    kij_pred = rational_form(Tr, *popt)
    residuals = kij - kij_pred
    mae = np.abs(residuals).mean()
    rmse = np.sqrt((residuals**2).mean())

    result = {
        'gas': gas, 'Tc': Tc, 'n': len(data),
        'A': popt[0], 'B': popt[1], 'C': popt[2],
        'A_err': perr[0], 'B_err': perr[1], 'C_err': perr[2],
        'mae': mae, 'rmse': rmse,
        'Tr_range': (Tr.min(), Tr.max()),
    }
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("HC kij Regression + Rational Form Fitting")
    print("=" * 60)

    # 1. Regress C2H6, C3H8, nC4H10
    print("\n--- Regressing C2H6, C3H8, nC4H10 ---")
    df_new = regress_extracted_gases()

    if len(df_new) > 0:
        # Append to existing results
        df_existing = pd.read_csv(CSV_RESULTS)
        # Remove any existing C2-C4 rows to avoid duplicates
        df_existing = df_existing[~df_existing['Gas'].isin(['C2H6', 'C3H8', 'nC4H10'])]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_RESULTS, index=False)
        print(f"\nAppended {len(df_new)} rows to {CSV_RESULTS}")
        print(f"Total rows: {len(df_combined)}")
    else:
        df_combined = pd.read_csv(CSV_RESULTS)

    # 2. Fit rational form to CH4, CO2
    print("\n--- Fitting Rational Form: kij = (A + Tr) / (B + C*Tr) ---")
    for gas in ['CH4', 'CO2', 'C3H8', 'nC4H10']:
        result = fit_rational(df_combined, gas)
        if result:
            r = result
            # Compare with S&W
            mask = (df_combined['Gas'] == gas) & (df_combined['kij_AQ_conv'] == True)
            if df_combined['kij_AQ'].dtype == object:
                mask = mask & (df_combined['kij_AQ'] != '')
            data = df_combined[mask].copy()
            data['kij_AQ'] = pd.to_numeric(data['kij_AQ'], errors='coerce')
            data = data.dropna(subset=['kij_AQ'])
            Tc = COMPONENTS[gas].Tc
            omega = COMPONENTS[gas].omega

            if gas == 'CO2':
                kij_sw = np.array([kij_aq_co2(T, 0.0) for T in data['T_K']])
            elif gas == 'CH4':
                kij_sw = np.array([kij_aq_ch4(T, 0.0) for T in data['T_K']])
            else:
                kij_sw = np.array([kij_aq_hydrocarbon(T, omega, Tc, 0.0) for T in data['T_K']])
            mae_sw = np.abs(data['kij_AQ'].values - kij_sw).mean()

            print(f"\n  {gas} (Tc={Tc:.2f}K, n={r['n']}):")
            print(f"    Rational: A={r['A']:.4f}, B={r['B']:.4f}, C={r['C']:.4f}")
            print(f"    MAE: rational={r['mae']:.4f}, S&W={mae_sw:.4f}  "
                  f"({(mae_sw-r['mae'])/mae_sw*100:.0f}% improvement)")
            print(f"    Tr range: {r['Tr_range'][0]:.3f} - {r['Tr_range'][1]:.3f}")

    print("\nDone.")
