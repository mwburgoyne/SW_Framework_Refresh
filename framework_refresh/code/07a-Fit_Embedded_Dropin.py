#!/usr/bin/env python3
"""
Fit Embedded Salinity BIP Correlations for Drop-in (Track 2) Framework
========================================================================

For each gas, fits kij_AQ(T, m) = kij_fw(T) + delta(T, m) that reproduces
the agreed Sechenov ks model when used inside the PR-EOS with S&W alpha.

CRITICAL DIFFERENCE from proposed (Track 1) fitting:
  S&W alpha embeds salinity (0.0103 * csw^1.1). A Track 2 user's VLE has
  alpha_water_soreide(Tr, m) when computing brine. So the root-finding
  must use a brine VLE:

    vle_fw = SWBinaryVLE(gas, 0.0, framework='dropin')   # S&W alpha at cs=0
    vle_br = SWBinaryVLE(gas, m,   framework='dropin')   # S&W alpha at cs=m

    x_fresh = vle_fw._calc_x_with_kij(T_K, P_Pa, kij_fw)
    x_target = x_fresh * 10**(-ks * m)

    # Root-find kij_eff in BRINE VLE:
    kij_eff = brentq(lambda k: vle_br._calc_x_with_kij(T_K, P_Pa, k) - x_target, ...)
    delta_kij = kij_eff - kij_fw

  This ensures delta absorbs both salting-out AND alpha-salinity coupling.

Uses Form D (solubility-error minimizing) linear-in-m fit:
  delta_kij = (a0 + a1*Tr + a2*Tr^2) * m
  CO2 gets optional quadratic-in-m (Form E) if improvement warrants.

Usage:
    cd framework_refresh/code
    python 07a-Fit_Embedded_Dropin.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
from scipy.optimize import minimize, brentq

from _lib_vle_engine import (
    SWBinaryVLE, get_sechenov_ks, COMPONENTS,
    celsius_to_kelvin, bar_to_pascal,
    kij_aq_co2_dropin, kij_aq_h2s_dropin, kij_aq_n2_dropin,
    kij_aq_h2_dropin, kij_aq_ch4_dropin, kij_aq_c2h6_dropin,
    kij_aq_c3h8_dropin, kij_aq_hydrocarbon,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'data')


# =============================================================================
# Gas-specific configuration (same grids as 07-Fit_Embedded_Salinity_All_Gases.py)
# =============================================================================
GAS_CONFIGS = {
    'CO2': {
        'Tc': 304.20,
        'T_range_C': [25, 50, 75, 100, 125, 150, 175, 200],
        'P_range_bar': [50, 100, 200, 500],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    },
    'H2S': {
        'Tc': 373.20,
        'T_range_C': [25, 50, 75, 100, 125, 150, 200],
        'P_range_bar': [50, 100, 150, 200],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    'CH4': {
        'Tc': 190.60,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    'N2': {
        'Tc': 126.10,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 500],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    'H2': {
        'Tc': 33.145,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 150, 200],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    'C2H6': {
        'Tc': 305.40,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    'C3H8': {
        'Tc': 369.80,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    'nC4H10': {
        'Tc': 425.20,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
}

# Dropin freshwater kij functions (S&W alpha)
KIJ_FW_FUNCS = {
    'CO2': lambda T_K: kij_aq_co2_dropin(T_K),
    'H2S': lambda T_K: kij_aq_h2s_dropin(T_K),
    'CH4': lambda T_K: kij_aq_ch4_dropin(T_K),
    'N2':  lambda T_K: kij_aq_n2_dropin(T_K),
    'H2':  lambda T_K: kij_aq_h2_dropin(T_K),
    'C2H6': lambda T_K: kij_aq_c2h6_dropin(T_K),
    'C3H8': lambda T_K: kij_aq_c3h8_dropin(T_K),
    'nC4H10': lambda T_K: kij_aq_hydrocarbon(T_K, COMPONENTS['nC4H10'].omega,
                                                COMPONENTS['nC4H10'].Tc, 0.0),
}


# =============================================================================
# Synthetic data generation (dropin-specific: brine VLE for root-finding)
# =============================================================================
def generate_synthetic_data(gas):
    """Generate synthetic kij data using dropin framework with brine VLE."""
    cfg = GAS_CONFIGS[gas]
    kij_fw_func = KIJ_FW_FUNCS[gas]
    Tc = cfg['Tc']

    # Freshwater VLE (dropin framework, cs=0)
    vle_fw = SWBinaryVLE(gas, salinity_molal=0.0, framework='dropin')
    data = []

    for T_C in cfg['T_range_C']:
        T_K = celsius_to_kelvin(T_C)
        Tr = T_K / Tc
        kij_fw = kij_fw_func(T_K)

        for P_bar in cfg['P_range_bar']:
            P_Pa = bar_to_pascal(P_bar)

            # Compute freshwater solubility
            try:
                x_fresh = vle_fw._calc_x_with_kij(T_K, P_Pa, kij_fw)
            except Exception:
                continue
            if x_fresh is None or x_fresh <= 0 or x_fresh >= 0.5:
                continue

            for m in cfg['m_range']:
                if m == 0:
                    data.append({
                        'T_C': T_C, 'T_K': T_K, 'Tr': Tr,
                        'P_bar': P_bar, 'P_Pa': P_Pa,
                        'csw': 0.0, 'x_fresh': x_fresh,
                        'x_target': x_fresh,
                        'kij_fw': kij_fw, 'kij_eff': kij_fw,
                        'delta_kij': 0.0,
                    })
                    continue

                # Target brine solubility from Sechenov
                ks = get_sechenov_ks(gas, T_K, m, P_bar)
                x_target = x_fresh * 10**(-ks * m)
                if x_target <= 0:
                    continue

                # Root-find kij_eff in BRINE VLE (S&W alpha with salinity)
                vle_br = SWBinaryVLE(gas, salinity_molal=m, framework='dropin')
                try:
                    kij_eff = _find_kij_brine(vle_br, T_K, P_Pa, x_target, kij_fw)
                except Exception:
                    continue

                delta_kij = kij_eff - kij_fw
                data.append({
                    'T_C': T_C, 'T_K': T_K, 'Tr': Tr,
                    'P_bar': P_bar, 'P_Pa': P_Pa,
                    'csw': m, 'x_fresh': x_fresh,
                    'x_target': x_target,
                    'kij_fw': kij_fw, 'kij_eff': kij_eff,
                    'delta_kij': delta_kij,
                })

    return data


def _find_kij_brine(vle_br, T_K, P_Pa, x_target, kij_fw):
    """Find kij in brine VLE that produces target solubility."""
    def objective(kij):
        try:
            x_calc = vle_br._calc_x_with_kij(T_K, P_Pa, kij)
            if x_calc is None:
                return 1.0
            return x_calc - x_target
        except Exception:
            return 1.0

    return brentq(objective, kij_fw - 3.0, kij_fw + 0.5, xtol=1e-8)


# =============================================================================
# Fitting: Form D (solubility-error minimizing, linear-in-m)
# =============================================================================
def fit_solubility_minimizing(data, Tc, gas, kij_fw_func, init_params=None):
    """
    Form D: delta_kij = (a0 + a1*Tr + a2*Tr^2) * m
    Minimizes solubility relative error using BRINE VLE (dropin framework).
    """
    brine_data = [d for d in data if d['csw'] > 0 and d['x_target'] > 0]

    def sol_objective(params):
        a0, a1, a2 = params
        sq_errors = []

        for d in brine_data:
            T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']
            Tr = T_K / Tc

            kij_fw = kij_fw_func(T_K)
            kij_brine = kij_fw + (a0 + a1*Tr + a2*Tr**2) * csw

            # Use BRINE VLE (dropin)
            vle_br = SWBinaryVLE(gas, salinity_molal=csw, framework='dropin')
            try:
                x_pred = vle_br._calc_x_with_kij(T_K, P_Pa, kij_brine)
            except Exception:
                sq_errors.append(1.0)
                continue

            x_target = d['x_target']
            if x_target > 0 and x_pred is not None and x_pred > 0:
                rel_err = (x_pred - x_target) / x_target
                sq_errors.append(rel_err**2)
            else:
                sq_errors.append(1.0)

        return np.mean(sq_errors) if sq_errors else 1e10

    if init_params is None:
        init_params = [0.0, 0.0, 0.0]

    result = minimize(sol_objective, init_params, method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-14})
    return result.x, result.fun


def fit_quad_m_solubility(data, Tc, gas, kij_fw_func, init_params=None):
    """
    Form E: delta_kij = (a0 + a1*Tr + a2*Tr^2) * m + (b0 + b1*Tr) * m^2
    Quadratic in m. Minimizes solubility relative error using BRINE VLE.
    """
    brine_data = [d for d in data if d['csw'] > 0 and d['x_target'] > 0]

    def sol_objective(params):
        a0, a1, a2, b0, b1 = params
        sq_errors = []

        for d in brine_data:
            T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']
            Tr = T_K / Tc

            kij_fw = kij_fw_func(T_K)
            kij_brine = kij_fw + (a0 + a1*Tr + a2*Tr**2)*csw + (b0 + b1*Tr)*csw**2

            vle_br = SWBinaryVLE(gas, salinity_molal=csw, framework='dropin')
            try:
                x_pred = vle_br._calc_x_with_kij(T_K, P_Pa, kij_brine)
            except Exception:
                sq_errors.append(1.0)
                continue

            x_target = d['x_target']
            if x_target > 0 and x_pred is not None and x_pred > 0:
                rel_err = (x_pred - x_target) / x_target
                sq_errors.append(rel_err**2)
            else:
                sq_errors.append(1.0)

        return np.mean(sq_errors) if sq_errors else 1e10

    if init_params is None:
        init_params = [0.0, 0.0, 0.0, 0.0, 0.0]

    result = minimize(sol_objective, init_params, method='Nelder-Mead',
                     options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-16})
    return result.x, result.fun


# =============================================================================
# Evaluate accuracy
# =============================================================================
def evaluate_accuracy(data, gas, Tc, kij_fw_func, params, form='linear'):
    """Evaluate embedded BIP accuracy on synthetic data using BRINE VLE."""
    results = []
    for d in data:
        if d['csw'] <= 0:
            continue
        T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']
        Tr = T_K / Tc

        kij_fw = kij_fw_func(T_K)
        if form == 'linear':
            delta = (params[0] + params[1]*Tr + params[2]*Tr**2) * csw
        else:
            delta = (params[0] + params[1]*Tr + params[2]*Tr**2)*csw + (params[3] + params[4]*Tr)*csw**2
        kij_brine = kij_fw + delta

        vle_br = SWBinaryVLE(gas, salinity_molal=csw, framework='dropin')
        try:
            x_pred = vle_br._calc_x_with_kij(T_K, P_Pa, kij_brine)
        except Exception:
            continue

        x_target = d['x_target']
        if x_target > 0 and x_pred is not None and x_pred > 0:
            rel_err_pct = abs(x_pred - x_target) / x_target * 100
            results.append(rel_err_pct)

    if not results:
        return {'mae_x_pct': np.nan, 'max_x_pct': np.nan, 'within_5pct': np.nan}

    arr = np.array(results)
    return {
        'mae_x_pct': np.mean(arr),
        'max_x_pct': np.max(arr),
        'within_5pct': np.mean(arr < 5) * 100,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 78)
    print("FIT EMBEDDED SALINITY BIP — DROPIN (TRACK 2) FRAMEWORK")
    print("=" * 78)
    print("Using S&W alpha with salinity in brine VLE for root-finding")

    all_results = {}
    report_lines = []
    report_lines.append("=" * 78)
    report_lines.append("EMBEDDED SALINITY BIP — DROPIN (TRACK 2)")
    report_lines.append("=" * 78)
    report_lines.append("")

    for gas in ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8', 'nC4H10']:
        cfg = GAS_CONFIGS[gas]
        kij_fw_func = KIJ_FW_FUNCS[gas]
        Tc = cfg['Tc']

        print(f"\n{'='*60}")
        print(f"  {gas}  (Tc = {Tc} K)")
        print(f"{'='*60}")

        # Generate synthetic data
        print(f"  Generating synthetic data...")
        data = generate_synthetic_data(gas)
        n_total = len(data)
        n_brine = len([d for d in data if d['csw'] > 0])
        print(f"  Generated {n_total} points ({n_brine} brine)")

        if n_brine < 3:
            print(f"  WARNING: Insufficient brine data for {gas}, skipping")
            continue

        # Use initial guess from least-squares on delta_kij data
        from scipy.optimize import least_squares
        brine = [d for d in data if d['csw'] > 0]
        def res_ls(p):
            return [d['delta_kij'] - (p[0] + p[1]*d['Tr'] + p[2]*d['Tr']**2)*d['csw']
                    for d in brine]
        ls_result = least_squares(res_ls, [0.0, 0.0, 0.0])
        init_params = list(ls_result.x)
        print(f"  LS initial guess: a0={init_params[0]:.4f}, a1={init_params[1]:.4f}, a2={init_params[2]:.4f}")

        # Fit Form D (solubility-error minimizing, linear-in-m)
        print(f"  Fitting Form D (x-error, linear-in-m)...")
        params_D, obj_D = fit_solubility_minimizing(data, Tc, gas, kij_fw_func, init_params)
        acc_D = evaluate_accuracy(data, gas, Tc, kij_fw_func, params_D, 'linear')
        print(f"    a0={params_D[0]:.4f}, a1={params_D[1]:.4f}, a2={params_D[2]:.4f}")
        print(f"    MAE(x%) = {acc_D['mae_x_pct']:.2f}%, Max = {acc_D['max_x_pct']:.2f}%")

        best_params = params_D
        best_form = 'linear'
        best_acc = acc_D

        # CO2: also try quadratic-in-m
        if gas == 'CO2':
            print(f"  Fitting Form E (x-error, quadratic-in-m)...")
            init_E = list(params_D) + [0.0, 0.0]
            params_E, obj_E = fit_quad_m_solubility(data, Tc, gas, kij_fw_func, init_E)
            acc_E = evaluate_accuracy(data, gas, Tc, kij_fw_func, params_E, 'quadratic')
            print(f"    a0={params_E[0]:.4f}, a1={params_E[1]:.4f}, a2={params_E[2]:.4f}")
            print(f"    b0={params_E[3]:.4f}, b1={params_E[4]:.4f}")
            print(f"    MAE(x%) = {acc_E['mae_x_pct']:.2f}%, Max = {acc_E['max_x_pct']:.2f}%")

            if acc_E['mae_x_pct'] < acc_D['mae_x_pct'] * 0.8:
                print(f"  → Using quadratic-in-m for CO2")
                best_params = params_E
                best_form = 'quadratic'
                best_acc = acc_E

        all_results[gas] = {
            'Tc': Tc, 'params': best_params, 'form': best_form,
            'acc': best_acc, 'n_brine': n_brine,
        }

        report_lines.append(f"{gas} (Tc={Tc}, n_brine={n_brine})")
        if best_form == 'linear':
            report_lines.append(f"  a0={best_params[0]:.4f}, a1={best_params[1]:.4f}, a2={best_params[2]:.4f}")
        else:
            report_lines.append(f"  a0={best_params[0]:.4f}, a1={best_params[1]:.4f}, a2={best_params[2]:.4f}")
            report_lines.append(f"  b0={best_params[3]:.4f}, b1={best_params[4]:.4f}")
        report_lines.append(f"  MAE(x%) = {best_acc['mae_x_pct']:.2f}%")
        report_lines.append("")

    # Summary and Python code
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Gas':>8} {'Tc':>8} {'a0':>8} {'a1':>8} {'a2':>8} {'b0':>8} {'b1':>8} {'MAE%':>8}")
    print("-" * 78)

    code_lines = []
    code_lines.append("EMBEDDED_SALINITY_PARAMS_DROPIN: Dict[str, Dict] = {")

    for gas in ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8', 'nC4H10']:
        if gas not in all_results:
            continue
        r = all_results[gas]
        p = r['params']
        if r['form'] == 'quadratic':
            print(f"{gas:>8} {r['Tc']:>8.2f} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f} {p[3]:>8.4f} {p[4]:>8.4f} {r['acc']['mae_x_pct']:>8.2f}")
            code_lines.append(f"    '{gas}': {{'Tc': {r['Tc']:.2f}, 'a0': {p[0]:.4f}, 'a1': {p[1]:.4f}, 'a2': {p[2]:.4f},")
            code_lines.append(f"               'b0': {p[3]:.4f}, 'b1': {p[4]:.4f}}},")
        else:
            print(f"{gas:>8} {r['Tc']:>8.2f} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f} {'---':>8} {'---':>8} {r['acc']['mae_x_pct']:>8.2f}")
            code_lines.append(f"    '{gas}': {{'Tc': {r['Tc']:.2f}, 'a0': {p[0]:.4f}, 'a1': {p[1]:.4f}, 'a2': {p[2]:.4f}}},")

    code_lines.append("}")

    print("\nPython code for _lib_vle_engine.py:")
    for line in code_lines:
        print(line)

    report_lines.append("PYTHON CODE")
    report_lines.extend(code_lines)

    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'embedded_dropin_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
