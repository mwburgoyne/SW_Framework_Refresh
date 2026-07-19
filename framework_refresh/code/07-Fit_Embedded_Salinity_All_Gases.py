#!/usr/bin/env python3
"""
Fit Embedded Salinity BIP Correlations for All Paper 2 Gases
=============================================================

For each gas, fits kij_AQ(T, m) = kij_fw(T) + delta(T, m) that reproduces
the agreed Sechenov ks model when used inside the PR-EOS flash.

Agreed Sechenov assignments:
  CO2:  Duan & Sun 2003 Pitzer model
  H2S:  Akinfiev et al. 2016 Pitzer model
  CH4:  S&W Equation 8 (already embedded in S&W Eq 12)
  N2:   S&W Equation 8 (already embedded in S&W Eq 13)
  H2:   S&W Equation 8 (Paper 1 result)

Template: h2_solubility/code/03-Fit_Embedded_Salinity_BIP.py (H2)

For CH4 and N2, the S&W kij_AQ functions already embed salinity via cs terms
in Eqs 11-13. We verify consistency with our proposed freshwater kij + S&W Eq 8.

For CO2 and H2S, the S&W kij_AQ functions also embed salinity (Eqs 14-15),
but the embedded ks doesn't match the recommended Pitzer models. We refit
delta(T, m) to reproduce the Pitzer ks.

Fitting approach (per gas):
  1. Synthetic grid: T × P × m conditions
  2. For each brine point: compute x_fresh → apply target ks → get x_target
  3. Root-find kij_eff that matches x_target in PR-EOS
  4. delta_kij = kij_eff - kij_fw
  5. Fit delta_kij = f(Tr, m) with several forms
  6. Objective: minimize implied ks error

Author: Mark Burgoyne
Date: 2026-02-08
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
from scipy.optimize import minimize, brentq
from typing import Tuple, Dict, List, Callable, Optional

from _lib_vle_engine import (
    SWBinaryVLE, sw_equation_8_ks, get_sechenov_ks,
    COMPONENTS, BIP_TC_H2,
    celsius_to_kelvin, bar_to_pascal, pascal_to_bar,
    kij_aq_h2, kij_aq_co2, kij_aq_n2, kij_aq_h2s,
    kij_aq_hydrocarbon,
    kij_aq_co2_proposed, kij_aq_h2s_proposed, kij_aq_n2_proposed,
    kij_aq_h2_proposed, kij_aq_c2h6_proposed, kij_aq_c3h8_proposed,
    kij_aq_ch4, KIJ_AQ_PROPOSED,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'data')


# =============================================================================
# Gas-specific configuration
# =============================================================================
GAS_CONFIGS = {
    'CO2': {
        'Tc': 304.20,
        'T_range_C': [25, 50, 75, 100, 125, 150, 175, 200],
        'P_range_bar': [50, 100, 200, 500],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        'ks_source': 'duan2003',
        'ks_P_dependent': True,
    },
    'H2S': {
        'Tc': 373.20,
        'T_range_C': [25, 50, 75, 100, 125, 150, 200],
        'P_range_bar': [50, 100, 150, 200],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        'ks_source': 'akinfiev',
        'ks_P_dependent': False,
    },
    'CH4': {
        'Tc': 190.60,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        'ks_source': 'sw_eq8',
        'ks_P_dependent': False,
    },
    'N2': {
        'Tc': 126.10,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 500],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        'ks_source': 'sw_eq8',
        'ks_P_dependent': False,
    },
    'H2': {
        'Tc': 33.145,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 150, 200],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        'ks_source': 'sw_eq8',
        'ks_P_dependent': False,
    },
    'C2H6': {
        'Tc': 305.40,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        'ks_source': 'sw_eq8',
        'ks_P_dependent': False,
    },
    'C3H8': {
        'Tc': 369.80,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        'ks_source': 'sw_eq8',
        'ks_P_dependent': False,
    },
    'nC4H10': {
        'Tc': 425.20,
        'T_range_C': [25, 50, 75, 100, 125, 150],
        'P_range_bar': [50, 100, 200, 300],
        'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        'ks_source': 'sw_eq8',
        'ks_P_dependent': False,
    },
}


# =============================================================================
# Freshwater BIP functions (from VLE engine — proposed Paper 2 base cases)
# =============================================================================
# KIJ_FW_FUNCS wraps the engine's proposed kij_AQ functions with single-arg
# signatures expected by the fitting loop below.
KIJ_FW_FUNCS = {
    'CO2': lambda T_K: kij_aq_co2_proposed(T_K),
    'H2S': lambda T_K: kij_aq_h2s_proposed(T_K),
    'CH4': lambda T_K: kij_aq_ch4(T_K),
    'N2':  lambda T_K: kij_aq_n2_proposed(T_K),
    'H2':  lambda T_K: kij_aq_h2_proposed(T_K),
    'C2H6': lambda T_K: kij_aq_c2h6_proposed(T_K),
    'C3H8': lambda T_K: kij_aq_c3h8_proposed(T_K),
    'nC4H10': lambda T_K: kij_aq_hydrocarbon(T_K, COMPONENTS['nC4H10'].omega,
                                               COMPONENTS['nC4H10'].Tc, 0.0),
}


# =============================================================================
# Target ks functions (agreed models)
# =============================================================================
def get_target_ks(gas, T_K, m, P_bar=100.0):
    """Get target ks from agreed model for each gas."""
    return get_sechenov_ks(gas, T_K, m, P_bar)


# =============================================================================
# Synthetic data generation
# =============================================================================
def generate_synthetic_data(gas):
    """Generate synthetic kij data for a given gas."""
    cfg = GAS_CONFIGS[gas]
    kij_fw_func = KIJ_FW_FUNCS[gas]
    vle = SWBinaryVLE(gas, salinity_molal=0.0)
    data = []

    for T_C in cfg['T_range_C']:
        T_K = celsius_to_kelvin(T_C)
        Tr = T_K / cfg['Tc']
        kij_fw = kij_fw_func(T_K)

        for P_bar in cfg['P_range_bar']:
            P_Pa = bar_to_pascal(P_bar)

            # Compute freshwater solubility using the VLE with proposed kij
            try:
                x_fresh = vle._calc_x_with_kij(T_K, P_Pa, kij_fw)
            except Exception:
                continue

            if x_fresh <= 0 or x_fresh >= 0.5:
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

                # Apply agreed ks to get target brine solubility
                ks = get_target_ks(gas, T_K, m, P_bar)
                x_target = x_fresh * 10**(-ks * m)

                if x_target <= 0:
                    continue

                # Root-find kij_eff that matches x_target
                try:
                    kij_eff = _find_kij_for_target(vle, T_K, P_Pa, x_target, kij_fw)
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


def _find_kij_for_target(vle, T_K, P_Pa, x_target, kij_fw):
    """Find kij that produces target solubility via root-finding."""
    def objective(kij):
        try:
            x_calc = vle._calc_x_with_kij(T_K, P_Pa, kij)
            return x_calc - x_target
        except Exception:
            return 1.0

    # Brine solubility is lower → need more negative kij
    try:
        kij_eff = brentq(objective, kij_fw - 3.0, kij_fw + 0.5, xtol=1e-8)
    except ValueError:
        result = minimize(lambda k: abs(objective(k[0])), [kij_fw - 0.5],
                         bounds=[(kij_fw - 5.0, kij_fw + 1.0)])
        kij_eff = result.x[0]

    return kij_eff


# =============================================================================
# Fitting forms
# =============================================================================
def fit_linear_in_m(data, Tc):
    """Form A: delta_kij = (a0 + a1*Tr + a2*Tr^2) * m  [3 params]"""
    from scipy.optimize import least_squares

    brine = [d for d in data if d['csw'] > 0]
    if len(brine) < 3:
        return None, None, np.inf

    def residuals(params):
        a0, a1, a2 = params
        return [d['delta_kij'] - (a0 + a1*d['Tr'] + a2*d['Tr']**2)*d['csw']
                for d in brine]

    result = least_squares(residuals, [0.0, 0.0, 0.0])
    params = result.x

    # Compute RMSE
    errs = [(d['delta_kij'] - (params[0]+params[1]*d['Tr']+params[2]*d['Tr']**2)*d['csw'])**2
            for d in data]
    rmse = np.sqrt(np.mean(errs))

    def kij_func(T_K, m, p=params, tc=Tc):
        Tr = T_K / tc
        return (p[0] + p[1]*Tr + p[2]*Tr**2) * m

    return params, kij_func, rmse


def fit_quadratic_in_m(data, Tc):
    """Form B: delta_kij = (a0+a1*Tr)*m + (b0+b1*Tr)*m^2  [4 params]"""
    from scipy.optimize import least_squares

    brine = [d for d in data if d['csw'] > 0]
    if len(brine) < 4:
        return None, None, np.inf

    def residuals(params):
        a0, a1, b0, b1 = params
        return [d['delta_kij'] - ((a0+a1*d['Tr'])*d['csw'] + (b0+b1*d['Tr'])*d['csw']**2)
                for d in brine]

    result = least_squares(residuals, [0.0, 0.0, 0.0, 0.0])
    params = result.x

    errs = [(d['delta_kij'] - ((params[0]+params[1]*d['Tr'])*d['csw'] +
             (params[2]+params[3]*d['Tr'])*d['csw']**2))**2
            for d in data]
    rmse = np.sqrt(np.mean(errs))

    def kij_func(T_K, m, p=params, tc=Tc):
        Tr = T_K / tc
        return (p[0]+p[1]*Tr)*m + (p[2]+p[3]*Tr)*m**2

    return params, kij_func, rmse


def fit_ks_minimizing(data, Tc, gas, kij_fw_func, init_params=None):
    """
    Form C: delta_kij = (a0 + a1*Tr + a2*Tr^2) * m
    Minimizes implied ks error (not kij error).
    """
    vle = SWBinaryVLE(gas, salinity_molal=0.0)

    def ks_objective(params):
        a0, a1, a2 = params
        sq_errors = []

        for d in data:
            if d['csw'] <= 0:
                continue

            T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']
            Tr = T_K / Tc

            # True ks from agreed model
            ks_true = get_target_ks(gas, T_K, csw, d['P_bar'])

            # Implied ks from embedded BIP
            kij_fw = kij_fw_func(T_K)
            kij_brine = kij_fw + (a0 + a1*Tr + a2*Tr**2) * csw

            try:
                x_fresh = vle._calc_x_with_kij(T_K, P_Pa, kij_fw)
                x_brine = vle._calc_x_with_kij(T_K, P_Pa, kij_brine)
            except Exception:
                continue

            if x_brine > 0 and x_fresh > 0:
                ks_implied = np.log10(x_fresh / x_brine) / csw
                sq_errors.append((ks_implied - ks_true)**2)

        return np.mean(sq_errors) if sq_errors else 1e10

    if init_params is None:
        init_params = [0.0, 0.0, 0.0]

    result = minimize(ks_objective, init_params, method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-12})
    params = result.x

    # Compute kij RMSE for comparison
    errs = [(d['delta_kij'] - (params[0]+params[1]*d['Tr']+params[2]*d['Tr']**2)*d['csw'])**2
            for d in data]
    rmse = np.sqrt(np.mean(errs))

    def kij_func(T_K, m, p=params, tc=Tc):
        Tr = T_K / tc
        return (p[0] + p[1]*Tr + p[2]*Tr**2) * m

    return params, kij_func, rmse


def fit_solubility_minimizing(data, Tc, gas, kij_fw_func, init_params=None):
    """
    Form D: delta_kij = (a0 + a1*Tr + a2*Tr^2) * m
    Minimizes solubility relative error directly. RECOMMENDED.
    Better than ks-minimizing for high-solubility gases (CO2, H2S).
    """
    vle = SWBinaryVLE(gas, salinity_molal=0.0)

    brine_data = [d for d in data if d['csw'] > 0 and d['x_target'] > 0]

    def sol_objective(params):
        a0, a1, a2 = params
        sq_errors = []

        for d in brine_data:
            T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']
            Tr = T_K / Tc

            kij_fw = kij_fw_func(T_K)
            kij_brine = kij_fw + (a0 + a1*Tr + a2*Tr**2) * csw

            try:
                x_pred = vle._calc_x_with_kij(T_K, P_Pa, kij_brine)
            except Exception:
                sq_errors.append(1.0)
                continue

            x_target = d['x_target']
            if x_target > 0 and x_pred > 0:
                rel_err = (x_pred - x_target) / x_target
                sq_errors.append(rel_err**2)
            else:
                sq_errors.append(1.0)

        return np.mean(sq_errors) if sq_errors else 1e10

    if init_params is None:
        init_params = [0.0, 0.0, 0.0]

    result = minimize(sol_objective, init_params, method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-14})
    params = result.x

    errs = [(d['delta_kij'] - (params[0]+params[1]*d['Tr']+params[2]*d['Tr']**2)*d['csw'])**2
            for d in data]
    rmse = np.sqrt(np.mean(errs))

    def kij_func(T_K, m, p=params, tc=Tc):
        Tr = T_K / tc
        return (p[0] + p[1]*Tr + p[2]*Tr**2) * m

    return params, kij_func, rmse


def fit_quad_m_solubility(data, Tc, gas, kij_fw_func, init_params=None):
    """
    Form E: delta_kij = (a0 + a1*Tr + a2*Tr^2) * m + (b0 + b1*Tr) * m^2
    Quadratic in m with T-dependent m^2 coefficient. [5 params]
    Minimizes solubility relative error. For gases where Duan ks depends on m.
    """
    vle = SWBinaryVLE(gas, salinity_molal=0.0)

    brine_data = [d for d in data if d['csw'] > 0 and d['x_target'] > 0]

    def sol_objective(params):
        a0, a1, a2, b0, b1 = params
        sq_errors = []

        for d in brine_data:
            T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']
            Tr = T_K / Tc

            kij_fw = kij_fw_func(T_K)
            kij_brine = kij_fw + (a0 + a1*Tr + a2*Tr**2) * csw + (b0 + b1*Tr) * csw**2

            try:
                x_pred = vle._calc_x_with_kij(T_K, P_Pa, kij_brine)
            except Exception:
                sq_errors.append(1.0)
                continue

            x_target = d['x_target']
            if x_target > 0 and x_pred > 0:
                rel_err = (x_pred - x_target) / x_target
                sq_errors.append(rel_err**2)
            else:
                sq_errors.append(1.0)

        return np.mean(sq_errors) if sq_errors else 1e10

    if init_params is None:
        init_params = [0.0, 0.0, 0.0, 0.0, 0.0]

    result = minimize(sol_objective, init_params, method='Nelder-Mead',
                     options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-16})
    params = result.x

    def kij_func(T_K, m, p=params, tc=Tc):
        Tr = T_K / tc
        return (p[0] + p[1]*Tr + p[2]*Tr**2) * m + (p[3] + p[4]*Tr) * m**2

    return params, kij_func, 0.0


# =============================================================================
# Evaluate implied ks accuracy
# =============================================================================
def evaluate_ks_accuracy(data, gas, Tc, kij_fw_func, delta_func):
    """Evaluate how well embedded BIP reproduces target ks."""
    vle = SWBinaryVLE(gas, salinity_molal=0.0)

    results = []
    for d in data:
        if d['csw'] <= 0:
            continue

        T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']

        ks_true = get_target_ks(gas, T_K, csw, d['P_bar'])

        kij_fw = kij_fw_func(T_K)
        kij_brine = kij_fw + delta_func(T_K, csw)

        try:
            x_fresh = vle._calc_x_with_kij(T_K, P_Pa, kij_fw)
            x_brine = vle._calc_x_with_kij(T_K, P_Pa, kij_brine)
        except Exception:
            continue

        if x_brine > 0 and x_fresh > 0:
            ks_implied = np.log10(x_fresh / x_brine) / csw
            x_target = d['x_target']
            x_error_pct = (x_brine - x_target) / x_target * 100 if x_target > 0 else 0

            results.append({
                'T_C': d['T_C'], 'P_bar': d['P_bar'], 'csw': csw,
                'ks_true': ks_true, 'ks_implied': ks_implied,
                'ks_error': ks_implied - ks_true,
                'x_target': x_target, 'x_pred': x_brine,
                'x_error_pct': x_error_pct,
            })

    if not results:
        return {'mae_ks': np.inf, 'max_ks_err': np.inf,
                'mae_x_pct': np.inf, 'max_x_pct': np.inf, 'n': 0}

    ks_errs = [abs(r['ks_error']) for r in results]
    x_errs = [abs(r['x_error_pct']) for r in results]

    return {
        'mae_ks': np.mean(ks_errs),
        'max_ks_err': np.max(ks_errs),
        'mae_x_pct': np.mean(x_errs),
        'max_x_pct': np.max(x_errs),
        'within_1pct': 100 * sum(1 for e in x_errs if e < 1) / len(x_errs),
        'within_2pct': 100 * sum(1 for e in x_errs if e < 2) / len(x_errs),
        'within_5pct': 100 * sum(1 for e in x_errs if e < 5) / len(x_errs),
        'n': len(results),
        'details': results,
    }


# =============================================================================
# S&W embedded salinity comparison (for CH4, N2, CO2, H2S)
# =============================================================================
def evaluate_sw_embedded(gas, data):
    """Evaluate S&W's original embedded salinity BIP (Eqs 12-15) vs target ks."""
    vle = SWBinaryVLE(gas, salinity_molal=0.0)

    results = []
    for d in data:
        if d['csw'] <= 0:
            continue

        T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']

        ks_true = get_target_ks(gas, T_K, csw, d['P_bar'])

        # S&W kij with embedded salinity
        if gas in ('CH4', 'C2H6', 'C3H8', 'iC4H10', 'nC4H10',
                   'iC5H12', 'nC5H12', 'nC6H14', 'nC7H16', 'nC8H18', 'nC10H22'):
            kij_sw_brine = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                               COMPONENTS[gas].Tc, csw)
            kij_sw_fresh = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                               COMPONENTS[gas].Tc, 0.0)
        elif gas == 'CO2':
            kij_sw_brine = kij_aq_co2(T_K, csw)
            kij_sw_fresh = kij_aq_co2(T_K, 0.0)
        elif gas == 'N2':
            kij_sw_brine = kij_aq_n2(T_K, csw)
            kij_sw_fresh = kij_aq_n2(T_K, 0.0)
        elif gas == 'H2S':
            # H2S S&W Eq 15 has no salinity terms
            kij_sw_brine = kij_aq_h2s(T_K, csw)
            kij_sw_fresh = kij_aq_h2s(T_K, 0.0)
        else:
            continue

        try:
            x_sw_fresh = vle._calc_x_with_kij(T_K, P_Pa, kij_sw_fresh)
            x_sw_brine = vle._calc_x_with_kij(T_K, P_Pa, kij_sw_brine)
        except Exception:
            continue

        if x_sw_brine > 0 and x_sw_fresh > 0:
            ks_sw_implied = np.log10(x_sw_fresh / x_sw_brine) / csw
        else:
            ks_sw_implied = 0.0

        x_target = d['x_target']
        x_error_pct = (x_sw_brine - x_target) / x_target * 100 if x_target > 0 else 0

        results.append({
            'T_C': d['T_C'], 'P_bar': d['P_bar'], 'csw': csw,
            'ks_true': ks_true, 'ks_sw_implied': ks_sw_implied,
            'ks_error': ks_sw_implied - ks_true,
            'x_target': x_target, 'x_sw_brine': x_sw_brine,
            'x_error_pct': x_error_pct,
        })

    if not results:
        return None

    ks_errs = [abs(r['ks_error']) for r in results]
    x_errs = [abs(r['x_error_pct']) for r in results]

    return {
        'mae_ks': np.mean(ks_errs),
        'max_ks_err': np.max(ks_errs),
        'mae_x_pct': np.mean(x_errs),
        'max_x_pct': np.max(x_errs),
        'within_5pct': 100 * sum(1 for e in x_errs if e < 5) / len(x_errs),
        'n': len(results),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 78)
    print("FIT EMBEDDED SALINITY BIP CORRELATIONS — ALL PAPER 2 GASES")
    print("=" * 78)

    all_results = {}

    # Paper 1 published coefficients for H2 — NOT refitted
    H2_PAPER1_PARAMS = np.array([0.3833, -0.06595, 0.003321])

    for gas in ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8', 'nC4H10']:
        cfg = GAS_CONFIGS[gas]
        kij_fw_func = KIJ_FW_FUNCS[gas]
        Tc = cfg['Tc']

        print(f"\n{'='*78}")
        print(f"  {gas}  (Tc = {Tc} K, ks source = {cfg['ks_source']})")
        print(f"{'='*78}")

        # Generate synthetic data
        print(f"\n  Generating synthetic data...")
        data = generate_synthetic_data(gas)
        n_total = len(data)
        n_brine = len([d for d in data if d['csw'] > 0])
        print(f"  Generated {n_total} points ({n_brine} brine)")

        if n_brine < 3:
            print(f"  WARNING: Insufficient brine data for {gas}, skipping")
            continue

        # H2: use Paper 1 published coefficients (ks-error minimized)
        if gas == 'H2':
            print(f"\n  Using Paper 1 published coefficients (not refitting):")
            params_fixed = H2_PAPER1_PARAMS
            print(f"    a0={params_fixed[0]:.4f}, a1={params_fixed[1]:.5f}, a2={params_fixed[2]:.6f}")

            def h2_delta_func(T_K, m, p=params_fixed, tc=Tc):
                Tr = T_K / tc
                return (p[0] + p[1]*Tr + p[2]*Tr**2) * m

            forms = {}
            acc_h2 = evaluate_ks_accuracy(data, gas, Tc, kij_fw_func, h2_delta_func)
            forms['Paper 1 (ks-err)'] = {'params': params_fixed, 'func': h2_delta_func, 'acc': acc_h2}

            print(f"\n  --- Accuracy (Paper 1 coefficients) ---")
            print(f"  {'Form':<45} {'MAE(ks)':>8} {'Max(ks)':>8} {'MAE(x%)':>8} {'Max(x%)':>8} {'<5%':>6}")
            print(f"  {'-'*90}")
            print(f"  {'Paper 1 (ks-err minimized)':<45} {acc_h2['mae_ks']:8.4f} {acc_h2['max_ks_err']:8.4f} "
                  f"{acc_h2['mae_x_pct']:8.2f} {acc_h2['max_x_pct']:8.2f} {acc_h2['within_5pct']:5.0f}%")

            all_results[gas] = {
                'Tc': Tc,
                'ks_source': cfg['ks_source'],
                'params': params_fixed,
                'forms': forms,
                'best': 'Paper 1 (ks-err)',
                'n_points': n_total,
                'n_brine': n_brine,
            }

            print(f"\n  RECOMMENDED for {gas} (Paper 1 published):")
            print(f"    kij(T, m) = kij_fw(T) + ({params_fixed[0]:.4f} + {params_fixed[1]:.5f}*Tr "
                  f"+ {params_fixed[2]:.6f}*Tr^2) * m")
            print(f"    Implied ks MAE = {acc_h2['mae_ks']:.4f}")
            print(f"    Solubility MAE = {acc_h2['mae_x_pct']:.2f}%, Max = {acc_h2['max_x_pct']:.2f}%")
            continue

        # Fit Form A: linear in m (kij-error minimization)
        print(f"\n  Form A: delta = (a0 + a1*Tr + a2*Tr^2) * m  [kij-error]")
        params_A, func_A, rmse_A = fit_linear_in_m(data, Tc)
        if params_A is not None:
            print(f"    a0={params_A[0]:.6f}, a1={params_A[1]:.6f}, a2={params_A[2]:.8f}")
            print(f"    kij RMSE = {rmse_A:.6f}")

        # Fit Form B: quadratic in m
        print(f"\n  Form B: delta = (a0+a1*Tr)*m + (b0+b1*Tr)*m^2  [kij-error]")
        params_B, func_B, rmse_B = fit_quadratic_in_m(data, Tc)
        if params_B is not None:
            print(f"    a0={params_B[0]:.6f}, a1={params_B[1]:.6f}")
            print(f"    b0={params_B[2]:.6f}, b1={params_B[3]:.6f}")
            print(f"    kij RMSE = {rmse_B:.6f}")

        # Fit Form C: linear in m (ks-error minimization)
        print(f"\n  Form C: delta = (a0 + a1*Tr + a2*Tr^2) * m  [ks-error]")
        init_C = list(params_A) if params_A is not None else [0.0, 0.0, 0.0]
        params_C, func_C, rmse_C = fit_ks_minimizing(data, Tc, gas, kij_fw_func, init_C)
        print(f"    a0={params_C[0]:.6f}, a1={params_C[1]:.6f}, a2={params_C[2]:.8f}")
        print(f"    kij RMSE = {rmse_C:.6f}")

        # Fit Form D: linear in m (solubility-error minimization) — RECOMMENDED
        print(f"\n  Form D: delta = (a0 + a1*Tr + a2*Tr^2) * m  [x-error, RECOMMENDED]")
        init_D = list(params_A) if params_A is not None else [0.0, 0.0, 0.0]
        params_D, func_D, rmse_D = fit_solubility_minimizing(data, Tc, gas, kij_fw_func, init_D)
        print(f"    a0={params_D[0]:.6f}, a1={params_D[1]:.6f}, a2={params_D[2]:.8f}")
        print(f"    kij RMSE = {rmse_D:.6f}")

        # Evaluate all forms
        print(f"\n  --- Accuracy comparison ---")
        print(f"  {'Form':<45} {'MAE(ks)':>8} {'Max(ks)':>8} {'MAE(x%)':>8} {'Max(x%)':>8} {'<5%':>6}")
        print(f"  {'-'*90}")

        forms = {}
        if func_A is not None:
            acc_A = evaluate_ks_accuracy(data, gas, Tc, kij_fw_func, func_A)
            forms['A: linear-m (kij-err)'] = {'params': params_A, 'func': func_A, 'acc': acc_A}
            print(f"  {'A: linear-m (kij-err)':<45} {acc_A['mae_ks']:8.4f} {acc_A['max_ks_err']:8.4f} "
                  f"{acc_A['mae_x_pct']:8.2f} {acc_A['max_x_pct']:8.2f} {acc_A['within_5pct']:5.0f}%")

        if func_B is not None:
            acc_B = evaluate_ks_accuracy(data, gas, Tc, kij_fw_func, func_B)
            forms['B: quad-m (kij-err)'] = {'params': params_B, 'func': func_B, 'acc': acc_B}
            print(f"  {'B: quad-m (kij-err)':<45} {acc_B['mae_ks']:8.4f} {acc_B['max_ks_err']:8.4f} "
                  f"{acc_B['mae_x_pct']:8.2f} {acc_B['max_x_pct']:8.2f} {acc_B['within_5pct']:5.0f}%")

        acc_C = evaluate_ks_accuracy(data, gas, Tc, kij_fw_func, func_C)
        forms['C: linear-m (ks-err)'] = {'params': params_C, 'func': func_C, 'acc': acc_C}
        print(f"  {'C: linear-m (ks-err)':<45} {acc_C['mae_ks']:8.4f} {acc_C['max_ks_err']:8.4f} "
              f"{acc_C['mae_x_pct']:8.2f} {acc_C['max_x_pct']:8.2f} {acc_C['within_5pct']:5.0f}%")

        acc_D = evaluate_ks_accuracy(data, gas, Tc, kij_fw_func, func_D)
        forms['D: linear-m (x-err)'] = {'params': params_D, 'func': func_D, 'acc': acc_D}
        print(f"  {'D: linear-m (x-err)':<45} {acc_D['mae_ks']:8.4f} {acc_D['max_ks_err']:8.4f} "
              f"{acc_D['mae_x_pct']:8.2f} {acc_D['max_x_pct']:8.2f} {acc_D['within_5pct']:5.0f}%")

        # Form E: quadratic in m (for gases where ks depends on m, e.g. CO2 Duan)
        print(f"\n  Form E: delta = (a0+a1*Tr+a2*Tr^2)*m + (b0+b1*Tr)*m^2  [x-error, 5 params]")
        init_E = list(params_D) + [0.0, 0.0]
        params_E, func_E, _ = fit_quad_m_solubility(data, Tc, gas, kij_fw_func, init_E)
        print(f"    a0={params_E[0]:.6f}, a1={params_E[1]:.6f}, a2={params_E[2]:.8f}")
        print(f"    b0={params_E[3]:.6f}, b1={params_E[4]:.6f}")

        acc_E = evaluate_ks_accuracy(data, gas, Tc, kij_fw_func, func_E)
        forms['E: quad-m (x-err)'] = {'params': params_E, 'func': func_E, 'acc': acc_E}
        print(f"  {'E: quad-m (x-err)':<45} {acc_E['mae_ks']:8.4f} {acc_E['max_ks_err']:8.4f} "
              f"{acc_E['mae_x_pct']:8.2f} {acc_E['max_x_pct']:8.2f} {acc_E['within_5pct']:5.0f}%")

        # S&W embedded comparison (for gases that have S&W salinity terms)
        if gas in ('CH4', 'N2', 'CO2', 'C2H6', 'C3H8', 'nC4H10'):
            print(f"\n  --- S&W original embedded salinity ---")
            sw_acc = evaluate_sw_embedded(gas, data)
            if sw_acc:
                print(f"  {'S&W original embedded':<45} {sw_acc['mae_ks']:8.4f} {sw_acc['max_ks_err']:8.4f} "
                      f"{sw_acc['mae_x_pct']:8.2f} {sw_acc['max_x_pct']:8.2f} {sw_acc['within_5pct']:5.0f}%")
                forms['S&W original embedded'] = {'acc': sw_acc}

        # Select best form: always use Form D (linear-in-m) for consistency
        # Form E (quadratic-in-m) was previously used for CO2 but the marginal
        # improvement (2.44% vs 3.14% MAE) does not justify the added complexity.
        candidates = [
            ('D: linear-m (x-err)', params_D, acc_D),
        ]
        best_form, best_params, best_acc = candidates[0][0], candidates[0][1], candidates[0][2]

        # For reporting: mark the recommended form in the forms dict
        forms[best_form + ' [REC]'] = forms.pop(best_form)
        best_form = best_form + ' [REC]'

        all_results[gas] = {
            'Tc': Tc,
            'ks_source': cfg['ks_source'],
            'params': best_params,
            'forms': forms,
            'best': best_form,
            'n_points': n_total,
            'n_brine': n_brine,
            'n_best_params': len(best_params),
        }

        # Print recommended correlation
        print(f"\n  RECOMMENDED for {gas} ({best_form}):")
        if len(best_params) == 3:
            print(f"    kij(T, m) = kij_fw(T) + ({best_params[0]:.6f} + {best_params[1]:.6f}*Tr "
                  f"+ {best_params[2]:.8f}*Tr^2) * m")
        elif len(best_params) == 5:
            print(f"    kij(T, m) = kij_fw(T) + ({best_params[0]:.6f} + {best_params[1]:.6f}*Tr "
                  f"+ {best_params[2]:.8f}*Tr^2) * m")
            print(f"              + ({best_params[3]:.6f} + {best_params[4]:.6f}*Tr) * m^2")
        print(f"    where Tr = T / {Tc}")
        print(f"    Implied ks MAE = {best_acc['mae_ks']:.4f}")
        print(f"    Solubility MAE = {best_acc['mae_x_pct']:.2f}%, Max = {best_acc['max_x_pct']:.2f}%")

    # Write summary
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    print(f"\n{'Gas':<8} {'Tc':>6} {'a0':>10} {'a1':>10} {'a2':>12} "
          f"{'b0':>10} {'b1':>10} {'MAE(ks)':>8} {'MAE(x%)':>8} {'<5%':>6}")
    print("-" * 110)

    for gas, res in all_results.items():
        p = res['params']
        acc = res['forms'][res['best']]['acc']
        b0_str = f"{p[3]:10.6f}" if len(p) > 3 else f"{'—':>10}"
        b1_str = f"{p[4]:10.6f}" if len(p) > 4 else f"{'—':>10}"
        print(f"{gas:<8} {res['Tc']:6.2f} {p[0]:10.6f} {p[1]:10.6f} {p[2]:12.8f} "
              f"{b0_str} {b1_str} {acc['mae_ks']:8.4f} {acc['mae_x_pct']:8.2f} "
              f"{acc['within_5pct']:5.0f}%")

    # Write CSV report
    write_csv_report(all_results)
    write_text_report(all_results)

    # Generate manuscript-ready ks plots
    print(f"\n{'='*78}")
    print("GENERATING KS PLOTS")
    print(f"{'='*78}")
    generate_ks_plots(all_results)

    # Cache slow VLE-computed ks band data for generate_figures.py
    print(f"\n{'='*78}")
    print("CACHING KS BAND DATA")
    print(f"{'='*78}")
    cache_ks_band_data(all_results)

    print(f"\n{'='*78}")
    print("FITTING COMPLETE")
    print(f"{'='*78}")


def write_csv_report(all_results):
    """Write CSV with all fitting results."""
    filepath = os.path.join(OUTPUT_DIR, 'embedded_salinity_bip_all_gases.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("gas,ks_source,Tc_K,form,a0,a1,a2,b0,b1,"
                "mae_ks,max_ks_err,mae_x_pct,max_x_pct,within_5pct,n_points\n")

        for gas, res in all_results.items():
            for form_name, form_data in res['forms'].items():
                if 'params' not in form_data:
                    # S&W comparison row
                    acc = form_data['acc']
                    f.write(f"{gas},{res['ks_source']},{res['Tc']},"
                            f"S&W original,,,,,,{acc['mae_ks']:.6f},{acc['max_ks_err']:.6f},"
                            f"{acc['mae_x_pct']:.4f},{acc['max_x_pct']:.4f},"
                            f"{acc['within_5pct']:.1f},{acc['n']}\n")
                    continue

                p = form_data['params']
                acc = form_data['acc']
                # Handle 3-param vs 4-param forms
                if len(p) == 3:
                    f.write(f"{gas},{res['ks_source']},{res['Tc']},"
                            f"{form_name},{p[0]:.8f},{p[1]:.8f},{p[2]:.10f},,,"
                            f"{acc['mae_ks']:.6f},{acc['max_ks_err']:.6f},"
                            f"{acc['mae_x_pct']:.4f},{acc['max_x_pct']:.4f},"
                            f"{acc['within_5pct']:.1f},{acc['n']}\n")
                elif len(p) == 4:
                    f.write(f"{gas},{res['ks_source']},{res['Tc']},"
                            f"{form_name},{p[0]:.8f},{p[1]:.8f},,{p[2]:.8f},{p[3]:.8f},"
                            f"{acc['mae_ks']:.6f},{acc['max_ks_err']:.6f},"
                            f"{acc['mae_x_pct']:.4f},{acc['max_x_pct']:.4f},"
                            f"{acc['within_5pct']:.1f},{acc['n']}\n")

    print(f"\n  CSV saved: {filepath}")


def write_text_report(all_results):
    """Write human-readable text report."""
    filepath = os.path.join(OUTPUT_DIR, 'embedded_salinity_bip_all_gases_report.txt')

    with open(filepath, 'w') as f:
        f.write("=" * 78 + "\n")
        f.write("EMBEDDED SALINITY BIP CORRELATIONS — ALL PAPER 2 GASES\n")
        f.write("=" * 78 + "\n\n")

        f.write("General form:\n")
        f.write("  kij(T, m) = kij_fw(T) + (a0 + a1*Tr + a2*Tr^2) * m\n")
        f.write("  where Tr = T/Tc, m = NaCl molality (mol/kg)\n\n")

        f.write("Sechenov models used as fitting targets:\n")
        f.write("  CO2: Duan & Sun 2003 (Pitzer, P-dependent)\n")
        f.write("  H2S: Akinfiev et al. 2016 (Pitzer)\n")
        f.write("  CH4, N2, H2: S&W Equation 8 (Tb-based)\n\n")

        f.write("=" * 78 + "\n")
        f.write("RECOMMENDED COEFFICIENTS (ks-error minimizing)\n")
        f.write("=" * 78 + "\n\n")

        f.write(f"{'Gas':<6} {'Tc (K)':<10} {'a0':>12} {'a1':>12} {'a2':>14} "
                f"{'MAE(ks)':>8} {'MAE(x%)':>8}\n")
        f.write("-" * 78 + "\n")

        for gas, res in all_results.items():
            p = res['params']
            acc = res['forms'][res['best']]['acc']
            f.write(f"{gas:<6} {res['Tc']:<10.2f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:14.8f} "
                    f"{acc['mae_ks']:8.4f} {acc['mae_x_pct']:8.2f}\n")

        f.write("\n\n")
        f.write("=" * 78 + "\n")
        f.write("PER-GAS DETAILS\n")
        f.write("=" * 78 + "\n")

        for gas, res in all_results.items():
            f.write(f"\n{'─'*78}\n")
            f.write(f"{gas}  (Tc = {res['Tc']} K, ks source: {res['ks_source']})\n")
            f.write(f"{'─'*78}\n\n")

            f.write(f"  Freshwater kij:\n")
            if gas == 'CO2':
                f.write(f"    kij_fw = -2.290187 + 1.673983e-2*T - 4.388171e-5*T^2 + 4.078155e-8*T^3\n")
            elif gas == 'H2S':
                f.write(f"    kij_fw = -70.8170/T + 1540.9516*exp(-4532.56/T) + 0.21517\n")
            elif gas == 'CH4':
                f.write(f"    kij_fw = (-2.1642 + Tr) / (1.7325 + 0.2105*Tr),  Tr = T/190.60\n")
            elif gas == 'N2':
                f.write(f"    kij_fw = -1.6669 + 3.447873e-3*T  (T in K)\n")
            elif gas == 'H2':
                f.write(f"    kij_fw = (-14.6157 + Tr) / (3.5494 + 0.2230*Tr),  Tr = T/33.145\n")

            f.write(f"\n  Fitted forms:\n\n")

            for form_name, form_data in res['forms'].items():
                acc = form_data['acc']
                f.write(f"    {form_name}:\n")
                if 'params' in form_data:
                    p = form_data['params']
                    f.write(f"      Params: {', '.join(f'{v:.8f}' for v in p)}\n")
                f.write(f"      MAE(ks) = {acc['mae_ks']:.4f}, Max(ks) = {acc['max_ks_err']:.4f}\n")
                f.write(f"      MAE(x%) = {acc['mae_x_pct']:.2f}%, Max(x%) = {acc['max_x_pct']:.2f}%\n")
                if 'within_5pct' in acc:
                    f.write(f"      Within ±5%: {acc['within_5pct']:.0f}%\n")
                f.write(f"\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("NOTES\n")
        f.write("=" * 78 + "\n\n")
        f.write("1. This is NOT the recommended salinity approach for rigorous work.\n")
        f.write("   The gamma-phi method (K = gamma * phi_L / phi_V) with explicit\n")
        f.write("   Sechenov activity coefficients is more accurate and flexible.\n\n")
        f.write("2. Embedded salinity BIPs are provided for users of commercial\n")
        f.write("   simulators that cannot be modified to support gamma-phi.\n\n")
        f.write("3. For CO2, the ks from Duan & Sun 2003 depends on pressure.\n")
        f.write("   The embedded BIP was fitted at representative P grid.\n")
        f.write("   P-sensitivity is small (<10% over 50-500 bar).\n\n")
        f.write("4. For CH4 and N2, S&W's original embedded forms (Eqs 11-13)\n")
        f.write("   already target S&W Eq 8 ks. Our refit should be consistent\n")
        f.write("   or very close.\n")

    print(f"  Report saved: {filepath}")


# =============================================================================
# Manuscript-ready ks plots
# =============================================================================
def generate_ks_plots(all_results):
    """
    Per-gas ks plots showing:
      - Experimental ks data points (from extracted_ks_values.csv)
      - Selected explicit ks model (Duan 2003 / Akinfiev / S&W Eq 8)
      - S&W Eq 8 reference curve (if different from selected model)
      - Implied ks from S&W original embedded BIP
      - Implied ks from this work embedded BIP
    All implied ks computed at m = 1.0 mol/kg, P = 100 bar.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    from _lib_salting_library import (
        ks_sw_eq8 as ks_sw_eq8_lib, ks_duan2003_co2, ks_akinfiev_h2s,
        TB_K,
    )

    # Load experimental ks data
    ks_csv = os.path.join(OUTPUT_DIR, 'extracted_ks_values.csv')
    if os.path.exists(ks_csv):
        df_ks = pd.read_csv(ks_csv)
        print(f"\n  Loaded {len(df_ks)} experimental ks values from {ks_csv}")
    else:
        df_ks = pd.DataFrame()
        print(f"\n  WARNING: {ks_csv} not found, no experimental data points")

    M_BRINE = 1.0     # molal for implied ks
    P_BAR = 100.0      # bar for implied ks (main line)
    # Pressure band: 35-350 bar
    P_BAND_BAR = [35.0, 70.0, 105.0, 140.0, 210.0, 280.0, 350.0]
    T_RANGE = np.linspace(273.15, 473.15, 120)
    T_C_RANGE = T_RANGE - 273.15

    # All gases to plot (consistent order across all overview figures)
    gas_list = [g for g in ['CO2', 'H2S', 'N2', 'H2', 'CH4',
                            'C2H6', 'C3H8', 'nC4H10'] if g in all_results]
    n_gases = len(gas_list)
    n_cols = 2
    n_rows = 4

    # Per-gas plot config
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    # HC gases: min 0.1, max auto-scaled; others: manual limits
    HC_GASES = {'CH4', 'C2H6', 'C3H8', 'nC4H10'}
    gas_ylim = {
        'CO2': (0, 0.25), 'H2S': (-0.02, 0.22),
        'N2': (0, 0.16), 'H2': (0, 0.14),
        # HCs set below after plotting to auto-scale upper limit
    }

    # Scatter config for experimental data sources
    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 20))
    axes = axes.flatten()

    P_Pa = bar_to_pascal(P_BAR)

    for idx, gas in enumerate(gas_list):
        ax = axes[idx]
        res = all_results[gas]
        params = res['params']
        Tc = res['Tc']
        kij_fw_func_local = KIJ_FW_FUNCS[gas]

        # --- Explicit ks model curves ---
        # S&W Eq 8 shown only for HCs and H2 (where it is the target model)
        if gas == 'CO2':
            ks_model = np.array([ks_duan2003_co2(T, M_BRINE, P_BAR) for T in T_RANGE])
            # Duan 2003 pressure band (P-dependent)
            ks_duan_by_P = []
            for P_b in P_BAND_BAR:
                ks_at_P = []
                for T_K in T_RANGE:
                    try:
                        ks_at_P.append(ks_duan2003_co2(T_K, M_BRINE, P_b))
                    except Exception:
                        ks_at_P.append(np.nan)
                ks_duan_by_P.append(np.array(ks_at_P))
            ks_duan_arr = np.array(ks_duan_by_P)
            ks_duan_min = np.nanmin(ks_duan_arr, axis=0)
            ks_duan_max = np.nanmax(ks_duan_arr, axis=0)
            ax.fill_between(T_C_RANGE, ks_duan_min, ks_duan_max,
                            alpha=0.15, color='red', zorder=19,
                            label='Duan 2003 (35\u2013350 bar)')
            ax.plot(T_C_RANGE, ks_model, 'r-', lw=2.5,
                    label='Duan 2003 (target, 100 bar)', zorder=20)
        elif gas == 'H2S':
            T_valid = T_RANGE[(T_RANGE >= 283) & (T_RANGE <= 570)]
            T_valid_C = T_valid - 273.15
            ks_model = np.array([ks_akinfiev_h2s(T, M_BRINE) for T in T_valid])
            ax.plot(T_valid_C, ks_model, 'g-', lw=2.5,
                    label='Akinfiev 2016 (target)', zorder=20)
        elif gas == 'N2':
            # N2 uses S&W Eq 8 as target but don't show it (not an HC)
            # Plot as target line without the S&W Eq 8 label
            ks_sw = np.array([sw_equation_8_ks(T_C, COMPONENTS[gas].Tb) for T_C in T_C_RANGE])
            ax.plot(T_C_RANGE, ks_sw, 'k-', lw=2.5,
                    label='Target $k_s$', zorder=20)
        else:
            # HCs and H2: S&W Eq 8 is the target model — show with label
            ks_sw = np.array([sw_equation_8_ks(T_C, COMPONENTS[gas].Tb) for T_C in T_C_RANGE])
            ax.plot(T_C_RANGE, ks_sw, 'k-', lw=2.5,
                    label='S&W Eq 8 (target)', zorder=20)

        # --- Implied ks from THIS WORK embedded BIP ---
        # Use the best-form delta function from all_results
        best_form_key = res['best']
        delta_func = res['forms'][best_form_key]['func']
        vle = SWBinaryVLE(gas, salinity_molal=0.0)

        # Consistent T limit: 200°C for all gases
        T_max_K = celsius_to_kelvin(200)

        # Compute implied ks at multiple pressures for shaded band
        ks_by_pressure = []
        for P_b in P_BAND_BAR:
            P_Pa_band = bar_to_pascal(P_b)
            ks_at_P = []
            for T_K in T_RANGE:
                if T_K > T_max_K + 10:
                    ks_at_P.append(np.nan)
                    continue
                kij_fw = kij_fw_func_local(T_K)
                kij_brine = kij_fw + delta_func(T_K, M_BRINE)
                try:
                    x_fw = vle._calc_x_with_kij(T_K, P_Pa_band, kij_fw)
                    x_br = vle._calc_x_with_kij(T_K, P_Pa_band, kij_brine)
                    if x_fw > 0 and x_br > 0:
                        ks_at_P.append(np.log10(x_fw / x_br) / M_BRINE)
                    else:
                        ks_at_P.append(np.nan)
                except Exception:
                    ks_at_P.append(np.nan)
            ks_by_pressure.append(np.array(ks_at_P))

        ks_arr = np.array(ks_by_pressure)
        ks_band_min = np.nanmin(ks_arr, axis=0)
        ks_band_max = np.nanmax(ks_arr, axis=0)

        # Main line at representative pressure (100 bar)
        ks_this_work = []
        for T_K in T_RANGE:
            kij_fw = kij_fw_func_local(T_K)
            kij_brine = kij_fw + delta_func(T_K, M_BRINE)
            try:
                x_fw = vle._calc_x_with_kij(T_K, P_Pa, kij_fw)
                x_br = vle._calc_x_with_kij(T_K, P_Pa, kij_brine)
                if x_fw > 0 and x_br > 0:
                    ks_this_work.append(np.log10(x_fw / x_br) / M_BRINE)
                else:
                    ks_this_work.append(np.nan)
            except Exception:
                ks_this_work.append(np.nan)

        # Plot pressure band behind main line
        ax.fill_between(T_C_RANGE, ks_band_min, ks_band_max,
                        alpha=0.2, color='darkgreen', zorder=15,
                        label='This work embedded (35\u2013350 bar)')
        ax.plot(T_C_RANGE, ks_this_work, color='darkgreen',
                linestyle=(0, (3, 1, 1, 1)), lw=2.5,
                label='This work embedded (100 bar)', zorder=16)

        # --- Implied ks from S&W original embedded BIP ---
        gases_with_sw_embedded = {'CH4', 'N2', 'CO2', 'C2H6', 'C3H8',
                                  'iC4H10', 'nC4H10'}
        if gas in gases_with_sw_embedded:
            ks_sw_emb = []
            for T_K in T_RANGE:
                if gas in ('CH4', 'C2H6', 'C3H8', 'nC4H10',
                           'iC4H10', 'nC5H12', 'iC5H12'):
                    kij_sw_fresh = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                                       COMPONENTS[gas].Tc, 0.0)
                    kij_sw_brine = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                                       COMPONENTS[gas].Tc, M_BRINE)
                elif gas == 'CO2':
                    kij_sw_fresh = kij_aq_co2(T_K, 0.0)
                    kij_sw_brine = kij_aq_co2(T_K, M_BRINE)
                elif gas == 'N2':
                    kij_sw_fresh = kij_aq_n2(T_K, 0.0)
                    kij_sw_brine = kij_aq_n2(T_K, M_BRINE)
                else:
                    ks_sw_emb.append(np.nan)
                    continue
                try:
                    x_sw_fw = vle._calc_x_with_kij(T_K, P_Pa, kij_sw_fresh)
                    x_sw_br = vle._calc_x_with_kij(T_K, P_Pa, kij_sw_brine)
                    if x_sw_fw > 0 and x_sw_br > 0:
                        ks_sw_emb.append(np.log10(x_sw_fw / x_sw_br) / M_BRINE)
                    else:
                        ks_sw_emb.append(np.nan)
                except Exception:
                    ks_sw_emb.append(np.nan)

            ax.plot(T_C_RANGE, ks_sw_emb, color='gray', linestyle=':',
                    lw=2, label='S&W embedded (implied)', zorder=14)

        # --- Experimental ks data points ---
        if not df_ks.empty:
            gas_data = df_ks[df_ks['Gas'] == gas]
            if len(gas_data) > 0:
                sources = gas_data['Source'].unique()
                for si, src in enumerate(sources):
                    src_data = gas_data[gas_data['Source'] == src]
                    mi = si % len(markers)
                    ci = si % len(colors)
                    src_short = src.split(' ')[0]
                    if len(src.split(' ')) > 1:
                        src_short += ' ' + src.split(' ')[1][:4]
                    ax.scatter(src_data['T_C'], src_data['ks_data'],
                               marker=markers[mi], c=colors[ci],
                               s=50, alpha=0.7, edgecolors='k', linewidths=0.5,
                               zorder=10, label=f'{src_short}')

        # --- Panel formatting ---
        ax.set_title(f'{labels[idx]} {gas}', fontweight='bold')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel(r'$k_s$ (log$_{10}$ / molality)')
        ax.set_xlim(0, 200)
        if gas in HC_GASES:
            # HC gases: fixed lower limit of 0.1, auto-scale upper to encompass data
            ax.set_ylim(bottom=0.1)
            ax.autoscale(axis='y')
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0.1, max(ymax, 0.12))  # ensure minimum visible range
        else:
            ax.set_ylim(gas_ylim.get(gas, (0, 0.18)))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right', ncol=1)

    # Hide unused panels
    for i in range(n_gases, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Sechenov Coefficients: Explicit Models vs Embedded BIP Implied $k_s$\n'
                 f'(implied at m = {M_BRINE:.1f} mol/kg; line at {P_BAR:.0f} bar, band at 35\u2013350 bar)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'manuscript', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    for ext in ['pdf', 'png']:
        filepath = os.path.join(fig_dir, f'embedded_bip_ks_comparison.{ext}')
        fig.savefig(filepath, dpi=300)
        print(f"  Saved: {filepath}")

    plt.close(fig)


def cache_ks_band_data(all_results):
    """Cache the slow VLE-computed implied ks arrays for the overview figure.

    Saves per-gas arrays to an .npz file so generate_figures.py can
    plot the embedded BIP comparison without re-running VLE.
    Delete the cache file to force recalculation.
    """
    cache_path = os.path.join(OUTPUT_DIR, 'embedded_ks_band_cache.npz')

    M_BRINE = 1.0
    P_BAR = 100.0
    P_BAND_BAR = [35.0, 70.0, 105.0, 140.0, 210.0, 280.0, 350.0]
    T_RANGE = np.linspace(273.15, 473.15, 120)

    cache = {'T_RANGE': T_RANGE}

    gas_list = [g for g in ['CO2', 'H2S', 'N2', 'H2', 'CH4',
                            'C2H6', 'C3H8', 'nC4H10'] if g in all_results]

    for gas in gas_list:
        print(f"  Caching {gas}...")
        res = all_results[gas]
        Tc = res['Tc']
        kij_fw_func_local = KIJ_FW_FUNCS[gas]
        best_form_key = res['best']
        delta_func = res['forms'][best_form_key]['func']
        vle = SWBinaryVLE(gas, salinity_molal=0.0)

        # Consistent T limit: 200°C for all gases
        T_max_K = celsius_to_kelvin(200)
        P_Pa = bar_to_pascal(P_BAR)

        # --- This work: implied ks at main pressure ---
        ks_this_work = []
        for T_K in T_RANGE:
            kij_fw = kij_fw_func_local(T_K)
            kij_brine = kij_fw + delta_func(T_K, M_BRINE)
            try:
                x_fw = vle._calc_x_with_kij(T_K, P_Pa, kij_fw)
                x_br = vle._calc_x_with_kij(T_K, P_Pa, kij_brine)
                if x_fw > 0 and x_br > 0:
                    ks_this_work.append(np.log10(x_fw / x_br) / M_BRINE)
                else:
                    ks_this_work.append(np.nan)
            except Exception:
                ks_this_work.append(np.nan)
        cache[f'{gas}_ks_this_work'] = np.array(ks_this_work)

        # --- This work: pressure band ---
        ks_by_pressure = []
        for P_b in P_BAND_BAR:
            P_Pa_band = bar_to_pascal(P_b)
            ks_at_P = []
            for T_K in T_RANGE:
                if T_K > T_max_K + 10:
                    ks_at_P.append(np.nan)
                    continue
                kij_fw = kij_fw_func_local(T_K)
                kij_brine = kij_fw + delta_func(T_K, M_BRINE)
                try:
                    x_fw = vle._calc_x_with_kij(T_K, P_Pa_band, kij_fw)
                    x_br = vle._calc_x_with_kij(T_K, P_Pa_band, kij_brine)
                    if x_fw > 0 and x_br > 0:
                        ks_at_P.append(np.log10(x_fw / x_br) / M_BRINE)
                    else:
                        ks_at_P.append(np.nan)
                except Exception:
                    ks_at_P.append(np.nan)
            ks_by_pressure.append(np.array(ks_at_P))
        ks_arr = np.array(ks_by_pressure)
        cache[f'{gas}_ks_band_min'] = np.nanmin(ks_arr, axis=0)
        cache[f'{gas}_ks_band_max'] = np.nanmax(ks_arr, axis=0)

        # --- S&W original embedded BIP implied ks ---
        gases_with_sw_embedded = {'CH4', 'N2', 'CO2', 'C2H6', 'C3H8',
                                  'iC4H10', 'nC4H10'}
        if gas in gases_with_sw_embedded:
            ks_sw_emb = []
            for T_K in T_RANGE:
                if gas in ('CH4', 'C2H6', 'C3H8', 'nC4H10',
                           'iC4H10', 'nC5H12', 'iC5H12'):
                    kij_sw_fresh = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                                       COMPONENTS[gas].Tc, 0.0)
                    kij_sw_brine = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                                       COMPONENTS[gas].Tc, M_BRINE)
                elif gas == 'CO2':
                    kij_sw_fresh = kij_aq_co2(T_K, 0.0)
                    kij_sw_brine = kij_aq_co2(T_K, M_BRINE)
                elif gas == 'N2':
                    kij_sw_fresh = kij_aq_n2(T_K, 0.0)
                    kij_sw_brine = kij_aq_n2(T_K, M_BRINE)
                else:
                    ks_sw_emb.append(np.nan)
                    continue
                try:
                    x_sw_fw = vle._calc_x_with_kij(T_K, P_Pa, kij_sw_fresh)
                    x_sw_br = vle._calc_x_with_kij(T_K, P_Pa, kij_sw_brine)
                    if x_sw_fw > 0 and x_sw_br > 0:
                        ks_sw_emb.append(np.log10(x_sw_fw / x_sw_br) / M_BRINE)
                    else:
                        ks_sw_emb.append(np.nan)
                except Exception:
                    ks_sw_emb.append(np.nan)
            cache[f'{gas}_ks_sw_emb'] = np.array(ks_sw_emb)

    np.savez_compressed(cache_path, **cache)
    print(f"\n  Cached ks band data to {cache_path}")


if __name__ == '__main__':
    main()
