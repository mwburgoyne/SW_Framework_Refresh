#!/usr/bin/env python3
"""
Fit S&W-specific K-value correlations for flash initialization.

This script generates the component-specific K-value parameters stored in
_SW_KVALUE_PARAMS, _SW_KVALUE_HEAVY, and _SW_KVALUE_WATER in vle_engine.py.
Run it to reproduce or update the fits from scratch.

Method
------
1. For each of the 15 S&W gas components, run binary VLE calculations
   across a (T, P) grid spanning 14.7–15000 psia and 32–300°F.
2. From each binary VLE result, compute the "true" S&W K-values:
     K_gas = (1 - y_H2O) / x_gas
     K_H2O = y_H2O / (1 - x_gas)
3. Fit component-specific correlations in ln(K) space:
   - Light gases (H2–nC4H10): Cross form with 6 parameters
       ln(K) = a + b(Tc/T) + c·ln(P/Pc) + d(Tc/T)² + e·ln(P/Pc)² + f(Tc/T)·ln(P/Pc)
   - Heavy HCs (iC5H12–nC10H22): LogLinear form with 3 params + floor(K=10)
       ln(K) = a + b(Tc/T) + c·ln(P/Pc)
   - Water: Universal 6-param fit in T and P
       ln(K) = a + b/T + c·ln(P) + d/T² + e·ln(P)/T + f·ln(P)²
4. Fitting uses robust Huber loss (scipy least_squares, f_scale=0.5)
   with 10 random restarts to avoid local minima.
5. Report per-component MARE and K < 1 violation counts.

Why the Cross form?
-------------------
The standard Wilson correlation K = (Pc/P)·exp(5.373(1+ω)(1-Tc/T)) was
developed for hydrocarbon-hydrocarbon equilibria. For gas-water systems in
the S&W framework, it gives K_gas < 1 (qualitatively wrong) for:
  - H2S at 44% of conditions
  - nC4H10 at 69% of conditions
  - All C5+ at 80–100% of conditions

The Cross form adds a T-P interaction term f(Tc/T)·ln(P/Pc) that captures
how the temperature dependence of gas-water K-values changes with pressure.
This is real physics: at high pressure, the EOS non-ideality affects both
phases differently, and the simple Wilson separation of T and P effects
breaks down. The Cross form is uniformly best for all 9 light gases
(MARE 13–28% vs 95–100% for Wilson).

Why not a universal form?
-------------------------
Each gas has individually fitted parameters because the S&W framework uses
gas-specific BIP correlations (kij_AQ and kij_NA) that are NOT captured by
Wilson's Tc/Pc/ω parameterization. The K-values emerge from the full EOS
solution with these BIPs, and they cannot be predicted from critical
properties alone.

Heavy HCs (C5+) use a simpler LogLinear form because their K-values span
3–11 orders of magnitude (from ~2 at 1034 bar to 10^11 at 1 bar), which
no polynomial form can accurately capture. Since K >> 1 always holds for
these components, the floor(K=10) ensures qualitative correctness; the
exact magnitude matters little for SS convergence.

Usage
-----
  python fit_flash_kvalues.py          # Run full fitting + validation
  python fit_flash_kvalues.py --quick  # Skip VLE, just show current params

Author: Mark Burgoyne
"""

import sys
import os
import time
import warnings
import numpy as np
from scipy.optimize import least_squares
from collections import OrderedDict

warnings.filterwarnings('ignore')

# Add parent directory (code/) to path for shared module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

from _lib_vle_engine import (SWBinaryVLE, COMPONENTS, GAS_SPECIES,
                        SWMultiComponentFlash, _SW_KVALUE_PARAMS,
                        _SW_KVALUE_HEAVY, _SW_KVALUE_WATER,
                        _sw_kvalue_init, kij_aq_hydrocarbon,
                        kij_aq_co2_proposed, kij_aq_h2s_proposed,
                        kij_aq_ch4, kij_aq_h2_proposed,
                        kij_aq_c2h6_proposed, kij_aq_c3h8_proposed)

# ─────────────────────────────────────────────────────────────────────
# Proposed freshwater kij_AQ functions (from VLE engine — Paper 2 base cases)
# ─────────────────────────────────────────────────────────────────────
# For gases not listed here, the S&W original (via calc_gas_solubility) is used.
PROPOSED_KIJ = {
    'CO2': lambda T_K: kij_aq_co2_proposed(T_K),
    'H2S': lambda T_K: kij_aq_h2s_proposed(T_K),
    'CH4': lambda T_K: kij_aq_ch4(T_K),
    'H2':  lambda T_K: kij_aq_h2_proposed(T_K),
    'C2H6': lambda T_K: kij_aq_c2h6_proposed(T_K),
    'C3H8': lambda T_K: kij_aq_c3h8_proposed(T_K),
}

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
PSI_TO_PA = 6894.757
F_TO_K = lambda F: (F - 32) * 5/9 + 273.15

# Grid: 20 temperatures × 25 pressures = 500 points per gas
P_PSIA = np.geomspace(14.7, 15000, 25)
T_F = np.linspace(32, 300, 20)
P_PA = P_PSIA * PSI_TO_PA
T_K = np.array([F_TO_K(t) for t in T_F])

LIGHT_GASES = ['H2', 'CO2', 'N2', 'H2S', 'CH4', 'C2H6', 'C3H8',
               'iC4H10', 'nC4H10']
HEAVY_GASES = ['iC5H12', 'nC5H12', 'nC6H14', 'nC7H16', 'nC8H18', 'nC10H22']


def wilson_standard(T, P, Tc, Pc, omega, A=5.373):
    """Standard Wilson K-value correlation."""
    return (Pc/P) * np.exp(A * (1+omega) * (1 - Tc/T))


# ─────────────────────────────────────────────────────────────────────
# Functional forms
# ─────────────────────────────────────────────────────────────────────
def form_cross(params, T, P, Tc, Pc):
    """Cross form: ln(K) = a + b(Tc/T) + c·ln(P/Pc) + d(Tc/T)² + e·ln(P/Pc)² + f(Tc/T)·ln(P/Pc)"""
    a, b, c, d, e, f = params
    Tr_inv = Tc / T
    lnPr = np.log(P / Pc)
    return a + b*Tr_inv + c*lnPr + d*Tr_inv**2 + e*lnPr**2 + f*Tr_inv*lnPr


def form_log_linear(params, T, P, Tc, Pc):
    """LogLinear form: ln(K) = a + b(Tc/T) + c·ln(P/Pc)"""
    a, b, c = params
    return a + b*(Tc/T) + c*np.log(P/Pc)


def form_water(params, T, P):
    """Water form: ln(K) = a + b/T + c·ln(P) + d/T² + e·ln(P)/T + f·ln(P)²"""
    a, b, c, d, e, f = params
    return a + b/T + c*np.log(P) + d/T**2 + e*np.log(P)/T + f*np.log(P)**2


def fit_robust(func, n_params, bounds, T, P, Tc, Pc, lnK_data, n_restarts=10):
    """Fit using robust Huber loss with multiple random restarts."""
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    best_result = None
    best_cost = np.inf

    for trial in range(n_restarts):
        if trial == 0:
            x0 = np.zeros(n_params)
            x0[0] = 5.0
            if n_params >= 2: x0[1] = -5.0
            if n_params >= 3: x0[2] = -1.0
        else:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])

        def residual(params):
            return func(params, T, P, Tc, Pc) - lnK_data

        try:
            result = least_squares(residual, x0, bounds=(lb, ub),
                                   loss='huber', f_scale=0.5,
                                   max_nfev=10000, ftol=1e-14, xtol=1e-14)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None, 999, 999

    params = best_result.x
    lnK_pred = func(params, T, P, Tc, Pc)
    K_pred = np.exp(lnK_pred)
    K_data = np.exp(lnK_data)
    rmse = np.sqrt(np.mean((lnK_pred - lnK_data)**2))
    mare = np.mean(np.abs((K_pred - K_data) / K_data)) * 100

    return params, rmse, mare


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    quick_mode = '--quick' in sys.argv

    if quick_mode:
        print("="*80)
        print("CURRENT K-VALUE PARAMETERS (from vle_engine.py)")
        print("="*80)
        print(f"\n  Light gases — Cross form:")
        print(f"  ln(K) = a + b(Tc/T) + c·ln(P/Pc) + d(Tc/T)² + e·ln(P/Pc)² + f(Tc/T)·ln(P/Pc)")
        print(f"\n  {'Gas':<8s} {'a':>8s} {'b':>8s} {'c':>8s} {'d':>8s} {'e':>8s} {'f':>8s}  MARE%")
        print(f"  {'─'*74}")
        for gas, (p, mare) in _SW_KVALUE_PARAMS.items():
            print(f"  {gas:<8s} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f} "
                  f"{p[3]:>8.4f} {p[4]:>8.4f} {p[5]:>8.4f}  {mare:.1f}")

        print(f"\n  Heavy HCs — LogLinear + floor(K=10):")
        print(f"  ln(K) = a + b(Tc/T) + c·ln(P/Pc)")
        print(f"\n  {'Gas':<8s} {'a':>8s} {'b':>8s} {'c':>8s}")
        print(f"  {'─'*34}")
        for gas, p in _SW_KVALUE_HEAVY.items():
            print(f"  {gas:<8s} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f}")

        print(f"\n  Water — Universal 6-param:")
        print(f"  ln(K) = a + b/T + c·ln(P) + d/T² + e·ln(P)/T + f·ln(P)²")
        wp = _SW_KVALUE_WATER
        print(f"  [{', '.join(f'{p:.4f}' for p in wp)}]")
        return

    print("="*80)
    print("S&W K-VALUE CORRELATION FITTING")
    print("="*80)
    print(f"\nGrid: {len(T_K)}T × {len(P_PA)}P = {len(T_K)*len(P_PA)} pts/gas")
    print(f"  T: {T_K[0]:.1f}–{T_K[-1]:.1f} K ({T_F[0]:.0f}–{T_F[-1]:.0f} °F)")
    print(f"  P: {P_PA[0]/1e5:.2f}–{P_PA[-1]/1e5:.1f} bar "
          f"({P_PSIA[0]:.1f}–{P_PSIA[-1]:.0f} psia)")

    # Phase 1: Compute K-values
    print(f"\n{'─'*80}")
    print("PHASE 1: Computing binary VLE K-values")
    print(f"{'─'*80}")

    results = {}
    t0 = time.time()

    for gi, gas in enumerate(GAS_SPECIES):
        props = COMPONENTS[gas]
        vle = SWBinaryVLE(gas, salinity_molal=0.0)

        Tg, Pg, Kg, Kw = [], [], [], []
        n_fail = 0

        kij_func = PROPOSED_KIJ.get(gas, None)

        for T in T_K:
            for P in P_PA:
                try:
                    if kij_func is not None:
                        kij_val = kij_func(T)
                        x_gas = vle._calc_x_with_kij(T, P, kij_val)
                    else:
                        x_gas = vle.calc_gas_solubility(T, P)
                    y_H2O = vle.calc_water_content(T, P)
                    if (x_gas is None or y_H2O is None or
                            x_gas <= 0 or x_gas >= 1 or y_H2O < 0 or y_H2O >= 1):
                        n_fail += 1; continue
                    K_gas = (1.0 - y_H2O) / x_gas
                    K_H2O = y_H2O / (1.0 - x_gas)
                    if K_gas <= 0 or K_H2O <= 0 or not np.isfinite(K_gas*K_H2O):
                        n_fail += 1; continue
                    Tg.append(T); Pg.append(P); Kg.append(K_gas); Kw.append(K_H2O)
                except Exception:
                    n_fail += 1

        results[gas] = {
            'T': np.array(Tg), 'P': np.array(Pg),
            'K_gas': np.array(Kg), 'K_H2O': np.array(Kw),
            'Tc': props.Tc, 'Pc': props.Pc, 'omega': props.omega
        }

        elapsed = time.time() - t0
        print(f"  [{gi+1:2d}/15] {gas:8s}: {len(Kg)} pts  "
              f"K=[{min(Kg):.1e}, {max(Kg):.1e}]  ({elapsed:.0f}s)")

    print(f"\n  Total VLE time: {time.time()-t0:.0f}s")

    # Phase 2: Fit light gases
    print(f"\n{'─'*80}")
    print("PHASE 2: Fitting Cross form to light gases")
    print(f"{'─'*80}")
    print(f"\n  Cross form: ln(K) = a + b(Tc/T) + c·ln(P/Pc) + d(Tc/T)²")
    print(f"                      + e·ln(P/Pc)² + f(Tc/T)·ln(P/Pc)")
    print(f"\n  {'Gas':<8s} {'RMSE(lnK)':>10s} {'MARE%':>8s} {'K<1':>5s} "
          f"{'Wilson MARE%':>12s} {'Wilson K<1':>10s}")
    print(f"  {'─'*60}")

    fitted_params = {}
    cross_bounds = [(-50, 50)] * 6

    for gas in LIGHT_GASES:
        r = results[gas]
        T, P = r['T'], r['P']
        Tc, Pc = r['Tc'], r['Pc']
        lnK = np.log(r['K_gas'])

        params, rmse, mare = fit_robust(
            form_cross, 6, cross_bounds, T, P, Tc, Pc, lnK)

        # Count violations
        K_fit = np.exp(form_cross(params, T, P, Tc, Pc))
        fit_bad = np.sum((K_fit < 1) & (r['K_gas'] > 1))

        K_wil = wilson_standard(T, P, Tc, Pc, r['omega'])
        wil_mare = np.mean(np.abs((K_wil - r['K_gas'])/r['K_gas'])) * 100
        wil_bad = np.sum(K_wil < 1)

        fitted_params[gas] = (list(params), mare)
        print(f"  {gas:<8s} {rmse:>10.4f} {mare:>8.1f} {fit_bad:>5d} "
              f"{wil_mare:>12.1f} {wil_bad:>10d}")

    # Phase 3: Fit heavy gases
    print(f"\n{'─'*80}")
    print("PHASE 3: Fitting LogLinear to heavy HCs (+ floor K=10)")
    print(f"{'─'*80}")

    heavy_params = {}
    ll_bounds = [(-50, 50)] * 3

    for gas in HEAVY_GASES:
        r = results[gas]
        T, P = r['T'], r['P']
        Tc, Pc = r['Tc'], r['Pc']
        lnK = np.log(r['K_gas'])

        params, rmse, mare = fit_robust(
            form_log_linear, 3, ll_bounds, T, P, Tc, Pc, lnK)

        K_wil = wilson_standard(T, P, Tc, Pc, r['omega'])
        wil_bad = np.sum(K_wil < 1)

        K_fit = np.exp(form_log_linear(params, T, P, Tc, Pc))
        fit_bad = np.sum((K_fit < 1) & (r['K_gas'] > 1))

        heavy_params[gas] = list(params)
        print(f"  {gas:<8s}: RMSE={rmse:.2f}  K<1(fit)={fit_bad}  "
              f"K<1(Wilson)={wil_bad}  [{', '.join(f'{p:.4f}' for p in params)}]")

    # Phase 4: Fit water K-values
    print(f"\n{'─'*80}")
    print("PHASE 4: Fitting universal water K-value correlation")
    print(f"{'─'*80}")

    T_all, P_all, lnKw_all = [], [], []
    for gas in GAS_SPECIES:
        r = results[gas]
        T_all.append(r['T']); P_all.append(r['P'])
        lnKw_all.append(np.log(r['K_H2O']))
    T_all = np.concatenate(T_all)
    P_all = np.concatenate(P_all)
    lnKw_all = np.concatenate(lnKw_all)

    water_bounds = [(-100, 100), (-20000, 20000), (-5, 5),
                    (-5e6, 5e6), (-1000, 1000), (-1, 1)]

    # Use water form (no Tc/Pc args needed, adapt interface)
    def water_residual(params):
        return form_water(params, T_all, P_all) - lnKw_all

    best_cost = np.inf
    best_water = None
    for trial in range(10):
        if trial == 0:
            x0 = [20, -5000, -1, -500000, 30, 0.05]
        else:
            x0 = [np.random.uniform(l, u) for l, u in water_bounds]
        try:
            result = least_squares(water_residual, x0,
                                   bounds=([b[0] for b in water_bounds],
                                           [b[1] for b in water_bounds]),
                                   loss='huber', f_scale=0.5,
                                   max_nfev=10000, ftol=1e-14, xtol=1e-14)
            if result.cost < best_cost:
                best_cost = result.cost
                best_water = result.x
        except Exception:
            pass

    lnKw_pred = form_water(best_water, T_all, P_all)
    Kw_pred = np.exp(lnKw_pred)
    Kw_data = np.exp(lnKw_all)
    w_rmse = np.sqrt(np.mean((lnKw_pred - lnKw_all)**2))
    w_mare = np.mean(np.abs((Kw_pred - Kw_data) / Kw_data)) * 100
    w_med = np.median(np.abs((Kw_pred - Kw_data) / Kw_data)) * 100

    water_params = list(best_water)
    print(f"  {len(T_all)} pooled points from {len(GAS_SPECIES)} gas partners")
    print(f"  RMSE={w_rmse:.4f}  MARE={w_mare:.1f}%  MedARE={w_med:.1f}%")
    print(f"  Params: [{', '.join(f'{p:.4f}' for p in water_params)}]")

    # Phase 5: Output for vle_engine.py
    print(f"\n{'─'*80}")
    print("FITTED PARAMETERS (for vle_engine.py)")
    print(f"{'─'*80}")

    print("\n_SW_KVALUE_PARAMS = {")
    for gas, (p, mare) in fitted_params.items():
        ps = ', '.join(f'{v:.4f}' for v in p)
        print(f"    '{gas}': ([{ps}], {mare:.1f}),")
    print("}")

    print("\n_SW_KVALUE_HEAVY = {")
    for gas, p in heavy_params.items():
        ps = ', '.join(f'{v:.4f}' for v in p)
        print(f"    '{gas}': [{ps}],")
    print("}")

    print(f"\n_SW_KVALUE_WATER = [{', '.join(f'{p:.4f}' for p in water_params)}]")

    # Phase 6: Validation
    print(f"\n{'─'*80}")
    print("VALIDATION: Previously problematic conditions")
    print(f"{'─'*80}")

    test_conditions = [
        ('H2S', 298.15, 50e5,   "H2S 25°C/50bar"),
        ('H2S', 323.15, 100e5,  "H2S 50°C/100bar"),
        ('H2S', 373.15, 300e5,  "H2S 100°C/300bar"),
        ('CO2', 298.15, 100e5,  "CO2 25°C/100bar"),
        ('CO2', 323.15, 300e5,  "CO2 50°C/300bar"),
        ('CO2', 373.15, 500e5,  "CO2 100°C/500bar"),
        ('C2H6', 298.15, 100e5, "C2H6 25°C/100bar"),
        ('C3H8', 323.15, 100e5, "C3H8 50°C/100bar"),
        ('CH4', 323.15, 500e5,  "CH4 50°C/500bar"),
        ('H2', 298.15, 300e5,   "H2 25°C/300bar"),
    ]

    print(f"\n  {'Condition':<22s} {'K_true':>10s} {'K_wilson':>10s} {'K_fitted':>10s}  "
          f"{'Wil Err%':>9s} {'Fit Err%':>9s}")
    print(f"  {'─'*70}")

    for gas, T, P, label in test_conditions:
        props = COMPONENTS[gas]
        Tc, Pc, omega = props.Tc, props.Pc, props.omega

        vle = SWBinaryVLE(gas, salinity_molal=0.0)
        kij_func = PROPOSED_KIJ.get(gas, None)
        if kij_func is not None:
            x_gas = vle._calc_x_with_kij(T, P, kij_func(T))
        else:
            x_gas = vle.calc_gas_solubility(T, P)
        y_H2O = vle.calc_water_content(T, P)
        K_true = (1 - y_H2O) / x_gas

        K_wil = wilson_standard(T, P, Tc, Pc, omega)
        wil_err = (K_wil - K_true) / K_true * 100

        # Use our fitted form
        names = ['H2O', gas]
        Tc_arr = np.array([COMPONENTS['H2O'].Tc, Tc])
        Pc_arr = np.array([COMPONENTS['H2O'].Pc, Pc])
        om_arr = np.array([COMPONENTS['H2O'].omega, omega])
        K_arr = _sw_kvalue_init(names, Tc_arr, Pc_arr, om_arr, T, P)
        K_fit = K_arr[1]
        fit_err = (K_fit - K_true) / K_true * 100

        wf = " ✗" if K_wil < 1 else ""
        ff = " ✗" if K_fit < 1 else ""
        print(f"  {label:<22s} {K_true:>10.1f} {K_wil:>9.4f}{wf:<2s} {K_fit:>10.1f}{ff:<2s}  "
              f"{wil_err:>+8.1f}% {fit_err:>+8.1f}%")


if __name__ == '__main__':
    main()
