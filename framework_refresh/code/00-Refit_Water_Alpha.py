"""
00-Refit_Water_Alpha.py — Refit the PR-EOS water alpha function to IAPWS-95 vapor pressure data.

Prerequisite for all subsequent pipeline steps (pointwise kij regression uses the alpha function).

Approach (analogous to pointwise kij regression):
  Step 1: At each T, find alpha that makes PR-EOS Pvap = IAPWS-95 Pvap
  Step 2: Fit functional forms to the pointwise alpha values
  Step 3: Compute delta_alpha for NaCl brine at various salinities
  Step 4: Fit a functional form for delta_alpha(T, csw)

Uses S&W critical properties (Tc=647.3 K, Pc=221.2 bar) to maintain
compatibility with existing kij regressions.

Dependency: 00a-Archer_Water_Activity.py (Pitzer model for NaCl water activity)
Output: data/water_alpha_refit_report.txt
"""

import sys
import os
import importlib
import numpy as np
from scipy.optimize import brentq, minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))
sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# Constants
# =============================================================================
# S&W water critical properties (their Table 3) — kept for framework compatibility
TC_W = 647.3      # K
PC_W = 221.2e5    # Pa (221.2 bar)
R = 8.314462      # J/(mol·K)

OMEGA_A = 0.45724
OMEGA_B = 0.07780

# IAPWS-95 critical properties (for reference Pvap calculation)
TC_IAPWS = 647.096  # K
PC_IAPWS = 220.64   # bar

# PR EOS pure-component parameters for water (using S&W Tc, Pc)
b_w = OMEGA_B * R * TC_W / PC_W  # m^3/mol


# =============================================================================
# IAPWS-95 Wagner Equation
# =============================================================================
def pvap_iapws95(T_K):
    """IAPWS-95 pure water saturation pressure (bar)."""
    tau = 1.0 - T_K / TC_IAPWS
    ln_ratio = (TC_IAPWS / T_K) * (
        -7.85951783 * tau
        + 1.84408259 * tau**1.5
        - 11.7866497 * tau**3
        + 22.6807411 * tau**3.5
        - 15.9618719 * tau**4
        + 1.80122502 * tau**7.5
    )
    return PC_IAPWS * np.exp(ln_ratio)


# =============================================================================
# PR-EOS Vapor Pressure for Pure Water
# =============================================================================
def pr_eos_pvap(T_K, alpha_val, max_iter=200, tol=1e-10):
    """Compute PR-EOS vapor pressure for pure water given alpha.

    Returns Pvap in bar, or NaN if not converged.
    """
    a_w = OMEGA_A * R**2 * TC_W**2 * alpha_val / PC_W

    # Initial guess from IAPWS
    P_Pa = pvap_iapws95(T_K) * 1e5

    for _ in range(max_iter):
        A = a_w * P_Pa / (R * T_K)**2
        B = b_w * P_Pa / (R * T_K)

        # Solve cubic: Z^3 - (1-B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B**2 - 2.0 * B
        c0 = -(A * B - B**2 - B**3)

        roots = np.roots([1.0, c2, c1, c0])
        real_roots = sorted(roots[np.isreal(roots)].real)
        real_roots = [z for z in real_roots if z > B + 1e-15]

        if len(real_roots) < 2:
            return np.nan

        Z_L = real_roots[0]   # Smallest valid root (liquid)
        Z_V = real_roots[-1]  # Largest valid root (vapor)

        sqrt2 = np.sqrt(2)
        def ln_phi(Z):
            return ((Z - 1) - np.log(Z - B)
                    - A / (2 * sqrt2 * B) * np.log(
                        (Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B)))

        ln_phi_L = ln_phi(Z_L)
        ln_phi_V = ln_phi(Z_V)

        # Successive substitution: P_new = P * exp(ln_phi_L - ln_phi_V)
        correction = np.exp(ln_phi_L - ln_phi_V)
        P_Pa_new = P_Pa * correction

        if abs(P_Pa_new - P_Pa) / max(P_Pa, 1e-10) < tol:
            return P_Pa_new / 1e5  # bar

        P_Pa = P_Pa_new

    return np.nan


def find_pointwise_alpha(T_K, target_pvap_bar):
    """Find alpha that makes PR-EOS Pvap match target at given T.

    PR-EOS Pvap is monotonically DECREASING with alpha (higher alpha = stronger
    attraction = lower vapor pressure). Scan for valid bracket then use brentq.
    """
    def objective(alpha):
        pvap = pr_eos_pvap(T_K, alpha)
        if np.isnan(pvap):
            return np.nan
        return pvap - target_pvap_bar

    # Scan alpha values to find a bracket where objective changes sign
    # PR Pvap decreases with alpha, so objective goes from positive to negative
    alphas = np.linspace(0.8, 5.0, 200)
    obj_vals = np.array([objective(a) for a in alphas])

    # Find valid (non-NaN) evaluations
    valid = ~np.isnan(obj_vals)
    if np.sum(valid) < 2:
        return np.nan

    alphas_v = alphas[valid]
    obj_v = obj_vals[valid]

    # Find sign change
    for i in range(len(obj_v) - 1):
        if obj_v[i] * obj_v[i+1] < 0:
            try:
                # Refine with brentq between these two alpha values
                def safe_obj(alpha):
                    val = objective(alpha)
                    return val if not np.isnan(val) else (1e10 if alpha < alphas_v[i] else -1e10)
                alpha = brentq(safe_obj, alphas_v[i], alphas_v[i+1],
                               xtol=1e-12, rtol=1e-12)
                return alpha
            except ValueError:
                continue

    return np.nan


# =============================================================================
# Alpha Function Forms
# =============================================================================
def alpha_sw_original(Tr, csw=0.0):
    """S&W original (2 params, Eq 9)."""
    sqrt_a = 1.0 + 0.4530 * (1.0 - Tr * (1.0 - 0.0103 * csw**1.1)) + 0.0034 * (Tr**(-3) - 1.0)
    return sqrt_a**2

def form_sw2(Tr, params):
    """S&W-type 2-param: sqrt(alpha) = 1 + a*(1-Tr) + b*(Tr^-3 - 1)."""
    a, b = params
    sqrt_a = 1.0 + a * (1.0 - Tr) + b * (Tr**(-3) - 1.0)
    return sqrt_a**2

def form_sw3(Tr, params):
    """Extended S&W 3-param: + c*(1-Tr)^2."""
    a, b, c = params
    sqrt_a = 1.0 + a * (1.0 - Tr) + b * (Tr**(-3) - 1.0) + c * (1.0 - Tr)**2
    return sqrt_a**2

def form_mc3(Tr, params):
    """Mathias-Copeman 3-param: sqrt(alpha) = 1 + c1*(1-sqrt(Tr)) + c2*(1-sqrt(Tr))^2 + c3*(1-sqrt(Tr))^3."""
    c1, c2, c3 = params
    x = 1.0 - np.sqrt(Tr)
    sqrt_a = 1.0 + c1 * x + c2 * x**2 + c3 * x**3
    return sqrt_a**2

def form_sw4(Tr, params):
    """Extended S&W 4-param: + c*(1-Tr)^2 + d*(Tr^-6 - 1)."""
    a, b, c, d = params
    sqrt_a = 1.0 + a * (1.0 - Tr) + b * (Tr**(-3) - 1.0) + c * (1.0 - Tr)**2 + d * (Tr**(-6) - 1.0)
    return sqrt_a**2


# =============================================================================
# Brine Vapor Pressure Models
# =============================================================================
def aw_katz(csw):
    """Water activity from Katz et al. (S&W Eq 16).

    a_w ≈ P_brine/P_freshwater = 1 - 0.02865 * csw^1.44
    Valid for NaCl, csw in mol/kg.
    """
    if csw <= 0:
        return 1.0
    return max(1.0 - 0.02865 * csw**1.44, 0.01)

def aw_raoult(csw):
    """Ideal Raoult's law water activity for NaCl (nu=2).

    x_w = n_w / (n_w + nu*n_salt) = 55.508 / (55.508 + 2*csw)
    """
    return 55.508 / (55.508 + 2.0 * csw)


# =============================================================================
# Fitting
# =============================================================================
def fit_form(Tr_arr, alpha_arr, form_func, n_params, label, x0=None):
    """Fit a functional form to pointwise alpha values using L1 regression."""
    def objective(params):
        pred = np.array([form_func(Tr, params) for Tr in Tr_arr])
        return np.sum(np.abs(pred - alpha_arr))  # L1

    if x0 is None:
        if n_params == 2:
            x0 = [0.45, 0.003]
        elif n_params == 3:
            x0 = [0.45, 0.003, 0.01]
        elif n_params == 4:
            x0 = [0.45, 0.003, 0.01, 0.0001]

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12})

    params = result.x
    pred = np.array([form_func(Tr, params) for Tr in Tr_arr])
    residuals = pred - alpha_arr
    mae = np.mean(np.abs(residuals))
    max_err = np.max(np.abs(residuals))

    # Compute Pvap error
    pvap_errs = []
    for i, Tr in enumerate(Tr_arr):
        T_K = Tr * TC_W
        pvap_pred = pr_eos_pvap(T_K, pred[i])
        pvap_ref = pvap_iapws95(T_K)
        if not np.isnan(pvap_pred) and pvap_ref > 0:
            pvap_errs.append(abs(pvap_pred - pvap_ref) / pvap_ref * 100)

    pvap_mare = np.mean(pvap_errs) if pvap_errs else np.nan
    pvap_max = np.max(pvap_errs) if pvap_errs else np.nan

    return params, mae, max_err, pvap_mare, pvap_max


# =============================================================================
# Main
# =============================================================================
def main():
    outfile = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'water_alpha_refit_report.txt'))

    # Temperature grid: 20 to 200 deg C (Tr ~ 0.453 to 0.731)
    T_degC = np.arange(20, 201, 5)
    T_K_arr = T_degC + 273.15
    Tr_arr = T_K_arr / TC_W

    # =========================================================================
    # Step 1: Pointwise Alpha Regression
    # =========================================================================
    print("Step 1: Pointwise alpha regression...")
    alpha_pw = np.zeros(len(T_K_arr))
    pvap_ref = np.zeros(len(T_K_arr))

    for i, T_K in enumerate(T_K_arr):
        pvap_ref[i] = pvap_iapws95(T_K)
        alpha_pw[i] = find_pointwise_alpha(T_K, pvap_ref[i])
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1}/{len(T_K_arr)}: T={T_K-273.15:.0f}°C, "
                  f"Pvap={pvap_ref[i]:.4f} bar, alpha={alpha_pw[i]:.6f}")

    valid = ~np.isnan(alpha_pw)
    print(f"  Valid points: {np.sum(valid)}/{len(alpha_pw)}")

    Tr_valid = Tr_arr[valid]
    alpha_valid = alpha_pw[valid]
    T_degC_valid = T_degC[valid]

    # =========================================================================
    # Step 2: Fit Functional Forms
    # =========================================================================
    print("\nStep 2: Fitting functional forms...")

    # S&W original values for comparison
    alpha_sw = np.array([alpha_sw_original(Tr) for Tr in Tr_valid])
    sw_residuals = alpha_sw - alpha_valid
    sw_mae = np.mean(np.abs(sw_residuals))

    # Compute S&W Pvap errors
    sw_pvap_errs = []
    for i, Tr in enumerate(Tr_valid):
        T_K = Tr * TC_W
        pvap_pred = pr_eos_pvap(T_K, alpha_sw[i])
        pvap_target = pvap_iapws95(T_K)
        if not np.isnan(pvap_pred) and pvap_target > 0:
            sw_pvap_errs.append(abs(pvap_pred - pvap_target) / pvap_target * 100)
    sw_pvap_mare = np.mean(sw_pvap_errs)
    sw_pvap_max = np.max(sw_pvap_errs)

    forms = [
        ("SW-2 (refit)", form_sw2, 2, [0.45, 0.003]),
        ("SW-3 (extended)", form_sw3, 3, [0.45, 0.003, 0.01]),
        ("MC-3 (Mathias-Copeman)", form_mc3, 3, [0.9, -0.3, 0.3]),
        ("SW-4 (extended)", form_sw4, 4, [0.45, 0.003, 0.01, 0.0001]),
    ]

    results = {}
    for label, func, n_params, x0 in forms:
        print(f"  Fitting {label}...")
        params, mae, max_err, pvap_mare, pvap_max = fit_form(
            Tr_valid, alpha_valid, func, n_params, label, x0)
        results[label] = (params, mae, max_err, pvap_mare, pvap_max, func)
        print(f"    MAE(alpha)={mae:.6f}, Pvap MARE={pvap_mare:.4f}%, "
              f"Max Pvap err={pvap_max:.4f}%")

    # =========================================================================
    # Step 3: Brine Delta Alpha using Archer (1992) Water Activity
    # =========================================================================
    print("\nStep 3: Computing brine delta_alpha (Archer 1992 a_w)...")

    _archer = importlib.import_module('00a-Archer_Water_Activity')
    water_activity_archer = _archer.water_activity_archer
    csw_to_molality = _archer.csw_to_molality
    molality_to_csw = _archer.molality_to_csw

    # Use best freshwater form for base alpha
    best_label = min(results, key=lambda k: results[k][3])  # Lowest Pvap MARE
    best_params, _, _, _, _, best_func = results[best_label]
    print(f"  Using base form: {best_label}")

    # Grid in molality (natural Pitzer variable) — covers 0-26 wt% NaCl
    m_arr = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    T_brine_degC = np.arange(20, 201, 10)
    T_brine_K = T_brine_degC + 273.15
    Tr_brine = T_brine_K / TC_W

    # For each (T, m), find alpha_brine and compute delta
    delta_results = []  # (T_degC, Tr, m, csw, alpha_fw, alpha_brine, delta, aw)

    for m in m_arr:
        csw = molality_to_csw(m)
        print(f"  m={m:.1f} mol/kg (csw={csw:.1f} wt%)...")
        for j, T_K in enumerate(T_brine_K):
            Tr = T_K / TC_W
            aw = water_activity_archer(T_K, m)
            pvap_fw = pvap_iapws95(T_K)
            pvap_brine = aw * pvap_fw  # Target brine Pvap

            alpha_fw = best_func(Tr, best_params)
            alpha_brine = find_pointwise_alpha(T_K, pvap_brine)

            if not np.isnan(alpha_brine):
                delta = alpha_brine - alpha_fw
                delta_results.append((T_brine_degC[j], Tr, m, csw, alpha_fw,
                                      alpha_brine, delta, aw))

    print(f"  Total brine points: {len(delta_results)}")

    # =========================================================================
    # Step 4: Fit Delta Form — delta_alpha(Tr, m)
    # =========================================================================
    print("\nStep 4: Fitting delta_alpha(Tr, m) forms...")

    delta_data = np.array(delta_results)
    Tr_d = delta_data[:, 1]
    m_d = delta_data[:, 2]    # molality
    csw_d = delta_data[:, 3]  # wt% (for reference)
    delta_d = delta_data[:, 6]  # delta_alpha

    # Form A: delta = (d0 + d1*Tr) * m  (linear in m, 2 params)
    def delta_form_A(Tr, m, params):
        d0, d1 = params
        return (d0 + d1 * Tr) * m

    # Form B: delta = (d0 + d1*Tr + d2*Tr^2) * m  (quadratic in Tr, linear in m, 3 params)
    def delta_form_B(Tr, m, params):
        d0, d1, d2 = params
        return (d0 + d1 * Tr + d2 * Tr**2) * m

    # Form C: delta = (d0 + d1*Tr) * m^d2  (power law in m, 3 params)
    def delta_form_C(Tr, m, params):
        d0, d1, d2 = params
        return (d0 + d1 * Tr) * m**d2

    # Form D: delta = (d0 + d1*Tr + d2*Tr^2) * m + (d3 + d4*Tr) * m^2
    #         (quadratic in both, 5 params)
    def delta_form_D(Tr, m, params):
        d0, d1, d2, d3, d4 = params
        return (d0 + d1 * Tr + d2 * Tr**2) * m + (d3 + d4 * Tr) * m**2

    # Form E: S&W-like power law: delta = d0 * m^d1  (T-independent, 2 params)
    def delta_form_E(Tr, m, params):
        d0, d1 = params
        return d0 * m**d1

    # Form F: delta = (d0 + d1*Tr + d2*Tr^2) * m + (d3 + d4*Tr + d5*Tr^2) * m^2
    #         (full quadratic, 6 params)
    def delta_form_F(Tr, m, params):
        d0, d1, d2, d3, d4, d5 = params
        return ((d0 + d1 * Tr + d2 * Tr**2) * m
                + (d3 + d4 * Tr + d5 * Tr**2) * m**2)

    delta_forms = [
        ("A: lin-m (2p)", delta_form_A, 2, [0.01, -0.005]),
        ("B: quad-Tr lin-m (3p)", delta_form_B, 3, [0.01, -0.01, 0.005]),
        ("C: power-m (3p)", delta_form_C, 3, [0.01, -0.005, 1.1]),
        ("D: quad-Tr quad-m (5p)", delta_form_D, 5, [0.01, -0.01, 0.005, 0.001, -0.001]),
        ("E: T-indep power-m (2p)", delta_form_E, 2, [0.005, 1.1]),
        ("F: full quad (6p)", delta_form_F, 6, [0.01, -0.01, 0.005, 0.001, -0.001, 0.0005]),
    ]

    delta_fit_results = {}
    for label, func, n_params, x0 in delta_forms:
        def objective(params, func=func):
            pred = np.array([func(Tr_d[i], m_d[i], params) for i in range(len(Tr_d))])
            return np.sum(np.abs(pred - delta_d))

        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-14})
        params = res.x
        pred = np.array([func(Tr_d[i], m_d[i], params) for i in range(len(Tr_d))])
        mae = np.mean(np.abs(pred - delta_d))
        max_err = np.max(np.abs(pred - delta_d))
        delta_fit_results[label] = (params, mae, max_err)
        print(f"  {label}: MAE(delta)={mae:.6f}, Max={max_err:.6f}, params={params}")

    # =========================================================================
    # Write Report
    # =========================================================================
    print(f"\nWriting report to {outfile}...")

    with open(outfile, 'w') as f:
        f.write("Water Alpha Function Refit Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: 2026-02-14\n")
        f.write(f"S&W water Tc = {TC_W} K, Pc = {PC_W/1e5:.1f} bar\n")
        f.write(f"IAPWS-95 Tc = {TC_IAPWS} K, Pc = {PC_IAPWS:.2f} bar\n")
        f.write(f"Temperature range: {T_degC_valid[0]:.0f}-{T_degC_valid[-1]:.0f} deg C "
                f"(Tr = {Tr_valid[0]:.4f}-{Tr_valid[-1]:.4f})\n")
        f.write(f"Points: {len(Tr_valid)}\n")
        f.write("=" * 80 + "\n\n")

        # Pointwise results
        f.write("POINTWISE ALPHA VALUES\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'T (degC)':>10}  {'Tr':>8}  {'Pvap IAPWS':>14}  "
                f"{'alpha_pw':>12}  {'alpha_SW':>12}  {'SW err':>10}\n")
        f.write(f"{'':>10}  {'':>8}  {'(bar)':>14}  "
                f"{'':>12}  {'(original)':>12}  {'(%)':>10}\n")
        f.write("-" * 72 + "\n")
        for i in range(len(Tr_valid)):
            T_K = Tr_valid[i] * TC_W
            pv = pvap_iapws95(T_K)
            sw = alpha_sw_original(Tr_valid[i])
            err_pct = (sw - alpha_valid[i]) / alpha_valid[i] * 100
            f.write(f"{T_degC_valid[i]:10.0f}  {Tr_valid[i]:8.4f}  {pv:14.6f}  "
                    f"{alpha_valid[i]:12.6f}  {sw:12.6f}  {err_pct:10.4f}\n")
        f.write("-" * 72 + "\n\n")

        # Correlation fitting results
        f.write("CORRELATION FITTING RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"S&W Original (a=0.4530, b=0.0034):\n")
        f.write(f"  MAE(alpha) = {sw_mae:.6f}\n")
        f.write(f"  Pvap MARE  = {sw_pvap_mare:.4f}%\n")
        f.write(f"  Pvap Max   = {sw_pvap_max:.4f}%\n\n")

        for label in results:
            params, mae, max_err, pvap_mare, pvap_max, func = results[label]
            f.write(f"{label}:\n")
            f.write(f"  Params: {', '.join(f'{p:.8f}' for p in params)}\n")
            f.write(f"  MAE(alpha) = {mae:.6f}\n")
            f.write(f"  Max(alpha) = {max_err:.6f}\n")
            f.write(f"  Pvap MARE  = {pvap_mare:.4f}%\n")
            f.write(f"  Pvap Max   = {pvap_max:.4f}%\n\n")

        # Comparison of fitted vs pointwise for best form
        f.write(f"\nBEST FORM COMPARISON: {best_label}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'T (degC)':>10}  {'Tr':>8}  {'alpha_pw':>12}  "
                f"{'alpha_fit':>12}  {'alpha_SW':>12}  {'fit err%':>10}  {'SW err%':>10}\n")
        f.write("-" * 80 + "\n")
        for i in range(len(Tr_valid)):
            alpha_fit = best_func(Tr_valid[i], best_params)
            sw = alpha_sw_original(Tr_valid[i])
            fit_err = (alpha_fit - alpha_valid[i]) / alpha_valid[i] * 100
            sw_err = (sw - alpha_valid[i]) / alpha_valid[i] * 100
            f.write(f"{T_degC_valid[i]:10.0f}  {Tr_valid[i]:8.4f}  {alpha_valid[i]:12.6f}  "
                    f"{alpha_fit:12.6f}  {sw:12.6f}  {fit_err:10.4f}  {sw_err:10.4f}\n")
        f.write("-" * 80 + "\n\n")

        # Brine delta results
        f.write("BRINE DELTA_ALPHA RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Base alpha form: {best_label}\n")
        f.write(f"Brine Pvap: a_w(Archer 1992) * Pvap_IAPWS\n")
        f.write(f"a_w from Pitzer model: Archer, JPCRD 21(4), 793-829, 1992\n")
        f.write(f"DOI: 10.1063/1.555987\n\n")

        f.write(f"{'T (degC)':>10}  {'Tr':>8}  {'m':>6}  {'csw%':>6}  {'a_w':>8}  "
                f"{'alpha_fw':>12}  {'alpha_br':>12}  {'delta':>12}\n")
        f.write("-" * 90 + "\n")
        for row in delta_results[::5]:  # Every 5th row to keep output manageable
            f.write(f"{row[0]:10.0f}  {row[1]:8.4f}  {row[2]:6.1f}  {row[3]:6.1f}  "
                    f"{row[7]:8.4f}  {row[4]:12.6f}  {row[5]:12.6f}  {row[6]:12.6f}\n")
        f.write("-" * 90 + "\n\n")

        # Delta fitting results
        f.write("DELTA CORRELATION FITTING\n")
        f.write("=" * 80 + "\n")
        f.write("delta_alpha(Tr, m) = alpha_brine - alpha_freshwater\n")
        f.write("where m = NaCl molality (mol/kg H2O), Tr = T/Tc_w\n\n")

        for label in delta_fit_results:
            params, mae, max_err = delta_fit_results[label]
            f.write(f"{label}:\n")
            f.write(f"  Params: {', '.join(f'{p:.8f}' for p in params)}\n")
            f.write(f"  MAE(delta) = {mae:.6f}\n")
            f.write(f"  Max(delta) = {max_err:.6f}\n\n")

        # Summary comparison with S&W original
        f.write("\nSUMMARY: S&W ORIGINAL vs REFITTED ALPHA\n")
        f.write("=" * 80 + "\n")

        # Compute Pvap errors for S&W original at key temperatures
        key_temps = [20, 25, 50, 80, 100, 120, 150, 175, 200]
        f.write(f"\n{'T (degC)':>10}  {'Pvap IAPWS':>14}  "
                f"{'Pvap SW (bar)':>14}  {'SW err%':>10}  "
                f"{'Pvap refit':>14}  {'refit err%':>10}\n")
        f.write("-" * 80 + "\n")
        for T in key_temps:
            T_K = T + 273.15
            Tr = T_K / TC_W
            pv_ref = pvap_iapws95(T_K)
            pv_sw = pr_eos_pvap(T_K, alpha_sw_original(Tr))
            pv_new = pr_eos_pvap(T_K, best_func(Tr, best_params))
            sw_err = (pv_sw - pv_ref) / pv_ref * 100 if not np.isnan(pv_sw) else np.nan
            new_err = (pv_new - pv_ref) / pv_ref * 100 if not np.isnan(pv_new) else np.nan
            f.write(f"{T:10.0f}  {pv_ref:14.6f}  {pv_sw:14.6f}  {sw_err:10.4f}  "
                    f"{pv_new:14.6f}  {new_err:10.4f}\n")
        f.write("-" * 80 + "\n")

    print(f"Report written to {outfile}")


if __name__ == '__main__':
    main()
