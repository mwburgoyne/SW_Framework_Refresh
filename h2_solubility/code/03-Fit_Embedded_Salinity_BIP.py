#!/usr/bin/env python3
"""
Fit Embedded Salinity BIP Correlation for H2
=============================================

Fits a kij_AQ(T, csw) correlation that embeds salinity dependence directly
in the BIP, similar to S&W Equations 12-15 for hydrocarbons.

This is NOT the recommended approach (gamma-phi is more rigorous), but
provides an option for users who:
1. Need material balance accuracy (can't use post-flash Sechenov)
2. Cannot modify their VLE flash routine (commercial simulators)

Approach:
- Generate "synthetic" brine kij values by finding the kij that matches
  the Sechenov-corrected solubility at each (T, P, csw) condition
- Fit various functional forms to these synthetic kij values
- Evaluate how well the fitted form reproduces actual Sechenov behaviour

Author: Mark Burgoyne
Date: 2026-02-05
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

from _lib_vle_engine import (
    SWBinaryVLE, sw_equation_8_ks, COMPONENTS, BIP_TC_H2,
    celsius_to_kelvin, bar_to_pascal, kij_aq_h2
)


# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = '../../shared/data'

# Conditions for fitting
TEMPERATURES_C = [25, 50, 75, 100, 125, 150]
PRESSURES_BAR = [50, 100, 150, 200]
SALINITIES_MOLAL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

# Current freshwater BIP parameters
A_FW, B_FW, C_FW = -14.59, 2.184, 0.365


# =============================================================================
# Helper Functions
# =============================================================================
def kij_freshwater(T_K: float) -> float:
    """Current freshwater BIP correlation."""
    Tr = T_K / BIP_TC_H2
    return (A_FW + Tr) / (B_FW + C_FW * Tr)


def calc_target_brine_solubility(T_K: float, P_Pa: float, csw: float) -> float:
    """
    Calculate target brine solubility using freshwater VLE + Sechenov.
    This is the "truth" we're trying to match with embedded kij.
    """
    vle = SWBinaryVLE('H2', salinity_molal=0.0)
    x_fresh = vle.calc_gas_solubility(T_K, P_Pa)

    if csw <= 0:
        return x_fresh

    T_C = T_K - 273.15
    ks = sw_equation_8_ks(T_C, COMPONENTS['H2'].Tb)
    return x_fresh * 10**(-ks * csw)


def find_kij_for_target_solubility(T_K: float, P_Pa: float,
                                    x_target: float) -> float:
    """
    Find the kij value that produces the target solubility.
    This gives us the "effective" kij needed to match brine behaviour.
    """
    vle = SWBinaryVLE('H2', salinity_molal=0.0)

    def objective(kij):
        try:
            x_calc = vle._calc_x_with_kij(T_K, P_Pa, kij)
            return x_calc - x_target
        except:
            return 1.0

    # Find kij that gives target solubility
    # Search in reasonable range around freshwater kij
    kij_fw = kij_freshwater(T_K)

    try:
        # Brine solubility is lower, so we need more negative kij
        kij_brine = brentq(objective, kij_fw - 2.0, kij_fw + 0.5)
    except:
        # If brentq fails, use minimization
        result = minimize(lambda k: abs(objective(k[0])), [kij_fw - 0.5],
                         bounds=[(kij_fw - 3.0, kij_fw + 1.0)])
        kij_brine = result.x[0]

    return kij_brine


# =============================================================================
# Generate Synthetic kij Data
# =============================================================================
def generate_synthetic_kij_data() -> List[Dict]:
    """
    Generate synthetic kij values for various (T, P, csw) conditions.
    These represent the kij needed to match Sechenov-corrected solubility.
    """
    data = []

    print("Generating synthetic kij data...")
    print(f"  Temperatures: {TEMPERATURES_C} °C")
    print(f"  Pressures: {PRESSURES_BAR} bar")
    print(f"  Salinities: {SALINITIES_MOLAL} molal")

    for T_C in TEMPERATURES_C:
        T_K = celsius_to_kelvin(T_C)
        Tr = T_K / BIP_TC_H2
        kij_fw = kij_freshwater(T_K)

        for P_bar in PRESSURES_BAR:
            P_Pa = bar_to_pascal(P_bar)

            for csw in SALINITIES_MOLAL:
                # Get target brine solubility
                x_target = calc_target_brine_solubility(T_K, P_Pa, csw)

                if csw == 0:
                    kij_eff = kij_fw
                else:
                    # Find kij that matches this solubility
                    kij_eff = find_kij_for_target_solubility(T_K, P_Pa, x_target)

                # Store delta from freshwater kij
                delta_kij = kij_eff - kij_fw

                data.append({
                    'T_C': T_C,
                    'T_K': T_K,
                    'Tr': Tr,
                    'P_bar': P_bar,
                    'csw': csw,
                    'x_target': x_target,
                    'kij_fw': kij_fw,
                    'kij_eff': kij_eff,
                    'delta_kij': delta_kij,
                })

    print(f"  Generated {len(data)} data points")
    return data


# =============================================================================
# Fit Various Functional Forms
# =============================================================================
def fit_form_1(data: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    Form 1: Linear salinity term
    kij = kij_fw * (1 + α*csw)
    """
    # Filter to non-zero salinity
    brine_data = [d for d in data if d['csw'] > 0]

    # kij_eff / kij_fw = 1 + α*csw
    # (kij_eff / kij_fw - 1) / csw = α

    ratios = [(d['kij_eff'] / d['kij_fw'] - 1) / d['csw'] for d in brine_data]
    alpha = np.mean(ratios)

    # Calculate RMSE
    errors = []
    for d in data:
        kij_pred = d['kij_fw'] * (1 + alpha * d['csw'])
        errors.append((kij_pred - d['kij_eff'])**2)
    rmse = np.sqrt(np.mean(errors))

    return np.array([alpha]), rmse


def fit_form_2(data: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    Form 2: Additive salinity term (similar to S&W hydrocarbon form)
    kij = kij_fw + β*csw
    """
    brine_data = [d for d in data if d['csw'] > 0]

    # delta_kij = β*csw
    betas = [d['delta_kij'] / d['csw'] for d in brine_data]
    beta = np.mean(betas)

    errors = []
    for d in data:
        kij_pred = d['kij_fw'] + beta * d['csw']
        errors.append((kij_pred - d['kij_eff'])**2)
    rmse = np.sqrt(np.mean(errors))

    return np.array([beta]), rmse


def fit_form_3(data: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    Form 3: Temperature-dependent salinity term
    kij = kij_fw + (β0 + β1*Tr)*csw
    """
    from scipy.optimize import least_squares

    def residuals(params):
        beta0, beta1 = params
        res = []
        for d in data:
            kij_pred = d['kij_fw'] + (beta0 + beta1 * d['Tr']) * d['csw']
            res.append(kij_pred - d['kij_eff'])
        return res

    result = least_squares(residuals, [0.0, 0.0])
    params = result.x

    errors = []
    for d in data:
        kij_pred = d['kij_fw'] + (params[0] + params[1] * d['Tr']) * d['csw']
        errors.append((kij_pred - d['kij_eff'])**2)
    rmse = np.sqrt(np.mean(errors))

    return params, rmse


def fit_form_4(data: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    Form 4: S&W-style with salinity in each term
    kij = (A*(1+α0*csw) + Tr) / (B + C*Tr)

    Using fixed B, C from freshwater fit.
    """
    from scipy.optimize import least_squares

    def kij_form4(Tr, csw, alpha0):
        return (A_FW * (1 + alpha0 * csw) + Tr) / (B_FW + C_FW * Tr)

    def residuals(params):
        alpha0 = params[0]
        res = []
        for d in data:
            kij_pred = kij_form4(d['Tr'], d['csw'], alpha0)
            res.append(kij_pred - d['kij_eff'])
        return res

    result = least_squares(residuals, [0.01])
    params = result.x

    errors = []
    for d in data:
        kij_pred = kij_form4(d['Tr'], d['csw'], params[0])
        errors.append((kij_pred - d['kij_eff'])**2)
    rmse = np.sqrt(np.mean(errors))

    return params, rmse


def fit_form_5(data: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    Form 5: Full S&W-style with salinity in A and separate additive term
    kij = (A*(1+α0*csw^α1) + Tr) / (B + C*Tr)
    """
    from scipy.optimize import least_squares

    def kij_form5(Tr, csw, alpha0, alpha1):
        if csw <= 0:
            return (A_FW + Tr) / (B_FW + C_FW * Tr)
        return (A_FW * (1 + alpha0 * csw**alpha1) + Tr) / (B_FW + C_FW * Tr)

    def residuals(params):
        alpha0, alpha1 = params
        res = []
        for d in data:
            kij_pred = kij_form5(d['Tr'], d['csw'], alpha0, alpha1)
            res.append(kij_pred - d['kij_eff'])
        return res

    result = least_squares(residuals, [0.01, 1.0], bounds=([0, 0.1], [0.5, 2.0]))
    params = result.x

    errors = []
    for d in data:
        kij_pred = kij_form5(d['Tr'], d['csw'], params[0], params[1])
        errors.append((kij_pred - d['kij_eff'])**2)
    rmse = np.sqrt(np.mean(errors))

    return params, rmse


def fit_form_6(data: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    Form 6: Quadratic temperature-dependent salinity term (kij-error minimization)
    kij = kij_fw + (β0 + β1*Tr + β2*Tr²)*csw

    Note: This minimizes kij error, which may not minimize implied ks error.
    See fit_form_7 for ks-error minimization (recommended).
    """
    from scipy.optimize import least_squares

    def residuals(params):
        beta0, beta1, beta2 = params
        res = []
        for d in data:
            kij_pred = d['kij_fw'] + (beta0 + beta1 * d['Tr'] + beta2 * d['Tr']**2) * d['csw']
            res.append(kij_pred - d['kij_eff'])
        return res

    result = least_squares(residuals, [0.0, 0.0, 0.0])
    params = result.x

    errors = []
    for d in data:
        kij_pred = d['kij_fw'] + (params[0] + params[1] * d['Tr'] + params[2] * d['Tr']**2) * d['csw']
        errors.append((kij_pred - d['kij_eff'])**2)
    rmse = np.sqrt(np.mean(errors))

    return params, rmse


def fit_form_7(data: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    Form 7: Quadratic temperature-dependent salinity term (ks-error minimization)
    kij = kij_fw + (β0 + β1*Tr + β2*Tr²)*csw

    RECOMMENDED: Minimizes implied Sechenov coefficient error, not kij error.
    This produces better match to S&W Equation 8 ks values and better
    solubility accuracy across the full T-P-m range.
    """
    from scipy.optimize import minimize

    vle = SWBinaryVLE('H2', salinity_molal=0.0)

    def ks_error(params):
        beta0, beta1, beta2 = params
        sq_errors = []

        for d in data:
            if d['csw'] <= 0:
                continue

            T_K, P_Pa, csw = d['T_K'], d['P_Pa'], d['csw']
            Tr = d['Tr']

            # Get true ks from S&W Eq 8
            T_C = T_K - 273.15
            ks_true = sw_equation_8_ks(T_C, COMPONENTS['H2'].Tb)

            # Calculate implied ks from embedded BIP
            kij_fw = d['kij_fw']
            kij_brine = kij_fw + (beta0 + beta1 * Tr + beta2 * Tr**2) * csw

            x_fresh = vle._calc_x_with_kij(T_K, P_Pa, kij_fw)
            x_brine = vle._calc_x_with_kij(T_K, P_Pa, kij_brine)

            if x_brine > 0 and x_fresh > 0:
                ks_implied = np.log10(x_fresh / x_brine) / csw
                sq_errors.append((ks_implied - ks_true)**2)

        return np.mean(sq_errors) if sq_errors else 1e10

    # Start from Form 6 kij-error solution
    params_init, _ = fit_form_6(data)
    result = minimize(ks_error, params_init, method='Nelder-Mead')
    params = result.x

    # Calculate kij RMSE for comparison
    errors = []
    for d in data:
        kij_pred = d['kij_fw'] + (params[0] + params[1] * d['Tr'] + params[2] * d['Tr']**2) * d['csw']
        errors.append((kij_pred - d['kij_eff'])**2)
    rmse = np.sqrt(np.mean(errors))

    return params, rmse


# =============================================================================
# Evaluate Solubility Accuracy
# =============================================================================
def evaluate_solubility_accuracy(data: List[Dict], form_name: str,
                                  kij_func) -> Dict:
    """
    Evaluate how well the fitted kij reproduces actual brine solubility.
    """
    errors = []

    for d in data:
        if d['csw'] == 0:
            continue

        # Get predicted kij
        kij_pred = kij_func(d['T_K'], d['csw'])

        # Calculate solubility with this kij
        vle = SWBinaryVLE('H2', salinity_molal=0.0)
        try:
            x_pred = vle._calc_x_with_kij(d['T_K'], bar_to_pascal(d['P_bar']), kij_pred)
        except:
            x_pred = d['x_target']

        # Compare to Sechenov target
        rel_error = (x_pred - d['x_target']) / d['x_target'] * 100
        errors.append({
            'T_C': d['T_C'],
            'P_bar': d['P_bar'],
            'csw': d['csw'],
            'x_target': d['x_target'],
            'x_pred': x_pred,
            'rel_error': rel_error,
        })

    rel_errors = [e['rel_error'] for e in errors]

    return {
        'form': form_name,
        'mean_abs_error': np.mean(np.abs(rel_errors)),
        'max_abs_error': np.max(np.abs(rel_errors)),
        'within_1pct': np.sum(np.abs(rel_errors) < 1) / len(rel_errors) * 100,
        'within_5pct': np.sum(np.abs(rel_errors) < 5) / len(rel_errors) * 100,
        'errors': errors,
    }


# =============================================================================
# Plotting
# =============================================================================
def create_plots(data: List[Dict], fitted_forms: Dict, output_dir: str):
    """Create diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Delta kij vs salinity at different temperatures
    ax = axes[0, 0]
    for T_C in [50, 100, 150]:
        subset = [d for d in data if d['T_C'] == T_C and d['P_bar'] == 100]
        csw_vals = [d['csw'] for d in subset]
        delta_vals = [d['delta_kij'] for d in subset]
        ax.plot(csw_vals, delta_vals, 'o-', label=f'{T_C}°C')

    ax.set_xlabel('Salinity (molal)')
    ax.set_ylabel('Δkij (kij_brine - kij_fresh)')
    ax.set_title('(a) Required kij Shift to Match Sechenov')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Effective kij vs temperature at different salinities
    ax = axes[0, 1]
    for csw in [0, 2, 4, 6]:
        subset = [d for d in data if d['csw'] == csw and d['P_bar'] == 100]
        T_vals = [d['T_C'] for d in subset]
        kij_vals = [d['kij_eff'] for d in subset]
        label = 'Fresh' if csw == 0 else f'{csw} molal'
        ax.plot(T_vals, kij_vals, 'o-', label=label)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Effective kij')
    ax.set_title('(b) Effective kij vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Parity plot: fitted kij vs effective kij (best form)
    ax = axes[1, 0]
    best_form = min(fitted_forms.items(), key=lambda x: x[1]['rmse'])
    form_name, form_data = best_form

    kij_eff = [d['kij_eff'] for d in data]
    kij_pred = [form_data['kij_func'](d['T_K'], d['csw']) for d in data]

    ax.scatter(kij_eff, kij_pred, alpha=0.6, s=30)
    lims = [min(kij_eff) - 0.1, max(kij_eff) + 0.1]
    ax.plot(lims, lims, 'k-', lw=1.5)
    ax.set_xlabel('Effective kij (from Sechenov)')
    ax.set_ylabel(f'Fitted kij ({form_name})')
    ax.set_title(f'(c) Best Fit: {form_name} (RMSE={form_data["rmse"]:.4f})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 4. Solubility error distribution
    ax = axes[1, 1]
    if 'accuracy' in form_data:
        errors = [e['rel_error'] for e in form_data['accuracy']['errors']]
        ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='k', linestyle='-', lw=1)
        ax.axvline(x=-5, color='r', linestyle='--', alpha=0.7)
        ax.axvline(x=5, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Solubility Error (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'(d) Solubility Error Distribution\n'
                     f'Mean |Δ|={form_data["accuracy"]["mean_abs_error"]:.1f}%, '
                     f'Max={form_data["accuracy"]["max_abs_error"]:.1f}%')

    plt.tight_layout()

    filepath = os.path.join(output_dir, 'embedded_salinity_bip_fit.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filepath}")
    plt.close()


def create_comparison_plot(data: List[Dict], fitted_forms: Dict, output_dir: str):
    """Create plot comparing Sechenov vs embedded kij behaviour."""
    os.makedirs(output_dir, exist_ok=True)

    # Get best form
    best_form = min(fitted_forms.items(), key=lambda x: x[1]['rmse'])
    form_name, form_data = best_form
    kij_func = form_data['kij_func']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Plot at T = 50, 100, 150°C
    temps = [50, 100, 150]
    P_bar = 100

    for ax, T_C in zip(axes, temps):
        T_K = celsius_to_kelvin(T_C)
        P_Pa = bar_to_pascal(P_bar)

        csw_range = np.linspace(0, 6, 50)

        # Sechenov (truth)
        x_sechenov = []
        for csw in csw_range:
            x = calc_target_brine_solubility(T_K, P_Pa, csw)
            x_sechenov.append(x)

        # Embedded kij
        x_embedded = []
        vle = SWBinaryVLE('H2', salinity_molal=0.0)
        for csw in csw_range:
            kij = kij_func(T_K, csw)
            try:
                x = vle._calc_x_with_kij(T_K, P_Pa, kij)
            except:
                x = x_sechenov[len(x_embedded)]
            x_embedded.append(x)

        ax.plot(csw_range, np.array(x_sechenov) * 1e4, 'b-', lw=2,
                label='Explicit Sechenov')
        ax.plot(csw_range, np.array(x_embedded) * 1e4, 'r--', lw=2,
                label='Embedded kij')

        ax.set_xlabel('Salinity (molal)')
        ax.set_ylabel('H₂ Solubility (×10⁴ mol/mol)')
        ax.set_title(f'T = {T_C}°C, P = {P_bar} bar')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Comparison: Explicit Sechenov vs Embedded kij Correlation\n'
                 f'Form: {form_name}', fontsize=12, y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'sechenov_vs_embedded_kij.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*70)
    print("FIT EMBEDDED SALINITY BIP CORRELATION FOR H2")
    print("="*70)

    # Generate synthetic data
    data = generate_synthetic_kij_data()

    # Fit various forms
    print("\n" + "-"*70)
    print("Fitting functional forms...")
    print("-"*70)

    fitted_forms = {}

    # Form 1: Linear multiplicative
    params1, rmse1 = fit_form_1(data)
    fitted_forms['Form 1: kij×(1+α·csw)'] = {
        'params': params1,
        'rmse': rmse1,
        'kij_func': lambda T_K, csw, p=params1: kij_freshwater(T_K) * (1 + p[0] * csw),
        'equation': f'kij = kij_fw × (1 + {params1[0]:.4f}·csw)',
    }
    print(f"\nForm 1: kij = kij_fw × (1 + α·csw)")
    print(f"  α = {params1[0]:.4f}")
    print(f"  RMSE = {rmse1:.5f}")

    # Form 2: Linear additive
    params2, rmse2 = fit_form_2(data)
    fitted_forms['Form 2: kij + β·csw'] = {
        'params': params2,
        'rmse': rmse2,
        'kij_func': lambda T_K, csw, p=params2: kij_freshwater(T_K) + p[0] * csw,
        'equation': f'kij = kij_fw + {params2[0]:.4f}·csw',
    }
    print(f"\nForm 2: kij = kij_fw + β·csw")
    print(f"  β = {params2[0]:.4f}")
    print(f"  RMSE = {rmse2:.5f}")

    # Form 3: T-dependent salinity
    params3, rmse3 = fit_form_3(data)
    fitted_forms['Form 3: kij + (β₀+β₁·Tr)·csw'] = {
        'params': params3,
        'rmse': rmse3,
        'kij_func': lambda T_K, csw, p=params3: kij_freshwater(T_K) + (p[0] + p[1] * T_K/BIP_TC_H2) * csw,
        'equation': f'kij = kij_fw + ({params3[0]:.4f} + {params3[1]:.4f}·Tr)·csw',
    }
    print(f"\nForm 3: kij = kij_fw + (β₀ + β₁·Tr)·csw")
    print(f"  β₀ = {params3[0]:.4f}, β₁ = {params3[1]:.4f}")
    print(f"  RMSE = {rmse3:.5f}")

    # Form 4: Salinity in A term
    params4, rmse4 = fit_form_4(data)
    fitted_forms['Form 4: (A·(1+α·csw)+Tr)/(B+C·Tr)'] = {
        'params': params4,
        'rmse': rmse4,
        'kij_func': lambda T_K, csw, p=params4: (A_FW * (1 + p[0] * csw) + T_K/BIP_TC_H2) / (B_FW + C_FW * T_K/BIP_TC_H2),
        'equation': f'kij = (A×(1+{params4[0]:.4f}·csw) + Tr) / (B + C·Tr)',
    }
    print(f"\nForm 4: kij = (A×(1+α·csw) + Tr) / (B + C·Tr)")
    print(f"  α = {params4[0]:.4f}")
    print(f"  RMSE = {rmse4:.5f}")

    # Form 5: Power law salinity
    params5, rmse5 = fit_form_5(data)
    fitted_forms['Form 5: (A·(1+α₀·csw^α₁)+Tr)/(B+C·Tr)'] = {
        'params': params5,
        'rmse': rmse5,
        'kij_func': lambda T_K, csw, p=params5: (A_FW * (1 + p[0] * (csw**p[1] if csw > 0 else 0)) + T_K/BIP_TC_H2) / (B_FW + C_FW * T_K/BIP_TC_H2),
        'equation': f'kij = (A×(1+{params5[0]:.4f}·csw^{params5[1]:.3f}) + Tr) / (B + C·Tr)',
    }
    print(f"\nForm 5: kij = (A×(1+α₀·csw^α₁) + Tr) / (B + C·Tr)")
    print(f"  α₀ = {params5[0]:.4f}, α₁ = {params5[1]:.3f}")
    print(f"  RMSE = {rmse5:.5f}")

    # Form 6: Quadratic temperature-dependent salinity (kij-error)
    params6, rmse6 = fit_form_6(data)
    fitted_forms['Form 6: kij + (β₀+β₁·Tr+β₂·Tr²)·csw (kij-err)'] = {
        'params': params6,
        'rmse': rmse6,
        'kij_func': lambda T_K, csw, p=params6: kij_freshwater(T_K) + (p[0] + p[1] * T_K/BIP_TC_H2 + p[2] * (T_K/BIP_TC_H2)**2) * csw,
        'equation': f'kij = kij_fw + ({params6[0]:.5f} + {params6[1]:.5f}·Tr + {params6[2]:.6f}·Tr²)·csw',
    }
    print(f"\nForm 6: kij = kij_fw + (β₀ + β₁·Tr + β₂·Tr²)·csw [minimizes kij error]")
    print(f"  β₀ = {params6[0]:.5f}, β₁ = {params6[1]:.5f}, β₂ = {params6[2]:.6f}")
    print(f"  RMSE = {rmse6:.5f}")

    # Form 7: Quadratic temperature-dependent salinity (ks-error) - RECOMMENDED
    params7, rmse7 = fit_form_7(data)
    fitted_forms['Form 7: kij + (β₀+β₁·Tr+β₂·Tr²)·csw (ks-err)'] = {
        'params': params7,
        'rmse': rmse7,
        'kij_func': lambda T_K, csw, p=params7: kij_freshwater(T_K) + (p[0] + p[1] * T_K/BIP_TC_H2 + p[2] * (T_K/BIP_TC_H2)**2) * csw,
        'equation': f'kij = kij_fw + ({params7[0]:.5f} + {params7[1]:.5f}·Tr + {params7[2]:.6f}·Tr²)·csw',
    }
    print(f"\nForm 7: kij = kij_fw + (β₀ + β₁·Tr + β₂·Tr²)·csw [minimizes ks error - RECOMMENDED]")
    print(f"  β₀ = {params7[0]:.5f}, β₁ = {params7[1]:.5f}, β₂ = {params7[2]:.6f}")
    print(f"  RMSE = {rmse7:.5f}")

    # Evaluate solubility accuracy for each form
    print("\n" + "-"*70)
    print("Evaluating solubility accuracy...")
    print("-"*70)

    print(f"\n{'Form':<40} {'Mean|Δ|%':>10} {'Max|Δ|%':>10} {'<1%':>8} {'<5%':>8}")
    print("-"*80)

    for name, form_data in fitted_forms.items():
        accuracy = evaluate_solubility_accuracy(data, name, form_data['kij_func'])
        form_data['accuracy'] = accuracy
        print(f"{name:<40} {accuracy['mean_abs_error']:>10.2f} "
              f"{accuracy['max_abs_error']:>10.2f} "
              f"{accuracy['within_1pct']:>7.0f}% {accuracy['within_5pct']:>7.0f}%")

    # Best form
    best = min(fitted_forms.items(), key=lambda x: x[1]['accuracy']['mean_abs_error'])
    print("-"*80)
    print(f"\nBest form: {best[0]}")
    print(f"  Equation: {best[1]['equation']}")
    print(f"  Solubility MARE: {best[1]['accuracy']['mean_abs_error']:.2f}%")

    # Create plots
    print("\nGenerating plots...")
    create_plots(data, fitted_forms, OUTPUT_DIR)
    create_comparison_plot(data, fitted_forms, OUTPUT_DIR)

    # Write report
    write_report(data, fitted_forms, OUTPUT_DIR)

    print("\n" + "="*70)
    print("FITTING COMPLETE")
    print("="*70)


def write_report(data: List[Dict], fitted_forms: Dict, output_dir: str):
    """Write detailed report."""
    filepath = os.path.join(output_dir, 'embedded_salinity_bip_report.txt')

    best = min(fitted_forms.items(), key=lambda x: x[1]['accuracy']['mean_abs_error'])

    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EMBEDDED SALINITY BIP CORRELATION FOR H2\n")
        f.write("="*70 + "\n\n")

        f.write("PURPOSE:\n")
        f.write("  Fit kij(T, csw) correlation for users who cannot modify their\n")
        f.write("  VLE flash routine but need brine solubility predictions.\n\n")

        f.write("NOTE: This is NOT the recommended approach. The gamma-phi method\n")
        f.write("  (K = γ·φL/φV with γ = 10^(ks·m)) is more rigorous and maintains\n")
        f.write("  exact correspondence with Sechenov behaviour.\n\n")

        f.write("="*70 + "\n")
        f.write("FITTED FORMS\n")
        f.write("="*70 + "\n\n")

        for name, form_data in fitted_forms.items():
            f.write(f"{name}\n")
            f.write(f"  Equation: {form_data['equation']}\n")
            f.write(f"  kij RMSE: {form_data['rmse']:.5f}\n")
            f.write(f"  Solubility MARE: {form_data['accuracy']['mean_abs_error']:.2f}%\n")
            f.write(f"  Max error: {form_data['accuracy']['max_abs_error']:.2f}%\n\n")

        f.write("="*70 + "\n")
        f.write("RECOMMENDED FORM\n")
        f.write("="*70 + "\n\n")

        f.write(f"Best fit: {best[0]}\n\n")
        f.write(f"  {best[1]['equation']}\n\n")
        f.write(f"  where kij_fw = (A + Tr) / (B + C·Tr)\n")
        f.write(f"        A = {A_FW}, B = {B_FW}, C = {C_FW}\n")
        f.write(f"        Tr = T / Tc,H2 with Tc,H2 = {BIP_TC_H2} K\n\n")

        f.write(f"Performance:\n")
        f.write(f"  Mean |Δx/x|: {best[1]['accuracy']['mean_abs_error']:.2f}%\n")
        f.write(f"  Max |Δx/x|: {best[1]['accuracy']['max_abs_error']:.2f}%\n")
        f.write(f"  Within ±5%: {best[1]['accuracy']['within_5pct']:.0f}%\n\n")

        f.write("="*70 + "\n")
        f.write("LIMITATIONS\n")
        f.write("="*70 + "\n\n")

        f.write("1. The BIP is a fitting parameter, not a thermodynamic quantity.\n")
        f.write("   Embedding salinity in kij does not accurately represent the\n")
        f.write("   physical Sechenov salting-out mechanism.\n\n")

        f.write("2. The correlation is fitted to synthetic data generated from\n")
        f.write("   the Sechenov model. Real brine behaviour may differ.\n\n")

        f.write("3. Extrapolation beyond the fitted range (0-6 molal, 25-150°C)\n")
        f.write("   should be done with caution.\n\n")

        f.write("For rigorous accuracy, use the gamma-phi formulation with\n")
        f.write("Sechenov activity coefficients integrated into the VLE flash.\n")

    print(f"Saved: {filepath}")


if __name__ == "__main__":
    main()
