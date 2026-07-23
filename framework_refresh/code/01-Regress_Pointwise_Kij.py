"""
Point-by-Point kij Regression Tool
===================================
Regresses individual kij_AQ and kij_NA values for each experimental data point.

Unlike the correlation-based refit, this tool finds the EXACT kij value that
matches each experimental point, enabling:
- Visual identification of outliers
- Development of new correlation forms
- Assessment of kij trends with T, P

SUPPORTED GASES:
- CO2, N2, H2S, CH4 (with S&W correlations for comparison)
- H2 (uses N2 correlation as initial guess, no S&W dashed lines on plots)
- C2H6, C3H8 (limited support)

METHODOLOGY:
- kij_AQ regressed to match x_gas (gas solubility in aqueous phase)
- kij_NA regressed to match y_H2O (water content in gas phase)
- Salinity > 0 points EXCLUDED (pure water only)

Author: Based on Søreide-Whitson EOS framework
Reference: Søreide & Whitson, Fluid Phase Equilibria 77 (1992) 217-240
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import sys
import os
import warnings

# Suppress convergence warnings during regression
warnings.filterwarnings('ignore')

# Add parent directory (code/) to path for shared module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

# Import from unified VLE engine
from _lib_vle_engine import (
    COMPONENTS, R_GAS, OMEGA_A, OMEGA_B,
    SWBinaryVLE, alpha_water_soreide, alpha_standard_pr,
    solve_cubic_eos, calc_fugacity_coeff,
    kij_aq_h2, kij_aq_co2, kij_aq_n2, kij_aq_h2s, kij_aq_ch4,
    kij_na_h2s_sw_eq17,
    get_kij_na, get_kij_aq,
)

@dataclass
class ExperimentalPoint:
    """Container for a single experimental data point."""
    gas: str
    T_K: float
    P_Pa: float
    salinity_molal: float = 0.0
    x_gas: Optional[float] = None
    y_H2O: Optional[float] = None
    source: str = ""

@dataclass
class PointwiseResult:
    """Container for regression result of a single point."""
    gas: str
    source: str
    T_K: float
    P_Pa: float
    x_gas_exp: Optional[float] = None
    y_H2O_exp: Optional[float] = None
    kij_aq: Optional[float] = None
    kij_na: Optional[float] = None
    kij_aq_converged: bool = False
    kij_na_converged: bool = False

# Unit conversions
def degC_to_K(T_C): return T_C + 273.15
def K_to_degC(T_K): return T_K - 273.15
def bar_to_Pa(P_bar): return P_bar * 1e5
def Pa_to_bar(P_Pa): return P_Pa / 1e5

# =============================================================================
# S&W BIP Correlations - Import from vle_engine, define local dispatch dicts
# =============================================================================
# Original S&W 1992 CO2 correlation for comparison plots
def kij_aq_co2_sw_1992(T_K, sal=0.0):
    """ORIGINAL Søreide-Whitson 1992 correlation for CO2-brine (Equation 14)."""
    Tr = T_K / COMPONENTS['CO2'].Tc
    cs = sal
    term1 = -0.31092 * (1.0 + 0.15587 * cs**0.7505)
    term2 = 0.23580 * (1.0 + 0.17837 * cs**0.979) * Tr
    term3 = -21.2566 * np.exp(-6.7222 * Tr - cs)
    return term1 + term2 + term3

def kij_na_co2_sw_1992(T_K):
    """ORIGINAL Søreide-Whitson 1992 kij_NA for CO2-water."""
    return 0.1896

# Yan et al. 2011 (used as default for CO2)
def kij_aq_co2_yan2011(T_K, sal=0.0):
    """Yan et al. 2011 improved correlation for CO2-brine."""
    cs = sal
    return (0.30823655 + 0.11820367 * cs - 0.00095381166 * cs**2
            - 126.42095 / T_K - 0.00062924435 * cs * T_K
            + 0.00000092946667 * cs * T_K**2)

def kij_na_co2_yan2011(T_K):
    """Yan et al. 2011 kij_NA for CO2-water."""
    return 0.18756

# H2S proposed correlations for plotting
def kij_na_h2s_sw(T_K):
    """S&W 1992 Equation 17: kij_NA for H2S-water."""
    return kij_na_h2s_sw_eq17(T_K)

def kij_na_h2s_proposed(T_K):
    """PROPOSED kij_NA for H2S-water."""
    Tr = T_K / COMPONENTS['H2S'].Tc
    if Tr <= 0.70:
        return 0.30 - 0.066 / 0.01
    return 0.30 - 0.066 / (Tr - 0.69)

def kij_aq_h2s_proposed(T_K):
    """PROPOSED kij_AQ for H2S-water."""
    A, B, C = -1.04E-01, 7.25E-03, 2.95E+00
    Tr = T_K / COMPONENTS['H2S'].Tc
    return A + B * np.exp(C * Tr)

# Use vle_engine functions with local wrappers for dispatch
def kij_aq_h2_proposed(T_K, sal=0.0):
    """H2 kij_AQ - uses rational form from vle_engine."""
    return kij_aq_h2(T_K, sal)

def kij_na_h2_proposed(T_K):
    """H2 kij_NA - constant 0.468."""
    return get_kij_na('H2', T_K)

# Dispatch dictionaries for plotting
KIJ_AQ_SW = {
    'CO2': kij_aq_co2_yan2011,
    'N2': lambda T, s=0: kij_aq_n2(T, s),
    'H2S': lambda T, s=0: kij_aq_h2s(T, s),
    'CH4': lambda T, s=0: kij_aq_ch4(T, s),
    'H2': kij_aq_h2_proposed,
}
KIJ_NA_SW = {
    'CO2': kij_na_co2_yan2011,
    'N2': lambda T: get_kij_na('N2', T),
    'H2S': lambda T: get_kij_na('H2S', T),
    'CH4': lambda T: get_kij_na('CH4', T),
    'H2': kij_na_h2_proposed,
}

# Gases that have actual S&W correlations (for plotting)
GASES_WITH_SW_CORRELATION = {'CO2', 'N2', 'H2S', 'CH4'}

# Gases with proposed correlations
GASES_WITH_PROPOSED_CORRELATION = {'H2S', 'H2'}

# =============================================================================
# VLE Calculator - Use SWBinaryVLE from vle_engine
# =============================================================================
# FastBinaryVLE is now provided by vle_engine.SWBinaryVLE
# Create an alias for backward compatibility
FastBinaryVLE = SWBinaryVLE

# =============================================================================
# Point-by-Point Regression Functions
# =============================================================================
def regress_kij_aq(vle: SWBinaryVLE, T_K: float, P_Pa: float,
                   x_gas_exp: float, kij_bounds=(-1.5, 1.0)) -> Tuple[Optional[float], bool]:
    """
    Find kij_AQ that reproduces experimental x_gas.
    Uses bounded minimization with grid-scan fallback for robustness
    (minimize_scalar can fail when VLE returns NaN in parts of the kij range).
    """
    def objective(kij):
        try:
            x_calc = vle._calc_x_with_kij(T_K, P_Pa, kij)
            # TODO(post-submission): x_calc can be np.nan (slips through the
            # comparisons below and poisons minimize_scalar); the 500-point
            # grid-scan fallback rescues those cases, so results are correct,
            # but an explicit isfinite guard here would make the fallback
            # rarely needed. Left unchanged for now so the published C2/C3
            # correlation fits remain bit-reproducible against this code path.
            if x_calc is None or x_calc <= 0 or x_calc >= 1:
                return 1e10
            # Relative error squared
            return ((x_calc - x_gas_exp) / max(x_gas_exp, 1e-6))**2
        except:
            return 1e10

    try:
        # Use bounded scalar minimization
        result = minimize_scalar(objective, bounds=kij_bounds, method='bounded',
                                options={'xatol': 1e-5, 'maxiter': 100})

        if result.fun < 0.01:  # Less than 10% relative error
            return result.x, True
        else:
            # Try wider bounds
            result2 = minimize_scalar(objective, bounds=(-2.0, 1.5), method='bounded',
                                     options={'xatol': 1e-5, 'maxiter': 100})
            if result2.fun < result.fun:
                result = result2

            if result.fun < 0.04:  # < 20% error
                return result.x, True

            # Grid-scan fallback: optimizer may have failed due to NaN regions
            best_kij = result.x
            best_fun = result.fun if np.isfinite(result.fun) else 1e10
            for kij_test in np.linspace(kij_bounds[0], kij_bounds[1], 500):
                f = objective(kij_test)
                if f < best_fun:
                    best_kij, best_fun = kij_test, f

            if best_fun < 0.04:  # < 20% error from grid scan
                # Refine with local minimization around grid-scan best
                try:
                    delta = (kij_bounds[1] - kij_bounds[0]) / 500
                    r3 = minimize_scalar(objective,
                                         bounds=(best_kij - 5*delta, best_kij + 5*delta),
                                         method='bounded',
                                         options={'xatol': 1e-8, 'maxiter': 200})
                    if r3.fun < best_fun:
                        best_kij, best_fun = r3.x, r3.fun
                except:
                    pass
                return best_kij, best_fun < 0.04

            return best_kij, False
    except:
        return None, False

def regress_kij_na(vle: SWBinaryVLE, T_K: float, P_Pa: float,
                   y_H2O_exp: float, kij_bounds=(-0.5, 1.0)) -> Tuple[Optional[float], bool]:
    """
    Find kij_NA that reproduces experimental y_H2O.
    Uses bounded minimization for robustness.

    Note: PR EOS cannot produce y_H2O values significantly below Raoult's law.
    If y_H2O_exp < 0.3 * (Psat/P), the model will fail to converge.
    """
    # Check for physically unreachable y_H2O values
    Psat_H2O = np.exp(73.649 - 7258.2/T_K - 7.3037*np.log(T_K) + 4.1653e-6*T_K**2)
    y_raoult = Psat_H2O / P_Pa

    # PR EOS cannot produce y_H2O much below Raoult's law
    # If experimental is <30% of Raoult, it's outside model capability
    if y_H2O_exp < 0.3 * y_raoult:
        return 1.5, False

    def objective(kij):
        try:
            y_calc = vle.calc_water_content_with_kij(T_K, P_Pa, kij)
            # NOTE: the hardened solver returns np.nan (not None) when no root
            # exists; nan slips through <=/>= comparisons and poisons
            # minimize_scalar, so it must be caught explicitly.
            if y_calc is None or not np.isfinite(y_calc) or y_calc <= 0 or y_calc >= 1:
                return 1e10
            # Relative error squared
            return ((y_calc - y_H2O_exp) / max(y_H2O_exp, 1e-6))**2
        except:
            return 1e10

    def refine(lo, hi):
        return minimize_scalar(objective, bounds=(lo, hi), method='bounded',
                               options={'xatol': 1e-5, 'maxiter': 100})

    try:
        # Coarse scan first: the hardened water-content solver (2026-07-19)
        # returns no root outside the valid kij basin, so the objective has wide
        # 1e10 plateaus that defeat bounded minimization directly. Locate the
        # valid basin on a grid, then refine locally around the best grid point.
        grid = np.linspace(kij_bounds[0], kij_bounds[1], 31)
        scanned = [(objective(k), k) for k in grid]
        finite = [(f, k) for f, k in scanned if f < 1e9]
        if not finite:
            grid2 = np.linspace(-1.0, 1.5, 51)
            scanned = [(objective(k), k) for k in grid2]
            finite = [(f, k) for f, k in scanned if f < 1e9]
            if not finite:
                return None, False
        _, k_best = min(finite)
        step = 0.1
        result = refine(max(-1.0, k_best - step), min(1.5, k_best + step))

        if result.fun < 0.01:
            return result.x, True
        else:
            # Try wider bounds
            result2 = refine(-1.0, 1.5)
            if result2.fun < result.fun:
                return result2.x, result2.fun < 0.04
            return result.x, result.fun < 0.04
    except:
        return None, False

# =============================================================================
# Main Regression Function with Progress Indicator
# =============================================================================
def regress_all_points(data: List[ExperimentalPoint],
                       exclude_salinity: bool = True,
                       include_H2: bool = True,
                       framework: str = 'proposed') -> List[PointwiseResult]:
    """
    Regress kij_AQ and kij_NA for each experimental point.
    
    Args:
        data: List of experimental points
        exclude_salinity: If True, skip points with salinity > 0
        include_H2: If True, include H2 data points (default: True)
    
    Returns:
        List of PointwiseResult objects
    """
    # Filter data
    filtered_data = []
    for pt in data:
        # Skip salinity points if requested
        if exclude_salinity and pt.salinity_molal >= 0.001:
            continue
        # Skip H2 if not requested
        if not include_H2 and pt.gas == 'H2':
            continue
        filtered_data.append(pt)
    
    # Count points to process
    n_total = len(filtered_data)
    n_aq = sum(1 for pt in filtered_data if pt.x_gas is not None)
    n_na = sum(1 for pt in filtered_data if pt.y_H2O is not None)
    
    print("="*70)
    print("POINT-BY-POINT kij REGRESSION")
    print("="*70)
    print(f"Total points: {n_total} (pure water only)")
    print(f"  Points with x_gas data: {n_aq} → regressing kij_AQ")
    print(f"  Points with y_H2O data: {n_na} → regressing kij_NA")
    print()
    
    results = []
    vle_cache = {}  # Cache VLE objects by gas
    
    # Progress tracking
    processed = 0
    n_aq_success = 0
    n_na_success = 0
    
    print("Processing: ", end="", flush=True)
    
    for i, pt in enumerate(filtered_data):
        # Progress indicator (every 5% or 10 points)
        if (i + 1) % max(1, n_total // 20) == 0 or i == n_total - 1:
            pct = 100 * (i + 1) / n_total
            print(f"\rProcessing: [{i+1}/{n_total}] {pct:.0f}% ", end="", flush=True)
        
        # Get or create VLE calculator
        if pt.gas not in vle_cache:
            if pt.gas in COMPONENTS:
                vle_cache[pt.gas] = SWBinaryVLE(pt.gas, 0.0, framework=framework)
            else:
                continue
        
        vle = vle_cache[pt.gas]
        
        # Create result container
        result = PointwiseResult(
            gas=pt.gas,
            source=pt.source,
            T_K=pt.T_K,
            P_Pa=pt.P_Pa,
            x_gas_exp=pt.x_gas,
            y_H2O_exp=pt.y_H2O
        )
        
        # Regress kij_AQ if x_gas data exists
        if pt.x_gas is not None and pt.x_gas > 0:
            kij_aq, converged = regress_kij_aq(vle, pt.T_K, pt.P_Pa, pt.x_gas)
            result.kij_aq = kij_aq
            result.kij_aq_converged = converged
            if converged:
                n_aq_success += 1
        
        # Regress kij_NA if y_H2O data exists
        if pt.y_H2O is not None and pt.y_H2O > 0:
            kij_na, converged = regress_kij_na(vle, pt.T_K, pt.P_Pa, pt.y_H2O)
            result.kij_na = kij_na
            result.kij_na_converged = converged
            if converged:
                n_na_success += 1
        
        results.append(result)
        processed += 1
    
    print(f"\rProcessing: [{n_total}/{n_total}] 100% - DONE!")
    print()
    print(f"Results:")
    print(f"  kij_AQ: {n_aq_success}/{n_aq} converged ({100*n_aq_success/max(1,n_aq):.1f}%)")
    print(f"  kij_NA: {n_na_success}/{n_na} converged ({100*n_na_success/max(1,n_na):.1f}%)")
    
    # Diagnostic for kij_NA failures - check for physically impossible y_H2O
    if n_na > 0 and n_na_success < n_na:
        na_failures = [r for r in results if r.y_H2O_exp is not None and not r.kij_na_converged]
        impossible_yh2o = 0
        for r in na_failures:
            Psat = np.exp(73.649 - 7258.2/r.T_K - 7.3037*np.log(r.T_K) + 4.1653e-6*r.T_K**2)
            y_raoult = Psat / r.P_Pa
            if r.y_H2O_exp < 0.3 * y_raoult:
                impossible_yh2o += 1
        
        if impossible_yh2o > 0:
            print()
            print(f"  WARNING: {impossible_yh2o}/{n_na - n_na_success} kij_NA failures have")
            print(f"           y_H2O_exp << Raoult's law (Psat/P).")
            print(f"           PR EOS fundamentally cannot produce y_H2O this low.")
            print(f"           Possible causes:")
            print(f"             - Data may be in different units (partial pressure bar?)")
            print(f"             - Different measurement methodology")
            print(f"             - Check original data source for units clarification")
    
    return results

# =============================================================================
# Plotting Functions
# =============================================================================
def _ensure_images_dir(base_dir: str = '.') -> str:
    """Ensure images subdirectory exists and return its path."""
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    return images_dir

def plot_kij_vs_temperature(results: List[PointwiseResult], gas: str = None,
                            save_dir: str = '.'):
    """
    Create 2×1 scatter plots of kij_AQ and kij_NA vs Temperature for each gas.
    Color-coded by SOURCE to help identify off-trend data sources.
    
    Plots are displayed inline AND saved as PNG files to images/ subdirectory.
    
    Shows multiple correlation lines:
    - CO2: S&W 1992 (original) and Yan et al. 2011
    - N2, CH4: S&W 1992 only
    - H2S: S&W 1992 + Proposed (for kij_NA)
    - H2: Proposed only
    
    Args:
        results: List of PointwiseResult objects
        gas: Optional gas name to filter (None = all gases)
        save_dir: Base directory for saving images (default: current dir)
    """
    # Ensure images directory exists
    images_dir = _ensure_images_dir(save_dir)
    
    # Group by gas
    by_gas = defaultdict(list)
    for r in results:
        by_gas[r.gas].append(r)
    
    gases_to_plot = [gas] if gas else sorted(by_gas.keys())
    
    # Define a good set of distinguishable colors (tab20 for up to 20 sources)
    color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*', 'X', 'P', 'd', '8']
    
    for gas_name in gases_to_plot:
        gas_results = by_gas[gas_name]
        if not gas_results:
            continue
        
        # Group by source
        by_source = defaultdict(list)
        for r in gas_results:
            by_source[r.source].append(r)
        
        # Sort sources by number of points (descending) for consistent ordering
        sorted_sources = sorted(by_source.keys(), 
                               key=lambda s: len(by_source[s]), reverse=True)
        
        # Assign colors and markers to sources
        source_style = {}
        for idx, source in enumerate(sorted_sources):
            source_style[source] = {
                'color': color_palette[idx % len(color_palette)],
                'marker': markers[idx % len(markers)],
            }
        
        # Separate AQ and NA results by source
        aq_by_source = defaultdict(list)
        na_by_source = defaultdict(list)
        
        for r in gas_results:
            if r.kij_aq is not None and r.kij_aq_converged:
                aq_by_source[r.source].append(r)
            if r.kij_na is not None and r.kij_na_converged:
                na_by_source[r.source].append(r)
        
        total_aq = sum(len(v) for v in aq_by_source.values())
        total_na = sum(len(v) for v in na_by_source.values())
        
        if total_aq == 0 and total_na == 0:
            print(f"No converged data for {gas_name}")
            continue
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
        fig.suptitle(f'{gas_name} - Point-by-Point Regressed kij Values (colored by source)', 
                    fontsize=14, fontweight='bold')
        
        # =====================================================================
        # Plot kij_AQ vs T (by source)
        # =====================================================================
        if total_aq > 0:
            for source in sorted_sources:
                pts = aq_by_source.get(source, [])
                if not pts:
                    continue
                
                T_vals = [K_to_degC(r.T_K) for r in pts]
                kij_vals = [r.kij_aq for r in pts]
                style = source_style[source]
                
                # Truncate long source names for legend
                label = source if len(source) <= 25 else source[:22] + '...'
                label = f'{label} ({len(pts)})'
                
                ax1.scatter(T_vals, kij_vals, 
                           c=[style['color']], 
                           marker=style['marker'],
                           s=70, edgecolors='k', linewidths=0.5, alpha=0.85,
                           label=label, zorder=5)
            
            # Add correlation lines for kij_AQ
            all_T = [K_to_degC(r.T_K) for r in gas_results if r.kij_aq is not None]
            if all_T:
                T_range = np.linspace(min(all_T), max(all_T), 50)
                T_K_range = [degC_to_K(t) for t in T_range]
                
                if gas_name == 'CO2':
                    # Show both S&W 1992 and Yan 2011
                    kij_sw1992 = [kij_aq_co2_sw_1992(T, 0.0) for T in T_K_range]
                    kij_yan = [kij_aq_co2_yan2011(T, 0.0) for T in T_K_range]
                    ax1.plot(T_range, kij_sw1992, 'k--', linewidth=2.5, 
                            label='S&W 1992', zorder=1)
                    ax1.plot(T_range, kij_yan, 'b-', linewidth=2.5, 
                            label='Yan et al. 2011', zorder=1)
                elif gas_name == 'H2':
                    # Show Proposed only
                    kij_prop = [kij_aq_h2_proposed(T, 0.0) for T in T_K_range]
                    ax1.plot(T_range, kij_prop, 'g-', linewidth=2.5, 
                            label='Proposed', zorder=1)
                elif gas_name in GASES_WITH_SW_CORRELATION:
                    # Show S&W only (N2, H2S, CH4)
                    kij_sw = [KIJ_AQ_SW[gas_name](T, 0.0) for T in T_K_range]
                    ax1.plot(T_range, kij_sw, 'k--', linewidth=2.5, 
                            label='S&W 1992', zorder=1)
            
            ax1.set_xlabel('Temperature (°C)', fontsize=11)
            ax1.set_ylabel('$k_{ij}^{AQ}$', fontsize=11)
            ax1.set_title(f'Aqueous Phase BIP (from x_{{{gas_name}}} data) - {total_aq} points')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Legend outside plot if many sources, otherwise inside
            n_sources_aq = len([s for s in sorted_sources if s in aq_by_source])
            if n_sources_aq > 6:
                ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, 
                          framealpha=0.9, ncol=1)
            else:
                ax1.legend(loc='best', fontsize=9, framealpha=0.9, ncol=1)
        else:
            ax1.text(0.5, 0.5, 'No x_gas data available', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title(f'Aqueous Phase BIP - No data')
        
        # =====================================================================
        # Plot kij_NA vs T (by source)
        # =====================================================================
        if total_na > 0:
            for source in sorted_sources:
                pts = na_by_source.get(source, [])
                if not pts:
                    continue
                
                T_vals = [K_to_degC(r.T_K) for r in pts]
                kij_vals = [r.kij_na for r in pts]
                style = source_style[source]
                
                label = source if len(source) <= 25 else source[:22] + '...'
                label = f'{label} ({len(pts)})'
                
                ax2.scatter(T_vals, kij_vals,
                           c=[style['color']],
                           marker=style['marker'],
                           s=70, edgecolors='k', linewidths=0.5, alpha=0.85,
                           label=label, zorder=5)
            
            # Add correlation lines for kij_NA
            all_T = [K_to_degC(r.T_K) for r in gas_results if r.kij_na is not None]
            if all_T:
                T_range = np.linspace(min(all_T), max(all_T), 50)
                T_K_range = [degC_to_K(t) for t in T_range]
                
                if gas_name == 'CO2':
                    # Show both S&W 1992 and Yan 2011
                    kij_sw1992 = [kij_na_co2_sw_1992(T) for T in T_K_range]
                    kij_yan = [kij_na_co2_yan2011(T) for T in T_K_range]
                    ax2.plot(T_range, kij_sw1992, 'k--', linewidth=2.5, 
                            label='S&W 1992', zorder=1)
                    ax2.plot(T_range, kij_yan, 'b-', linewidth=2.5, 
                            label='Yan et al. 2011', zorder=1)
                elif gas_name == 'H2S':
                    # Show S&W and Proposed
                    kij_sw = [kij_na_h2s_sw(T) for T in T_K_range]
                    kij_prop = [kij_na_h2s_proposed(T) for T in T_K_range]
                    ax2.plot(T_range, kij_sw, 'k--', linewidth=2.5, 
                            label='S&W 1992', zorder=1)
                    ax2.plot(T_range, kij_prop, 'g-', linewidth=2.5, 
                            label='Proposed', zorder=1)
                    kij_prop2 = [kij_aq_h2s_proposed(T) for T in T_K_range]
                    ax1.plot(T_range, kij_prop2, 'g-', linewidth=2.5, 
                            label='Proposed', zorder=1)
                elif gas_name == 'H2':
                    # Show Proposed only
                    kij_prop = [kij_na_h2_proposed(T) for T in T_K_range]
                    ax2.plot(T_range, kij_prop, 'g-', linewidth=2.5, 
                            label='Proposed', zorder=1)
                elif gas_name in GASES_WITH_SW_CORRELATION:
                    # Show S&W only (N2, CH4)
                    kij_sw = [KIJ_NA_SW[gas_name](T) for T in T_K_range]
                    ax2.plot(T_range, kij_sw, 'k--', linewidth=2.5, 
                            label='S&W 1992', zorder=1)
            
            ax2.set_xlabel('Temperature (°C)', fontsize=11)
            ax2.set_ylabel('$k_{ij}^{NA}$', fontsize=11)
            ax2.set_title(f'Non-Aqueous Phase BIP (from y_{{H₂O}} data) - {total_na} points')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Legend
            n_sources_na = len([s for s in sorted_sources if s in na_by_source])
            if n_sources_na > 6:
                ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8,
                          framealpha=0.9, ncol=1)
            else:
                ax2.legend(loc='best', fontsize=9, framealpha=0.9, ncol=1)
        else:
            ax2.text(0.5, 0.5, 'No y_H2O data available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title(f'Non-Aqueous Phase BIP - No data')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.82)  # Make room for legend if outside
        
        # Save to images directory
        filename = f'kij_vs_T_{gas_name}_by_source.png'
        filepath = os.path.join(images_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")
        
        plt.show()


def plot_kij_vs_temperature_by_pressure(results: List[PointwiseResult], gas: str = None,
                                         save_dir: str = '.'):
    """
    Alternative plot: kij vs Temperature color-coded by PRESSURE.
    Use this to see pressure dependence of kij values.
    
    Plots are displayed inline AND saved as PNG files to images/ subdirectory.
    
    Args:
        results: List of PointwiseResult objects
        gas: Optional gas name to filter (None = all gases)
        save_dir: Base directory for saving images (default: current dir)
    """
    # Ensure images directory exists
    images_dir = _ensure_images_dir(save_dir)
    
    # Group by gas
    by_gas = defaultdict(list)
    for r in results:
        by_gas[r.gas].append(r)
    
    gases_to_plot = [gas] if gas else sorted(by_gas.keys())
    
    for gas_name in gases_to_plot:
        gas_results = by_gas[gas_name]
        if not gas_results:
            continue
        
        # Separate AQ and NA results
        aq_data = [(r.T_K, r.P_Pa, r.kij_aq, r.source) for r in gas_results 
                   if r.kij_aq is not None and r.kij_aq_converged]
        na_data = [(r.T_K, r.P_Pa, r.kij_na, r.source) for r in gas_results 
                   if r.kij_na is not None and r.kij_na_converged]
        
        if not aq_data and not na_data:
            print(f"No converged data for {gas_name}")
            continue
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{gas_name} - Point-by-Point Regressed kij Values (colored by pressure)', fontsize=14)
        
        # Plot kij_AQ vs T
        if aq_data:
            T_vals = [K_to_degC(d[0]) for d in aq_data]
            P_vals = [Pa_to_bar(d[1]) for d in aq_data]
            kij_vals = [d[2] for d in aq_data]
            
            scatter = ax1.scatter(T_vals, kij_vals, c=P_vals, cmap='viridis', 
                                 s=60, edgecolors='k', linewidths=0.5, alpha=0.8)
            plt.colorbar(scatter, ax=ax1, label='Pressure (bar)')
            
            if gas_name in GASES_WITH_SW_CORRELATION:
                T_range = np.linspace(min(T_vals), max(T_vals), 50)
                T_K_range = [degC_to_K(t) for t in T_range]
                kij_sw = [KIJ_AQ_SW[gas_name](T, 0.0) for T in T_K_range]
                ax1.plot(T_range, kij_sw, 'r--', linewidth=2, label='S&W Correlation', zorder=1)
                ax1.legend(loc='best')
            
            ax1.set_xlabel('Temperature (°C)')
            ax1.set_ylabel('$k_{ij}^{AQ}$')
            ax1.set_title(f'Aqueous Phase BIP - {len(aq_data)} points')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No x_gas data available', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14)
        
        # Plot kij_NA vs T
        if na_data:
            T_vals = [K_to_degC(d[0]) for d in na_data]
            P_vals = [Pa_to_bar(d[1]) for d in na_data]
            kij_vals = [d[2] for d in na_data]
            
            scatter = ax2.scatter(T_vals, kij_vals, c=P_vals, cmap='plasma',
                                 s=60, edgecolors='k', linewidths=0.5, alpha=0.8)
            plt.colorbar(scatter, ax=ax2, label='Pressure (bar)')
            
            if gas_name in GASES_WITH_SW_CORRELATION:
                T_range = np.linspace(min(T_vals), max(T_vals), 50)
                T_K_range = [degC_to_K(t) for t in T_range]
                kij_sw = [KIJ_NA_SW[gas_name](T) for T in T_K_range]
                ax2.plot(T_range, kij_sw, 'r--', linewidth=2, label='S&W Correlation', zorder=1)
                ax2.legend(loc='best')
            
            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('$k_{ij}^{NA}$')
            ax2.set_title(f'Non-Aqueous Phase BIP - {len(na_data)} points')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No y_H2O data available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
        
        plt.tight_layout()
        
        # Save to images directory
        filename = f'kij_vs_T_{gas_name}_by_pressure.png'
        filepath = os.path.join(images_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")
        
        plt.show()

# =============================================================================
# Tabular Export Functions
# =============================================================================
def results_to_table(results: List[PointwiseResult]) -> str:
    """
    Export results as formatted table, grouped by gas.
    """
    by_gas = defaultdict(list)
    for r in results:
        by_gas[r.gas].append(r)
    
    lines = []
    lines.append("="*100)
    lines.append("POINT-BY-POINT kij REGRESSION RESULTS")
    lines.append("="*100)
    lines.append("")
    
    for gas in sorted(by_gas.keys()):
        gas_results = by_gas[gas]
        
        # Sort by source, then temperature, then pressure
        gas_results.sort(key=lambda r: (r.source, r.T_K, r.P_Pa))
        
        lines.append(f"\n{'='*100}")
        lines.append(f"GAS: {gas}")
        lines.append(f"{'='*100}")
        
        # Header
        header = f"{'Source':<25} {'T (°C)':>10} {'P (bar)':>10} {'kij_AQ':>12} {'kij_NA':>12}"
        lines.append(header)
        lines.append("-"*len(header))
        
        for r in gas_results:
            T_C = K_to_degC(r.T_K)
            P_bar = Pa_to_bar(r.P_Pa)
            
            # Format kij values
            if r.kij_aq is not None and r.kij_aq_converged:
                kij_aq_str = f"{r.kij_aq:+.6f}"
            elif r.kij_aq is not None:
                kij_aq_str = f"{r.kij_aq:+.6f}*"  # Mark unconverged
            else:
                kij_aq_str = "-"
            
            if r.kij_na is not None and r.kij_na_converged:
                kij_na_str = f"{r.kij_na:+.6f}"
            elif r.kij_na is not None:
                kij_na_str = f"{r.kij_na:+.6f}*"  # Mark unconverged
            else:
                kij_na_str = "-"
            
            # Truncate source name if too long
            source_short = r.source[:24] if len(r.source) > 24 else r.source
            
            line = f"{source_short:<25} {T_C:>10.1f} {P_bar:>10.1f} {kij_aq_str:>12} {kij_na_str:>12}"
            lines.append(line)
        
        # Summary for this gas
        n_aq = sum(1 for r in gas_results if r.kij_aq_converged)
        n_na = sum(1 for r in gas_results if r.kij_na_converged)
        lines.append("-"*len(header))
        lines.append(f"Total: {len(gas_results)} points, {n_aq} kij_AQ, {n_na} kij_NA converged")
    
    lines.append("")
    lines.append("* = unconverged (>20% error)")
    
    return "\n".join(lines)

def export_to_csv(results: List[PointwiseResult], filepath: str):
    """Export results to CSV file."""
    lines = ["Gas,Source,T_K,T_C,P_Pa,P_bar,x_gas_exp,y_H2O_exp,kij_AQ,kij_AQ_conv,kij_NA,kij_NA_conv"]

    for r in results:
        T_C = K_to_degC(r.T_K)
        P_bar = Pa_to_bar(r.P_Pa)

        x_gas = f"{r.x_gas_exp:.6e}" if r.x_gas_exp else ""
        y_H2O = f"{r.y_H2O_exp:.6e}" if r.y_H2O_exp else ""
        kij_aq = f"{r.kij_aq:.6f}" if r.kij_aq is not None else ""
        kij_na = f"{r.kij_na:.6f}" if r.kij_na is not None else ""

        line = f"{r.gas},{r.source},{r.T_K:.2f},{T_C:.2f},{r.P_Pa:.0f},{P_bar:.2f},"
        line += f"{x_gas},{y_H2O},{kij_aq},{r.kij_aq_converged},{kij_na},{r.kij_na_converged}"
        lines.append(line)

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))

    print(f"Results exported to {filepath}")


def load_results_from_csv(filepath: str) -> List[PointwiseResult]:
    """
    Load existing results from CSV file.

    Returns empty list if file doesn't exist.
    """
    import pandas as pd

    if not os.path.exists(filepath):
        return []

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Warning: Could not read existing results file: {e}")
        return []

    results = []
    for _, row in df.iterrows():
        result = PointwiseResult(
            gas=row['Gas'],
            source=row['Source'],
            T_K=float(row['T_K']),
            P_Pa=float(row['P_Pa']),
            x_gas_exp=float(row['x_gas_exp']) if pd.notna(row.get('x_gas_exp')) and row.get('x_gas_exp') != '' else None,
            y_H2O_exp=float(row['y_H2O_exp']) if pd.notna(row.get('y_H2O_exp')) and row.get('y_H2O_exp') != '' else None,
            kij_aq=float(row['kij_AQ']) if pd.notna(row.get('kij_AQ')) and row.get('kij_AQ') != '' else None,
            kij_na=float(row['kij_NA']) if pd.notna(row.get('kij_NA')) and row.get('kij_NA') != '' else None,
            kij_aq_converged=row.get('kij_AQ_conv', False) == True,
            kij_na_converged=row.get('kij_NA_conv', False) == True
        )
        results.append(result)

    return results


def make_point_key(gas: str, T_K: float, P_Pa: float, source: str) -> tuple:
    """Create a unique key for matching data points."""
    return (gas, round(T_K, 1), round(P_Pa / 1e5, 1), source)


def find_missing_points(data: List[ExperimentalPoint],
                        existing_results: List[PointwiseResult]) -> List[ExperimentalPoint]:
    """
    Find data points that don't have existing results.

    Matches on: gas, T_K (rounded to 0.1K), P_Pa (rounded to 0.1 bar), source
    """
    existing_keys = set()
    for r in existing_results:
        key = make_point_key(r.gas, r.T_K, r.P_Pa, r.source)
        existing_keys.add(key)

    missing = []
    for pt in data:
        key = make_point_key(pt.gas, pt.T_K, pt.P_Pa, pt.source)
        if key not in existing_keys:
            missing.append(pt)

    return missing


def merge_results(existing: List[PointwiseResult],
                  new: List[PointwiseResult]) -> List[PointwiseResult]:
    """
    Merge new results with existing, replacing any duplicates with new values.
    """
    # Index existing by key
    result_dict = {}
    for r in existing:
        key = make_point_key(r.gas, r.T_K, r.P_Pa, r.source)
        result_dict[key] = r

    # Add/replace with new results
    for r in new:
        key = make_point_key(r.gas, r.T_K, r.P_Pa, r.source)
        result_dict[key] = r

    # Sort by gas, source, T, P
    merged = sorted(result_dict.values(),
                   key=lambda r: (r.gas, r.source, r.T_K, r.P_Pa))
    return merged

# =============================================================================
# Excel Data Loader
# =============================================================================
# Sources excluded per gas (consistent with 12-Generate_Figures.py)
T_MIN_K = 273.15   # 0°C — exclude sub-zero data (consistent across all Paper 2 scripts)

EXCLUDE_SOURCES = {
    'CH4': {'Blount 1982', 'McGee 1981'},
    'CO2': {'Prutton & Savage 1945'},
}

def load_data_from_excel(filepath: str = '../../shared/data/solubility_points.xlsx',
                         sheet_name: int = 0,
                         gas_filter: List[str] = None) -> List[ExperimentalPoint]:
    """
    Load solubility data from Excel file.

    Args:
        filepath: Path to Excel file
        sheet_name: Sheet index or name
        gas_filter: List of gas names to include (e.g., ['H2'] or ['H2', 'CO2']).
                   If None, includes all gases.

    Note: Creates data points when EITHER:
        - x_gas is not NaN (gas solubility data), OR
        - y_H2O is not NaN AND z_gas=1 (water content data with gas indicator)
    """
    import pandas as pd

    df = pd.read_excel(filepath, sheet_name=sheet_name)

    gas_x_columns = {
        'CO2': 'x_CO2', 'H2S': 'x_H2S', 'N2': 'x_N2',
        'H2': 'x_H2', 'CH4': 'x_CH4', 'C2H6': 'x_C2H6', 'C3H8': 'x_C3H8',
    }

    # z_* columns indicate which gas a row belongs to (for y_H2O-only data)
    gas_z_columns = {
        'CO2': 'z_CO2', 'H2S': 'z_H2S', 'N2': 'z_N2',
        'H2': 'z_H2', 'CH4': 'z_CH4', 'C2H6': 'z_C2H6', 'C3H8': 'z_C3H8',
    }

    data = []

    for idx, row in df.iterrows():
        T_K = row['T_K']
        P_Pa = row['P_bar'] * 1e5
        salinity = row.get('Sal_m', 0.0)
        if pd.isna(salinity):
            salinity = 0.0

        source = row.get('Source', 'Unknown')

        y_H2O = row.get('y_H2O', None)
        if pd.isna(y_H2O):
            y_H2O = None

        for gas, x_col in gas_x_columns.items():
            # Apply gas filter if specified
            if gas_filter is not None and gas not in gas_filter:
                continue
            
            x_gas = row.get(x_col, None)
            x_gas_valid = pd.notna(x_gas)
            
            # Check z_* indicator column for y_H2O-only data
            z_col = gas_z_columns.get(gas)
            z_indicator = row.get(z_col, 0) if z_col else 0
            if pd.isna(z_indicator):
                z_indicator = 0
            z_gas_set = (z_indicator == 1)
            
            # Create point if x_gas exists OR (y_H2O exists AND z_gas indicates this gas)
            if x_gas_valid or (y_H2O is not None and z_gas_set):
                # Skip excluded sources
                if str(source) in EXCLUDE_SOURCES.get(gas, set()):
                    continue
                # Skip sub-ambient data (T < 20°C)
                if T_K < T_MIN_K:
                    continue
                point = ExperimentalPoint(
                    gas=gas,
                    T_K=float(T_K),
                    P_Pa=float(P_Pa),
                    salinity_molal=float(salinity),
                    x_gas=float(x_gas) if x_gas_valid else None,
                    y_H2O=float(y_H2O) if y_H2O is not None else None,
                    source=str(source)
                )
                data.append(point)
    
    return data

# =============================================================================
# Main Workflow Function
# =============================================================================
def run_pointwise_regression(data: List[ExperimentalPoint],
                             plot_results: bool = True,
                             export_csv: bool = True,
                             csv_path: str = '../../shared/data/pointwise_kij_results.csv',
                             include_H2: bool = True):
    """
    Run complete point-by-point regression workflow.
    
    1. Filter to pure water data only
    2. Regress individual kij values with progress indicator
    3. Plot kij vs T for each gas (2×1 layout)
    4. Export tabular results
    
    Args:
        data: List of experimental points
        plot_results: Generate scatter plots
        export_csv: Export results to CSV
        csv_path: Path for CSV output
        include_H2: Include H2 data in regression (default: True)
    """
    # Run regression
    results = regress_all_points(data, exclude_salinity=True, include_H2=include_H2)
    
    # Print table
    print()
    print(results_to_table(results))
    
    # Plot results
    if plot_results:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        # Get unique gases
        gases = sorted(set(r.gas for r in results))
        for gas in gases:
            print(f"\nPlotting {gas}...")
            plot_kij_vs_temperature(results, gas)
    
    # Export to CSV
    if export_csv:
        export_to_csv(results, csv_path)
    
    return results

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Configuration: framework for alpha function
    # 'proposed' = MC-3 alpha (Track 1), 'sw_original' = S&W alpha (Track 2 drop-in)
    FRAMEWORK = 'proposed'

    if FRAMEWORK == 'proposed':
        csv_path = '../../shared/data/pointwise_kij_results.csv'
    else:
        csv_path = '../../shared/data/pointwise_kij_results_sw_alpha.csv'

    # Configuration: which gases to process
    # Set to None to process all gases, or ['H2'] for H2 only (default for this project)
    GAS_FILTER = None  # Process all gases

    # Load data from Excel
    print(f"Framework: {FRAMEWORK}")
    print("Loading data from ../../shared/data/solubility_points.xlsx...")
    if GAS_FILTER:
        print(f"  Gas filter: {GAS_FILTER}")
    else:
        print("  Gas filter: ALL gases")
    data = load_data_from_excel('../../shared/data/solubility_points.xlsx', sheet_name=0, gas_filter=GAS_FILTER)
    print(f"Loaded {len(data)} data points")

    # Check for existing results
    existing_results = load_results_from_csv(csv_path)

    if existing_results:
        # Find missing points
        missing_data = find_missing_points(data, existing_results)
        print(f"\nExisting results file found: {len(existing_results)} results")
        print(f"New/missing data points: {len(missing_data)}")

        # Prompt user for mode
        print("\nOptions:")
        print("  [Enter] = Process only new/missing points (default)")
        print("  [F]     = Force full recalculation of all points")
        print("  [S]     = Skip regression, just plot existing results")

        choice = input("\nChoice [Enter/F/S]: ").strip().upper()

        if choice == 'F':
            print("\n>>> Full recalculation requested")
            data_to_process = data
            existing_results = []  # Clear existing
        elif choice == 'S':
            print("\n>>> Skipping regression, using existing results")
            data_to_process = []
        else:
            if missing_data:
                print(f"\n>>> Processing {len(missing_data)} new points only")
                data_to_process = missing_data
            else:
                print("\n>>> No new points to process")
                data_to_process = []
    else:
        print(f"\nNo existing results file found at {csv_path}")
        print(">>> Running full calculation")
        data_to_process = data

    # Process data if needed
    if data_to_process:
        new_results = regress_all_points(data_to_process, exclude_salinity=True, include_H2=True, framework=FRAMEWORK)

        # Merge with existing
        all_results = merge_results(existing_results, new_results)

        # Export merged results
        export_to_csv(all_results, csv_path)
    else:
        all_results = existing_results

    # Print summary table
    if all_results:
        print()
        print(results_to_table(all_results))

        # Generate plots
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)

        gases = sorted(set(r.gas for r in all_results))
        for gas in gases:
            print(f"\nPlotting {gas}...")
            plot_kij_vs_temperature(all_results, gas)
    else:
        print("\nNo results to display.")
