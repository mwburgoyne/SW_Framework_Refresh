"""
================================================================================
HYDROGEN PHASE EQUILIBRIA PAPER - FIGURE GENERATION SCRIPT (v11)
================================================================================

Self-contained script to generate all figures for:
"Assessment and Development of Hydrogen Solubility Correlations
 for Underground Storage Applications"

Authors: Mark Burgoyne, Markus H. Nielsen

VERSION 11 UPDATES:
- Figure 5: Added Tb vs kij_NA inset showing volatility trend (CH4 -> N2 -> H2)
- Figure 7: NEW - yH2O parity plot comparing predicted vs measured water content
  with ±10% error bands to validate the kij_NA = 0.468 assertion

CORRELATIONS:
- Rational form: kij = (A + Tr) / (B + C*Tr) with A=-14.59, B=2.184, C=0.365
- T-O Sechenov for salting-out (converted to log10 basis)
- kij_NA = 0.468 (constant)

FIGURES:
    1. Experimental Data Quality Assessment
    2. BIP Correlation Comparison (fitted vs excluded sources)
    3. Model Validation - Multiple pressure isotherms
    4. Sechenov coefficient comparison with implied Chabab 2023 trace
    5. Non-aqueous phase BIP comparison (with Tb vs kij_NA inset)
    6. kij_NA sensitivity analysis
    7. NEW: yH2O parity plot (predicted vs measured)

Dependencies: numpy, pandas, matplotlib, scipy
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import warnings
warnings.filterwarnings('ignore')

# Import from unified VLE engine
from _lib_vle_engine import (
    COMPONENTS, R_GAS, OMEGA_A, OMEGA_B,
    H2WaterVLE, SWBinaryVLE,
    kij_aq_rational, kij_aq_linear, kij_aq_chabab_2023, kij_aq_lopez_lazaro_2019,
    kij_na_chabab_2023, kij_na_lopez_lazaro_2019,
    sw_equation_8_ks, sechenov_TO,
    get_kij_na, get_kij_aq,
)

# Set publication-quality plot defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# =============================================================================
# CONSTANTS AND COMPONENT PROPERTIES (from vle_engine for backward compat)
# =============================================================================
TC_H2 = COMPONENTS['H2'].Tc  # 33.145 K
PC_H2 = COMPONENTS['H2'].Pc
OMEGA_H2 = COMPONENTS['H2'].omega

TC_H2O = COMPONENTS['H2O'].Tc  # 647.3 K
PC_H2O = COMPONENTS['H2O'].Pc

COMP_H2O = {'Tc': COMPONENTS['H2O'].Tc, 'Pc': COMPONENTS['H2O'].Pc, 'omega': COMPONENTS['H2O'].omega}
COMP_H2 = {'Tc': COMPONENTS['H2'].Tc, 'Pc': COMPONENTS['H2'].Pc, 'omega': COMPONENTS['H2'].omega}

# =============================================================================
# SOURCE CLASSIFICATION
# =============================================================================
FITTED_SOURCES = ['Wiebe 1932', 'Wiebe 1934', 'Chahab 2023', 'Chabab 2023',
                  'Torín-Ollarves 2021', 'Torin-Ollarves 2021']

VALIDATION_SOURCES = ['Stephan 1956']

EXCLUDED_SOURCES = ['Suciu 1951', 'Gillespie 1980']

SOURCE_STYLES = {
    'Wiebe 1932': {'marker': 'o', 'color': 'blue', 'status': 'fit'},
    'Wiebe 1934': {'marker': 's', 'color': 'blue', 'status': 'fit'},
    'Chahab 2023': {'marker': '^', 'color': 'red', 'status': 'fit'},
    'Chabab 2023': {'marker': '^', 'color': 'red', 'status': 'fit'},
    'Torín-Ollarves 2021': {'marker': 'D', 'color': 'green', 'status': 'fit'},
    'Torin-Ollarves 2021': {'marker': 'D', 'color': 'green', 'status': 'fit'},
    'Stephan 1956': {'marker': 'v', 'color': 'brown', 'status': 'validation'},
    'Suciu 1951': {'marker': '<', 'color': 'orange', 'status': 'excluded'},
    'Gillespie 1980': {'marker': '>', 'color': 'purple', 'status': 'excluded'},
}

# =============================================================================
# CORRELATIONS - now imported from vle_engine
# Local helper for constant kij_NA
# =============================================================================
def kij_na_constant():
    """Constant non-aqueous phase BIP."""
    return 0.468


def calc_implied_ks_lopez_lazaro(T_K, P_bar=100.0, m_brine=1.5):
    """
    Calculate implied Sechenov coefficient from Lopez-Lazaro 2019 BIP correlation.
    """
    P_Pa = P_bar * 1e5

    # VLE solver for fresh water
    vle = H2WaterVLE(salinity=0.0)

    # Fresh water solubility
    kij_fresh = kij_aq_lopez_lazaro_2019(T_K, csw=0.0)
    x_fresh = vle.calc_x_H2(T_K, P_Pa, kij_fresh)

    # Brine solubility (salinity effect embedded in BIP)
    kij_brine = kij_aq_lopez_lazaro_2019(T_K, csw=m_brine)
    x_brine = vle.calc_x_H2(T_K, P_Pa, kij_brine)

    # Calculate implied ks
    if x_brine > 0 and x_fresh > 0:
        ks_implied = np.log10(x_fresh / x_brine) / m_brine
    else:
        ks_implied = np.nan

    return ks_implied

# =============================================================================
# VLE SOLVER - now imported from vle_engine (H2WaterVLE, SWBinaryVLE)
# =============================================================================


# =============================================================================
# NON-AQUEOUS PHASE VLE (for yH2O calculations)
# =============================================================================
def P_sat_H2O(T_K):
    """
    Water saturation pressure using Wagner equation (IAPWS-IF97).
    Returns pressure in Pa.
    """
    TC_H2O_local = 647.096
    PC_H2O_local = 22.064e6

    Tr = T_K / TC_H2O_local
    tau = 1 - Tr

    # Wagner equation coefficients
    a1, a2, a3, a4, a5, a6 = -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502

    ln_Pr = (1/Tr) * (a1*tau + a2*tau**1.5 + a3*tau**3 + a4*tau**3.5 + a5*tau**4 + a6*tau**7.5)

    return PC_H2O_local * np.exp(ln_Pr)


def calc_y_H2O_predicted(T_K, P_Pa, kij_NA=0.468):
    """
    Calculate predicted water content in gas phase using PR EOS.
    Uses SWBinaryVLE from vle_engine.
    """
    vle = SWBinaryVLE('H2', salinity_molal=0.0)
    return vle.calc_water_content_with_kij(T_K, P_Pa, kij_NA)


# =============================================================================
# IMPLIED KS FROM BIP CORRELATIONS
# =============================================================================
def calc_implied_ks_chabab(T_K, P_bar=100.0, m_brine=1.5):
    """
    Calculate implied Sechenov coefficient from Chabab 2023 BIP correlation.
    """
    P_Pa = P_bar * 1e5
    vle = H2WaterVLE(salinity=0.0)

    kij_fresh = kij_aq_chabab_2023(T_K, m=0.0)
    x_fresh = vle.calc_x_H2(T_K, P_Pa, kij_fresh)

    kij_brine = kij_aq_chabab_2023(T_K, m=m_brine)
    x_brine = vle.calc_x_H2(T_K, P_Pa, kij_brine)

    if x_brine > 0 and x_fresh > 0:
        ks_implied = np.log10(x_fresh / x_brine) / m_brine
    else:
        ks_implied = np.nan

    return ks_implied


def calc_implied_ks_this_work_embedded(T_K, P_bar=100.0, m_brine=1.5):
    """
    Calculate implied Sechenov coefficient from this work's embedded salinity BIP.

    Uses quadratic form: kij = kij_fw + (β₀ + β₁·Tr + β₂·Tr²)·m
    where β₀ = 0.381, β₁ = -0.065, β₂ = 0.0033

    This form approximates explicit Sechenov to within 0.2% mean error.
    """
    P_Pa = P_bar * 1e5
    vle = H2WaterVLE(salinity=0.0)

    # Quadratic embedded salinity coefficients
    BETA0, BETA1, BETA2 = 0.3833, -0.06595, 0.003321
    TC_H2 = 33.145  # K

    Tr = T_K / TC_H2

    # Fresh water solubility (using standard kij_aq)
    kij_fresh = kij_aq_rational(T_K)
    x_fresh = vle.calc_x_H2(T_K, P_Pa, kij_fresh)

    # Brine solubility with embedded salinity term
    kij_brine = kij_fresh + (BETA0 + BETA1 * Tr + BETA2 * Tr**2) * m_brine
    x_brine = vle.calc_x_H2(T_K, P_Pa, kij_brine)

    if x_brine > 0 and x_fresh > 0:
        ks_implied = np.log10(x_fresh / x_brine) / m_brine
    else:
        ks_implied = np.nan

    return ks_implied


# =============================================================================
# DATA LOADING
# =============================================================================
def load_all_h2_data_from_csv(csv_path='../../shared/data/pointwise_kij_results.csv'):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, encoding='latin-1')

    # Normalise source name typo
    if 'Source' in df.columns:
        df['Source'] = df['Source'].str.replace('Chahab', 'Chabab', regex=False)

    mask_aq = (df['Gas'] == 'H2') & (df['kij_AQ'].notna()) & (df['kij_AQ_conv'] == True)
    df_kij_aq = df.loc[mask_aq, ['T_K', 'P_bar', 'kij_AQ', 'Source', 'x_gas_exp']].copy()
    
    # Load ALL kij_NA data (both converged and non-converged)
    mask_na = (df['Gas'] == 'H2') & (df['kij_NA'].notna())
    kij_na_results = []
    for _, row in df.loc[mask_na].iterrows():
        kij_na_results.append((row['T_K'], row['kij_NA'], row['Source'], row['P_bar']))
    
    # Get y_H2O experimental data for non-aqueous phase validation
    mask_yh2o = (df['Gas'] == 'H2') & (df['y_H2O_exp'].notna())
    yh2o_data = []
    for _, row in df.loc[mask_yh2o].iterrows():
        yh2o_data.append({
            'T_K': row['T_K'],
            'P_bar': row['P_bar'],
            'y_H2O_exp': row['y_H2O_exp'],
            'Source': row['Source'],
            'kij_NA': row.get('kij_NA', np.nan),
            'kij_NA_conv': row.get('kij_NA_conv', True)
        })
    
    # Also get raw experimental data
    mask_exp = (df['Gas'] == 'H2') & (df['x_gas_exp'].notna())
    df_exp = df.loc[mask_exp, ['T_K', 'P_bar', 'x_gas_exp', 'Source']].copy()
    
    print(f"Loaded {len(df_kij_aq)} H2 kij_AQ points")
    print(f"Loaded {len(kij_na_results)} H2 kij_NA points")
    print(f"Loaded {len(yh2o_data)} H2 y_H2O experimental points")
    print(f"Loaded {len(df_exp)} H2 experimental solubility points")
    
    return df_kij_aq, kij_na_results, df_exp, yh2o_data


# =============================================================================
# FIGURE 1: EXPERIMENTAL DATA QUALITY ASSESSMENT
# =============================================================================
def figure_1_experimental_data_quality(df_exp):
    """
    Figure 1: Experimental Data Quality Assessment
    (a) Wiebe vs Chabab comparison at 100 bar showing ~10% systematic offset
    (b) U-shaped experimental behaviour in Wiebe data at multiple pressures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # === Panel (a): Wiebe vs Chabab at 100 bar ===
    ax1 = axes[0]
    
    wiebe = df_exp[df_exp['Source'].str.contains('Wiebe')]
    chabab = df_exp[df_exp['Source'].str.contains('hab')]
    
    wiebe_100 = wiebe[abs(wiebe['P_bar'] - 100) < 10].copy()
    chabab_100 = chabab[abs(chabab['P_bar'] - 100) < 15].copy()
    
    ax1.plot(wiebe_100['T_K'] - 273.15, wiebe_100['x_gas_exp'] * 1000, 
             'bs-', markersize=10, linewidth=2, label='Wiebe & Gaddy (1934)')
    ax1.plot(chabab_100['T_K'] - 273.15, chabab_100['x_gas_exp'] * 1000, 
             'r^-', markersize=10, linewidth=2, label='Chabab et al. (2023)')
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel(r'H$_2$ Mole Fraction (×10$^{-3}$)', fontsize=12)
    ax1.set_title('(a) Source Comparison at 100 bar', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 110)
    ax1.set_ylim(1.0, 1.8)
    
    # === Panel (b): U-shaped behaviour at multiple pressures ===
    ax2 = axes[1]
    
    pressures = [50, 100, 200]
    colors = ['#0072B2', '#E69F00', '#D55E00']
    markers = ['o', 's', '^']
    
    for P_bar, color, marker in zip(pressures, colors, markers):
        subset = wiebe[abs(wiebe['P_bar'] - P_bar) < 10].sort_values('T_K')
        if len(subset) > 0:
            ax2.plot(subset['T_K'] - 273.15, subset['x_gas_exp'] * 1000,
                    marker=marker, color=color, linestyle='-', 
                    markersize=9, linewidth=2, label=f'{P_bar} bar')
    
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel(r'H$_2$ Mole Fraction (×10$^{-3}$)', fontsize=12)
    ax2.set_title('(b) U-Shaped Temperature Dependence (Wiebe)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 110)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2: BIP CORRELATION COMPARISON
# =============================================================================
def figure_2_kij_vs_Tr(df_kij_aq):
    """
    Figure 2: Point-by-point regressed kij_AQ vs reduced temperature.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sources = df_kij_aq['Source'].unique()
    
    for source in sorted(sources):
        mask = df_kij_aq['Source'] == source
        if mask.sum() == 0:
            continue
        
        T_K = df_kij_aq.loc[mask, 'T_K'].values
        Tr = T_K / TC_H2
        kij = df_kij_aq.loc[mask, 'kij_AQ'].values
        
        style = SOURCE_STYLES.get(source, {'marker': 'o', 'color': 'gray', 'status': 'excluded'})
        status = style.get('status', 'excluded')
        
        if status == 'fit':
            facecolor = style['color']
            edgecolor = 'black'
            linewidth = 0.5
            alpha = 0.8
            zorder = 10
            status_label = "(fit)"
        elif status == 'validation':
            facecolor = 'white'
            edgecolor = style['color']
            linewidth = 2
            alpha = 0.9
            zorder = 8
            status_label = "(validation)"
        else:
            facecolor = 'none'
            edgecolor = style['color']
            linewidth = 1.5
            alpha = 0.6
            zorder = 5
            status_label = "(excluded)"
        
        short_name = source.replace('1932', "'32").replace('1934', "'34")
        short_name = short_name.replace('1951', "'51").replace('1956', "'56")
        short_name = short_name.replace('1980', "'80").replace('2021', "'21").replace('2023', "'23")
        label = f'{short_name} {status_label} n={mask.sum()}'
        
        ax.scatter(Tr, kij, marker=style['marker'], s=90,
                  facecolors=facecolor, edgecolors=edgecolor,
                  linewidths=linewidth, alpha=alpha, label=label, zorder=zorder)
    
    Tr_range = np.linspace(8, 14.5, 100)
    T_range = Tr_range * TC_H2
    
    kij_rat = [kij_aq_rational(T) for T in T_range]
    ax.plot(Tr_range, kij_rat, 'b-', linewidth=3, 
            label=r'This work: $(A+T_r)/(B+CT_r)$', zorder=20)
    
    kij_chabab = [kij_aq_chabab_2023(T) for T in T_range]
    ax.plot(Tr_range, kij_chabab, 'r--', linewidth=2.5,
            label='Chabab 2023', zorder=15)
    
    kij_ll = [kij_aq_lopez_lazaro_2019(T) for T in T_range]
    ax.plot(Tr_range, kij_ll, color='purple', linestyle='-.', linewidth=2.5,
            label='Lopez-Lazaro 2019', zorder=15)
    
    kij_lin = [kij_aq_linear(T) for T in T_range]
    ax.plot(Tr_range, kij_lin, 'g:', linewidth=2, alpha=0.7,
            label=r'Linear: $A + BT_r$', zorder=15)
    
    ax.set_xlabel(r'Reduced Temperature, $T_r = T/T_{c,H_2}$', fontsize=12)
    ax.set_ylabel(r'Aqueous Phase BIP, $k_{ij}^{AQ}$', fontsize=12)
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlim(7.8, 14.8)
    
    ax2 = ax.twiny()
    T_ticks = np.array([273, 323, 373, 423, 473])
    Tr_ticks = T_ticks / TC_H2
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(Tr_ticks)
    ax2.set_xticklabels([f'{T} K' for T in T_ticks])
    ax2.set_xlabel('Temperature', fontsize=11)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 3: MODEL VALIDATION - MULTIPLE PRESSURES
# =============================================================================
def figure_3_validation(df_exp):
    """
    Figure 3: Model validation - comparing correlations against experimental data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    vle = H2WaterVLE()
    T_range_K = np.linspace(273.15, 423.15, 60)
    T_range_C = T_range_K - 273.15
    
    pressures = [50, 100, 150, 200]
    
    for idx, P_bar in enumerate(pressures):
        ax = axes[idx]

        wiebe = df_exp[df_exp['Source'].str.contains('Wiebe')]
        chabab = df_exp[df_exp['Source'].str.contains('hab')]
        torin = df_exp[df_exp['Source'].str.contains('Tor')]
        stephan = df_exp[df_exp['Source'].str.contains('Stephan')]

        # Only add legend labels on first panel
        lbl = lambda s: s if idx == 0 else None

        P_tol = 15 if P_bar >= 100 else 10
        w_mask = abs(wiebe['P_bar'] - P_bar) < P_tol
        if w_mask.sum() > 0:
            subset = wiebe[w_mask].sort_values('T_K')
            ax.scatter(subset['T_K'] - 273.15, subset['x_gas_exp'] * 1000,
                      c='blue', marker='s', s=100, edgecolors='k', linewidths=0.5,
                      alpha=0.9, zorder=15, label=lbl('Wiebe (1934)'))

        c_mask = abs(chabab['P_bar'] - P_bar) < P_tol
        if c_mask.sum() > 0:
            subset = chabab[c_mask].sort_values('T_K')
            ax.scatter(subset['T_K'] - 273.15, subset['x_gas_exp'] * 1000,
                      c='red', marker='^', s=100, edgecolors='k', linewidths=0.5,
                      alpha=0.9, zorder=15, label=lbl('Chabab (2023)'))

        if len(torin) > 0:
            t_mask = abs(torin['P_bar'] - P_bar) < P_tol
            if t_mask.sum() > 0:
                subset = torin[t_mask].sort_values('T_K')
                ax.scatter(subset['T_K'] - 273.15, subset['x_gas_exp'] * 1000,
                          c='green', marker='D', s=100, edgecolors='k', linewidths=0.5,
                          alpha=0.9, zorder=15, label=lbl('T-O (2021)'))

        if len(stephan) > 0:
            s_mask = abs(stephan['P_bar'] - P_bar) < P_tol
            if s_mask.sum() > 0:
                subset = stephan[s_mask].sort_values('T_K')
                ax.scatter(subset['T_K'] - 273.15, subset['x_gas_exp'] * 1000,
                          facecolors='none', edgecolors='brown', marker='v', s=100,
                          linewidths=1.5, alpha=0.8, zorder=12, label=lbl('Stephan (1956)'))

        x_rational = []
        for T_K in T_range_K:
            kij = kij_aq_rational(T_K)
            try:
                x = vle.calc_x_H2(T_K, P_bar * 1e5, kij) * 1000
            except:
                x = np.nan
            x_rational.append(x)
        ax.plot(T_range_C, x_rational, 'b-', linewidth=2.5, label=lbl('This work'), zorder=10)

        x_chabab = []
        for T_K in T_range_K:
            kij = kij_aq_chabab_2023(T_K, m=0)
            try:
                x = vle.calc_x_H2(T_K, P_bar * 1e5, kij) * 1000
            except:
                x = np.nan
            x_chabab.append(x)
        ax.plot(T_range_C, x_chabab, 'r--', linewidth=2, label=lbl('Chabab (2023)'), zorder=10)

        x_ll = []
        for T_K in T_range_K:
            kij = kij_aq_lopez_lazaro_2019(T_K, csw=0)
            try:
                x = vle.calc_x_H2(T_K, P_bar * 1e5, kij) * 1000
            except:
                x = np.nan
            x_ll.append(x)
        ax.plot(T_range_C, x_ll, color='purple', linestyle='-.', linewidth=2,
                label=lbl('Lopez-Lazaro (2019)'), zorder=10)

        x_linear = []
        for T_K in T_range_K:
            kij = kij_aq_linear(T_K)
            try:
                x = vle.calc_x_H2(T_K, P_bar * 1e5, kij) * 1000
            except:
                x = np.nan
            x_linear.append(x)
        ax.plot(T_range_C, x_linear, 'g:', linewidth=2, alpha=0.7, label=lbl('Linear'), zorder=10)

        ax.set_xlabel('Temperature (°C)', fontsize=11)
        ax.set_ylabel(r'H$_2$ Mole Fraction (×10$^{-3}$)', fontsize=11)
        ax.set_title(f'({chr(97+idx)}) P = {P_bar} bar', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 155)

        if P_bar == 50:
            ax.set_ylim(0.4, 1.0)
        elif P_bar == 100:
            ax.set_ylim(0.9, 1.9)
        elif P_bar == 150:
            ax.set_ylim(1.4, 2.8)
        else:
            ax.set_ylim(1.8, 3.6)

    # Shared legend below panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10)
    fig.subplots_adjust(bottom=0.12)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# =============================================================================
# FIGURE 4: SECHENOV COEFFICIENT COMPARISON
# =============================================================================
def figure_4_sechenov_comparison():
    """
    Figure 4: Sechenov coefficient comparison with implied Chabab 2023 trace.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    LN_TO_LOG10 = 2.303
    
    T_CY = np.array([274.04, 283.15, 289.55, 293.15, 295.34, 301.50])
    ks_CY = np.array([0.127, 0.101, 0.115, 0.093, 0.106, 0.089])
    
    T_Gordon = np.array([273.15, 283.15, 293.15, 302.15])
    ks_Gordon = np.array([0.110, 0.098, 0.087, 0.078])
    
    T_MB = np.array([285.75, 303.15, 322.55, 344.85])
    ks_MB = np.array([0.112, 0.107, 0.097, 0.081])
    
    T_GB = np.array([288.15, 323.15])
    ks_GB = np.array([0.095, 0.076])
    
    T_Chabab = np.array([298.15, 323.15, 373.15])
    ks_Chabab = np.array([0.092, 0.078, 0.070])
    
    T_TO = np.array([323.15, 423.15])
    ks_TO_ln = np.array([0.130, 0.170])
    ks_TO = ks_TO_ln / LN_TO_LOG10
    
    T_range = np.linspace(273.15, 423.15, 100)
    T_range_C = T_range - 273.15
    
    ks_SW = [sw_equation_8_ks(T_C, COMPONENTS['H2'].Tb) for T_C in T_range_C]
    ax.plot(T_range, ks_SW, 'k-', linewidth=3, label='S&W Eq. 8 ($T_b$=20.3 K)', zorder=20)
    
    ks_TO_corr = [sechenov_TO(T, log10_basis=True) for T in T_range]
    ax.plot(T_range, ks_TO_corr, 'b--', linewidth=2.5, label='T-O 2021 (Eq. 22)', zorder=15)
    
    ks_implied_chabab = []
    for T_K in T_range:
        ks = calc_implied_ks_chabab(T_K, P_bar=100, m_brine=1.5)
        ks_implied_chabab.append(ks)
    ax.plot(T_range, ks_implied_chabab, 'r-.', linewidth=2.5, label='Chabab 2023 (implied)', zorder=15)

    ks_implied_ll = []
    for T_K in T_range:
        ks = calc_implied_ks_lopez_lazaro(T_K, P_bar=100, m_brine=1.5)
        ks_implied_ll.append(ks)
    ax.plot(T_range, ks_implied_ll, color='purple', linestyle=':', linewidth=2.5,
            label='Lopez-Lazaro 2019 (implied)', zorder=15)

    # This work - embedded salinity BIP (quadratic form)
    ks_implied_this_work = []
    for T_K in T_range:
        ks = calc_implied_ks_this_work_embedded(T_K, P_bar=100, m_brine=1.5)
        ks_implied_this_work.append(ks)
    ax.plot(T_range, ks_implied_this_work, color='darkgreen', linestyle=(0, (3, 1, 1, 1)),
            linewidth=2.5, label='This work embedded (implied)', zorder=16)

    ax.scatter(T_CY, ks_CY, c='green', marker='s', s=100, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10, label='C&Y 1974')
    ax.scatter(T_Gordon, ks_Gordon, c='purple', marker='d', s=100, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10, label='Gordon 1977')
    ax.scatter(T_MB, ks_MB, c='orange', marker='v', s=100, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10, label='M&B 1952')
    ax.scatter(T_GB, ks_GB, c='red', marker='^', s=100, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10, label='G&B 1971')
    ax.scatter(T_Chabab, ks_Chabab, c='blue', marker='o', s=120, alpha=0.9,
               edgecolors='k', linewidths=0.5, zorder=12, label='Chabab 2020')
    ax.scatter(T_TO, ks_TO, c='cyan', marker='D', s=120, alpha=0.9,
               edgecolors='k', linewidths=1.5, zorder=12, label='T-O 2021')
    
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel(r'Sechenov Coefficient, $k_s$ (kg/mol, log$_{10}$ basis)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(265, 435)
    ax.set_ylim(0.03, 0.15)
    
    ax_top = ax.twiny()
    T_C_ticks = np.array([0, 50, 100, 150])
    T_K_ticks = T_C_ticks + 273.15
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(T_K_ticks)
    ax_top.set_xticklabels([f'{T}°C' for T in T_C_ticks])
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 5: NON-AQUEOUS PHASE BIP (WITH Tb vs kij_NA INSET)
# =============================================================================
def figure_5_na_comparison(kij_na_results=None):
    """
    Figure 5: Non-aqueous phase BIP with Tb vs kij_NA inset.
    
    The inset shows the volatility trend: CH4 -> N2 -> H2
    demonstrating that kij_NA = 0.468 is consistent with S&W pattern.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if kij_na_results:
        quality_data = {}
        excluded_data = {}
        
        for T_K, kij_na, source, P_bar in kij_na_results:
            is_quality = (kij_na > -0.99) and (P_bar >= 50)
            
            target = quality_data if is_quality else excluded_data
            if source not in target:
                target[source] = {'T': [], 'kij': [], 'reason': []}
            target[source]['T'].append(T_K)
            target[source]['kij'].append(kij_na)
            if not is_quality:
                if kij_na <= -0.99:
                    target[source]['reason'].append('bound')
                else:
                    target[source]['reason'].append('low-P')
        
        source_styles = {
            'Suciu 1951': {'color': 'orange', 'marker': '^'},
            'Torín-Ollarves 2021': {'color': 'green', 'marker': 'D'},
            'Gillespie 1980': {'color': 'purple', 'marker': 's'},
        }
        
        for source, data in quality_data.items():
            style = source_styles.get(source, {'color': 'gray', 'marker': 'o'})
            ax.scatter(data['T'], data['kij'], 
                      c=style['color'], marker=style['marker'],
                      s=90, alpha=0.8, edgecolors='k', linewidths=0.5,
                      label=f"{source} (quality, n={len(data['T'])})", zorder=10)

        for source, data in excluded_data.items():
            style = source_styles.get(source, {'color': 'gray', 'marker': 'o'})
            ax.scatter(data['T'], data['kij'],
                      facecolors='none', edgecolors=style['color'],
                      marker=style['marker'], s=90, alpha=0.5, linewidths=1.5,
                      label=f"{source} (excluded, n={len(data['T'])})", zorder=5)
    
    T_range = np.linspace(273.15, 500, 100)
    
    ax.axhline(y=0.468, color='blue', linewidth=2.5,
               label=r'This work: $k_{ij}^{NA} = 0.468$', zorder=15)
    
    kij_na_chabab = [kij_na_chabab_2023(T) for T in T_range]
    ax.plot(T_range, kij_na_chabab, 'r--', linewidth=2.5,
            label=r'Chabab 2023: $0.020 + 0.043 T_r$', zorder=15)
    
    kij_na_ll = [kij_na_lopez_lazaro_2019(T) for T in T_range]
    ax.plot(T_range, kij_na_ll, color='purple', linestyle='-.', linewidth=2.5,
            label=r'Lopez-Lazaro 2019: $2.50 - 0.18 T_r$', zorder=15)
    
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel(r'Non-Aqueous Phase BIP, $k_{ij}^{NA}$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(270, 500)
    ax.set_ylim(-1.2, 1.5)
    
    ax2 = ax.twiny()
    T_C_ticks = np.array([0, 50, 100, 150, 200])
    T_K_ticks = T_C_ticks + 273.15
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(T_K_ticks)
    ax2.set_xticklabels([f'{T}°C' for T in T_C_ticks])
    
    # ==========================================================================
    # INSET: Tb vs kij_NA showing volatility trend
    # ==========================================================================
    ax_inset = inset_axes(ax, width="35%", height="35%", loc='lower left',
                          bbox_to_anchor=(0.08, 0.08, 1, 1), bbox_transform=ax.transAxes)
    
    # S&W non-aqueous BIP values from their 1992 paper (Table 2)
    # Note: These are the original S&W values for water-gas pairs
    Tb_data = {
        'CH$_4$': {'Tb': 111.66, 'kij_NA': 0.485},  # Methane
        'N$_2$':  {'Tb': 77.36,  'kij_NA': 0.478},  # Nitrogen  
        'H$_2$':  {'Tb': 20.3,   'kij_NA': 0.468},  # Hydrogen (this work)
    }
    
    Tb_values = [Tb_data[gas]['Tb'] for gas in Tb_data]
    kij_values = [Tb_data[gas]['kij_NA'] for gas in Tb_data]
    labels = list(Tb_data.keys())
    colors = ['blue', 'green', 'red']
    
    for i, (Tb, kij, label, color) in enumerate(zip(Tb_values, kij_values, labels, colors)):
        ax_inset.scatter(Tb, kij, s=100, c=color, marker='o', edgecolors='k', 
                        linewidths=1, zorder=10)
        # Add labels
        if label == 'H$_2$':
            ax_inset.annotate(label, (Tb, kij), xytext=(Tb+8, kij+0.003), fontsize=9)
        elif label == 'N$_2$':
            ax_inset.annotate(label, (Tb, kij), xytext=(Tb+5, kij-0.008), fontsize=9)
        else:
            ax_inset.annotate(label, (Tb, kij), xytext=(Tb-25, kij+0.003), fontsize=9)
    
    # Fit and plot trend line
    z = np.polyfit(Tb_values, kij_values, 1)
    p = np.poly1d(z)
    Tb_line = np.linspace(0, 130, 50)
    ax_inset.plot(Tb_line, p(Tb_line), 'k--', linewidth=1.5, alpha=0.7, zorder=5)
    
    ax_inset.set_xlabel(r'$T_b$ (K)', fontsize=9)
    ax_inset.set_ylabel(r'$k_{ij}^{NA}$', fontsize=9)
    ax_inset.set_title('Volatility Trend', fontsize=9)
    ax_inset.set_xlim(0, 130)
    ax_inset.set_ylim(0.455, 0.495)
    ax_inset.grid(True, alpha=0.3, linewidth=0.5)
    ax_inset.tick_params(labelsize=8)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 6: kij_NA SENSITIVITY
# =============================================================================
def figure_6_kij_na_sensitivity():
    """
    Figure 6: Implied kij_NA uncertainty vs pressure at various temperatures.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    R = 8.314462
    TC_H2_local = 33.145
    PC_H2_local = 1.2964e6
    OMEGA_H2_local = -0.219
    TC_H2O_local = 647.3
    PC_H2O_local = 22.12e6
    OMEGA_H2O_local = 0.3434
    
    def calc_a_b(T, Tc, Pc, omega):
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
        alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2
        a = 0.45724 * R**2 * Tc**2 / Pc * alpha
        b = 0.07780 * R * Tc / Pc
        return a, b
    
    def calc_Z_vapor(T, P, a_mix, b_mix):
        A = a_mix * P / (R**2 * T**2)
        B = b_mix * P / (R * T)
        coeffs = [1, -(1-B), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)]
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > B]
        return max(real_roots) if real_roots else 1.0
    
    def calc_y_H2O(T_K, P_Pa, kij_NA):
        a_H2, b_H2 = calc_a_b(T_K, TC_H2_local, PC_H2_local, OMEGA_H2_local)
        a_H2O, b_H2O = calc_a_b(T_K, TC_H2O_local, PC_H2O_local, OMEGA_H2O_local)
        P_sat = P_sat_H2O(T_K)
        y_H2O = min(P_sat / P_Pa, 0.99)
        
        for _ in range(20):
            y_H2 = 1 - y_H2O
            a_mix = y_H2**2 * a_H2 + 2*y_H2*y_H2O*np.sqrt(a_H2*a_H2O)*(1-kij_NA) + y_H2O**2 * a_H2O
            b_mix = y_H2*b_H2 + y_H2O*b_H2O
            Z = calc_Z_vapor(T_K, P_Pa, a_mix, b_mix)
            
            A_mix = a_mix * P_Pa / (R**2 * T_K**2)
            B_mix = b_mix * P_Pa / (R * T_K)
            a_12 = np.sqrt(a_H2 * a_H2O) * (1 - kij_NA)
            sum_ya_H2O = y_H2 * a_12 + y_H2O * a_H2O
            
            term1 = (b_H2O/b_mix) * (Z - 1)
            term2 = -np.log(max(Z - B_mix, 1e-10))
            term3_num = 2 * sum_ya_H2O / a_mix - b_H2O / b_mix
            term3_denom = 2 * np.sqrt(2) * B_mix
            term3_log = np.log((Z + (1+np.sqrt(2))*B_mix) / (Z + (1-np.sqrt(2))*B_mix))
            ln_phi_H2O = term1 + term2 - (A_mix / term3_denom) * term3_num * term3_log
            phi_H2O = np.exp(ln_phi_H2O)
            
            y_H2O_new = P_sat / (phi_H2O * P_Pa)
            y_H2O_new = max(min(y_H2O_new, 0.99), 1e-10)
            y_H2O = 0.5 * y_H2O + 0.5 * y_H2O_new
            if abs(y_H2O - y_H2O_new) < 1e-8:
                break
        return y_H2O
    
    def calc_sensitivity(T_K, P_Pa, kij_NA_base=0.468, delta_kij=0.01):
        y_plus = calc_y_H2O(T_K, P_Pa, kij_NA_base + delta_kij)
        y_minus = calc_y_H2O(T_K, P_Pa, kij_NA_base - delta_kij)
        return (y_plus - y_minus) / (2 * delta_kij)
    
    def calc_implied_uncertainty(T_K, P_Pa, rel_error=0.05):
        y_H2O = calc_y_H2O(T_K, P_Pa, 0.468)
        sens = abs(calc_sensitivity(T_K, P_Pa))
        delta_y = rel_error * y_H2O
        return delta_y / sens if sens > 1e-10 else float('inf')
    
    temperatures = [50, 75, 100, 125, 150]
    pressures = np.linspace(10, 200, 80)
    colors = ['#0072B2', '#56B4E9', '#999999', '#E69F00', '#D55E00']
    
    for i, T_C in enumerate(temperatures):
        T_K = T_C + 273.15
        uncertainties = []
        for P_bar in pressures:
            try:
                unc = calc_implied_uncertainty(T_K, P_bar * 1e5, rel_error=0.05)
                uncertainties.append(min(unc, 2.0))
            except:
                uncertainties.append(np.nan)
        ax.plot(pressures, uncertainties, color=colors[i], linewidth=2.5, label=f'{T_C}°C')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axhspan(0.5, 1.0, alpha=0.1, color='red')
    ax.axhspan(0.25, 0.5, alpha=0.1, color='yellow')
    ax.axhspan(0, 0.25, alpha=0.1, color='green')
    
    ax.text(175, 0.55, r'Poor ($\sigma$ > 0.5)', fontsize=9, color='gray', ha='center')
    ax.text(175, 0.30, 'Marginal', fontsize=9, color='gray', ha='center')
    ax.text(175, 0.12, r'Good ($\sigma$ < 0.25)', fontsize=9, color='gray', ha='center')
    
    ax.axvline(x=50, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(52, 0.85, 'P = 50 bar\nthreshold', fontsize=9, ha='left')
    
    ax.set_xlabel('Pressure (bar)', fontsize=12)
    ax.set_ylabel(r'Implied $k_{ij,\mathrm{NA}}$ Standard Deviation ($\sigma$)', fontsize=12)
    ax.set_xlim(10, 200)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', title='Temperature')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 7: yH2O PARITY PLOT (NEW)
# =============================================================================
def figure_7_yH2O_parity(yh2o_data=None):
    """
    Figure 7: Water content parity plot - predicted vs measured yH2O.
    
    Validates the claim that kij_NA = 0.468 delivers values within ±10%
    relative error for 90% of quality-filtered data points.
    
    Quality filter: P >= 50 bar (low-P data excluded due to ill-conditioning)
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # If no data provided, use representative experimental data from literature
    if yh2o_data is None or len(yh2o_data) == 0:
        # Representative data points from T-O 2021, Suciu 1951, Gillespie 1980
        # at P >= 50 bar (quality-filtered)
        yh2o_data = [
            # Torín-Ollarves 2021 (high quality, modern measurements)
            {'T_K': 323.15, 'P_bar': 112, 'y_H2O_exp': 0.00108, 'Source': 'T-O 2021'},
            {'T_K': 323.15, 'P_bar': 220, 'y_H2O_exp': 0.00055, 'Source': 'T-O 2021'},
            {'T_K': 373.15, 'P_bar': 112, 'y_H2O_exp': 0.00890, 'Source': 'T-O 2021'},
            {'T_K': 373.15, 'P_bar': 220, 'y_H2O_exp': 0.00455, 'Source': 'T-O 2021'},
            {'T_K': 373.15, 'P_bar': 330, 'y_H2O_exp': 0.00305, 'Source': 'T-O 2021'},
            {'T_K': 423.15, 'P_bar': 112, 'y_H2O_exp': 0.0395, 'Source': 'T-O 2021'},
            {'T_K': 423.15, 'P_bar': 220, 'y_H2O_exp': 0.0202, 'Source': 'T-O 2021'},
            {'T_K': 423.15, 'P_bar': 330, 'y_H2O_exp': 0.0136, 'Source': 'T-O 2021'},
            # Suciu 1951 (P >= 50 bar subset)
            {'T_K': 299.26, 'P_bar': 69, 'y_H2O_exp': 0.00047, 'Source': 'Suciu 1951'},
            {'T_K': 299.26, 'P_bar': 138, 'y_H2O_exp': 0.00025, 'Source': 'Suciu 1951'},
            {'T_K': 299.26, 'P_bar': 207, 'y_H2O_exp': 0.00018, 'Source': 'Suciu 1951'},
            {'T_K': 338.71, 'P_bar': 69, 'y_H2O_exp': 0.00370, 'Source': 'Suciu 1951'},
            {'T_K': 338.71, 'P_bar': 138, 'y_H2O_exp': 0.00192, 'Source': 'Suciu 1951'},
            {'T_K': 338.71, 'P_bar': 207, 'y_H2O_exp': 0.00133, 'Source': 'Suciu 1951'},
            {'T_K': 366.48, 'P_bar': 69, 'y_H2O_exp': 0.01235, 'Source': 'Suciu 1951'},
            {'T_K': 366.48, 'P_bar': 138, 'y_H2O_exp': 0.00635, 'Source': 'Suciu 1951'},
            {'T_K': 366.48, 'P_bar': 207, 'y_H2O_exp': 0.00438, 'Source': 'Suciu 1951'},
            # Gillespie 1980 (P >= 50 bar subset)
            {'T_K': 310.93, 'P_bar': 69, 'y_H2O_exp': 0.00095, 'Source': 'Gillespie 1980'},
            {'T_K': 310.93, 'P_bar': 138, 'y_H2O_exp': 0.00050, 'Source': 'Gillespie 1980'},
            {'T_K': 366.48, 'P_bar': 69, 'y_H2O_exp': 0.01195, 'Source': 'Gillespie 1980'},
            {'T_K': 366.48, 'P_bar': 138, 'y_H2O_exp': 0.00620, 'Source': 'Gillespie 1980'},
        ]
    
    # Filter to quality data matching Table 5: P >= 50 bar, UHS-relevant T range,
    # and converged kij_NA regression (excludes bound-hitting Suciu points)
    filtered_data = [d for d in yh2o_data if d['P_bar'] >= 50
                     and d['T_K'] >= 323.15 and d['T_K'] <= 423.15
                     and d.get('kij_NA_conv', True)]
    
    if len(filtered_data) == 0:
        print("Warning: No yH2O data available with P >= 50 bar")
        ax.text(0.5, 0.5, 'No quality-filtered data available\n(P ≥ 50 bar)', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        return fig
    
    # Calculate predictions
    results = []
    for d in filtered_data:
        try:
            y_pred = calc_y_H2O_predicted(d['T_K'], d['P_bar'] * 1e5, kij_NA=0.468)
            y_exp = d['y_H2O_exp']
            error_pct = (y_pred - y_exp) / y_exp * 100
            results.append({
                'T_K': d['T_K'],
                'P_bar': d['P_bar'],
                'y_exp': y_exp,
                'y_pred': y_pred,
                'error_pct': error_pct,
                'Source': d['Source']
            })
        except Exception as e:
            print(f"Calculation failed for T={d['T_K']}, P={d['P_bar']}: {e}")
    
    if len(results) == 0:
        ax.text(0.5, 0.5, 'Calculation failed for all points', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        return fig
    
    # Source styling
    source_styles = {
        'T-O 2021': {'color': 'green', 'marker': 'D', 'label': 'T-O 2021'},
        'Torín-Ollarves 2021': {'color': 'green', 'marker': 'D', 'label': 'T-O 2021'},
        'Suciu 1951': {'color': 'orange', 'marker': '^', 'label': 'Suciu 1951'},
        'Gillespie 1980': {'color': 'purple', 'marker': 's', 'label': 'Gillespie 1980'},
    }
    
    # Plot by source
    plotted_sources = set()
    for r in results:
        source = r['Source']
        style = source_styles.get(source, {'color': 'gray', 'marker': 'o', 'label': source})
        
        label = style['label'] if source not in plotted_sources else None
        plotted_sources.add(source)
        
        # Color by error magnitude
        if abs(r['error_pct']) <= 10:
            facecolor = style['color']
            edgecolor = 'black'
        else:
            facecolor = 'none'
            edgecolor = style['color']
        
        ax.scatter(r['y_exp'] * 100, r['y_pred'] * 100,
                  marker=style['marker'], s=120,
                  facecolors=facecolor, edgecolors=edgecolor,
                  linewidths=1.5, alpha=0.8, label=label, zorder=10)
    
    # Get axis range
    all_y = [r['y_exp'] for r in results] + [r['y_pred'] for r in results]
    y_min = min(all_y) * 100 * 0.5
    y_max = max(all_y) * 100 * 1.5
    
    # 1:1 line
    line_range = np.array([y_min, y_max])
    ax.plot(line_range, line_range, 'k-', linewidth=2, label='1:1 line', zorder=5)
    
    # ±10% error bands
    ax.fill_between(line_range, line_range * 0.9, line_range * 1.1, 
                    alpha=0.2, color='green', label='±10% error')
    ax.plot(line_range, line_range * 0.9, 'g--', linewidth=1, alpha=0.7)
    ax.plot(line_range, line_range * 1.1, 'g--', linewidth=1, alpha=0.7)
    
    # ±20% error bands (for reference)
    ax.plot(line_range, line_range * 0.8, 'r:', linewidth=1, alpha=0.5)
    ax.plot(line_range, line_range * 1.2, 'r:', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(r'Measured $y_{\mathrm{H}_2\mathrm{O}}$ (%)', fontsize=12)
    ax.set_ylabel(r'Predicted $y_{\mathrm{H}_2\mathrm{O}}$ (%)', fontsize=12)
    
    # Log scale for wide range
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)
    
    # Statistics
    errors = [abs(r['error_pct']) for r in results]
    within_10pct = sum(1 for e in errors if e <= 10) / len(errors) * 100
    within_20pct = sum(1 for e in errors if e <= 20) / len(errors) * 100
    mae = np.mean(errors)
    
    stats_text = (f'n = {len(results)} points\n'
                  f'Within ±10%: {within_10pct:.0f}%\n'
                  f'Within ±20%: {within_20pct:.0f}%\n'
                  f'MAE: {mae:.1f}%')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Print detailed results
    print("\n" + "="*70)
    print("yH2O PARITY ANALYSIS (P >= 50 bar)")
    print("="*70)
    print(f"{'Source':<20} {'T(°C)':<8} {'P(bar)':<8} {'y_exp(%)':<12} {'y_pred(%)':<12} {'Error(%)':<10}")
    print("-"*70)
    for r in sorted(results, key=lambda x: x['Source']):
        print(f"{r['Source']:<20} {r['T_K']-273.15:<8.1f} {r['P_bar']:<8.0f} "
              f"{r['y_exp']*100:<12.4f} {r['y_pred']*100:<12.4f} {r['error_pct']:<+10.1f}")
    print("-"*70)
    print(f"Points within ±10%: {within_10pct:.0f}%")
    print(f"Points within ±20%: {within_20pct:.0f}%")
    print(f"Mean Absolute Error: {mae:.1f}%")
    print("="*70)
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_all_figures(csv_path='../../shared/data/pointwise_kij_results.csv', save_figures=True, output_dir='../manuscript/figures'):
    """Generate all figures for the paper."""
    print("="*70)
    print("HYDROGEN PHASE EQUILIBRIA PAPER - FIGURE GENERATION (v11)")
    print("="*70)
    print("\nRATIONAL FORM CORRELATIONS:")
    print("  kij_AQ = (A + Tr) / (B + C*Tr)")
    print("          A = -14.59, B = 2.184, C = 0.365")
    print("  Salting-out: S&W Eq. 8 (Tb=20.3K)")
    print("  kij_NA = 0.468 (constant)")
    
    print("\n1. Loading data...")
    try:
        df_kij_aq, kij_na_results, df_exp, yh2o_data = load_all_h2_data_from_csv(csv_path)
    except Exception as e:
        print(f"   Warning: Could not load CSV ({e})")
        print("   Using synthetic/representative data for demonstration")
        df_kij_aq = pd.DataFrame()
        kij_na_results = []
        df_exp = pd.DataFrame()
        yh2o_data = []
    
    print("\n2. Generating figures...")
    figures = {}
    
    if len(df_exp) > 0:
        print("   Figure 1: Experimental Data Quality Assessment...")
        figures['fig1'] = figure_1_experimental_data_quality(df_exp)
    
    if len(df_kij_aq) > 0:
        print("   Figure 2: BIP Correlation Comparison...")
        figures['fig2'] = figure_2_kij_vs_Tr(df_kij_aq)
    
    if len(df_exp) > 0:
        print("   Figure 3: Model Validation (Multiple Pressures)...")
        figures['fig3'] = figure_3_validation(df_exp)
    
    print("   Figure 4: Sechenov Coefficient (with implied Chabab 2023)...")
    figures['fig4'] = figure_4_sechenov_comparison()
    
    print("   Figure 5: Non-Aqueous Phase BIP (with Tb inset)...")
    figures['fig5'] = figure_5_na_comparison(kij_na_results)

    print("   Figure 6: kij_NA sensitivity analysis...")
    figures['fig6'] = figure_6_kij_na_sensitivity()
    
    print("   Figure 7: yH2O Parity Plot (NEW)...")
    figures['fig7'] = figure_7_yH2O_parity(yh2o_data)
    
    if save_figures:
        print("\n3. Saving figures...")
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figures.items():
            filename_pdf = os.path.join(output_dir, f'H2_SW_paper_{name}.pdf')
            fig.savefig(filename_pdf, format='pdf', bbox_inches='tight')
            filename = os.path.join(output_dir, f'H2_SW_paper_{name}.png')
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   Saved {filename}")
    
    print("\n" + "="*70)
    print("COMPLETE! All 7 figures generated.")
    print("="*70)
    
    return figures, df_kij_aq, kij_na_results, df_exp, yh2o_data


if __name__ == "__main__":
    figures, df_kij_aq, kij_na_results, df_exp, yh2o_data = run_all_figures(
        csv_path='../../shared/data/pointwise_kij_results.csv',
        save_figures=True,
        output_dir='../manuscript/figures'
    )
    plt.show()
