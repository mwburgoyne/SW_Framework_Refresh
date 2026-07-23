#!/usr/bin/env python3
"""
================================================================================
S&W FRAMEWORK REFRESH PAPER (Paper 2) - FIGURE GENERATION
================================================================================

Figures for: "Revisiting the Soreide-Whitson Framework: Updated BIPs and
Salting-Out Correlations for CO2, H2S, CH4, N2, and H2"

Planned figures:
  1. All-gas kij_AQ overview — 5-panel with S&W overlays
  2. All-gas kij_NA overview — same layout for non-aqueous BIPs
  3. S&W kij performance — MAE bar chart per gas
  4. Sechenov comparison — ks vs T per gas: S&W Eq 8 vs modern Pitzer
  5. Sechenov S&W Eq 8 family — Tb-parameterized curves for all gases
  6. Solubility validation — parity plots per gas (predicted vs measured x_gas)
  7. Water content validation — parity plots per gas (predicted vs measured y_H2O)

Dependencies: numpy, pandas, matplotlib, scipy
Shared modules: vle_engine.py, salting_library.py (from parent code/ dir)
================================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from _lib_vle_engine import (
    COMPONENTS, SWBinaryVLE, SWMultiComponentFlash,
    get_kij_aq, get_kij_na,
    kij_aq_h2, kij_aq_co2, kij_aq_n2, kij_aq_h2s, kij_aq_ch4,
    kij_aq_hydrocarbon,
    kij_aq_co2_proposed, kij_aq_h2s_proposed, kij_aq_n2_proposed,
    kij_aq_h2_proposed, kij_aq_c2h6_proposed, kij_aq_c3h8_proposed,
    KIJ_AQ_PROPOSED, KIJ_AQ_SW_ORIGINAL,
    kij_na_h2s, kij_na_h2s_sw_eq17, KIJ_NA,
    kij_aq_chabab_2023, kij_aq_lopez_lazaro_2019, sechenov_TO,
    get_sechenov_ks, sw_equation_8_ks,
    solve_rachford_rice, _sw_kvalue_init,
    _SW_KVALUE_PARAMS, _SW_KVALUE_HEAVY, _SW_KVALUE_WATER,
    EMBEDDED_SALINITY_PARAMS, EMBEDDED_SALINITY_PARAMS_DROPIN,
    calc_embedded_delta_kij,
    kij_aq_co2_dropin, kij_aq_h2s_dropin, kij_aq_n2_dropin,
    kij_aq_h2_dropin, kij_aq_ch4_dropin, kij_aq_c2h6_dropin,
    kij_aq_c3h8_dropin, KIJ_AQ_DROPIN,
)
from _lib_salting_library import (
    ks_sw_eq8, ks_duan2003_co2, ks_akinfiev_h2s,
    ks_dubessy_co2, ks_dubessy_h2s,
    ks_li2015, ks_mao2006_n2, TB_K,
)

# Plot defaults — publication quality
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
    'savefig.bbox': 'tight',
})

# =============================================================================
# SHARED PLOT STYLE — one consistent visual language across every figure.
#
# Colour encodes ROLE, not model identity. The curve that represents this
# work's recommendation is ALWAYS the same green and weight, so a reader
# learns the legend once. Where S&W Eq 8 is itself the recommendation
# (CH4, H2) it is drawn in the recommended style; where S&W is only a
# reference (CO2, N2) it is drawn in the baseline style. Palette is the
# colourblind-safe Okabe–Ito set.
# =============================================================================
C_REC     = '#009E73'   # recommended / this work (bluish green)
C_BASE    = 'black'     # S&W original, shown as baseline / reference
C_EXP     = '0.35'      # experimental data points (neutral grey)
C_ALT     = ['#E69F00', '#D55E00', '#56B4E9']  # alternative published models
C_MC3     = '#56B4E9'   # MC-3 alpha (parity only, de-emphasised)
C_IMPLIED = '0.55'      # S&W embedded-implied k_s (grey, secondary)

STYLE_REC  = dict(color=C_REC,  linestyle='-',  linewidth=2.5)   # recommendation
STYLE_BASE = dict(color=C_BASE, linestyle='--', linewidth=2.0)   # S&W baseline
ALT_LINESTYLES = [':', '-.', (0, (3, 1, 1, 1))]


def alt_style(i, **kw):
    """Line style for the i-th alternative published model (colourblind-safe)."""
    d = dict(color=C_ALT[i % len(C_ALT)],
             linestyle=ALT_LINESTYLES[i % len(ALT_LINESTYLES)],
             linewidth=1.6)
    d.update(kw)
    return d


GASES = ['CO2', 'H2S', 'N2', 'H2', 'CH4', 'C2H6', 'C3H8', 'nC4H10']

# Temperature bounds for all regressions, plots, and validation metrics
T_MIN_K = 273.15   # 0 deg C — exclude sub-zero data
T_MAX_K = 473.15   # 200 deg C
T_MAX_C = 200.0

# Output directory for Paper 2 figures
OUTPUT_DIR = '../manuscript/figures'

# S&W kij_AQ functions (freshwater)
def _sw_kij_aq(gas, T_K):
    """S&W kij_AQ for any supported gas at T_K."""
    if gas == 'CO2':
        return kij_aq_co2(T_K, 0.0)
    elif gas == 'H2S':
        return kij_aq_h2s(T_K, 0.0)
    elif gas == 'N2':
        return kij_aq_n2(T_K, 0.0)
    elif gas == 'H2':
        return kij_aq_h2(T_K, 0.0)
    else:
        comp = COMPONENTS[gas]
        return kij_aq_hydrocarbon(T_K, comp.omega, comp.Tc, 0.0)

KIJ_AQ_SW = {gas: (lambda T, g=gas: _sw_kij_aq(g, T)) for gas in GASES}


def get_sw_kij_na(gas, T_K):
    """Get S&W original kij_NA value for a gas (for overlay comparison)."""
    if gas == 'H2S':
        return kij_na_h2s_sw_eq17(T_K)
    elif gas == 'H2':
        return 0.468  # This work constant
    return get_kij_na(gas, T_K)


# Sources excluded per gas (consistent with generate_appendix_figures.py)
EXCLUDE_SOURCES = {
    'CH4': {'Blount 1982', 'McGee 1981'},
    'CO2': {'Prutton & Savage 1945'},
    'H2S': {'Barrett 1988'},
    'H2': {'Gillespie 1980'},
}


def load_data(csv_path='../../shared/data/pointwise_kij_results_sw_alpha.csv'):
    """Load pointwise kij results, excluding known-problematic sources."""
    df = pd.read_csv(csv_path)
    if 'Source' in df.columns:
        df['Source'] = df['Source'].str.replace('Chahab', 'Chabab', regex=False)
        for gas, sources in EXCLUDE_SOURCES.items():
            mask = (df['Gas'] == gas) & (df['Source'].isin(sources))
            df = df[~mask]
    return df


# Gas display names for figures
GAS_DISPLAY = {
    'CO2': 'CO$_2$', 'H2S': 'H$_2$S', 'CH4': 'CH$_4$', 'N2': 'N$_2$',
    'H2': 'H$_2$', 'C2H6': 'C$_2$H$_6$', 'C3H8': 'C$_3$H$_8$',
    'nC4H10': 'n-C$_4$H$_{10}$',
}

# ── Proposed kij_AQ correlation functions (inlined from plot_kij_per_gas.py) ──

# Yan et al. 2011 CO2 correlation
def kij_aq_co2_yan2011(T_K, sal=0.0):
    """Yan et al. 2011 improved kij_AQ for CO2-water/brine."""
    cs = sal
    return (0.30823655 + 0.11820367 * cs - 0.00095381166 * cs**2
            - 126.42095 / T_K - 0.00062924435 * cs * T_K
            + 0.00000092946667 * cs * T_K**2)

# Proposed kij_AQ functions imported from VLE engine.
# _proposed_kij_aq dispatches to the engine's KIJ_AQ_PROPOSED functions.
def _proposed_kij_aq(gas, T_K):
    """This work's proposed kij_AQ for a gas (from VLE engine, MC-3 alpha)."""
    if gas in KIJ_AQ_PROPOSED:
        return KIJ_AQ_PROPOSED[gas](T_K, 0.0)
    return None  # nC4H10+ uses S&W generalized at cs=0

def _dropin_kij_aq(gas, T_K):
    """This work's dropin kij_AQ for a gas (from VLE engine, S&W alpha)."""
    if gas in KIJ_AQ_DROPIN:
        return KIJ_AQ_DROPIN[gas](T_K, 0.0)
    return None  # nC4H10+ uses S&W generalized at cs=0


# Proposed kij_NA constants (this work)
PROPOSED_KIJ_NA = {
    'CO2': 0.1896,   # S&W retained
    'H2S': 0.1610,   # This work (constant, replaces S&W Eq 17)
    'CH4': 0.4850,   # S&W retained
    'N2': 0.4778,    # S&W retained
    'H2': 0.4680,    # This work
    'C2H6': 0.4920,  # S&W retained
    'C3H8': 0.5525,  # S&W retained (S&W 1992 Table 5; 0.5070 pre-2026-07-23 was a transcription error)
    'nC4H10': 0.5091, # S&W retained (S&W 1992 Table 5; 0.5080 pre-2026-07-23 was a transcription error)
}


# =============================================================================
# FIGURE 1: All-Gas kij_AQ Overview (4x2 single page)
# =============================================================================
def figure_1_kij_aq_overview(df):
    """kij_AQ vs T(C) with S&W and proposed correlation overlays.

    One panel per gas that has aqueous data; pointwise-regressed points are
    pooled into a single neutral marker style so the correlation overlays
    read clearly (source-level identity belongs in the data tables, not an
    eight-panel overview). Panels without data are omitted, not left blank.
    """
    def _panel_data(gas):
        mask = ((df['Gas'] == gas) & (df['kij_AQ'].notna())
                & (df['kij_AQ_conv'] == True)
                & (df['T_K'] >= T_MIN_K) & (df['T_K'] <= T_MAX_K))
        return df[mask]

    gas_panels = [(g, _panel_data(g)) for g in GASES]
    gas_panels = [(g, d) for g, d in gas_panels if len(d) > 0]

    n = len(gas_panels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, (gas, data) in enumerate(gas_panels):
        ax = axes[idx]
        display = GAS_DISPLAY.get(gas, gas)

        # Pooled pointwise-regressed points (single neutral style)
        ax.scatter(data['T_K'] - 273.15, data['kij_AQ'],
                   c=C_EXP, marker='o', s=26, edgecolors='k',
                   linewidths=0.2, alpha=0.55,
                   label='Pointwise-regressed', zorder=5)

        # Temperature array for overlays (capped at T_MAX)
        T_lo = data['T_K'].min() - 10
        T_hi = min(data['T_K'].max() + 10, T_MAX_K)
        T_arr = np.linspace(T_lo, T_hi, 200)

        # S&W correlation overlay (baseline)
        kij_sw = [KIJ_AQ_SW[gas](T) for T in T_arr]
        ax.plot(T_arr - 273.15, kij_sw, label='S&W', zorder=10, **STYLE_BASE)

        # Dropin correlation overlay (this work — only if different from S&W)
        dropin_vals = [_dropin_kij_aq(gas, T) for T in T_arr]
        if dropin_vals[0] is not None:
            # Check if dropin differs from S&W
            diff = max(abs(p - s) for p, s in zip(dropin_vals, kij_sw) if p is not None)
            if diff > 0.001:
                ax.plot(T_arr - 273.15, dropin_vals, label='This work',
                        zorder=11, **STYLE_REC)

        # CO2: also show Yan et al. 2011 (alternative model)
        if gas == 'CO2':
            kij_yan = [kij_aq_co2_yan2011(T) for T in T_arr]
            ax.plot(T_arr - 273.15, kij_yan, label='Yan 2011', zorder=9,
                    **alt_style(0))

        # Per-gas Y-axis limits
        ylims = {
            'CO2': (-0.2, 0.4), 'H2S': (-0.2, 0.4),
            'N2': (-0.7, 0.2), 'H2': (-1.3, 0.2),
            'CH4': (-0.4, 0.3), 'C2H6': (-0.4, 0.3),
            'C3H8': (-0.4, 0.3), 'nC4H10': (-0.4, 0.3),
        }
        if gas in ylims:
            ax.set_ylim(ylims[gas])

        ax.set_xlabel('Temperature (\u00b0C)')
        ax.set_ylabel('$k_{ij}^{\\mathrm{AQ}}$')
        ax.set_title(f'({chr(97+idx)}) {display} (n={len(data)})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    # Drop any unused trailing axes rather than leaving blank panels.
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2: All-Gas kij_NA Overview (4x2 single page)
# =============================================================================
def figure_2_kij_na_overview(df):
    """kij_NA vs T(C) with S&W values and this-work overlays.

    One panel per gas that has non-aqueous data; pointwise-regressed points
    are pooled into a single neutral marker style. Panels without data are
    omitted, not left blank.
    """
    def _panel_data(gas):
        mask = ((df['Gas'] == gas) & (df['kij_NA'].notna())
                & (df['kij_NA_conv'] == True)
                & (df['T_K'] >= T_MIN_K) & (df['T_K'] <= T_MAX_K))
        return df[mask]

    gas_panels = [(g, _panel_data(g)) for g in GASES]
    gas_panels = [(g, d) for g, d in gas_panels if len(d) > 0]

    n = len(gas_panels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, (gas, data) in enumerate(gas_panels):
        ax = axes[idx]
        display = GAS_DISPLAY.get(gas, gas)

        # Pooled pointwise-regressed points (single neutral style)
        ax.scatter(data['T_K'] - 273.15, data['kij_NA'],
                   c=C_EXP, marker='o', s=26, edgecolors='k',
                   linewidths=0.2, alpha=0.55,
                   label='Pointwise-regressed', zorder=5)

        T_lo = data['T_K'].min() - 10
        T_hi = min(data['T_K'].max() + 10, T_MAX_K)
        T_arr = np.linspace(T_lo, T_hi, 200)

        # S&W overlay (original values — baseline)
        kij_na_sw = [get_sw_kij_na(gas, T) for T in T_arr]
        sw_label = 'S&W Eq 17' if gas == 'H2S' else 'S&W'
        ax.plot(T_arr - 273.15, kij_na_sw, label=sw_label, zorder=10,
                **STYLE_BASE)

        # This work overlay (recommended constant)
        this_work_val = PROPOSED_KIJ_NA.get(gas)
        if this_work_val is not None:
            ax.axhline(this_work_val, label=f'This work ({this_work_val:.3f})',
                       zorder=11, **STYLE_REC)

        # Cap Y-axis upper limit at 0.9
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, min(y_hi, 0.9))

        ax.set_xlabel('Temperature (\u00b0C)')
        ax.set_ylabel('$k_{ij}^{\\mathrm{NA}}$')
        ax.set_title(f'({chr(97+idx)}) {display} (n={len(data)})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    # Drop any unused trailing axes rather than leaving blank panels.
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2 ALT: kij_NA with pressure-intensity coloring
# =============================================================================
def figure_2_kij_na_pressure(df):
    """4x2 kij_NA vs T(C) colored by pressure (dark=high P, light=low P)."""
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    axes = axes.flatten()

    for idx, gas in enumerate(GASES):
        ax = axes[idx]
        mask = ((df['Gas'] == gas) & (df['kij_NA'].notna())
                & (df['kij_NA_conv'] == True)
                & (df['T_K'] >= T_MIN_K) & (df['T_K'] <= T_MAX_K))
        data = df[mask]
        display = GAS_DISPLAY.get(gas, gas)

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'({chr(97+idx)}) {display}')
            continue

        T_C = data['T_K'].values - 273.15
        P_bar = data['P_bar'].values
        kij_na = data['kij_NA'].values

        # Log-scale pressure for better visual spread
        P_log = np.log10(np.clip(P_bar, 1.0, None))
        P_log_min, P_log_max = P_log.min(), P_log.max()
        if P_log_max - P_log_min < 0.1:
            P_log_min, P_log_max = P_log_min - 0.5, P_log_max + 0.5
        norm = mcolors.Normalize(vmin=P_log_min, vmax=P_log_max)

        sc = ax.scatter(T_C, kij_na, c=P_log, cmap='inferno',
                       norm=norm, s=35, edgecolors='k', linewidths=0.3,
                       alpha=0.85, zorder=5)

        # Colorbar with real pressure labels
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30)
        # Tick at nice round pressures
        tick_vals = [1, 3, 10, 30, 100, 300, 1000, 3000]
        tick_log = [np.log10(v) for v in tick_vals
                    if P_log_min - 0.1 <= np.log10(v) <= P_log_max + 0.1]
        tick_labels = [str(int(10**v)) if 10**v >= 1 else f'{10**v:.1f}'
                       for v in tick_log]
        cbar.set_ticks(tick_log)
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('P (bar)', fontsize=9)

        # Temperature array for overlays
        T_lo = data['T_K'].min() - 10
        T_hi = min(data['T_K'].max() + 10, T_MAX_K)
        T_arr = np.linspace(T_lo, T_hi, 200)

        # S&W overlay
        kij_na_sw = [get_sw_kij_na(gas, T) for T in T_arr]
        sw_label = 'S&W Eq 17' if gas == 'H2S' else 'S&W'
        ax.plot(T_arr - 273.15, kij_na_sw, 'k--', linewidth=2,
                label=sw_label, zorder=10)

        # This work overlay
        this_work_val = PROPOSED_KIJ_NA.get(gas)
        if this_work_val is not None:
            ax.axhline(this_work_val, color='tab:purple', linestyle='-',
                       linewidth=2, label=f'This work ({this_work_val:.3f})',
                       zorder=11)

        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, min(y_hi, 0.9))

        ax.set_xlabel('Temperature (\u00b0C)')
        ax.set_ylabel('$k_{ij}^{\\mathrm{NA}}$')
        ax.set_title(f'({chr(97+idx)}) {display} (n={len(data)})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1, loc='best')

    fig.suptitle('Non-Aqueous Phase BIP: Colored by Pressure',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# =============================================================================
# FIGURE 3: S&W kij Performance Summary (bar chart)
# =============================================================================
def figure_3_sw_performance(df):
    """MAE bar chart of S&W kij_AQ and kij_NA by gas."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    gas_labels = [GAS_DISPLAY.get(g, g) for g in GASES]

    # kij_AQ
    mae_aq, n_aq = [], []
    for gas in GASES:
        mask = (df['Gas'] == gas) & (df['kij_AQ'].notna()) & (df['kij_AQ_conv'] == True)
        data = df[mask]
        if len(data) > 0:
            kij_sw = data['T_K'].apply(lambda T: KIJ_AQ_SW[gas](T))
            mae = (data['kij_AQ'] - kij_sw).abs().mean()
        else:
            mae = 0
        mae_aq.append(mae)
        n_aq.append(len(data))

    x = np.arange(len(GASES))
    bars1 = ax1.bar(x, mae_aq, color='steelblue', edgecolor='k', alpha=0.85)
    for i, (b, n) in enumerate(zip(bars1, n_aq)):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f'n={n}', ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gas_labels)
    ax1.set_ylabel('MAE($k_{ij}^{AQ}$)')
    ax1.set_title('(a) Aqueous Phase BIP')
    ax1.grid(True, alpha=0.3, axis='y')

    # kij_NA
    mae_na, n_na = [], []
    for gas in GASES:
        mask = (df['Gas'] == gas) & (df['kij_NA'].notna()) & (df['kij_NA_conv'] == True)
        data = df[mask]
        if len(data) > 0:
            kij_na_ref = data['T_K'].apply(lambda T: get_sw_kij_na(gas, T))
            mae = (data['kij_NA'] - kij_na_ref).abs().mean()
        else:
            mae = 0
        mae_na.append(mae)
        n_na.append(len(data))

    bars2 = ax2.bar(x, mae_na, color='darkorange', edgecolor='k', alpha=0.85)
    for i, (b, n) in enumerate(zip(bars2, n_na)):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f'n={n}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gas_labels)
    ax2.set_ylabel('MAE($k_{ij}^{NA}$)')
    ax2.set_title('(b) Non-Aqueous Phase BIP')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('S&W 1992 BIP Correlation Performance',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 4: Sechenov Comparison (S&W Eq 8 vs Modern Pitzer)
# =============================================================================
def figure_4_sechenov_comparison():
    """Per-gas ks vs T: S&W Eq 8 (dashed) vs best Pitzer model (solid)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    T_range = np.linspace(273.15, 473.15, 100)
    T_C = T_range - 273.15

    # CO2
    ax = axes[0]
    ax.plot(T_C, ks_sw_eq8(T_range, 'CO2'), 'k--', lw=2.5, label='S&W Eq 8')
    T_d = T_range[T_range <= 530]
    ax.plot(T_d - 273.15, [ks_duan2003_co2(T, 0, 100) for T in T_d],
            'r-', lw=2.5, label='Duan 2003')
    ax.set_title('(a) CO2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.22)

    # H2S
    ax = axes[1]
    ax.plot(T_C, ks_sw_eq8(T_range, 'H2S'), 'k--', lw=2.5, label='S&W Eq 8')
    T_a = T_range[(T_range >= 283) & (T_range <= 570)]
    ax.plot(T_a - 273.15, [ks_akinfiev_h2s(T, 1.0) for T in T_a],
            'g-', lw=2.5, label='Akinfiev 2016')
    ax.set_title('(b) H2S')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.22)

    # CH4
    ax = axes[2]
    ax.plot(T_C, ks_sw_eq8(T_range, 'CH4'), 'k--', lw=2.5, label='S&W Eq 8')
    ax.plot(T_C, [ks_li2015(T, 'CH4', 0, 100) for T in T_range],
            'b-', lw=2.5, label='Li 2015')
    ax.set_title('(c) CH4')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.18)

    # N2
    ax = axes[3]
    ax.plot(T_C, ks_sw_eq8(T_range, 'N2'), 'k--', lw=2.5, label='S&W Eq 8')
    T_m = T_range[T_range <= 398]
    ax.plot(T_m - 273.15, [ks_mao2006_n2(T, 0, 100) for T in T_m],
            color='darkorange', ls='-', lw=2.5, label='Mao 2006')
    ax.set_title('(d) N2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.16)

    # H2
    ax = axes[4]
    ax.plot(T_C, ks_sw_eq8(T_range, 'H2'), 'k-', lw=2.5, label='S&W Eq 8 (only model)')
    ax.set_title('(e) H2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.14)

    axes[-1].set_visible(False)

    for ax in axes[:5]:
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
        ax.set_xlim(0, 200)

    fig.suptitle('Sechenov Coefficients: S&W Eq 8 vs Modern Pitzer Models',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 5: S&W Eq 8 Family (Tb-parameterized)
# =============================================================================
def figure_5_sw_eq8_family():
    """S&W Eq 8 curves for all gases, colored by Tb."""
    fig, ax = plt.subplots(figsize=(10, 7))

    T_range = np.linspace(273.15, 523.15, 100)
    T_C = T_range - 273.15

    gases_by_tb = sorted(TB_K.items(), key=lambda x: x[1], reverse=True)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(gases_by_tb)))

    for i, (gas, Tb) in enumerate(gases_by_tb):
        ks = ks_sw_eq8(T_range, gas)
        ax.plot(T_C, ks, color=cmap[i], linewidth=2,
                label=f'{gas} ($T_b$={Tb:.0f} K)')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title('S&W Equation 8: Sechenov Coefficient Family')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 6: Solubility Validation (parity per gas)
# =============================================================================
def figure_6_solubility_parity(df):
    """Parity plots of predicted vs measured x_gas for each gas."""
    gases_5 = ['CO2', 'H2S', 'CH4', 'N2', 'H2']
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    for idx, gas in enumerate(gases_5):
        ax = axes[idx]
        mask = (df['Gas'] == gas) & (df['kij_AQ'].notna()) & (df['kij_AQ_conv'] == True) & (df['x_gas_exp'].notna())
        data = df[mask]

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'({chr(97+idx)}) {gas}')
            continue

        # Calculate predicted x_gas using S&W kij
        x_pred = []
        x_exp = []
        for _, row in data.iterrows():
            try:
                vle = SWBinaryVLE(gas, 0.0)
                kij = KIJ_AQ_SW[gas](row['T_K'])
                x_calc = vle._calc_x_with_kij(row['T_K'], row['P_bar'] * 1e5, kij)
                if x_calc is not None and x_calc > 0:
                    x_pred.append(x_calc)
                    x_exp.append(row['x_gas_exp'])
            except Exception:
                pass

        if len(x_pred) == 0:
            ax.text(0.5, 0.5, 'VLE failed', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'({chr(97+idx)}) {gas}')
            continue

        x_pred = np.array(x_pred)
        x_exp = np.array(x_exp)

        ax.scatter(x_exp * 1e3, x_pred * 1e3, s=30, alpha=0.7,
                  edgecolors='k', linewidths=0.3, zorder=5)

        # 1:1 line and ±10% bands
        lo = min(x_exp.min(), x_pred.min()) * 1e3 * 0.8
        hi = max(x_exp.max(), x_pred.max()) * 1e3 * 1.2
        line = np.array([lo, hi])
        ax.plot(line, line, 'k-', lw=1.5, zorder=10)
        ax.fill_between(line, line * 0.9, line * 1.1, alpha=0.15, color='green')

        # MARE
        rel_err = np.abs(x_pred - x_exp) / x_exp * 100
        mare = np.mean(rel_err)
        ax.text(0.05, 0.92, f'MARE={mare:.1f}%\nn={len(x_pred)}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(f'Measured $x_{{{gas}}}$ (×10$^{{-3}}$)')
        ax.set_ylabel(f'Predicted $x_{{{gas}}}$ (×10$^{{-3}}$)')
        ax.set_title(f'({chr(97+idx)}) {gas}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')

    axes[-1].set_visible(False)
    fig.suptitle('Gas Solubility: S&W Predicted vs Measured',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 7: Water Content Validation (parity per gas)
# =============================================================================
def figure_7_water_content_parity(df):
    """Parity plots of predicted vs measured y_H2O for each gas."""
    gases_5 = ['CO2', 'H2S', 'CH4', 'N2', 'H2']
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    for idx, gas in enumerate(gases_5):
        ax = axes[idx]
        mask = (df['Gas'] == gas) & (df['kij_NA'].notna()) & (df['y_H2O_exp'].notna())
        data = df[mask]

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'({chr(97+idx)}) {gas}')
            continue

        y_pred = []
        y_exp = []
        for _, row in data.iterrows():
            try:
                vle = SWBinaryVLE(gas, 0.0)
                kij_na = get_sw_kij_na(gas, row['T_K'])
                y_calc = vle.calc_water_content_with_kij(row['T_K'], row['P_bar'] * 1e5, kij_na)
                if y_calc is not None and y_calc > 0:
                    y_pred.append(y_calc)
                    y_exp.append(row['y_H2O_exp'])
            except Exception:
                pass

        if len(y_pred) == 0:
            ax.text(0.5, 0.5, 'VLE failed', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'({chr(97+idx)}) {gas}')
            continue

        y_pred = np.array(y_pred)
        y_exp = np.array(y_exp)

        ax.scatter(y_exp * 100, y_pred * 100, s=30, alpha=0.7,
                  edgecolors='k', linewidths=0.3, zorder=5)

        lo = min(y_exp.min(), y_pred.min()) * 100 * 0.5
        hi = max(y_exp.max(), y_pred.max()) * 100 * 1.5
        lo = max(lo, 1e-4)
        line = np.array([lo, hi])
        ax.plot(line, line, 'k-', lw=1.5, zorder=10)
        ax.fill_between(line, line * 0.9, line * 1.1, alpha=0.15, color='green')

        rel_err = np.abs(y_pred - y_exp) / y_exp * 100
        mare = np.mean(rel_err)
        within_10 = np.sum(rel_err <= 10) / len(rel_err) * 100
        ax.text(0.05, 0.88, f'MARE={mare:.1f}%\n±10%: {within_10:.0f}%\nn={len(y_pred)}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(r'Measured $y_{\mathrm{H_2O}}$ (%)')
        ax.set_ylabel(r'Predicted $y_{\mathrm{H_2O}}$ (%)')
        ax.set_title(f'({chr(97+idx)}) {gas}')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')

    axes[-1].set_visible(False)
    fig.suptitle('Water Content: S&W Predicted vs Measured',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# INDIVIDUAL PER-GAS KS FIGURES (with experimental data overlays)
# =============================================================================

def _load_ks_data(csv_path='../../shared/data/extracted_ks_values.csv'):
    """Load experimental ks data."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _scatter_ks_data(ax, df_ks, gas, min_sal=0.5, max_sal=4.0,
                     exclude_sources=(), pooled=False):
    """Add experimental ks scatter to axis, grouped by source. Return count.

    Args:
        min_sal: Minimum salinity (molal) to include. At low m, the ks
                 calculation amplifies measurement noise (e.g. at m=0.06,
                 a 2% x error gives ks error of 0.14). Default 0.5 m.
        max_sal: Maximum salinity (molal) to include. Limits to typical
                 oilfield reservoir range. Default 4.0 m (~200,000 ppm).
        exclude_sources: Sources to exclude from ks plots.
    """
    if df_ks.empty:
        return 0
    T_MIN_C = T_MIN_K - 273.15
    gas_data = df_ks[(df_ks['Gas'] == gas)
                     & (df_ks['Sal_m'] >= min_sal)
                     & (df_ks['Sal_m'] <= max_sal)
                     & (df_ks['T_C'] >= T_MIN_C) & (df_ks['T_C'] <= T_MAX_C)]
    # Merge explicit exclusions with global EXCLUDE_SOURCES
    all_excl = set(exclude_sources) | EXCLUDE_SOURCES.get(gas, set())
    for src in all_excl:
        gas_data = gas_data[gas_data['Source'] != src]
    if len(gas_data) == 0:
        return 0

    # Pooled mode: a single neutral marker style + one legend entry, so
    # multi-source overview panels keep readable legends.
    if pooled:
        ax.scatter(gas_data['T_C'], gas_data['ks_data'],
                   marker='o', c='0.30', s=34, alpha=0.6,
                   edgecolors='k', linewidths=0.4, zorder=10,
                   label=f'Experimental (n={len(gas_data)})')
        return len(gas_data)

    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h',
               'o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h']
    source_colors = plt.cm.tab10.colors

    sources = sorted(gas_data['Source'].unique())
    for si, src in enumerate(sources):
        sd = gas_data[gas_data['Source'] == src]
        ax.scatter(sd['T_C'], sd['ks_data'],
                   marker=markers[si % len(markers)],
                   c=[source_colors[si % len(source_colors)]],
                   s=50, alpha=0.7, edgecolors='k', linewidths=0.5,
                   zorder=10,
                   label=f'{src} (n={len(sd)})')
    return len(gas_data)


def figure_ks_co2(df_ks):
    """CO2 ks: Duan 2003 + Dubessy 2005 + experimental data (no S&W Eq 8)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    T_range = np.linspace(273.15, 573.15, 150)
    T_C = T_range - 273.15

    # Duan & Sun 2003 (valid to 533 K)
    T_d = T_range[T_range <= 530]
    T_C_d = T_d - 273.15
    ks_d1 = [ks_duan2003_co2(T, 1.0, 100) for T in T_d]
    ks_d3 = [ks_duan2003_co2(T, 3.0, 100) for T in T_d]
    ax.plot(T_C_d, ks_d1, 'b--', lw=2.5, label='Duan & Sun 2003 (m=1)')
    ax.plot(T_C_d, ks_d3, 'b:', lw=2, label='Duan & Sun 2003 (m=3)')

    # Dubessy 2005
    ks_db1 = [ks_dubessy_co2(T, 1.0) for T in T_range]
    ks_db3 = [ks_dubessy_co2(T, 3.0) for T in T_range]
    ax.plot(T_C, ks_db1, color='darkgreen', ls='-.', lw=2, label='Dubessy 2005 (m=1)')
    ax.plot(T_C, ks_db3, color='darkgreen', ls='-.', lw=1.5, alpha=0.6,
            label='Dubessy 2005 (m=3)')

    # S&W Eq 14 implied ks (embedded salinity in BIP)
    try:
        vle = SWBinaryVLE('CO2', salinity_molal=0.0)
        P_Pa_100 = 100e5
        M_BRINE = 1.0
        ks_sw_emb = []
        for T_K in T_range:
            kij_fw = kij_aq_co2(T_K, 0.0)
            kij_br = kij_aq_co2(T_K, M_BRINE)
            try:
                x_fw = vle._calc_x_with_kij(T_K, P_Pa_100, kij_fw)
                x_br = vle._calc_x_with_kij(T_K, P_Pa_100, kij_br)
                if x_fw > 0 and x_br > 0 and np.isfinite(x_fw) and np.isfinite(x_br):
                    ks_sw_emb.append(np.log10(x_fw / x_br) / M_BRINE)
                else:
                    ks_sw_emb.append(np.nan)
            except Exception:
                ks_sw_emb.append(np.nan)
        ax.plot(T_C, ks_sw_emb, color='gray', ls=':', lw=2,
                label='S&W Eq 14 implied $k_s$', zorder=14)
    except Exception:
        pass

    # Experimental data
    n = _scatter_ks_data(ax, df_ks, 'CO2')

    ax.set_xlabel('Temperature (\u00b0C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title(f'CO$_2$ Sechenov Coefficient  (n = {n})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.01, 0.22)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    return fig


def figure_ks_h2s(df_ks):
    """H2S ks: Akinfiev 2016 + Dubessy 2005 + experimental data (no S&W Eq 8)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    T_range = np.linspace(273.15, 573.15, 150)
    T_C = T_range - 273.15

    # Akinfiev 2016 (valid 283-570 K)
    T_a = T_range[(T_range >= 283) & (T_range <= 570)]
    T_C_a = T_a - 273.15
    ks_a1 = [ks_akinfiev_h2s(T, 1.0) for T in T_a]
    ks_a3 = [ks_akinfiev_h2s(T, 3.0) for T in T_a]
    ax.plot(T_C_a, ks_a1, 'b--', lw=2.5, label='Akinfiev 2016 (m=1)')
    ax.plot(T_C_a, ks_a3, 'b:', lw=2, label='Akinfiev 2016 (m=3)')

    # Dubessy 2005 (valid to ~520 K)
    T_db = T_range[T_range <= 520]
    T_C_db = T_db - 273.15
    ks_db1 = [ks_dubessy_h2s(T, 1.0) for T in T_db]
    ks_db3 = [ks_dubessy_h2s(T, 3.0) for T in T_db]
    ax.plot(T_C_db, ks_db1, color='darkgreen', ls='-.', lw=2, label='Dubessy 2005 (m=1)')
    ax.plot(T_C_db, ks_db3, color='darkgreen', ls='-.', lw=1.5, alpha=0.6,
            label='Dubessy 2005 (m=3)')

    # Experimental data
    n = _scatter_ks_data(ax, df_ks, 'H2S')

    ax.set_xlabel('Temperature (\u00b0C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title(f'H$_2$S Sechenov Coefficient  (n = {n})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 340)
    ax.set_ylim(-0.02, 0.20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    return fig


def figure_ks_n2(df_ks):
    """N2 ks: S&W Eq 8 + Mao 2006 + S&W Eq 13 implied ks + experimental data."""
    fig, ax = plt.subplots(figsize=(10, 7))

    T_range = np.linspace(273.15, 473.15, 100)
    T_C = T_range - 273.15

    # S&W Eq 8 (original, for reference)
    ks_sw = ks_sw_eq8(T_range, 'N2')
    ax.plot(T_C, ks_sw, 'k:', lw=1.5, alpha=0.5, label='S&W Eq 8 (original)', zorder=19)
    # Modified Eq 8 (+0.02 offset, recommended)
    ks_sw_mod = ks_sw + 0.02
    ax.plot(T_C, ks_sw_mod, 'k-', lw=2.5, label='Eq 8 + 0.02 (rec.)', zorder=20)

    # Mao 2006 (valid to 400 K = 127 C)
    T_m = T_range[T_range <= 398]
    T_C_m = T_m - 273.15
    ks_m0 = [ks_mao2006_n2(T, 0, 100) for T in T_m]
    ax.plot(T_C_m, ks_m0, color='darkorange', ls='--', lw=2,
            label='Mao 2006', zorder=15)

    # S&W Eq 13 implied ks (embedded salinity in BIP)
    try:
        vle = SWBinaryVLE('N2', salinity_molal=0.0)
        P_Pa_100 = 100e5
        M_BRINE = 1.0
        ks_sw_emb = []
        for T_K in T_range:
            kij_fw = kij_aq_n2(T_K, 0.0)
            kij_br = kij_aq_n2(T_K, M_BRINE)
            try:
                x_fw = vle._calc_x_with_kij(T_K, P_Pa_100, kij_fw)
                x_br = vle._calc_x_with_kij(T_K, P_Pa_100, kij_br)
                if x_fw > 0 and x_br > 0:
                    ks_sw_emb.append(np.log10(x_fw / x_br) / M_BRINE)
                else:
                    ks_sw_emb.append(np.nan)
            except Exception:
                ks_sw_emb.append(np.nan)
        ax.plot(T_C, ks_sw_emb, color='gray', ls=':', lw=2,
                label='S&W Eq 13 implied $k_s$', zorder=14)
    except Exception:
        pass

    # Experimental data
    n = _scatter_ks_data(ax, df_ks, 'N2')

    ax.set_xlabel('Temperature (\u00b0C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title(f'N$_2$ Sechenov Coefficient  (n = {n})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    return fig


def figure_ks_h2(df_ks):
    """H2 ks: S&W Eq 8 + implied ks from published BIPs + historical data.

    Mirrors the richness of Paper 1 Figure 4, showing:
    - S&W Eq 8 (recommended model)
    - T-O 2021 polynomial correlation
    - Implied ks from Chabab 2023 BIP
    - Implied ks from Lopez-Lazaro 2019 BIP
    - Implied ks from this work's embedded salinity BIP
    - Six historical experimental sources + Chabab 2023 from CSV
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    T_range = np.linspace(273.15, 423.15, 100)
    T_C = T_range - 273.15

    # --- Model lines ---

    # S&W Eq 8 (recommended)
    ks_sw = ks_sw_eq8(T_range, 'H2')
    ax.plot(T_C, ks_sw, 'k-', lw=3, label='S&W Eq 8 ($T_b$=20.3 K)',
            zorder=20)

    # T-O 2021 polynomial correlation
    ks_to = [sechenov_TO(T, log10_basis=True) for T in T_range]
    ax.plot(T_C, ks_to, 'b--', lw=2.5, label='T-O 2021 correlation',
            zorder=15)

    # Implied ks from Chabab 2023 BIP
    vle_h2 = SWBinaryVLE('H2', salinity_molal=0.0)
    P_Pa_100 = 100e5
    M_IMP = 1.5  # molality for implied ks calculation

    ks_chabab_impl = []
    for T_K in T_range:
        try:
            kij_fw = kij_aq_chabab_2023(T_K, m=0.0)
            kij_br = kij_aq_chabab_2023(T_K, m=M_IMP)
            x_fw = vle_h2._calc_x_with_kij(T_K, P_Pa_100, kij_fw)
            x_br = vle_h2._calc_x_with_kij(T_K, P_Pa_100, kij_br)
            if x_fw > 0 and x_br > 0:
                ks_chabab_impl.append(np.log10(x_fw / x_br) / M_IMP)
            else:
                ks_chabab_impl.append(np.nan)
        except Exception:
            ks_chabab_impl.append(np.nan)
    ax.plot(T_C, ks_chabab_impl, 'r-.', lw=2.5,
            label='Chabab 2023 BIP (implied)', zorder=15)

    # Implied ks from Lopez-Lazaro 2019 BIP
    ks_ll_impl = []
    for T_K in T_range:
        try:
            kij_fw = kij_aq_lopez_lazaro_2019(T_K, csw=0.0)
            kij_br = kij_aq_lopez_lazaro_2019(T_K, csw=M_IMP)
            x_fw = vle_h2._calc_x_with_kij(T_K, P_Pa_100, kij_fw)
            x_br = vle_h2._calc_x_with_kij(T_K, P_Pa_100, kij_br)
            if x_fw > 0 and x_br > 0:
                ks_ll_impl.append(np.log10(x_fw / x_br) / M_IMP)
            else:
                ks_ll_impl.append(np.nan)
        except Exception:
            ks_ll_impl.append(np.nan)
    ax.plot(T_C, ks_ll_impl, color='purple', ls=':', lw=2.5,
            label='Lopez-Lazaro 2019 (implied)', zorder=15)

    # Implied ks from this work's embedded salinity BIP
    BETA0, BETA1, BETA2 = 0.3833, -0.06595, 0.003321
    TC_H2 = 33.145
    ks_embed_impl = []
    for T_K in T_range:
        try:
            Tr = T_K / TC_H2
            kij_fw = kij_aq_h2(T_K)
            kij_br = kij_fw + (BETA0 + BETA1 * Tr + BETA2 * Tr**2) * M_IMP
            x_fw = vle_h2._calc_x_with_kij(T_K, P_Pa_100, kij_fw)
            x_br = vle_h2._calc_x_with_kij(T_K, P_Pa_100, kij_br)
            if x_fw > 0 and x_br > 0:
                ks_embed_impl.append(np.log10(x_fw / x_br) / M_IMP)
            else:
                ks_embed_impl.append(np.nan)
        except Exception:
            ks_embed_impl.append(np.nan)
    ax.plot(T_C, ks_embed_impl, color='darkgreen',
            linestyle=(0, (3, 1, 1, 1)), lw=2.5,
            label='This work embedded (implied)', zorder=16)

    # --- Historical experimental data (hardcoded from Paper 1 Table 4) ---

    # Crozier & Yamamoto 1974 (6 points, 0-30 C, 1 atm, NaCl)
    T_CY = np.array([274.04, 283.15, 289.55, 293.15, 295.34, 301.50]) - 273.15
    ks_CY = np.array([0.127, 0.101, 0.115, 0.093, 0.106, 0.089])
    ax.scatter(T_CY, ks_CY, c='green', marker='s', s=80, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10,
               label='Crozier & Yamamoto 1974')

    # Gordon et al. 1977 (4 points, 0-29 C, 1 atm, seawater)
    T_Gor = np.array([273.15, 283.15, 293.15, 302.15]) - 273.15
    ks_Gor = np.array([0.110, 0.098, 0.087, 0.078])
    ax.scatter(T_Gor, ks_Gor, c='purple', marker='d', s=80, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10,
               label='Gordon et al. 1977')

    # Morrison & Billett 1952 (4 points, 13-72 C, 1 atm, NaCl)
    T_MB = np.array([285.75, 303.15, 322.55, 344.85]) - 273.15
    ks_MB = np.array([0.112, 0.107, 0.097, 0.081])
    ax.scatter(T_MB, ks_MB, c='orange', marker='v', s=80, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10,
               label='Morrison & Billett 1952')

    # Gerecke & Bittrich 1971 (2 points, 15 and 50 C, 1 atm)
    T_GB = np.array([288.15, 323.15]) - 273.15
    ks_GB = np.array([0.095, 0.076])
    ax.scatter(T_GB, ks_GB, c='red', marker='^', s=80, alpha=0.8,
               edgecolors='k', linewidths=0.5, zorder=10,
               label='Gerecke & Bittrich 1971')

    # T-O 2021 at their two directly measured temperatures (pure-water reference
    # exists only at 323.15 and 423.15 K). Values from their Eq 22 cubic
    # ks(theta) with Table 5 coefficients, theta = T/273.15 - 1 (ln basis);
    # previous hardcoded values (0.130, 0.170) were digitised from their Fig 12.
    LN_TO_LOG10 = 2.303
    T_TO = np.array([323.15, 423.15]) - 273.15
    _TO_D = np.array([0.2898, -1.4330, 3.9584, -3.1666])
    _theta = np.array([323.15, 423.15]) / 273.15 - 1.0
    ks_TO_ln = np.polyval(_TO_D[::-1], _theta)
    ks_TO_pts = ks_TO_ln / LN_TO_LOG10
    ax.scatter(T_TO, ks_TO_pts, c='cyan', marker='D', s=100, alpha=0.9,
               edgecolors='k', linewidths=1.5, zorder=12,
               label='T-O 2021 data')

    # Chabab 2023 from CSV (high-pressure measurements: 28 points)
    n = _scatter_ks_data(ax, df_ks, 'H2')

    # Total count including hardcoded historical
    n_hist = len(T_CY) + len(T_Gor) + len(T_MB) + len(T_GB) + len(T_TO)
    n_total = n + n_hist

    ax.set_xlabel('Temperature (\u00b0C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title(f'H$_2$ Sechenov Coefficient  (n = {n_total})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 155)
    ax.set_ylim(0.03, 0.15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc='upper right', ncol=2)
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE: All-Gas ks Overview (2x4 single page)
# =============================================================================
def figure_ks_overview(df_ks):
    """3x2 Sechenov coefficient overview for 5 gases with brine data.

    Panel layout (row-major):
      (a) CO2   (b) H2S
      (c) N2    (d) H2
      (e) CH4   (f) [hidden]

    C2H6, C3H8, nC4H10 omitted — no experimental brine data at reservoir
    conditions; S&W Eq 8 retained by default.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()

    ks_gases = ['CO2', 'H2S', 'N2', 'H2', 'CH4']

    for idx, gas in enumerate(ks_gases):
        ax = axes[idx]
        display = GAS_DISPLAY.get(gas, gas)

        if gas == 'CO2':
            _ks_panel_co2(ax, df_ks)
        elif gas == 'H2S':
            _ks_panel_h2s(ax, df_ks)
        elif gas == 'N2':
            _ks_panel_n2(ax, df_ks)
        elif gas == 'H2':
            _ks_panel_h2(ax, df_ks)
        elif gas == 'CH4':
            _ks_panel_hc(ax, df_ks, gas)

        n = _count_ks_data(df_ks, gas) if gas != 'H2' else _count_ks_h2(df_ks)
        ax.set_title(f'({chr(97+idx)}) {display}' + (f' (n={n})' if n > 0 else ''),
                     fontsize=11)
        ax.set_xlabel('Temperature (\u00b0C)', fontsize=9)
        ax.set_ylabel('$k_s$ (log$_{10}$ / molality)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best', ncol=2)

    # Hide unused panel (f)
    axes[5].set_visible(False)

    plt.tight_layout()
    return fig


def _count_ks_data(df_ks, gas):
    """Count experimental ks data points for a gas (after exclusions, T <= T_MAX)."""
    if df_ks.empty:
        return 0
    T_MIN_C = T_MIN_K - 273.15
    data = df_ks[(df_ks['Gas'] == gas) & (df_ks['T_C'] >= T_MIN_C) & (df_ks['T_C'] <= T_MAX_C)]
    excl = EXCLUDE_SOURCES.get(gas, set())
    if excl and 'Source' in data.columns:
        data = data[~data['Source'].isin(excl)]
    return len(data)


def _count_ks_h2(df_ks):
    """Count H2 ks data (CSV + hardcoded historical)."""
    n_csv = _count_ks_data(df_ks, 'H2')
    n_hist = 6 + 4 + 4 + 2 + 2  # CY, Gordon, MB, GB, T-O
    return n_csv + n_hist


def _ks_panel_co2(ax, df_ks):
    """CO2 ks panel: Dubessy 2005 (rec.), Duan 2003, S&W Eq 8, S&W Eq 14 implied."""
    T_range = np.linspace(T_MIN_K, T_MAX_K, 150)
    T_C = T_range - 273.15

    # Dubessy 2005 (recommended) — with shift -0.011
    ks_db1 = np.array([ks_dubessy_co2(T, 1.0) for T in T_range])
    ax.plot(T_C, ks_db1 - 0.011, label='Dubessy 2005 (rec.)', **STYLE_REC)

    # Duan & Sun 2003 (alternative model)
    T_d = T_range[T_range <= 530]
    ks_d1 = [ks_duan2003_co2(T, 1.0, 100) for T in T_d]
    ax.plot(T_d - 273.15, ks_d1, label='Duan 2003', **alt_style(0))

    # S&W Eq 8 (reference baseline)
    ks_sw = ks_sw_eq8(T_range, 'CO2')
    ax.plot(T_C, ks_sw, label='S&W Eq 8', alpha=0.7, **STYLE_BASE)

    # S&W Eq 14 implied ks
    try:
        vle = SWBinaryVLE('CO2', salinity_molal=0.0)
        ks_sw_emb = []
        for T_K in T_range:
            kij_fw = kij_aq_co2(T_K, 0.0)
            kij_br = kij_aq_co2(T_K, 1.0)
            try:
                x_fw = vle._calc_x_with_kij(T_K, 100e5, kij_fw)
                x_br = vle._calc_x_with_kij(T_K, 100e5, kij_br)
                if x_fw > 0 and x_br > 0 and np.isfinite(x_fw) and np.isfinite(x_br):
                    ks_sw_emb.append(np.log10(x_fw / x_br) / 1.0)
                else:
                    ks_sw_emb.append(np.nan)
            except Exception:
                ks_sw_emb.append(np.nan)
        ax.plot(T_C, ks_sw_emb, color=C_IMPLIED, ls=':', lw=1.5,
                label='S&W Eq 14 implied')
    except Exception:
        pass

    # Experimental data
    _scatter_ks_data(ax, df_ks, 'CO2', pooled=True)
    ax.set_xlim(0, T_MAX_C)
    ax.set_ylim(-0.01, 0.22)


def _ks_panel_h2s(ax, df_ks):
    """H2S ks panel: Akinfiev 2016, Dubessy 2005."""
    T_range = np.linspace(T_MIN_K, T_MAX_K, 150)
    T_C = T_range - 273.15

    # Akinfiev 2016 (recommended)
    T_a = T_range[(T_range >= 283) & (T_range <= 570)]
    ks_a1 = [ks_akinfiev_h2s(T, 1.0) for T in T_a]
    ax.plot(T_a - 273.15, ks_a1, label='Akinfiev 2016 (rec.)', **STYLE_REC)

    # Dubessy 2005 (alternative model)
    T_db = T_range[T_range <= 520]
    ks_db1 = [ks_dubessy_h2s(T, 1.0) for T in T_db]
    ax.plot(T_db - 273.15, ks_db1, label='Dubessy 2005', **alt_style(0))

    # Experimental data
    _scatter_ks_data(ax, df_ks, 'H2S', pooled=True)
    ax.set_xlim(0, T_MAX_C)
    ax.set_ylim(-0.02, 0.20)


def _ks_panel_n2(ax, df_ks):
    """N2 ks panel: Modified S&W Eq 8 (+0.02), Mao 2006, S&W Eq 13 implied."""
    T_range = np.linspace(T_MIN_K, 473.15, 100)
    T_C = T_range - 273.15

    # S&W Eq 8 (original — reference baseline)
    ks_sw = ks_sw_eq8(T_range, 'N2')
    ax.plot(T_C, ks_sw, label='S&W Eq 8 (original)', alpha=0.7, **STYLE_BASE)
    # Modified Eq 8 (+0.02 offset, recommended)
    ks_sw_mod = ks_sw + 0.02
    ax.plot(T_C, ks_sw_mod, label='Eq 8 + 0.02 (rec.)', **STYLE_REC)

    # Mao 2006 (alternative model)
    T_m = T_range[T_range <= 398]
    ks_m0 = [ks_mao2006_n2(T, 0, 100) for T in T_m]
    ax.plot(T_m - 273.15, ks_m0, label='Mao 2006', **alt_style(0))

    # S&W Eq 13 implied ks
    try:
        vle = SWBinaryVLE('N2', salinity_molal=0.0)
        ks_sw_emb = []
        for T_K in T_range:
            kij_fw = kij_aq_n2(T_K, 0.0)
            kij_br = kij_aq_n2(T_K, 1.0)
            try:
                x_fw = vle._calc_x_with_kij(T_K, 100e5, kij_fw)
                x_br = vle._calc_x_with_kij(T_K, 100e5, kij_br)
                if x_fw > 0 and x_br > 0:
                    ks_sw_emb.append(np.log10(x_fw / x_br) / 1.0)
                else:
                    ks_sw_emb.append(np.nan)
            except Exception:
                ks_sw_emb.append(np.nan)
        ax.plot(T_C, ks_sw_emb, color=C_IMPLIED, ls=':', lw=1.5,
                label='S&W Eq 13 implied')
    except Exception:
        pass

    # Experimental data
    _scatter_ks_data(ax, df_ks, 'N2', pooled=True)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.16)


def _ks_panel_h2(ax, df_ks):
    """H2 ks panel: S&W Eq 8, T-O 2021 corr, implied ks lines, historical data."""
    T_range = np.linspace(T_MIN_K, 423.15, 100)
    T_C = T_range - 273.15

    # S&W Eq 8 (recommended for H2)
    ks_sw = ks_sw_eq8(T_range, 'H2')
    ax.plot(T_C, ks_sw, label='S&W Eq 8 (rec.)', **STYLE_REC)

    # T-O 2021 polynomial (alternative model)
    ks_to = [sechenov_TO(T, log10_basis=True) for T in T_range]
    ax.plot(T_C, ks_to, label='T-O 2021', **alt_style(0))

    # Implied ks from Chabab 2023 BIP
    vle_h2 = SWBinaryVLE('H2', salinity_molal=0.0)
    M_IMP = 1.5
    ks_chabab_impl = []
    for T_K in T_range:
        try:
            kij_fw = kij_aq_chabab_2023(T_K, m=0.0)
            kij_br = kij_aq_chabab_2023(T_K, m=M_IMP)
            x_fw = vle_h2._calc_x_with_kij(T_K, 100e5, kij_fw)
            x_br = vle_h2._calc_x_with_kij(T_K, 100e5, kij_br)
            if x_fw > 0 and x_br > 0:
                ks_chabab_impl.append(np.log10(x_fw / x_br) / M_IMP)
            else:
                ks_chabab_impl.append(np.nan)
        except Exception:
            ks_chabab_impl.append(np.nan)
    ax.plot(T_C, ks_chabab_impl, label='Chabab BIP (impl.)', **alt_style(1))

    # Implied ks from Lopez-Lazaro 2019
    ks_ll_impl = []
    for T_K in T_range:
        try:
            kij_fw = kij_aq_lopez_lazaro_2019(T_K, csw=0.0)
            kij_br = kij_aq_lopez_lazaro_2019(T_K, csw=M_IMP)
            x_fw = vle_h2._calc_x_with_kij(T_K, 100e5, kij_fw)
            x_br = vle_h2._calc_x_with_kij(T_K, 100e5, kij_br)
            if x_fw > 0 and x_br > 0:
                ks_ll_impl.append(np.log10(x_fw / x_br) / M_IMP)
            else:
                ks_ll_impl.append(np.nan)
        except Exception:
            ks_ll_impl.append(np.nan)
    ax.plot(T_C, ks_ll_impl, label='L-L 2019 (impl.)', **alt_style(2))

    # Historical experimental data (hardcoded) — pooled into one neutral
    # series for a readable legend, consistent with the other panels.
    ms = 32
    T_CY = np.array([274.04, 283.15, 289.55, 293.15, 295.34, 301.50]) - 273.15
    ks_CY = np.array([0.127, 0.101, 0.115, 0.093, 0.106, 0.089])

    T_Gor = np.array([273.15, 283.15, 293.15, 302.15]) - 273.15
    ks_Gor = np.array([0.110, 0.098, 0.087, 0.078])

    T_MB = np.array([285.75, 303.15, 322.55, 344.85]) - 273.15
    ks_MB = np.array([0.112, 0.107, 0.097, 0.081])

    T_GB = np.array([288.15, 323.15]) - 273.15
    ks_GB = np.array([0.095, 0.076])

    LN_TO_LOG10 = 2.303
    T_TO = np.array([323.15, 423.15]) - 273.15
    ks_TO_pts = np.array([0.130, 0.170]) / LN_TO_LOG10

    # Pool all historical sources (C&Y, Gordon, M&B, G&B, T-O) into one
    # neutral grey series.
    T_hist = np.concatenate([T_CY, T_Gor, T_MB, T_GB, T_TO])
    ks_hist = np.concatenate([ks_CY, ks_Gor, ks_MB, ks_GB, ks_TO_pts])
    ax.scatter(T_hist, ks_hist, c=C_EXP, marker='o', s=ms, alpha=0.6,
               edgecolors='k', linewidths=0.4, zorder=10,
               label='Historical data')

    # Chabab 2023 from CSV
    _scatter_ks_data(ax, df_ks, 'H2', pooled=True)

    ax.set_xlim(-5, 155)
    ax.set_ylim(0.03, 0.15)


def _ks_panel_hc(ax, df_ks, gas):
    """HC ks panel (CH4, C2H6, C3H8, nC4H10): S&W Eq 8 vs Li 2015."""
    display = GAS_DISPLAY.get(gas, gas)
    T_range = np.linspace(T_MIN_K, 473.15, 100)
    T_C = T_range - 273.15

    # S&W Eq 8 (recommended for all HCs — validated for CH4, retained for the rest)
    ks_sw = ks_sw_eq8(T_range, gas)
    ax.plot(T_C, ks_sw, label='S&W Eq 8 (rec.)', **STYLE_REC)

    # Li et al. 2015 (alternative model)
    try:
        ks_li = [ks_li2015(T, gas, m_NaCl=1.0, P_bar=100) for T in T_range]
        ax.plot(T_C, ks_li, label='Li 2015', **alt_style(0))
    except Exception:
        pass

    # Experimental data (usually none for HCs)
    _scatter_ks_data(ax, df_ks, gas, pooled=True)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, max(ks_sw) * 1.3 if max(ks_sw) > 0 else 0.2)


# =============================================================================
# EMBEDDED BIP KS COMPARISON (from cached VLE data)
# =============================================================================
def figure_embedded_bip_ks_comparison(df_ks):
    """Embedded BIP implied ks vs explicit models — loads cached VLE data.

    Cache file: ../../shared/data/embedded_ks_band_cache.npz
    Generated by: fit_embedded_salinity_all_gases.py
    Delete cache to force recalculation.
    """
    cache_path = '../../shared/data/embedded_ks_band_cache.npz'
    if not os.path.exists(cache_path):
        print("   WARNING: No cached ks band data found at", cache_path)
        print("   Run: python fit_embedded_salinity_all_gases.py  to generate cache")
        return None

    cache = np.load(cache_path)
    T_RANGE = cache['T_RANGE']
    T_C_RANGE = T_RANGE - 273.15

    gas_list = ['CO2', 'H2S', 'N2', 'H2', 'CH4', 'C2H6', 'C3H8', 'nC4H10']
    gas_list = [g for g in gas_list if f'{g}_ks_this_work' in cache]

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    HC_GASES = {'CH4', 'C2H6', 'C3H8', 'nC4H10'}
    gas_ylim = {
        'CO2': (0, 0.25), 'H2S': (-0.02, 0.22),
        'N2': (0, 0.16), 'H2': (0, 0.14),
    }
    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    M_BRINE = 1.0
    P_BAR = 100.0

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    axes = axes.flatten()

    for idx, gas in enumerate(gas_list):
        ax = axes[idx]

        # --- Explicit ks model curves (analytic, fast) ---
        if gas == 'CO2':
            ks_model = np.array([ks_dubessy_co2(T, M_BRINE) - 0.011
                                 for T in T_RANGE])
            ax.plot(T_C_RANGE, ks_model, color=C_BASE, linestyle='-', lw=2.5,
                    label='Dubessy 2005 (target)', zorder=20)
        elif gas == 'H2S':
            T_valid = T_RANGE[(T_RANGE >= 283) & (T_RANGE <= 570)]
            T_valid_C = T_valid - 273.15
            ks_model = np.array([ks_akinfiev_h2s(T, M_BRINE) for T in T_valid])
            ax.plot(T_valid_C, ks_model, color=C_BASE, linestyle='-', lw=2.5,
                    label='Akinfiev 2016 (target)', zorder=20)
        elif gas == 'N2':
            from _lib_salting_library import TB_K as _TB_K
            def _sw_eq8_ks(T_C, Tb):
                T_K = T_C + 273.15
                return 1.3012 + 4.45e-4*Tb - 8.769e-3*T_K + 2.0293e-5*T_K**2 - 1.5233e-8*T_K**3
            ks_sw = np.array([_sw_eq8_ks(T_C, COMPONENTS[gas].Tb) for T_C in T_C_RANGE])
            ks_sw_shifted = ks_sw + 0.02  # Modified Eq 8 for N2
            ax.plot(T_C_RANGE, ks_sw_shifted, 'k-', lw=2.5,
                    label='Target $k_s$ (Eq 8 + 0.02)', zorder=20)
        else:
            def _sw_eq8_ks(T_C, Tb):
                T_K = T_C + 273.15
                return 1.3012 + 4.45e-4*Tb - 8.769e-3*T_K + 2.0293e-5*T_K**2 - 1.5233e-8*T_K**3
            ks_sw = np.array([_sw_eq8_ks(T_C, COMPONENTS[gas].Tb) for T_C in T_C_RANGE])
            ax.plot(T_C_RANGE, ks_sw, 'k-', lw=2.5,
                    label='S&W Eq 8 (target)', zorder=20)

        # --- This work embedded BIP (from cache) ---
        ks_this_work = cache[f'{gas}_ks_this_work']
        ks_band_min = cache[f'{gas}_ks_band_min']
        ks_band_max = cache[f'{gas}_ks_band_max']

        ax.fill_between(T_C_RANGE, ks_band_min, ks_band_max,
                        alpha=0.18, color=C_REC, zorder=15,
                        label='This work embedded (35\u2013350 bar)')
        ax.plot(T_C_RANGE, ks_this_work, color=C_REC,
                linestyle=(0, (3, 1, 1, 1)), lw=2.5,
                label='This work embedded (100 bar)', zorder=16)

        # --- S&W original embedded BIP implied ks (from cache) ---
        sw_key = f'{gas}_ks_sw_emb'
        if sw_key in cache:
            ax.plot(T_C_RANGE, cache[sw_key], color=C_IMPLIED, linestyle=':',
                    lw=2, label='S&W embedded (implied)', zorder=14)

        # --- Experimental ks data points ---
        if not df_ks.empty:
            gas_data = df_ks[df_ks['Gas'] == gas]
            # Apply source exclusions
            excl = EXCLUDE_SOURCES.get(gas, set())
            if excl and 'Source' in gas_data.columns:
                gas_data = gas_data[~gas_data['Source'].isin(excl)]
            if len(gas_data) > 0:
                # Pool all sources into one neutral marker style + single
                # legend entry, keeping the per-panel legend readable.
                ax.scatter(gas_data['T_C'], gas_data['ks_data'],
                           marker='o', c='0.30', s=46, alpha=0.6,
                           edgecolors='k', linewidths=0.5, zorder=10,
                           label=f'Experimental (n={len(gas_data)})')

        # --- Panel formatting ---
        ax.set_title(f'{labels[idx]} {GAS_DISPLAY.get(gas, gas)}', fontweight='bold')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel(r'$k_s$ (log$_{10}$ / molality)')
        ax.set_xlim(20, 200)
        # Gas-specific y-axis limits
        if gas in ('CO2', 'H2S'):
            ax.set_ylim(0, 0.15)
        elif gas in ('CH4', 'C2H6', 'C3H8', 'nC4H10'):
            ax.set_ylim(0.05, 0.25)
        else:
            ax.set_ylim(0, 0.2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right', ncol=1)

    for i in range(len(gas_list), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================
def run_all_figures(csv_path='../../shared/data/pointwise_kij_results_sw_alpha.csv',
                    save_figures=True, output_dir=None):
    """Generate ALL Paper 2 manuscript figures from a single script.

    Figures produced (matching manuscript2.tex \\includegraphics):
      - fig1.pdf: kij_AQ overview (2x4)
      - fig2.pdf: kij_NA overview (2x4)
      - fig3.pdf: Sechenov overview (3x2) — models + experimental data
      - embedded_bip_ks_comparison.pdf: Embedded BIP implied ks (from cache)
      - parity_overview_1.pdf: Parity plots (CO2, H2S, N2, H2)
      - parity_overview_2.pdf: Parity plots (CH4, C2H6, C3H8, nC4H10)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print("=" * 70)
    print("S&W FRAMEWORK REFRESH (Paper 2) - FIGURE GENERATION")
    print("=" * 70)

    print("\n1. Loading data...")
    try:
        df = load_data(csv_path)
        print(f"   Loaded {len(df)} total results")
        for gas in GASES:
            n = (df['Gas'] == gas).sum()
            print(f"   {gas:5s}: {n} points")
    except Exception as e:
        print(f"   Error loading data: {e}")
        df = pd.DataFrame()

    # Load experimental ks data
    df_ks = _load_ks_data()
    if not df_ks.empty:
        print(f"   Loaded {len(df_ks)} experimental ks values")

    os.makedirs(output_dir, exist_ok=True)
    figures = {}

    def _save(name, fig):
        if fig is None:
            return
        figures[name] = fig
        if save_figures:
            for ext in ['pdf', 'png']:
                path = os.path.join(output_dir, f'{name}.{ext}')
                fig.savefig(path, format=ext if ext == 'pdf' else None,
                            dpi=300, bbox_inches='tight')
            print(f"   Saved {name}.pdf + .png")

    print("\n2. Generating manuscript figures...")

    # Figure 1: kij_AQ overview
    if len(df) > 0:
        print("   Figure 1: kij_AQ overview (2x4)...")
        _save('fig1', figure_1_kij_aq_overview(df))

        print("   Figure 2: kij_NA overview (2x4)...")
        _save('fig2', figure_2_kij_na_overview(df))

        print("   Figure 2 alt: kij_NA pressure-colored (2x4)...")
        _save('fig2_pressure', figure_2_kij_na_pressure(df))

    # Figure 3: Sechenov overview
    print("   Figure 3: Sechenov overview (3x2)...")
    _save('fig3', figure_ks_overview(df_ks))

    # Embedded BIP ks comparison (from cached VLE data)
    print("   Embedded BIP ks comparison (from cache)...")
    _save('embedded_bip_ks_comparison', figure_embedded_bip_ks_comparison(df_ks))

    # Parity overview (freshwater + brine per gas)
    print("   Parity overview plots...")
    try:
        df_xlsx = load_solubility_data()
        if len(df_xlsx) > 0:
            parity_figs = parity_overview(df_xlsx, output_dir=output_dir)
            for name, fig in parity_figs.items():
                figures[name] = fig
    except Exception as e:
        print(f"   WARNING: Could not generate parity overview: {e}")

    print("\n" + "=" * 70)
    print(f"COMPLETE! {len(figures)} figures generated.")
    print("=" * 70)

    return figures


# =============================================================================
# =============================================================================
#
#  APPENDIX FIGURES (merged from generate_appendix_figures.py)
#
# =============================================================================
# =============================================================================

XLSX_PATH = '../../shared/data/solubility_points.xlsx'
EMB_CSV_PATH = '../../shared/data/embedded_salinity_bip_all_gases.csv'

# Gas lists
PARITY_GASES = ['CO2', 'H2S', 'N2', 'H2', 'CH4', 'C2H6', 'C3H8', 'nC4H10']
KVALUE_GASES = ['CO2', 'H2S', 'N2', 'H2', 'CH4', 'C2H6', 'C3H8', 'nC4H10']

# Gases without an S&W (1992) original kij_AQ -- skip S&W comparison for these
NO_SW_ORIGINAL = {'H2'}

# Recommended validity envelopes per gas
VALIDITY_RANGES = {
    'CO2':    {'T_max': 473, 'Psat_limit': True},
    'H2S':    {'T_max': 443, 'Psat_limit': True},
    'CH4':    {'T_max': 473, 'Psat_limit': False},
    'N2':     {'T_max': 473, 'Psat_limit': False},
    'H2':     {'T_max': 443, 'Psat_limit': False},
    'C2H6':   {'T_max': 473, 'Psat_limit': True},
    'C3H8':   {'T_max': 422, 'Psat_limit': True},
    'nC4H10': {'T_max': 422, 'Psat_limit': True},
}


def _lee_kesler_psat(T_K, Tc, Pc_bar, omega):
    """Lee-Kesler vapor pressure correlation. Returns Psat in bar, or np.inf if T >= Tc."""
    Tr = T_K / Tc
    if Tr >= 1.0:
        return np.inf
    if Tr < 0.3:
        return 0.0
    f0 = 5.92714 - 6.09648/Tr - 1.28862*np.log(Tr) + 0.169347*Tr**6
    f1 = 15.2518 - 15.6875/Tr - 13.4721*np.log(Tr) + 0.43577*Tr**6
    return Pc_bar * np.exp(f0 + omega * f1)


def is_in_validity_range(gas, T_K, P_bar):
    """Check if (T, P) falls within the recommended validity envelope."""
    vr = VALIDITY_RANGES.get(gas)
    if vr is None:
        return True
    if T_K > vr['T_max']:
        return False
    if vr['Psat_limit']:
        comp = COMPONENTS[gas]
        Psat = _lee_kesler_psat(T_K, comp.Tc, comp.Pc / 1e5, comp.omega)
        if P_bar > Psat:
            return False
    return True


# Math-safe subscript labels for axis labels (used inside $x_{...}$)
GAS_MATH_SUB = {
    'CO2': r'\mathrm{CO_2}', 'H2S': r'\mathrm{H_2S}', 'CH4': r'\mathrm{CH_4}',
    'N2': r'\mathrm{N_2}', 'H2': r'\mathrm{H_2}',
    'C2H6': r'\mathrm{C_2H_6}', 'C3H8': r'\mathrm{C_3H_8}',
    'nC4H10': r'\mathrm{nC_4H_{10}}',
}

# Column in the Excel file that holds the experimental x_gas for each gas
X_COL = {
    'CO2': 'x_CO2', 'H2S': 'x_H2S', 'CH4': 'x_CH4', 'N2': 'x_N2',
    'H2': 'x_H2', 'C2H6': 'x_C2H6', 'C3H8': 'x_C3H8',
}

# Feed-composition column (z_gas == 1 identifies single-gas rows)
Z_COL = {
    'CO2': 'z_CO2', 'H2S': 'z_H2S', 'CH4': 'z_CH4', 'N2': 'z_N2',
    'H2': 'z_H2', 'C2H6': 'z_C2H6', 'C3H8': 'z_C3H8',
}


# =============================================================================
# PROPOSED kij_AQ FUNCTIONS (from VLE engine — Paper 2 base cases)
# =============================================================================
# Single-arg wrappers for parity plot code
PROPOSED_KIJ = {
    'CO2': lambda T_K: kij_aq_co2_proposed(T_K),
    'H2S': lambda T_K: kij_aq_h2s_proposed(T_K),
    'CH4': lambda T_K: kij_aq_ch4(T_K),
    'N2':  lambda T_K: kij_aq_n2_proposed(T_K),
    'H2':  lambda T_K: kij_aq_h2_proposed(T_K),
    'C2H6': lambda T_K: kij_aq_c2h6_proposed(T_K),
    'C3H8': lambda T_K: kij_aq_c3h8_proposed(T_K),
}

DROPIN_KIJ = {
    'CO2': lambda T_K: kij_aq_co2_dropin(T_K),
    'H2S': lambda T_K: kij_aq_h2s_dropin(T_K),
    'CH4': lambda T_K: kij_aq_ch4_dropin(T_K),
    'N2':  lambda T_K: kij_aq_n2_dropin(T_K),
    'H2':  lambda T_K: kij_aq_h2_dropin(T_K),
    'C2H6': lambda T_K: kij_aq_c2h6_dropin(T_K),
    'C3H8': lambda T_K: kij_aq_c3h8_dropin(T_K),
}


def _appendix_sw_kij_aq(gas, T_K, salinity_molal=0.0):
    """S&W original kij_AQ (appendix version with salinity arg)."""
    return get_kij_aq(gas, T_K, salinity_molal, framework='sw_original')


# =============================================================================
# EMBEDDED SALINITY BIP (uses engine's EMBEDDED_SALINITY_PARAMS)
# =============================================================================
EMBEDDED_PARAMS = EMBEDDED_SALINITY_PARAMS


def embedded_kij(gas, T_K, m):
    """Compute embedded salinity kij(T, m) = kij_fw(T) + delta(Tr, m)."""
    kij_fw = get_kij_aq(gas, T_K, 0.0, framework='proposed')
    delta = calc_embedded_delta_kij(gas, T_K, m)
    return kij_fw + delta


# =============================================================================
# DATA LOADING
# =============================================================================
def load_solubility_data(xlsx_path=XLSX_PATH):
    """Load experimental solubility data from Excel."""
    df = pd.read_excel(xlsx_path, sheet_name='QCd Data')
    return df


def get_gas_data(df, gas, freshwater_only=False):
    """Extract single-gas rows for a specific gas (T <= T_MAX_K)."""
    if gas == 'nC4H10':
        return pd.DataFrame()

    z_col = Z_COL.get(gas)
    x_col = X_COL.get(gas)
    if z_col is None or x_col is None or z_col not in df.columns:
        return pd.DataFrame()

    mask = (df[z_col] == 1.0) & (df[x_col].notna()) & (df[x_col] > 0) & (df['T_K'] >= T_MIN_K) & (df['T_K'] <= T_MAX_K)
    if freshwater_only:
        mask = mask & (df['Sal_m'] == 0)
    data = df[mask].copy()
    excl = EXCLUDE_SOURCES.get(gas, set())
    if excl and 'Source' in data.columns:
        data = data[~data['Source'].isin(excl)]
    data['x_gas_exp'] = data[x_col]
    return data


# =============================================================================
# APPENDIX A: PER-GAS SOLUBILITY PARITY PLOTS
# =============================================================================
def calc_mare(x_pred, x_exp):
    """Mean absolute relative error in percent."""
    valid = (x_exp > 0) & np.isfinite(x_pred) & (x_pred > 0)
    if valid.sum() == 0:
        return np.nan
    return np.mean(np.abs(x_pred[valid] - x_exp[valid]) / x_exp[valid]) * 100


def parity_plot_gas(gas, df, output_dir=OUTPUT_DIR):
    """Generate parity plot for one gas (freshwater + optional brine panels)."""

    display = GAS_DISPLAY.get(gas, gas)

    fw_data = get_gas_data(df, gas, freshwater_only=True)
    br_data = get_gas_data(df, gas, freshwater_only=False)
    br_data = br_data[br_data['Sal_m'] > 0] if len(br_data) > 0 else pd.DataFrame()

    has_fw = len(fw_data) > 0
    has_br = len(br_data) >= 5

    if not has_fw and not has_br:
        print(f"  {gas}: No data -- skipping parity plot.")
        return None

    if has_fw and has_br:
        fig, (ax_fw, ax_br) = plt.subplots(1, 2, figsize=(12, 5.5))
    elif has_fw:
        fig, ax_fw = plt.subplots(1, 1, figsize=(6.5, 5.5))
        ax_br = None
    else:
        fig, ax_br = plt.subplots(1, 1, figsize=(6.5, 5.5))
        ax_fw = None

    # Panel (a): Freshwater
    if has_fw and ax_fw is not None:
        x_exp_fw = fw_data['x_gas_exp'].values
        T_K_fw = fw_data['T_K'].values
        P_bar_fw = fw_data['P_bar'].values

        x_pred_tw = np.full_like(x_exp_fw, np.nan)
        has_sw = gas not in NO_SW_ORIGINAL
        x_pred_sw = np.full_like(x_exp_fw, np.nan)

        vle = SWBinaryVLE(gas, salinity_molal=0.0)
        for i in range(len(fw_data)):
            T, P = T_K_fw[i], P_bar_fw[i] * 1e5
            try:
                if gas in PROPOSED_KIJ:
                    kij_tw = PROPOSED_KIJ[gas](T)
                else:
                    kij_tw = _appendix_sw_kij_aq(gas, T, 0.0)
                x_tw = vle._calc_x_with_kij(T, P, kij_tw)
                if x_tw is not None and np.isfinite(x_tw) and x_tw > 0:
                    x_pred_tw[i] = x_tw
            except Exception:
                pass
            if has_sw:
                try:
                    kij_sw = _appendix_sw_kij_aq(gas, T, 0.0)
                    x_sw = vle._calc_x_with_kij(T, P, kij_sw)
                    if x_sw is not None and np.isfinite(x_sw) and x_sw > 0:
                        x_pred_sw[i] = x_sw
                except Exception:
                    pass

        solved_tw = np.isfinite(x_pred_tw) & (x_pred_tw > 0)
        solved_sw = np.isfinite(x_pred_sw) & (x_pred_sw > 0)

        mare_tw = calc_mare(x_pred_tw[solved_tw], x_exp_fw[solved_tw])

        lbl_tw = f'$k_{{ij}}(T)$ (MARE={mare_tw:.1f}%, n={solved_tw.sum()})'
        ax_fw.scatter(x_exp_fw[solved_tw], x_pred_tw[solved_tw],
                      s=25, alpha=0.6, edgecolors='k', linewidths=0.3,
                      c='tab:blue', label=lbl_tw, zorder=6)

        if has_sw and solved_sw.sum() > 0:
            mare_sw = calc_mare(x_pred_sw[solved_sw], x_exp_fw[solved_sw])
            lbl_sw = f'S&W original (MARE={mare_sw:.1f}%, n={solved_sw.sum()})'
            ax_fw.scatter(x_exp_fw[solved_sw], x_pred_sw[solved_sw],
                          s=25, alpha=0.5, edgecolors='k', linewidths=0.3,
                          c='tab:red', marker='s', label=lbl_sw, zorder=5)

        vals_list = [x_exp_fw[solved_tw], x_pred_tw[solved_tw]]
        if has_sw and solved_sw.sum() > 0:
            vals_list += [x_exp_fw[solved_sw], x_pred_sw[solved_sw]]
        all_vals = np.concatenate(vals_list)
        lo = max(all_vals.min() * 0.5, 1e-8)
        hi = all_vals.max() * 2
        line = np.array([lo, hi])
        ax_fw.plot(line, line, 'k-', lw=1.2, zorder=10)
        ax_fw.fill_between(line, line * 0.9, line * 1.1, alpha=0.12,
                           color='green', zorder=1, label='$\pm$10%')

        ax_fw.set_xscale('log')
        ax_fw.set_yscale('log')
        ax_fw.set_xlim(lo, hi)
        ax_fw.set_ylim(lo, hi)
        ax_fw.set_aspect('equal')
        panel_lbl = '(a) ' if has_br else ''
        ax_fw.set_title(f'{panel_lbl}Freshwater')
        ax_fw.set_xlabel(f'Experimental $x_{{{gas}}}$')
        ax_fw.set_ylabel(f'Predicted $x_{{{gas}}}$')
        ax_fw.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax_fw.grid(True, alpha=0.3, which='both')

    # Panel (b): Brine
    if has_br and ax_br is not None:
        x_exp_br = br_data['x_gas_exp'].values
        T_K_br = br_data['T_K'].values
        P_bar_br = br_data['P_bar'].values
        Sal_m_br = br_data['Sal_m'].values

        x_pred_sech = np.full_like(x_exp_br, np.nan)
        x_pred_emb = np.full_like(x_exp_br, np.nan)
        x_pred_sw_br = np.full_like(x_exp_br, np.nan)

        for i in range(len(br_data)):
            T = T_K_br[i]
            P_Pa = P_bar_br[i] * 1e5
            m = Sal_m_br[i]

            try:
                if gas in PROPOSED_KIJ:
                    kij_fw = PROPOSED_KIJ[gas](T)
                    x_fw = SWBinaryVLE(gas, 0.0)._calc_x_with_kij(T, P_Pa, kij_fw)
                    if np.isfinite(x_fw) and x_fw > 0:
                        ks = get_sechenov_ks(gas, T, m, P_bar_br[i])
                        x_sech = x_fw * 10**(-ks * m)
                    else:
                        x_sech = np.nan
                else:
                    vle_br = SWBinaryVLE(gas, salinity_molal=m)
                    x_sech = vle_br.calc_gas_solubility(T, P_Pa)
                if x_sech is not None and np.isfinite(x_sech) and x_sech > 0:
                    x_pred_sech[i] = x_sech
            except Exception:
                pass

            try:
                vle_fw = SWBinaryVLE(gas, salinity_molal=0.0)
                kij_emb = embedded_kij(gas, T, m)
                x_emb = vle_fw._calc_x_with_kij(T, P_Pa, kij_emb)
                if x_emb is not None and np.isfinite(x_emb) and x_emb > 0:
                    x_pred_emb[i] = x_emb
            except Exception:
                pass

            if has_sw:
                try:
                    vle_fw2 = SWBinaryVLE(gas, salinity_molal=m, framework='sw_original')
                    kij_sw_br = _appendix_sw_kij_aq(gas, T, m)
                    x_sw_br = vle_fw2._calc_x_with_kij(T, P_Pa, kij_sw_br)
                    if gas == 'H2S':
                        ks_sw = sw_equation_8_ks(T - 273.15, COMPONENTS[gas].Tb)
                        if np.isfinite(x_sw_br) and x_sw_br > 0:
                            x_sw_br = x_sw_br * 10**(-ks_sw * m)
                    if x_sw_br is not None and np.isfinite(x_sw_br) and x_sw_br > 0:
                        x_pred_sw_br[i] = x_sw_br
                except Exception:
                    pass

        v_sech = np.isfinite(x_pred_sech) & (x_pred_sech > 0)
        v_emb = np.isfinite(x_pred_emb) & (x_pred_emb > 0)
        v_sw = np.isfinite(x_pred_sw_br) & (x_pred_sw_br > 0)

        mare_sech = calc_mare(x_pred_sech[v_sech], x_exp_br[v_sech])
        mare_emb = calc_mare(x_pred_emb[v_emb], x_exp_br[v_emb])

        ax_br.scatter(x_exp_br[v_sech], x_pred_sech[v_sech],
                      s=25, alpha=0.6, edgecolors='k', linewidths=0.3,
                      c='tab:blue',
                      label=f'This work, Sechenov (MARE={mare_sech:.1f}%, n={v_sech.sum()})',
                      zorder=6)
        ax_br.scatter(x_exp_br[v_emb], x_pred_emb[v_emb],
                      s=25, alpha=0.5, edgecolors='k', linewidths=0.3,
                      c='tab:green', marker='^',
                      label=f'This work, embedded (MARE={mare_emb:.1f}%, n={v_emb.sum()})',
                      zorder=5)
        if has_sw and v_sw.sum() > 0:
            mare_sw = calc_mare(x_pred_sw_br[v_sw], x_exp_br[v_sw])
            ax_br.scatter(x_exp_br[v_sw], x_pred_sw_br[v_sw],
                          s=25, alpha=0.5, edgecolors='k', linewidths=0.3,
                          c='tab:red', marker='s',
                          label=f'S&W original (MARE={mare_sw:.1f}%, n={v_sw.sum()})',
                          zorder=4)

        vals_list_br = [
            x_exp_br[v_sech], x_pred_sech[v_sech],
            x_exp_br[v_emb], x_pred_emb[v_emb],
        ]
        if has_sw and v_sw.sum() > 0:
            vals_list_br += [x_exp_br[v_sw], x_pred_sw_br[v_sw]]
        all_vals_br = np.concatenate(vals_list_br)
        if len(all_vals_br) > 0:
            lo_br = max(all_vals_br.min() * 0.5, 1e-8)
            hi_br = all_vals_br.max() * 2
        else:
            lo_br, hi_br = 1e-6, 1e-1
        line_br = np.array([lo_br, hi_br])
        ax_br.plot(line_br, line_br, 'k-', lw=1.2, zorder=10)
        ax_br.fill_between(line_br, line_br * 0.9, line_br * 1.1, alpha=0.12,
                           color='green', zorder=1, label='$\pm$10%')

        ax_br.set_xscale('log')
        ax_br.set_yscale('log')
        ax_br.set_xlim(lo_br, hi_br)
        ax_br.set_ylim(lo_br, hi_br)
        ax_br.set_aspect('equal')
        panel_lbl = '(b) ' if has_fw else ''
        ax_br.set_title(f'{panel_lbl}Brine')
        ax_br.set_xlabel(f'Experimental $x_{{{gas}}}$')
        ax_br.set_ylabel(f'Predicted $x_{{{gas}}}$')
        ax_br.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax_br.grid(True, alpha=0.3, which='both')

    fig.suptitle(f'{display}' + u'\u2013' + 'H$_2$O Solubility: Predicted vs Experimental',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    stem = f'parity_{gas}'
    fig.savefig(os.path.join(output_dir, f'{stem}.pdf'),
                format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f'{stem}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  {gas}: saved {stem}.pdf + .png")
    return fig


# =============================================================================
# APPENDIX A: CONSOLIDATED PARITY OVERVIEW (3 pages, 3x2 each)
# =============================================================================

def _parity_panel_fw(ax, gas, df):
    """Draw freshwater parity panel on given axes. Returns True if data plotted."""
    display = GAS_DISPLAY.get(gas, gas)
    fw_data = get_gas_data(df, gas, freshwater_only=True)
    if len(fw_data) == 0:
        ax.text(0.5, 0.5, f'{display}\nNo freshwater data',
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    x_exp = fw_data['x_gas_exp'].values
    T_K = fw_data['T_K'].values
    P_bar = fw_data['P_bar'].values

    has_sw = gas not in NO_SW_ORIGINAL
    # Freshwater panels tell a two-way story: this work (drop-in) vs S&W
    # original. The MC-3 alpha equivalence lives quantitatively in Table 2
    # (tab:alpha_comparison), so it is not re-plotted here.
    x_pred_di = np.full_like(x_exp, np.nan)
    x_pred_sw = np.full_like(x_exp, np.nan)

    vle_di = SWBinaryVLE(gas, salinity_molal=0.0, framework='dropin')
    vle_sw = SWBinaryVLE(gas, salinity_molal=0.0, framework='sw_original') if has_sw else None
    for i in range(len(fw_data)):
        T, P = T_K[i], P_bar[i] * 1e5
        try:
            if gas in DROPIN_KIJ:
                kij_di = DROPIN_KIJ[gas](T)
                x_d = vle_di._calc_x_with_kij(T, P, kij_di)
                if x_d is not None and np.isfinite(x_d) and x_d > 0:
                    x_pred_di[i] = x_d
        except Exception:
            pass
        if has_sw:
            try:
                kij_sw = _appendix_sw_kij_aq(gas, T, 0.0)
                x_sw = vle_sw._calc_x_with_kij(T, P, kij_sw)
                if x_sw is not None and np.isfinite(x_sw) and x_sw > 0:
                    x_pred_sw[i] = x_sw
            except Exception:
                pass

    solved_di = np.isfinite(x_pred_di) & (x_pred_di > 0)
    solved_sw = np.isfinite(x_pred_sw) & (x_pred_sw > 0)

    if solved_di.sum() > 0:
        mare_di = calc_mare(x_pred_di[solved_di], x_exp[solved_di])
        lbl_di = f'This work ({mare_di:.1f}%, n={solved_di.sum()})'
        ax.scatter(x_exp[solved_di], x_pred_di[solved_di],
                   s=18, alpha=0.5, edgecolors='k', linewidths=0.2,
                   c=C_REC, label=lbl_di, zorder=6)

    if has_sw and solved_sw.sum() > 0:
        mare_sw = calc_mare(x_pred_sw[solved_sw], x_exp[solved_sw])
        lbl_sw = f'S&W original ({mare_sw:.1f}%, n={solved_sw.sum()})'
        ax.scatter(x_exp[solved_sw], x_pred_sw[solved_sw],
                   s=18, alpha=0.4, edgecolors='k', linewidths=0.2,
                   c=C_BASE, marker='s', label=lbl_sw, zorder=4)

    vals_list = []
    if solved_di.sum() > 0:
        vals_list += [x_exp[solved_di], x_pred_di[solved_di]]
    if has_sw and solved_sw.sum() > 0:
        vals_list += [x_exp[solved_sw], x_pred_sw[solved_sw]]
    all_vals = np.concatenate(vals_list)
    lo = max(all_vals.min() * 0.5, 1e-8)
    hi = all_vals.max() * 2
    line = np.array([lo, hi])
    ax.plot(line, line, color='0.45', linestyle='-', lw=1.2, zorder=10)
    ax.fill_between(line, line * 0.9, line * 1.1, alpha=0.10,
                     color=C_REC, zorder=1, label='$\\pm$10%')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_title(f'{display} — Freshwater', fontsize=10)
    math_sub = GAS_MATH_SUB.get(gas, gas)
    ax.set_xlabel(f'Experimental $x_{{{math_sub}}}$', fontsize=9)
    ax.set_ylabel(f'Predicted $x_{{{math_sub}}}$', fontsize=9)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    return True


def _parity_panel_br(ax, gas, df):
    """Draw brine parity panel on given axes. Returns True if data plotted."""
    display = GAS_DISPLAY.get(gas, gas)
    br_data = get_gas_data(df, gas, freshwater_only=False)
    br_data = br_data[br_data['Sal_m'] > 0] if len(br_data) > 0 else pd.DataFrame()

    if len(br_data) < 5:
        ax.text(0.5, 0.5, f'{display}\nNo brine data',
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    has_sw = gas not in NO_SW_ORIGINAL
    x_exp = br_data['x_gas_exp'].values
    T_K = br_data['T_K'].values
    P_bar_arr = br_data['P_bar'].values
    Sal_m = br_data['Sal_m'].values

    x_pred_sech = np.full_like(x_exp, np.nan)
    x_pred_di = np.full_like(x_exp, np.nan)
    x_pred_sw = np.full_like(x_exp, np.nan)

    for i in range(len(br_data)):
        T = T_K[i]
        P_Pa = P_bar_arr[i] * 1e5
        m = Sal_m[i]

        # Track 1 (proposed) — Sechenov
        try:
            if gas in PROPOSED_KIJ:
                kij_fw = PROPOSED_KIJ[gas](T)
                x_fw = SWBinaryVLE(gas, 0.0)._calc_x_with_kij(T, P_Pa, kij_fw)
                if np.isfinite(x_fw) and x_fw > 0:
                    ks = get_sechenov_ks(gas, T, m, P_bar_arr[i])
                    x_sech = x_fw * 10**(-ks * m)
                else:
                    x_sech = np.nan
            else:
                vle_br = SWBinaryVLE(gas, salinity_molal=m)
                x_sech = vle_br.calc_gas_solubility(T, P_Pa)
            if x_sech is not None and np.isfinite(x_sech) and x_sech > 0:
                x_pred_sech[i] = x_sech
        except Exception:
            pass

        # Track 2 (dropin) — embedded delta
        try:
            vle_di = SWBinaryVLE(gas, salinity_molal=m, framework='dropin')
            x_d = vle_di.calc_gas_solubility(T, P_Pa)
            if x_d is not None and np.isfinite(x_d) and x_d > 0:
                x_pred_di[i] = x_d
        except Exception:
            pass

        # S&W original — constructor salinity keeps the S&W alpha csw term;
        # the manual kij injection then matches the engine's auto path.
        if has_sw:
            try:
                vle_sw_br = SWBinaryVLE(gas, salinity_molal=m, framework='sw_original')
                kij_sw_br = _appendix_sw_kij_aq(gas, T, m)
                x_sw_br = vle_sw_br._calc_x_with_kij(T, P_Pa, kij_sw_br)
                if gas == 'H2S':
                    ks_sw = sw_equation_8_ks(T - 273.15, COMPONENTS[gas].Tb)
                    if np.isfinite(x_sw_br) and x_sw_br > 0:
                        x_sw_br = x_sw_br * 10**(-ks_sw * m)
                if x_sw_br is not None and np.isfinite(x_sw_br) and x_sw_br > 0:
                    x_pred_sw[i] = x_sw_br
            except Exception:
                pass

    v_sech = np.isfinite(x_pred_sech) & (x_pred_sech > 0)
    v_di = np.isfinite(x_pred_di) & (x_pred_di > 0)
    v_sw = np.isfinite(x_pred_sw) & (x_pred_sw > 0)

    mare_sech = calc_mare(x_pred_sech[v_sech], x_exp[v_sech])

    ax.scatter(x_exp[v_sech], x_pred_sech[v_sech],
               s=18, alpha=0.4, edgecolors='k', linewidths=0.2,
               c=C_MC3, marker='D',
               label=f'Explicit Sechenov ({mare_sech:.1f}%, n={v_sech.sum()})',
               zorder=5)
    if v_di.sum() > 0:
        mare_di = calc_mare(x_pred_di[v_di], x_exp[v_di])
        ax.scatter(x_exp[v_di], x_pred_di[v_di],
                   s=18, alpha=0.5, edgecolors='k', linewidths=0.2,
                   c=C_REC,
                   label=f'This work ({mare_di:.1f}%, n={v_di.sum()})',
                   zorder=6)
    if has_sw and v_sw.sum() > 0:
        mare_sw = calc_mare(x_pred_sw[v_sw], x_exp[v_sw])
        ax.scatter(x_exp[v_sw], x_pred_sw[v_sw],
                   s=18, alpha=0.4, edgecolors='k', linewidths=0.2,
                   c=C_BASE, marker='s',
                   label=f'S&W original ({mare_sw:.1f}%, n={v_sw.sum()})',
                   zorder=4)

    vals_list = [x_exp[v_sech], x_pred_sech[v_sech]]
    if v_di.sum() > 0:
        vals_list += [x_exp[v_di], x_pred_di[v_di]]
    if has_sw and v_sw.sum() > 0:
        vals_list += [x_exp[v_sw], x_pred_sw[v_sw]]
    all_vals = np.concatenate(vals_list)
    if len(all_vals) > 0:
        lo = max(all_vals.min() * 0.5, 1e-8)
        hi = all_vals.max() * 2
    else:
        lo, hi = 1e-6, 1e-1
    line = np.array([lo, hi])
    ax.plot(line, line, color='0.45', linestyle='-', lw=1.2, zorder=10)
    ax.fill_between(line, line * 0.9, line * 1.1, alpha=0.10,
                     color=C_REC, zorder=1, label='$\\pm$10%')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_title(f'{display} — Brine', fontsize=10)
    math_sub = GAS_MATH_SUB.get(gas, gas)
    ax.set_xlabel(f'Experimental $x_{{{math_sub}}}$', fontsize=9)
    ax.set_ylabel(f'Predicted $x_{{{math_sub}}}$', fontsize=9)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    return True


def parity_overview(df, output_dir=OUTPUT_DIR):
    """Generate consolidated solubility parity plots (freshwater | brine).

    Page 1: CO2, H2S, N2
    Page 2: H2, CH4, C2H6, C3H8 (C3H8 freshwater only — appended as the
            final panel; nC4H10 has no data and is omitted entirely)

    Only panels with data are drawn; gases with no data at all (nC4H10) and
    empty brine panels are omitted rather than rendered as blank placeholders.
    """
    pages = [
        (['CO2', 'H2S', 'N2'], 'parity_overview_1'),
        (['H2', 'CH4', 'C2H6', 'C3H8'], 'parity_overview_2'),
    ]

    def _has_brine(gas):
        br = get_gas_data(df, gas, freshwater_only=False)
        br = br[br['Sal_m'] > 0] if len(br) > 0 else br
        return len(br) >= 5

    def _has_fresh(gas):
        return len(get_gas_data(df, gas, freshwater_only=True)) > 0

    figs = {}
    letter = 0
    for gases, label in pages:
        # Build the list of panels that actually carry data.
        specs = []
        for gas in gases:
            if _has_fresh(gas):
                specs.append((gas, 'fw'))
            if _has_brine(gas):
                specs.append((gas, 'br'))

        if not specs:
            continue

        n = len(specs)
        ncols = 1 if n == 1 else 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.2 * nrows),
                                 squeeze=False)
        axes = axes.flatten()

        for i, (gas, kind) in enumerate(specs):
            ax = axes[i]
            if kind == 'fw':
                _parity_panel_fw(ax, gas, df)
            else:
                _parity_panel_br(ax, gas, df)
            # Sub-panel labels only when there is more than one panel.
            if n > 1:
                ax.text(0.02, 0.98, f'({chr(ord("a") + letter)})',
                        transform=ax.transAxes, fontsize=11,
                        fontweight='bold', va='top')
            letter += 1

        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(h_pad=3.0, w_pad=2.0)

        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{label}.pdf'),
                    format='pdf', bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, f'{label}.png'),
                    dpi=300, bbox_inches='tight')
        print(f"  Saved {label}.pdf + .png")
        figs[label] = fig

    return figs


# =============================================================================
# APPENDIX B: K-VALUE INITIALIZATION PERFORMANCE
# =============================================================================
def wilson_k(T, P, Tc, Pc, omega):
    """Standard Wilson K-value correlation."""
    return (Pc / P) * np.exp(5.373 * (1 + omega) * (1 - Tc / T))


def figure_b1_wilson_violations(output_dir=OUTPUT_DIR):
    """Bar chart: Wilson K < 1 violation rates vs Cross-form (0% for all)."""

    T_grid = np.linspace(273.15, 422.15, 10)
    P_grid = np.geomspace(1e5, 1034e5, 15)
    n_cond = len(T_grid) * len(P_grid)

    wilson_rates = []
    cross_rates = []

    for gas in KVALUE_GASES:
        props = COMPONENTS[gas]
        Tc, Pc, omega = props.Tc, props.Pc, props.omega
        wil_violations = 0
        cross_violations = 0

        for T in T_grid:
            for P in P_grid:
                K_wil = wilson_k(T, P, Tc, Pc, omega)
                if K_wil < 1:
                    wil_violations += 1

                names = ['H2O', gas]
                Tc_arr = np.array([COMPONENTS['H2O'].Tc, Tc])
                Pc_arr = np.array([COMPONENTS['H2O'].Pc, Pc])
                om_arr = np.array([COMPONENTS['H2O'].omega, omega])
                K_arr = _sw_kvalue_init(names, Tc_arr, Pc_arr, om_arr, T, P)
                if K_arr[1] < 1:
                    cross_violations += 1

        wilson_rates.append(wil_violations / n_cond * 100)
        cross_rates.append(cross_violations / n_cond * 100)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(KVALUE_GASES))
    width = 0.35

    display_labels = [GAS_DISPLAY.get(g, g) for g in KVALUE_GASES]

    bars_w = ax.bar(x - width/2, wilson_rates, width, color='tab:red', alpha=0.8,
                    edgecolor='k', linewidth=0.5, label='Wilson')
    bars_c = ax.bar(x + width/2, cross_rates, width, color='tab:blue', alpha=0.8,
                    edgecolor='k', linewidth=0.5, label='Cross-form')

    for b, rate in zip(bars_w, wilson_rates):
        if rate > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel('Conditions with $K_{\\mathrm{gas}} < 1$ (%)')
    ax.set_title('K-Value Initialization: Wilson Violation Rates vs Cross-Form',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(wilson_rates) * 1.2 + 5)

    ax.text(0.98, 0.95,
            f'{len(T_grid)} T ' + u'\u00d7' + f' {len(P_grid)} P = {n_cond} conditions/gas\n'
            f'T: {T_grid[0]-273.15:.0f}' + u'\u2013' + f'{T_grid[-1]-273.15:.0f} \u00b0C, '
            f'P: {P_grid[0]/1e5:.0f}' + u'\u2013' + f'{P_grid[-1]/1e5:.0f} bar',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'kvalue_wilson_violations.pdf'),
                format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'kvalue_wilson_violations.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved kvalue_wilson_violations.pdf + .png")
    return fig


def figure_b2_iteration_comparison(output_dir=OUTPUT_DIR):
    """Bar chart: median iterations to converge with Wilson vs Cross-form init."""

    T_grid = np.linspace(280, 420, 10)
    P_grid = np.geomspace(10e5, 500e5, 5)

    results_wilson = {}
    results_cross = {}

    for gas in KVALUE_GASES:
        props = COMPONENTS[gas]
        flash = SWMultiComponentFlash(['H2O', gas], salinity_molal=0.0)
        kij_matrix = flash.build_kij_matrix(300, 'AQ')

        iters_wil = []
        iters_cross = []

        for T_K in T_grid:
            kij_matrix = flash.build_kij_matrix(T_K, 'AQ')
            for P_Pa in P_grid:
                z = np.array([0.5, 0.5])
                gamma = np.ones(2)

                K_wil = np.array([
                    wilson_k(T_K, P_Pa, COMPONENTS['H2O'].Tc,
                             COMPONENTS['H2O'].Pc, COMPONENTS['H2O'].omega),
                    wilson_k(T_K, P_Pa, props.Tc, props.Pc, props.omega)
                ])
                K_wil[0] = min(K_wil[0], 0.01)
                K_wil = np.clip(K_wil, 1e-10, 1e10)

                conv_w, n_w = _run_ss_flash(flash, T_K, P_Pa, z, kij_matrix,
                                            gamma, K_wil)
                iters_wil.append(n_w)

                K_cross = _sw_kvalue_init(
                    flash.names,
                    flash.Tc, flash.Pc, flash.omega,
                    T_K, P_Pa
                )
                K_cross[0] = min(K_cross[0], 0.01)

                conv_c, n_c = _run_ss_flash(flash, T_K, P_Pa, z, kij_matrix,
                                            gamma, K_cross)
                iters_cross.append(n_c)

        results_wilson[gas] = np.array(iters_wil)
        results_cross[gas] = np.array(iters_cross)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(KVALUE_GASES))
    width = 0.35

    display_labels = [GAS_DISPLAY.get(g, g) for g in KVALUE_GASES]

    med_wil = [np.median(results_wilson[g]) for g in KVALUE_GASES]
    med_cross = [np.median(results_cross[g]) for g in KVALUE_GASES]

    fail_wil = [np.sum(results_wilson[g] >= 200) for g in KVALUE_GASES]
    fail_cross = [np.sum(results_cross[g] >= 200) for g in KVALUE_GASES]

    bars_w = ax.bar(x - width/2, med_wil, width, color='tab:red', alpha=0.8,
                    edgecolor='k', linewidth=0.5, label='Wilson init')
    bars_c = ax.bar(x + width/2, med_cross, width, color='tab:blue', alpha=0.8,
                    edgecolor='k', linewidth=0.5, label='Cross-form init')

    for i, (b, f) in enumerate(zip(bars_w, fail_wil)):
        if f > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{f} fail', ha='center', va='bottom', fontsize=7,
                    color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel('Median iterations to convergence')
    ax.set_title('Flash Convergence: Wilson vs Cross-Form Initialization',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    n_cond = len(T_grid) * len(P_grid)
    ax.text(0.98, 0.95,
            f'{n_cond} conditions/gas (SS, tol=10$^{{-10}}$, max 200 iter)',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'kvalue_iteration_comparison.pdf'),
                format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'kvalue_iteration_comparison.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved kvalue_iteration_comparison.pdf + .png")
    return fig


def _run_ss_flash(flash_obj, T_K, P_Pa, z, kij_matrix, gamma, K_init,
                  max_iter=200, tol=1e-10):
    """Run successive substitution flash with custom K_init.

    Returns (converged: bool, iterations: int).
    """
    K = K_init.copy()
    for it in range(max_iter):
        V, x, y = solve_rachford_rice(z, K)
        x = np.clip(x, 1e-15, None)
        y = np.clip(y, 1e-15, None)
        x = x / x.sum()
        y = y / y.sum()

        phi_L = flash_obj.calc_fugacity_coefficients(T_K, P_Pa, x, kij_matrix, 'liquid')
        phi_V = flash_obj.calc_fugacity_coefficients(T_K, P_Pa, y, kij_matrix, 'vapor')

        K_new = np.clip(gamma * phi_L / (phi_V + 1e-30), 1e-10, 1e10)

        if np.max(np.abs(K_new / K - 1.0)) < tol:
            return True, it + 1

        damp = 0.7 if it < 20 else 0.9
        K = K * (K_new / K)**damp

    return False, max_iter


if __name__ == '__main__':
    figures = run_all_figures()
    plt.show()
