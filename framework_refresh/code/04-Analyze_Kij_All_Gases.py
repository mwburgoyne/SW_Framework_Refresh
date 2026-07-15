#!/usr/bin/env python3
"""
Analyze pointwise kij regression results across all S&W gases.

For each gas (CO2, H2S, CH4, N2, H2):
  - Load pointwise kij results from CSV
  - Plot kij_AQ vs Tr with S&W correlation overlay
  - Plot kij_NA vs T with S&W values/correlations
  - Compute MARE of S&W correlation vs regressed kij values
  - Identify systematic biases (under/over-prediction regions)
  - For H2: show rational form vs S&W-style polynomial

Output:
  - Per-gas summary statistics
  - Comparison plots saved to images/
  - Text report to ../../shared/data/kij_all_gases_report.txt

Usage:
    cd framework_refresh/code
    python analyze_kij_all_gases.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from _lib_vle_engine import (
    COMPONENTS, get_kij_aq, get_kij_na,
    kij_aq_h2, kij_aq_co2, kij_aq_n2, kij_aq_h2s, kij_aq_ch4,
    kij_na_h2s_sw_eq17, KIJ_NA,
)

# Plot defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Gases to analyze (must have data in pointwise_kij_results.csv)
GASES = ['CO2', 'H2S', 'CH4', 'N2', 'H2']

# S&W kij_AQ correlation functions (freshwater)
KIJ_AQ_SW_FUNCS = {
    'CO2': lambda T: kij_aq_co2(T, 0.0),
    'H2S': lambda T: kij_aq_h2s(T, 0.0),
    'CH4': lambda T: kij_aq_ch4(T, 0.0),
    'N2':  lambda T: kij_aq_n2(T, 0.0),
    'H2':  lambda T: kij_aq_h2(T, 0.0),
}

# S&W kij_NA values/functions
def get_sw_kij_na(gas, T_K):
    """Get S&W kij_NA for a gas at given T."""
    if gas == 'H2S':
        return kij_na_h2s_sw_eq17(T_K)
    return get_kij_na(gas, T_K)


def load_pointwise_results(csv_path='../../shared/data/pointwise_kij_results.csv'):
    """Load pointwise kij regression results."""
    df = pd.read_csv(csv_path)
    # Normalize source names
    if 'Source' in df.columns:
        df['Source'] = df['Source'].str.replace('Chahab', 'Chabab', regex=False)
    print(f"Loaded {len(df)} total results from {csv_path}")

    # Summary by gas
    for gas in GASES:
        mask = df['Gas'] == gas
        n_total = mask.sum()
        n_aq = ((mask) & (df['kij_AQ'].notna()) & (df['kij_AQ_conv'] == True)).sum()
        n_na = ((mask) & (df['kij_NA'].notna()) & (df['kij_NA_conv'] == True)).sum()
        print(f"  {gas:5s}: {n_total:4d} total, {n_aq:4d} kij_AQ, {n_na:4d} kij_NA converged")
    return df


def compute_kij_aq_statistics(df, gas):
    """Compute MARE and bias statistics for kij_AQ S&W correlation vs regressed values."""
    mask = (df['Gas'] == gas) & (df['kij_AQ'].notna()) & (df['kij_AQ_conv'] == True)
    data = df[mask].copy()
    if len(data) == 0:
        return None

    kij_func = KIJ_AQ_SW_FUNCS.get(gas)
    if kij_func is None:
        return None

    data['kij_AQ_sw'] = data['T_K'].apply(kij_func)
    data['error'] = data['kij_AQ'] - data['kij_AQ_sw']
    data['abs_error'] = data['error'].abs()

    # For MARE, use relative error based on absolute kij values
    # Avoid division by zero for kij near zero
    kij_scale = data['kij_AQ'].abs().clip(lower=0.01)
    data['rel_error_pct'] = (data['error'] / kij_scale * 100)

    stats = {
        'gas': gas,
        'n_points': len(data),
        'mae_kij': data['abs_error'].mean(),
        'max_error': data['abs_error'].max(),
        'mean_bias': data['error'].mean(),
        'std_error': data['error'].std(),
        'mare_pct': data['abs_error'].mean() / kij_scale.mean() * 100,
        'T_range': (data['T_K'].min(), data['T_K'].max()),
        'P_range': (data['P_bar'].min(), data['P_bar'].max()),
    }

    # Check for systematic bias in temperature ranges
    T_mid = (data['T_K'].min() + data['T_K'].max()) / 2
    low_T = data[data['T_K'] < T_mid]
    high_T = data[data['T_K'] >= T_mid]

    if len(low_T) > 5:
        stats['bias_low_T'] = low_T['error'].mean()
    if len(high_T) > 5:
        stats['bias_high_T'] = high_T['error'].mean()

    return stats


def compute_kij_na_statistics(df, gas):
    """Compute statistics for kij_NA S&W values vs regressed values."""
    mask = (df['Gas'] == gas) & (df['kij_NA'].notna()) & (df['kij_NA_conv'] == True)
    data = df[mask].copy()
    if len(data) == 0:
        return None

    data['kij_NA_sw'] = data['T_K'].apply(lambda T: get_sw_kij_na(gas, T))
    data['error'] = data['kij_NA'] - data['kij_NA_sw']
    data['abs_error'] = data['error'].abs()

    stats = {
        'gas': gas,
        'n_points': len(data),
        'mae_kij': data['abs_error'].mean(),
        'max_error': data['abs_error'].max(),
        'mean_bias': data['error'].mean(),
        'std_error': data['error'].std(),
        'T_range': (data['T_K'].min(), data['T_K'].max()),
    }
    return stats


def plot_kij_aq_all_gases(df, save_dir='images'):
    """Create multi-panel kij_AQ comparison plot for all gases."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    color_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*']

    for idx, gas in enumerate(GASES):
        ax = axes[idx]
        mask = (df['Gas'] == gas) & (df['kij_AQ'].notna()) & (df['kij_AQ_conv'] == True)
        data = df[mask]

        if len(data) == 0:
            ax.text(0.5, 0.5, f'No kij_AQ data for {gas}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gas)
            continue

        # Group by source
        sources = sorted(data['Source'].unique())
        for si, source in enumerate(sources):
            src_data = data[data['Source'] == source]
            Tr = src_data['T_K'] / COMPONENTS[gas].Tc
            label = source if len(source) <= 20 else source[:17] + '...'
            ax.scatter(Tr, src_data['kij_AQ'],
                      c=[color_palette[si % len(color_palette)]],
                      marker=markers[si % len(markers)],
                      s=50, edgecolors='k', linewidths=0.3, alpha=0.8,
                      label=f'{label} ({len(src_data)})', zorder=5)

        # S&W correlation overlay
        T_min, T_max = data['T_K'].min(), data['T_K'].max()
        T_range = np.linspace(T_min - 10, T_max + 10, 100)
        Tr_range = T_range / COMPONENTS[gas].Tc
        kij_sw = [KIJ_AQ_SW_FUNCS[gas](T) for T in T_range]
        ax.plot(Tr_range, kij_sw, 'k-', linewidth=2.5,
                label='S&W correlation', zorder=10)

        ax.set_xlabel(f'$T_r = T/T_{{c,{gas}}}$')
        ax.set_ylabel('$k_{ij}^{AQ}$')
        ax.set_title(f'{gas} ({len(data)} pts)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1, loc='best')

    # Turn off unused subplot
    if len(GASES) < len(axes):
        for i in range(len(GASES), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle('Aqueous Phase BIP: Regressed vs S&W Correlation (All Gases)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_dir, 'kij_aq_all_gases_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")

    filepath_pdf = os.path.join(save_dir, 'kij_aq_all_gases_comparison.pdf')
    fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
    return fig


def plot_kij_na_all_gases(df, save_dir='images'):
    """Create multi-panel kij_NA comparison plot for all gases."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    color_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*']

    for idx, gas in enumerate(GASES):
        ax = axes[idx]
        mask = (df['Gas'] == gas) & (df['kij_NA'].notna()) & (df['kij_NA_conv'] == True)
        data = df[mask]

        if len(data) == 0:
            ax.text(0.5, 0.5, f'No kij_NA data for {gas}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gas)
            continue

        # Group by source
        sources = sorted(data['Source'].unique())
        for si, source in enumerate(sources):
            src_data = data[data['Source'] == source]
            T_C = src_data['T_K'] - 273.15
            label = source if len(source) <= 20 else source[:17] + '...'
            ax.scatter(T_C, src_data['kij_NA'],
                      c=[color_palette[si % len(color_palette)]],
                      marker=markers[si % len(markers)],
                      s=50, edgecolors='k', linewidths=0.3, alpha=0.8,
                      label=f'{label} ({len(src_data)})', zorder=5)

        # S&W kij_NA overlay
        T_min, T_max = data['T_K'].min(), data['T_K'].max()
        T_range = np.linspace(T_min - 10, T_max + 10, 100)
        kij_na_sw = [get_sw_kij_na(gas, T) for T in T_range]
        ax.plot(T_range - 273.15, kij_na_sw, 'k-', linewidth=2.5,
                label='S&W value', zorder=10)

        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('$k_{ij}^{NA}$')
        ax.set_title(f'{gas} ({len(data)} pts)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1, loc='best')

    # Turn off unused subplot
    if len(GASES) < len(axes):
        for i in range(len(GASES), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle('Non-Aqueous Phase BIP: Regressed vs S&W Values (All Gases)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_dir, 'kij_na_all_gases_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")

    filepath_pdf = os.path.join(save_dir, 'kij_na_all_gases_comparison.pdf')
    fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
    return fig


def plot_sw_performance_summary(aq_stats, na_stats, save_dir='images'):
    """Bar chart of S&W correlation performance (MAE) by gas."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # kij_AQ MAE
    gases_aq = [s['gas'] for s in aq_stats if s is not None]
    mae_aq = [s['mae_kij'] for s in aq_stats if s is not None]
    n_aq = [s['n_points'] for s in aq_stats if s is not None]
    bias_aq = [s['mean_bias'] for s in aq_stats if s is not None]

    x = np.arange(len(gases_aq))
    bars = ax1.bar(x, mae_aq, color='steelblue', edgecolor='k', alpha=0.8)
    for i, (bar, n) in enumerate(zip(bars, n_aq)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'n={n}', ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gases_aq)
    ax1.set_ylabel('MAE($k_{ij}^{AQ}$)')
    ax1.set_title('S&W Aqueous BIP Accuracy')
    ax1.grid(True, alpha=0.3, axis='y')

    # kij_NA MAE
    gases_na = [s['gas'] for s in na_stats if s is not None]
    mae_na = [s['mae_kij'] for s in na_stats if s is not None]
    n_na = [s['n_points'] for s in na_stats if s is not None]

    x2 = np.arange(len(gases_na))
    bars2 = ax2.bar(x2, mae_na, color='darkorange', edgecolor='k', alpha=0.8)
    for i, (bar, n) in enumerate(zip(bars2, n_na)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'n={n}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(gases_na)
    ax2.set_ylabel('MAE($k_{ij}^{NA}$)')
    ax2.set_title('S&W Non-Aqueous BIP Accuracy')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('S&W 1992 BIP Correlation Performance by Gas',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_dir, 'sw_kij_performance_summary.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    return fig


def generate_report(aq_stats, na_stats, report_path='../../shared/data/kij_all_gases_report.txt'):
    """Generate text report of kij analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("S&W FRAMEWORK: kij CORRELATION PERFORMANCE BY GAS")
    lines.append("=" * 80)
    lines.append("")

    lines.append("AQUEOUS PHASE BIP (kij_AQ)")
    lines.append("-" * 60)
    lines.append(f"{'Gas':>6} {'N':>6} {'MAE':>10} {'Max Err':>10} {'Bias':>10} {'Std':>10} {'T Range (K)':>20}")
    lines.append("-" * 80)
    for s in aq_stats:
        if s is None:
            continue
        T_lo, T_hi = s['T_range']
        lines.append(f"{s['gas']:>6} {s['n_points']:>6} {s['mae_kij']:>10.4f} "
                     f"{s['max_error']:>10.4f} {s['mean_bias']:>+10.4f} "
                     f"{s['std_error']:>10.4f} {T_lo:>8.0f}-{T_hi:.0f}")
    lines.append("")

    lines.append("NON-AQUEOUS PHASE BIP (kij_NA)")
    lines.append("-" * 60)
    lines.append(f"{'Gas':>6} {'N':>6} {'MAE':>10} {'Max Err':>10} {'Bias':>10} {'Std':>10}")
    lines.append("-" * 60)
    for s in na_stats:
        if s is None:
            continue
        lines.append(f"{s['gas']:>6} {s['n_points']:>6} {s['mae_kij']:>10.4f} "
                     f"{s['max_error']:>10.4f} {s['mean_bias']:>+10.4f} "
                     f"{s['std_error']:>10.4f}")
    lines.append("")

    lines.append("SYSTEMATIC BIAS ANALYSIS (kij_AQ)")
    lines.append("-" * 60)
    for s in aq_stats:
        if s is None:
            continue
        bias_lo = s.get('bias_low_T', None)
        bias_hi = s.get('bias_high_T', None)
        if bias_lo is not None and bias_hi is not None:
            direction = "over" if bias_lo > 0 else "under"
            lines.append(f"  {s['gas']}: Low-T bias = {bias_lo:+.4f} ({direction}-predicts), "
                        f"High-T bias = {bias_hi:+.4f}")
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    print(report_text)


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("ALL-GAS kij ANALYSIS (Paper 2)")
    print("=" * 70)

    # Load data
    df = load_pointwise_results()

    # Compute statistics
    print("\n--- kij_AQ Statistics ---")
    aq_stats = []
    for gas in GASES:
        stats = compute_kij_aq_statistics(df, gas)
        aq_stats.append(stats)
        if stats:
            print(f"  {gas}: MAE={stats['mae_kij']:.4f}, "
                  f"bias={stats['mean_bias']:+.4f}, n={stats['n_points']}")

    print("\n--- kij_NA Statistics ---")
    na_stats = []
    for gas in GASES:
        stats = compute_kij_na_statistics(df, gas)
        na_stats.append(stats)
        if stats:
            print(f"  {gas}: MAE={stats['mae_kij']:.4f}, "
                  f"bias={stats['mean_bias']:+.4f}, n={stats['n_points']}")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_kij_aq_all_gases(df)
    plot_kij_na_all_gases(df)
    plot_sw_performance_summary(aq_stats, na_stats)

    # Generate report
    generate_report(aq_stats, na_stats)

    plt.show()
