#!/usr/bin/env python3
"""
Extract effective Sechenov coefficients (ks) from experimental solubility data.

Method:
  For each brine measurement at (T, P, m_NaCl, Source), find the best
  freshwater measurement at very similar T and P.  Then:

    ks = log10(x_fresh / x_brine) / m_NaCl

  Only matched pairs where:
    - Same source
    - Same gas
    - |Delta T| <= 3 K
    - |Delta P / P| <= 10%  (relative pressure tolerance)
  are included.

  If multiple freshwater points match, use the one with smallest |Delta T|.

Outputs:
  - CSV of extracted ks values
  - Per-gas ks vs T plots with model overlays (S&W Eq 8, Duan, Akinfiev, etc.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib_salting_library import (
    ks_sw_eq8,
    ks_dubessy_co2, ks_dubessy_h2s,
    ks_akinfiev_h2s,
    ks_li2015,
    ks_mao2006_n2,
    ks_duan2003_co2,
)

# ── Plot defaults ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = '../manuscript/figures'
XLSX_PATH = '../../shared/data/solubility_points.xlsx'

# Matching tolerances
T_TOL_K = 3.0       # K
P_REL_TOL = 0.10    # 10% relative

# Gas columns in the xlsx
GAS_COLS = {
    'CO2': 'x_CO2',
    'H2S': 'x_H2S',
    'N2':  'x_N2',
    'H2':  'x_H2',
    'CH4': 'x_CH4',
    'C2H6': 'x_C2H6',
    'C3H8': 'x_C3H8',
}


def extract_ks():
    """Extract ks values from matched fresh/brine pairs."""

    df = pd.read_excel(XLSX_PATH)
    df['Sal_m'] = pd.to_numeric(df['Sal_m'], errors='coerce')
    df['T_K'] = pd.to_numeric(df['T_K'], errors='coerce')
    df['P_bar'] = pd.to_numeric(df['P_bar'], errors='coerce')

    results = []

    for gas, xcol in GAS_COLS.items():
        df[xcol] = pd.to_numeric(df[xcol], errors='coerce')

        # Fresh water measurements for this gas
        fresh = df[(df['Sal_m'] == 0) & (df[xcol].notna()) & (df[xcol] > 0)].copy()
        # Brine measurements for this gas
        brine = df[(df['Sal_m'] > 0) & (df[xcol].notna()) & (df[xcol] > 0)].copy()

        if len(fresh) == 0 or len(brine) == 0:
            continue

        n_matched = 0
        for _, br in brine.iterrows():
            T_br = br['T_K']
            P_br = br['P_bar']
            m_br = br['Sal_m']
            x_br = br[xcol]
            src_br = br['Source']

            # Find fresh matches from same source
            candidates = fresh[
                (fresh['Source'] == src_br)
                & ((fresh['T_K'] - T_br).abs() <= T_TOL_K)
                & (((fresh['P_bar'] - P_br) / P_br).abs() <= P_REL_TOL)
            ]

            if len(candidates) == 0:
                continue

            # Pick closest in T
            idx_best = (candidates['T_K'] - T_br).abs().idxmin()
            fw = candidates.loc[idx_best]

            x_fw = fw[xcol]
            T_fw = fw['T_K']
            P_fw = fw['P_bar']

            # Compute ks
            if x_fw > 0 and x_br > 0 and m_br > 0:
                ks = np.log10(x_fw / x_br) / m_br
                results.append({
                    'Gas': gas,
                    'Source': src_br,
                    'T_K': T_br,
                    'T_C': T_br - 273.15,
                    'P_bar': P_br,
                    'Sal_m': m_br,
                    'x_fresh': x_fw,
                    'x_brine': x_br,
                    'T_fresh': T_fw,
                    'P_fresh': P_fw,
                    'dT_K': abs(T_fw - T_br),
                    'dP_pct': abs(P_fw - P_br) / P_br * 100,
                    'ks_data': ks,
                })
                n_matched += 1

        if n_matched > 0:
            print(f"  {gas}: {n_matched} matched pairs from {len(brine)} brine points "
                  f"({n_matched/len(brine)*100:.0f}%)")
        else:
            print(f"  {gas}: 0 matched pairs from {len(brine)} brine points")

    return pd.DataFrame(results)


# ── Model overlays per gas ───────────────────────────────────────────────────
# Each entry: (label, function(T_K, m), color, linestyle, valid_range_K)
# The function signature is f(T_K, m_NaCl) -> ks
# For pressure-dependent models, use P=100 bar as representative

MODELS = {
    'CO2': [
        ('S&W Eq 8', lambda T, m: ks_sw_eq8(T, 'CO2'),
         'black', '-', (273, 573)),
        ('Duan & Sun 2003', lambda T, m: ks_duan2003_co2(T, m, P_bar=100),
         'tab:blue', '--', (273, 533)),
        ('Dubessy 2005', lambda T, m: ks_dubessy_co2(T, m),
         'tab:green', '-.', (273, 543)),
    ],
    'H2S': [
        ('S&W Eq 8', lambda T, m: ks_sw_eq8(T, 'H2S'),
         'black', '-', (273, 573)),
        ('Akinfiev 2016', lambda T, m: ks_akinfiev_h2s(T, m, m_h2s_approx=0.1),
         'tab:blue', '--', (283, 573)),
        ('Dubessy 2005', lambda T, m: ks_dubessy_h2s(T, m),
         'tab:green', '-.', (273, 523)),
    ],
    'N2': [
        ('S&W Eq 8', lambda T, m: ks_sw_eq8(T, 'N2'),
         'black', '-', (273, 573)),
        ('Mao & Duan 2006', lambda T, m: ks_mao2006_n2(T, m, P_bar=100),
         'tab:blue', '--', (273, 400)),
    ],
    'H2': [
        ('S&W Eq 8', lambda T, m: ks_sw_eq8(T, 'H2'),
         'black', '-', (273, 573)),
    ],
    'CH4': [
        ('S&W Eq 8', lambda T, m: ks_sw_eq8(T, 'CH4'),
         'black', '-', (273, 573)),
        ('Li et al. 2015', lambda T, m: ks_li2015(T, 'CH4', m, P_bar=100),
         'tab:blue', '--', (273, 473)),
    ],
    'C2H6': [
        ('S&W Eq 8', lambda T, m: ks_sw_eq8(T, 'C2H6'),
         'black', '-', (273, 573)),
        ('Li et al. 2015', lambda T, m: ks_li2015(T, 'C2H6', m, P_bar=100),
         'tab:blue', '--', (273, 473)),
    ],
    'C3H8': [
        ('S&W Eq 8', lambda T, m: ks_sw_eq8(T, 'C3H8'),
         'black', '-', (273, 573)),
        ('Li et al. 2015', lambda T, m: ks_li2015(T, 'C3H8', m, P_bar=100),
         'tab:blue', '--', (273, 473)),
    ],
}


GAS_DISPLAY = {
    'CO2': 'CO$_2$', 'H2S': 'H$_2$S', 'CH4': 'CH$_4$', 'N2': 'N$_2$',
    'H2': 'H$_2$',
    'C2H6': 'C$_2$H$_6$', 'C3H8': 'C$_3$H$_8$',
}


MARKERS = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*']

# Representative m values for model overlays (low, medium, high)
MODEL_M_VALUES = [1.0, 3.0]
MODEL_M_STYLES = {1.0: (2.0, 0.9), 3.0: (1.5, 0.6)}  # (linewidth, alpha)


def _salinity_bucket(m):
    """Round salinity to nearest 0.5 for grouping in legend."""
    return round(m * 2) / 2


def plot_ks_per_gas(df_ks, output_dir=OUTPUT_DIR):
    """Create per-gas ks vs T plots with model overlays."""

    os.makedirs(output_dir, exist_ok=True)

    gases_with_data = sorted(df_ks['Gas'].unique())
    if len(gases_with_data) == 0:
        print("  No gases with extracted ks data.")
        return

    for gas in gases_with_data:
        gdata = df_ks[df_ks['Gas'] == gas].copy()
        display_name = GAS_DISPLAY.get(gas, gas)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

        # ── Group by Source + salinity bucket for scatter ─────────────
        gdata['sal_bucket'] = gdata['Sal_m'].apply(_salinity_bucket)
        all_sources = sorted(gdata['Source'].unique())

        # Build colour map: one colour per source
        src_colors = {}
        for i, src in enumerate(all_sources):
            src_colors[src] = plt.cm.tab10.colors[i % 10]

        # Group by (source, sal_bucket) for legend entries
        groups = gdata.groupby(['Source', 'sal_bucket'], sort=True)
        marker_idx = 0
        for (src, sal_b), grp in groups:
            c = src_colors[src]
            m = MARKERS[marker_idx % len(MARKERS)]
            marker_idx += 1

            # Scale marker size with salinity (bigger = more saline)
            ms = 30 + 10 * sal_b

            ax.scatter(
                grp['T_C'], grp['ks_data'],
                c=[c], marker=m, s=ms, edgecolors='k', linewidths=0.3,
                alpha=0.8,
                label=f'{src} (m={sal_b:.1f}, n={len(grp)})',
                zorder=5,
            )

        # ── Temperature array for model overlays ─────────────────────
        T_lo = max(gdata['T_K'].min() - 20, 273.15)
        T_hi = min(gdata['T_K'].max() + 20, 573.15)
        T_arr = np.linspace(T_lo, T_hi, 200)

        # ── Model overlays ───────────────────────────────────────────
        if gas in MODELS:
            for label, func, color, ls, (T_valid_lo, T_valid_hi) in MODELS[gas]:
                T_plot = T_arr[(T_arr >= T_valid_lo) & (T_arr <= T_valid_hi)]
                if len(T_plot) == 0:
                    continue

                if 'S&W' in label:
                    # S&W Eq 8 is m-independent — single line
                    try:
                        ks_model = np.array([func(T, 0) for T in T_plot])
                        ax.plot(T_plot - 273.15, ks_model, color=color,
                                linestyle=ls, linewidth=2.5,
                                label=label, zorder=10)
                    except Exception as e:
                        print(f"    Warning: {label} failed for {gas}: {e}")
                else:
                    # m-dependent — show curves at representative salinities
                    for m_val in MODEL_M_VALUES:
                        lw, alpha = MODEL_M_STYLES[m_val]
                        try:
                            ks_model = np.array([func(T, m_val) for T in T_plot])
                            ax.plot(T_plot - 273.15, ks_model, color=color,
                                    linestyle=ls, linewidth=lw, alpha=alpha,
                                    label=f'{label} (m={m_val:.0f})', zorder=9)
                        except Exception as e:
                            print(f"    Warning: {label} m={m_val} failed: {e}")

        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('$k_s$ (log$_{10}$ basis, molality scale)')
        ax.set_title(f'{display_name} Sechenov Coefficient  (n = {len(gdata)})',
                     fontsize=14, fontweight='bold')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc='best', framealpha=0.9)

        plt.tight_layout()

        stem = f'ks_{gas}'
        pdf_path = os.path.join(output_dir, f'{stem}.pdf')
        png_path = os.path.join(output_dir, f'{stem}.png')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  {gas}: saved {png_path}")

        plt.close(fig)

    return


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Extract Sechenov coefficients from experimental data")
    print("=" * 60)
    print(f"\nMatching tolerances: dT <= {T_TOL_K} K, dP/P <= {P_REL_TOL*100:.0f}%")
    print()

    df_ks = extract_ks()

    if len(df_ks) == 0:
        print("\nNo matched pairs found. Check tolerances or data.")
    else:
        # Save CSV
        csv_path = '../../shared/data/extracted_ks_values.csv'
        df_ks.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\nSaved {len(df_ks)} ks values to {csv_path}")

        # Summary
        print("\n--- Summary ---")
        for gas in sorted(df_ks['Gas'].unique()):
            gd = df_ks[df_ks['Gas'] == gas]
            print(f"  {gas}: n={len(gd)}, ks range=[{gd['ks_data'].min():.4f}, "
                  f"{gd['ks_data'].max():.4f}], median={gd['ks_data'].median():.4f}, "
                  f"sources={sorted(gd['Source'].unique())}")

        # Plot
        print("\n--- Generating plots ---")
        plot_ks_per_gas(df_ks)

    print("\nDone.")
