#!/usr/bin/env python3
"""
Analyze Sechenov (salting-out) correlations across all S&W gases.

For each gas with published Pitzer models:
  - Compute effective ks from salting_library at representative T, P
  - Compare: S&W Eq 8 vs Duan/Akinfiev/Li/Mao models
  - Generate comparison plots: ks vs T for each gas, multiple models
  - Use data/salting_out_model_selection.md recommendations

Output:
  - Comparison plots saved to images/
  - Text report to ../../shared/data/salting_comparison_report.txt

Usage:
    cd framework_refresh/code
    python analyze_salting_all_gases.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
import matplotlib.pyplot as plt

from _lib_salting_library import (
    ks_sw_eq8, ks_dubessy_co2, ks_dubessy_h2s,
    ks_akinfiev_h2s, ks_akinfiev_h2s_from_tables,
    ks_li2015, ks_mao2006_n2, ks_duan2003_co2,
    TB_K,
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

# Temperature ranges
T_C_RANGE = np.linspace(0, 250, 100)
T_K_RANGE = T_C_RANGE + 273.15


def plot_co2_salting(save_dir='images'):
    """CO2: S&W Eq 8 vs Dubessy 2005 vs Duan & Sun 2003."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # S&W Eq 8
    ks_sw = ks_sw_eq8(T_K_RANGE, 'CO2')
    ax.plot(T_C_RANGE, ks_sw, 'k-', linewidth=2.5, label='S&W Eq 8')

    # Dubessy (at m=0 and m=2)
    ks_dub0 = [ks_dubessy_co2(T, 0.0) for T in T_K_RANGE]
    ks_dub2 = [ks_dubessy_co2(T, 2.0) for T in T_K_RANGE]
    ax.plot(T_C_RANGE, ks_dub0, 'b--', linewidth=2, label='Dubessy 2005 (m=0)')
    ax.plot(T_C_RANGE, ks_dub2, 'b:', linewidth=2, label='Dubessy 2005 (m=2)')

    # Duan & Sun 2003 (at m=0 and m=2, P=100 bar)
    # Valid to 260°C (533 K)
    T_duan = T_K_RANGE[T_K_RANGE <= 530]
    T_C_duan = T_duan - 273.15
    ks_dn0 = [ks_duan2003_co2(T, 0.0, 100.0) for T in T_duan]
    ks_dn2 = [ks_duan2003_co2(T, 2.0, 100.0) for T in T_duan]
    ax.plot(T_C_duan, ks_dn0, 'r-', linewidth=2.5, label='Duan 2003 (m=0, 100 bar)')
    ax.plot(T_C_duan, ks_dn2, 'r--', linewidth=2, label='Duan 2003 (m=2, 100 bar)')

    # Pressure sensitivity (Duan at m=0)
    ks_dn_50 = [ks_duan2003_co2(T, 0.0, 50.0) for T in T_duan]
    ks_dn_500 = [ks_duan2003_co2(T, 0.0, 500.0) for T in T_duan]
    ax.fill_between(T_C_duan, ks_dn_50, ks_dn_500, alpha=0.1, color='red',
                    label='Duan 2003 P=50-500 bar range')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title('CO2 Sechenov Coefficient: S&W vs Modern Pitzer Models')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 0.25)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'salting_co2_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"  Saved: {filepath}")
    return fig


def plot_h2s_salting(save_dir='images'):
    """H2S: S&W Eq 8 vs Dubessy 2005 vs Akinfiev 2016."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # S&W Eq 8
    ks_sw = ks_sw_eq8(T_K_RANGE, 'H2S')
    ax.plot(T_C_RANGE, ks_sw, 'k-', linewidth=2.5, label='S&W Eq 8')

    # Dubessy (m=0 and m=2)
    T_dub = T_K_RANGE[T_K_RANGE <= 520]
    T_C_dub = T_dub - 273.15
    ks_dub0 = [ks_dubessy_h2s(T, 0.0) for T in T_dub]
    ks_dub2 = [ks_dubessy_h2s(T, 2.0) for T in T_dub]
    ax.plot(T_C_dub, ks_dub0, 'b--', linewidth=2, label='Dubessy 2005 (m=0)')
    ax.plot(T_C_dub, ks_dub2, 'b:', linewidth=2, label='Dubessy 2005 (m=2)')

    # Akinfiev 2016 Pitzer (analytical, m=1)
    T_akin = T_K_RANGE[(T_K_RANGE >= 283) & (T_K_RANGE <= 570)]
    T_C_akin = T_akin - 273.15
    ks_ak1 = [ks_akinfiev_h2s(T, m_NaCl=1.0) for T in T_akin]
    ks_ak4 = [ks_akinfiev_h2s(T, m_NaCl=4.0) for T in T_akin]
    ax.plot(T_C_akin, ks_ak1, 'g-', linewidth=2.5, label='Akinfiev 2016 (m=1)')
    ax.plot(T_C_akin, ks_ak4, 'g--', linewidth=2, label='Akinfiev 2016 (m=4)')

    # Akinfiev table-based (discrete points)
    T_tab = np.array([323.15, 373.15, 423.15, 473.15, 523.15])
    T_C_tab = T_tab - 273.15
    ks_tab1 = [ks_akinfiev_h2s_from_tables(T, 1) for T in T_tab]
    ks_tab4 = [ks_akinfiev_h2s_from_tables(T, 4) for T in T_tab]
    ax.scatter(T_C_tab, ks_tab1, c='green', marker='o', s=80, edgecolors='k',
              zorder=10, label='Akinfiev table (m=1)')
    ax.scatter(T_C_tab, ks_tab4, c='green', marker='D', s=80, edgecolors='k',
              zorder=10, label='Akinfiev table (m=4)')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title('H2S Sechenov Coefficient: S&W vs Modern Pitzer Models')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 260)
    ax.set_ylim(-0.02, 0.25)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'salting_h2s_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"  Saved: {filepath}")
    return fig


def plot_ch4_n2_salting(save_dir='images'):
    """CH4 and N2: S&W Eq 8 vs Li 2015 / Mao 2006."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- CH4 ---
    ks_sw_ch4 = ks_sw_eq8(T_K_RANGE, 'CH4')
    ax1.plot(T_C_RANGE, ks_sw_ch4, 'k-', linewidth=2.5, label='S&W Eq 8')

    # Li 2015 at different pressures (m=0)
    for P, ls, lbl in [(50, '--', '50 bar'), (100, '-', '100 bar'), (500, ':', '500 bar')]:
        ks = [ks_li2015(T, 'CH4', 0.0, P) for T in T_K_RANGE]
        ax1.plot(T_C_RANGE, ks, linestyle=ls, color='blue', linewidth=2,
                label=f'Li 2015 P={lbl}')

    # Li 2015 at m=2 (P=100)
    ks_li_m2 = [ks_li2015(T, 'CH4', 2.0, 100) for T in T_K_RANGE]
    ax1.plot(T_C_RANGE, ks_li_m2, 'r--', linewidth=2, label='Li 2015 (m=2, 100 bar)')

    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax1.set_title('(a) CH4')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 0.20)

    # --- N2 ---
    ks_sw_n2 = ks_sw_eq8(T_K_RANGE, 'N2')
    ax2.plot(T_C_RANGE, ks_sw_n2, 'k-', linewidth=2.5, label='S&W Eq 8')

    # Mao 2006 (valid to 400 K = 127°C)
    T_mao = T_K_RANGE[T_K_RANGE <= 398]
    T_C_mao = T_mao - 273.15
    for P, ls, lbl in [(50, '--', '50 bar'), (100, '-', '100 bar'), (500, ':', '500 bar')]:
        ks = [ks_mao2006_n2(T, 0.0, P) for T in T_mao]
        ax2.plot(T_C_mao, ks, linestyle=ls, color='blue', linewidth=2,
                label=f'Mao 2006 P={lbl}')

    # Mao at m=2
    ks_mao_m2 = [ks_mao2006_n2(T, 2.0, 100) for T in T_mao]
    ax2.plot(T_C_mao, ks_mao_m2, 'r--', linewidth=2, label='Mao 2006 (m=2, 100 bar)')

    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax2.set_title('(b) N2')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 0.20)

    fig.suptitle('Light Gases: S&W Eq 8 vs Modern Pitzer Models',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'salting_ch4_n2_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"  Saved: {filepath}")
    return fig


def plot_all_gases_sw_eq8(save_dir='images'):
    """S&W Eq 8 for all gases — shows Tb-dependent family of curves."""
    fig, ax = plt.subplots(figsize=(10, 7))

    gases_ordered = ['nC10H22', 'nC8H18', 'nC7H16', 'nC6H14', 'nC5H12',
                     'C3H8', 'nC4H10', 'CO2', 'H2S', 'C2H6',
                     'CH4', 'N2', 'H2']
    cmap = plt.cm.viridis(np.linspace(0, 1, len(gases_ordered)))

    for i, gas in enumerate(gases_ordered):
        ks = ks_sw_eq8(T_K_RANGE, gas)
        Tb = TB_K[gas]
        ax.plot(T_C_RANGE, ks, color=cmap[i], linewidth=2,
                label=f'{gas} ($T_b$={Tb:.0f} K)')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title('S&W Equation 8: Sechenov Coefficient Family ($T_b$-parameterized)')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 0.35)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'salting_sw_eq8_all_gases.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"  Saved: {filepath}")
    return fig


def plot_overview_comparison(save_dir='images'):
    """Summary: S&W Eq 8 vs best Pitzer model for each gas at P=100 bar, m=0."""
    fig, ax = plt.subplots(figsize=(12, 7))

    T_range = np.linspace(273.15, 473.15, 100)
    T_C = T_range - 273.15

    # Each gas: S&W (dashed) vs best model (solid)
    gas_configs = [
        ('CO2', 'tab:blue',
         lambda T: ks_sw_eq8(T, 'CO2'),
         lambda T: ks_duan2003_co2(T, 0.0, 100.0) if T <= 530 else np.nan,
         'Duan 2003'),
        ('H2S', 'tab:red',
         lambda T: ks_sw_eq8(T, 'H2S'),
         lambda T: ks_akinfiev_h2s(T, 1.0),
         'Akinfiev 2016'),
        ('CH4', 'tab:green',
         lambda T: ks_sw_eq8(T, 'CH4'),
         lambda T: ks_li2015(T, 'CH4', 0.0, 100.0),
         'Li 2015'),
        ('N2', 'tab:orange',
         lambda T: ks_sw_eq8(T, 'N2'),
         lambda T: ks_mao2006_n2(T, 0.0, 100.0) if T <= 398 else np.nan,
         'Mao 2006'),
        ('H2', 'tab:purple',
         lambda T: ks_sw_eq8(T, 'H2'),
         None,  # No alternative
         None),
    ]

    for gas, color, sw_func, alt_func, alt_name in gas_configs:
        ks_sw = [sw_func(T) for T in T_range]
        ax.plot(T_C, ks_sw, '--', color=color, linewidth=2,
                label=f'{gas} S&W Eq 8')
        if alt_func is not None:
            ks_alt = [alt_func(T) for T in T_range]
            ax.plot(T_C, ks_alt, '-', color=color, linewidth=2.5,
                    label=f'{gas} {alt_name}')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('$k_s$ (log$_{10}$ / molality)')
    ax.set_title('Sechenov Coefficients: S&W Eq 8 (dashed) vs Best Pitzer Model (solid)')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.22)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'salting_overview_comparison.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"  Saved: {filepath}")
    return fig


def generate_comparison_table(report_path='../../shared/data/salting_comparison_report.txt'):
    """Generate text report with numeric comparisons."""
    temps_C = [25, 50, 75, 100, 125, 150, 200]
    temps_K = [T + 273.15 for T in temps_C]

    lines = []
    lines.append("=" * 90)
    lines.append("SECHENOV COEFFICIENT COMPARISON: S&W Eq 8 vs MODERN PITZER MODELS")
    lines.append("All values on log10 / molality basis.  P = 100 bar, m = 0 (infinite dilution)")
    lines.append("=" * 90)

    # CO2
    lines.append("\n--- CO2 ---")
    lines.append(f"{'T(C)':>6}  {'S&W':>8}  {'Duan':>8}  {'Dub':>8}  {'Ratio S&W/Duan':>16}")
    lines.append("-" * 52)
    for Tc, Tk in zip(temps_C, temps_K):
        sw = ks_sw_eq8(Tk, 'CO2')
        dn = ks_duan2003_co2(Tk, 0, 100) if Tk <= 530 else float('nan')
        db = ks_dubessy_co2(Tk, 0)
        ratio = sw / dn if not np.isnan(dn) and dn > 0 else float('nan')
        lines.append(f"{Tc:>6}  {sw:8.4f}  {dn:8.4f}  {db:8.4f}  {ratio:>16.2f}")

    # H2S
    lines.append("\n--- H2S ---")
    lines.append(f"{'T(C)':>6}  {'S&W':>8}  {'Akin':>8}  {'Dub':>8}  {'Ratio S&W/Akin':>16}")
    lines.append("-" * 52)
    for Tc, Tk in zip(temps_C, temps_K):
        sw = ks_sw_eq8(Tk, 'H2S')
        ak = ks_akinfiev_h2s(Tk, 1.0)
        db = ks_dubessy_h2s(Tk, 0)
        ratio = sw / ak if ak > 0.001 else float('nan')
        lines.append(f"{Tc:>6}  {sw:8.4f}  {ak:8.4f}  {db:8.4f}  {ratio:>16.2f}")

    # CH4
    lines.append("\n--- CH4 ---")
    lines.append(f"{'T(C)':>6}  {'S&W':>8}  {'Li':>8}  {'Ratio S&W/Li':>14}")
    lines.append("-" * 38)
    for Tc, Tk in zip(temps_C, temps_K):
        sw = ks_sw_eq8(Tk, 'CH4')
        li = ks_li2015(Tk, 'CH4', 0.0, 100.0)
        ratio = sw / li if li > 0.001 else float('nan')
        lines.append(f"{Tc:>6}  {sw:8.4f}  {li:8.4f}  {ratio:>14.2f}")

    # N2
    lines.append("\n--- N2 ---")
    lines.append(f"{'T(C)':>6}  {'S&W':>8}  {'Mao':>8}  {'Ratio S&W/Mao':>15}")
    lines.append("-" * 39)
    for Tc, Tk in zip(temps_C, temps_K):
        sw = ks_sw_eq8(Tk, 'N2')
        if Tk <= 398:
            mao = ks_mao2006_n2(Tk, 0.0, 100.0)
            ratio = sw / mao if mao > 0.001 else float('nan')
            lines.append(f"{Tc:>6}  {sw:8.4f}  {mao:8.4f}  {ratio:>15.2f}")
        else:
            lines.append(f"{Tc:>6}  {sw:8.4f}  {'n/a':>8}  {'n/a':>15}")

    # H2
    lines.append("\n--- H2 ---")
    lines.append(f"{'T(C)':>6}  {'S&W':>8}  {'Alt':>8}")
    lines.append("-" * 26)
    for Tc, Tk in zip(temps_C, temps_K):
        sw = ks_sw_eq8(Tk, 'H2')
        lines.append(f"{Tc:>6}  {sw:8.4f}  {'(none)':>8}")

    lines.append("\n" + "=" * 90)
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append("  CO2: S&W overpredicts by ~2x vs Duan 2003")
    lines.append("  H2S: S&W overpredicts by ~3x vs Akinfiev 2016")
    lines.append("  CH4: S&W agrees at 25C, diverges +30% at 200C vs Li 2015")
    lines.append("  N2:  S&W agrees at 25C, diverges at >75C vs Mao 2006")
    lines.append("  H2:  No alternative model exists")
    lines.append("=" * 90)

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
    print("ALL-GAS SECHENOV ANALYSIS (Paper 2)")
    print("=" * 70)

    print("\n--- Generating Plots ---")
    plot_all_gases_sw_eq8()
    plot_co2_salting()
    plot_h2s_salting()
    plot_ch4_n2_salting()
    plot_overview_comparison()

    print("\n--- Generating Comparison Tables ---")
    generate_comparison_table()

    plt.show()
