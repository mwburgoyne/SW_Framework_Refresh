"""
MARE/MAE Validation Script
==========================
Validates fit quality by calculating error metrics for solubility predictions:

MARE = Mean Absolute Relative Error = (1/n) * Σ|x_pred - x_exp| / x_exp * 100  [%]
MAE  = Mean Absolute Error = (1/n) * Σ|x_pred - x_exp|  [dimensionless]

Uses validated VLE class from vle_engine for accurate calculations.

Compares multiple correlations:
- This work (rational form)
- Chabab 2023
- Lopez-Lazaro 2019
- Linear fit

Output:
- Console output with summary tables
- Text report file: ../../shared/data/error_validation_report.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import pandas as pd
import numpy as np
from datetime import datetime

# Import validated VLE class and correlations from vle_engine
from _lib_vle_engine import (
    H2WaterVLE, SWBinaryVLE, COMPONENTS,
    kij_aq_rational,
    kij_aq_chabab_2023,
    kij_aq_lopez_lazaro_2019,
    kij_aq_linear,
    kij_na_chabab_2023,
    kij_na_lopez_lazaro_2019
)

TC_H2 = COMPONENTS['H2'].Tc  # 33.145


def calc_y_H2O_predicted(T_K, P_Pa, kij_NA=0.468):
    """Calculate predicted water content using SWBinaryVLE."""
    vle = SWBinaryVLE('H2', salinity_molal=0.0)
    return vle.calc_water_content_with_kij(T_K, P_Pa, kij_NA)

# =============================================================================
# DATA CLASSIFICATION
# =============================================================================
FITTING_SOURCES = ['Wiebe 1934', 'Wiebe 1932', 'Chabab 2023', 'Chahab 2023',
                   'Torín-Ollarves 2021', 'Torin-Ollarves 2021']

def is_fitting_source(source):
    return any(fit in source for fit in FITTING_SOURCES)

# =============================================================================
# REPORT WRITER
# =============================================================================
class ReportWriter:
    """Writes output to both console and a report file."""
    def __init__(self, report_path):
        self.report_path = report_path
        self.lines = []

    def write(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self):
        with open(self.report_path, 'w') as f:
            f.write('\n'.join(self.lines))
        print(f"\nReport saved to: {self.report_path}")

# =============================================================================
# MAIN VALIDATION
# =============================================================================
def run_mare_validation(csv_path='../../shared/data/pointwise_kij_results.csv',
                        report_path='../../shared/data/error_validation_report.txt'):
    """Run comprehensive MARE/MAE validation for all correlations"""

    report = ReportWriter(report_path)

    report.write("=" * 80)
    report.write("MARE/MAE VALIDATION REPORT")
    report.write("=" * 80)
    report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.write(f"Data source: {csv_path}")
    report.write()
    report.write("Metrics:")
    report.write("  MARE = Mean Absolute Relative Error = (1/n) × Σ|x_pred - x_exp|/x_exp × 100  [%]")
    report.write("  MAE  = Mean Absolute Error = (1/n) × Σ|x_pred - x_exp|  [dimensionless]")
    report.write()

    df = pd.read_csv(csv_path)
    df_h2 = df[df['Gas'] == 'H2'].copy()

    # =========================================================================
    # AQUEOUS PHASE: KIJ VALIDATION
    # =========================================================================
    report.write("=" * 80)
    report.write("AQUEOUS PHASE - KIJ ERRORS")
    report.write("(Comparing correlation kij to point-regressed kij)")
    report.write("=" * 80)

    mask_kij = df_h2['kij_AQ'].notna()
    df_kij = df_h2[mask_kij].copy()
    df_kij = df_kij[df_kij['Source'].apply(is_fitting_source)]
    # Exclude T-O 423K
    df_kij = df_kij[~((df_kij['Source'].str.contains('Tor')) & (abs(df_kij['T_K'] - 423.15) < 1))]

    kij_correlations = {
        'This work': kij_aq_rational,
        'Chabab 2023': lambda T: kij_aq_chabab_2023(T, m=0),
        'Lopez-Lazaro 2019': lambda T: kij_aq_lopez_lazaro_2019(T, csw=0),
        'Linear': kij_aq_linear,
    }

    report.write()
    report.write(f"{'Correlation':<25} {'MARE (%)':<12} {'MAE':<12} {'n':<6}")
    report.write("-" * 55)

    kij_summary = {}
    for name, kij_func in kij_correlations.items():
        rel_errors = []
        abs_errors = []
        for _, row in df_kij.iterrows():
            T_K = row['T_K']
            kij_exp = row['kij_AQ']
            kij_pred = kij_func(T_K)

            if abs(kij_exp) > 0.01:  # Avoid division by near-zero
                rel_errors.append(abs((kij_pred - kij_exp) / kij_exp) * 100)
            abs_errors.append(abs(kij_pred - kij_exp))

        mare = np.mean(rel_errors) if rel_errors else 0
        mae = np.mean(abs_errors) if abs_errors else 0
        report.write(f"{name:<25} {mare:<12.2f} {mae:<12.4f} {len(abs_errors):<6}")
        kij_summary[name] = {'MARE': mare, 'MAE': mae, 'n': len(abs_errors)}

    # =========================================================================
    # AQUEOUS PHASE: SOLUBILITY VALIDATION
    # =========================================================================
    report.write()
    report.write("=" * 80)
    report.write("AQUEOUS PHASE - SOLUBILITY ERRORS (x_H2)")
    report.write("(Using validated VLE calculations)")
    report.write("=" * 80)

    mask_x = df_h2['x_gas_exp'].notna()
    df_aq = df_h2[mask_x].copy()
    df_aq = df_aq[df_aq['Source'].apply(is_fitting_source)]
    # Exclude T-O 423K
    df_aq = df_aq[~((df_aq['Source'].str.contains('Tor')) & (abs(df_aq['T_K'] - 423.15) < 1))]

    vle = H2WaterVLE(salinity=0.0)

    report.write()
    report.write(f"{'Correlation':<25} {'MARE (%)':<12} {'MAE (x1e-4)':<15} {'n':<6}")
    report.write("-" * 60)

    sol_summary = {}
    for name, kij_func in kij_correlations.items():
        rel_errors = []
        abs_errors = []

        for _, row in df_aq.iterrows():
            T_K = row['T_K']
            P_Pa = row['P_bar'] * 1e5
            x_exp = row['x_gas_exp']

            kij = kij_func(T_K)
            x_pred = vle.calc_x_H2(T_K, P_Pa, kij)

            if x_pred is not None and x_exp > 0:
                rel_errors.append(abs((x_pred - x_exp) / x_exp) * 100)
                abs_errors.append(abs(x_pred - x_exp))

        mare = np.mean(rel_errors) if rel_errors else 0
        mae = np.mean(abs_errors) if abs_errors else 0
        report.write(f"{name:<25} {mare:<12.2f} {mae*1e4:<15.4f} {len(rel_errors):<6}")
        sol_summary[name] = {'MARE': mare, 'MAE': mae, 'n': len(rel_errors)}

    # =========================================================================
    # NON-AQUEOUS PHASE VALIDATION
    # =========================================================================
    report.write()
    report.write("=" * 80)
    report.write("NON-AQUEOUS PHASE (y_H2O) - P >= 50 bar")
    report.write("=" * 80)

    mask_y = df_h2['y_H2O_exp'].notna()
    df_na = df_h2[mask_y].copy()

    na_correlations = {
        'This work (0.468)': lambda T: 0.468,
        'Chabab 2023': kij_na_chabab_2023,
        'Lopez-Lazaro 2019': kij_na_lopez_lazaro_2019,
    }

    na_summary = {}  # Store results for report

    # All data and excluding Suciu
    for filter_name, filter_func in [('All sources', lambda s: True),
                                      ('Excluding Suciu', lambda s: 'Suciu' not in s)]:
        report.write()
        report.write(f"--- {filter_name} ---")

        results_na = {name: [] for name in na_correlations}

        for _, row in df_na.iterrows():
            T_K = row['T_K']
            P_bar = row['P_bar']
            P_Pa = P_bar * 1e5
            y_exp = row['y_H2O_exp']
            source = row['Source']

            if P_bar < 50:
                continue
            if not filter_func(source):
                continue

            for name, kij_func in na_correlations.items():
                kij = kij_func(T_K)
                y_pred = calc_y_H2O_predicted(T_K, P_Pa, kij)

                if y_pred is not None and y_exp > 0:
                    results_na[name].append({
                        'y_exp': y_exp,
                        'y_pred': y_pred,
                        'abs_error': abs(y_pred - y_exp),
                        'rel_error_pct': abs(y_pred - y_exp) / y_exp * 100
                    })

        report.write()
        report.write(f"{'Correlation':<25} {'MARE (%)':<12} {'MAE (x1e-2)':<15} {'Within +/-10%':<14} {'n':<6}")
        report.write("-" * 75)

        for name in na_correlations:
            data = results_na[name]
            if data:
                mare = np.mean([d['rel_error_pct'] for d in data])
                mae = np.mean([d['abs_error'] for d in data])
                within_10 = sum(1 for d in data if d['rel_error_pct'] <= 10) / len(data) * 100
                n = len(data)
                report.write(f"{name:<25} {mare:<12.2f} {mae*100:<15.4f} {within_10:<14.0f}% {n:<6}")
                # Store excluding Suciu results for summary
                if filter_name == 'Excluding Suciu':
                    na_summary[name] = {'MARE': mare, 'MAE': mae, 'within_10': within_10, 'n': n}
            else:
                report.write(f"{name:<25} {'N/A':<12} {'N/A':<15} {'N/A':<14} {0:<6}")


    # Save report to file
    report.save()

    return kij_summary, sol_summary, na_summary


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    run_mare_validation('../../shared/data/pointwise_kij_results.csv')
