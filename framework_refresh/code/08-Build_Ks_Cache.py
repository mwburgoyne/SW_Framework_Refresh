#!/usr/bin/env python3
"""
Build the embedded BIP ks band cache for generate_figures.py.

Uses the known fitted embedded BIP coefficients (hardcoded from previous fitting)
to compute implied ks at multiple pressures via VLE. Saves to .npz cache.

Much faster than running the full fit_embedded_salinity_all_gases.py script
since it skips synthetic data generation and fitting.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'vle_engine'))

import numpy as np
from _lib_vle_engine import (
    SWBinaryVLE, COMPONENTS,
    get_kij_aq,
    kij_aq_co2, kij_aq_n2, kij_aq_hydrocarbon,
    EMBEDDED_SALINITY_PARAMS_DROPIN,
)

# Embedded BIP coefficients: kij(T,m) = kij_fw(T) + (a0 + a1*Tr + a2*Tr^2) * m
#   CO2 uses quadratic-in-m (Form E): + (b0 + b1*Tr) * m^2
# DROPIN (manuscript default): sourced directly from the engine so this cache
# tracks the manuscript coefficients (S&W alpha + dropin freshwater kij).
EMBEDDED_COEFFS = EMBEDDED_SALINITY_PARAMS_DROPIN

# Framework for all freshwater/brine VLE evaluations (S&W alpha + dropin kij_fw)
FRAMEWORK = 'dropin'

M_BRINE = 1.0
P_BAR = 100.0
P_BAND_BAR = [35.0, 70.0, 105.0, 140.0, 210.0, 280.0, 350.0]
T_RANGE = np.linspace(273.15, 473.15, 120)
T_MAX_K = 273.15 + 200 + 10  # Consistent 200°C limit for shading

# Maximum allowed deviation of band ks from main-line (100 bar) ks.
# Near-critical conditions cause ks to diverge before solver fails outright;
# capping removes these pathological values that create sawtooth artifacts.
KS_BAND_MAX_DEVIATION = 0.05

GAS_LIST = ['CO2', 'H2S', 'N2', 'H2', 'CH4', 'C2H6', 'C3H8', 'nC4H10']
CACHE_PATH = '../../shared/data/embedded_ks_band_cache.npz'


def delta_kij(T_K, m, coeffs):
    """Embedded BIP delta: (a0 + a1*Tr + a2*Tr^2) * m [+ (b0 + b1*Tr) * m^2]"""
    Tr = T_K / coeffs['Tc']
    delta = (coeffs['a0'] + coeffs['a1'] * Tr + coeffs['a2'] * Tr**2) * m
    if 'b0' in coeffs:
        delta += (coeffs['b0'] + coeffs['b1'] * Tr) * m**2
    return delta


def compute_implied_ks(gas, T_arr, P_Pa, kij_fw_func, delta_func, m=1.0):
    """Compute implied ks at given T array and pressure."""
    vle = SWBinaryVLE(gas, salinity_molal=0.0, framework=FRAMEWORK)
    ks_arr = []
    for T_K in T_arr:
        kij_fw = kij_fw_func(T_K)
        kij_br = kij_fw + delta_func(T_K, m)
        try:
            x_fw = vle._calc_x_with_kij(T_K, P_Pa, kij_fw)
            x_br = vle._calc_x_with_kij(T_K, P_Pa, kij_br)
            if x_fw > 0 and x_br > 0:
                ks_arr.append(np.log10(x_fw / x_br) / m)
            else:
                ks_arr.append(np.nan)
        except Exception:
            ks_arr.append(np.nan)
    return np.array(ks_arr)


def main():
    print("Building embedded BIP ks band cache...")
    cache = {'T_RANGE': T_RANGE}

    for gas in GAS_LIST:
        coeffs = EMBEDDED_COEFFS[gas]
        kij_fw_func = lambda T_K, g=gas: get_kij_aq(g, T_K, 0.0, framework=FRAMEWORK)
        delta_func = lambda T_K, m, c=coeffs: delta_kij(T_K, m, c)

        print(f"  {gas}...", end='', flush=True)

        # Main line at 100 bar
        P_Pa = P_BAR * 1e5
        ks_main = compute_implied_ks(gas, T_RANGE, P_Pa, kij_fw_func, delta_func)
        cache[f'{gas}_ks_this_work'] = ks_main
        print(f" main", end='', flush=True)

        # Pressure band
        vle_band = SWBinaryVLE(gas, salinity_molal=0.0, framework=FRAMEWORK)
        ks_by_P = []
        for P_b in P_BAND_BAR:
            P_Pa_b = P_b * 1e5
            ks_at_P = []
            for i_T, T_K in enumerate(T_RANGE):
                if T_K > T_MAX_K:
                    ks_at_P.append(np.nan)
                    continue
                kij_fw = kij_fw_func(T_K)
                kij_br = kij_fw + delta_func(T_K, M_BRINE)
                try:
                    x_fw = vle_band._calc_x_with_kij(T_K, P_Pa_b, kij_fw)
                    x_br = vle_band._calc_x_with_kij(T_K, P_Pa_b, kij_br)
                    if x_fw > 0 and x_br > 0:
                        ks_val = np.log10(x_fw / x_br) / M_BRINE
                        # Filter near-critical divergence: if band ks deviates
                        # too far from the smooth main-line (100 bar), NaN it out
                        ks_ref = ks_main[i_T]
                        if (not np.isnan(ks_ref) and
                                abs(ks_val - ks_ref) > KS_BAND_MAX_DEVIATION):
                            ks_at_P.append(np.nan)
                        else:
                            ks_at_P.append(ks_val)
                    else:
                        ks_at_P.append(np.nan)
                except Exception:
                    ks_at_P.append(np.nan)
            ks_by_P.append(np.array(ks_at_P))
        ks_arr = np.array(ks_by_P)
        band_min_raw = np.nanmin(ks_arr, axis=0)
        band_max_raw = np.nanmax(ks_arr, axis=0)

        # Smooth band edges to eliminate sawtooth artifacts.
        # Near-critical gases (H2S, CO2) lose high-P solutions at elevated T,
        # causing abrupt band narrowing. Smoothing the min/max produces clean
        # shaded regions while preserving the physical trend.
        def _smooth_band(arr, window=7):
            """Moving-average smooth, NaN-aware, preserving NaN tails."""
            out = arr.copy()
            hw = window // 2
            for i in range(len(arr)):
                lo = max(0, i - hw)
                hi = min(len(arr), i + hw + 1)
                chunk = arr[lo:hi]
                valid = chunk[~np.isnan(chunk)]
                if len(valid) >= 2:
                    out[i] = np.mean(valid)
                elif len(valid) == 1:
                    out[i] = valid[0]
                # else: stays NaN
            return out

        cache[f'{gas}_ks_band_min'] = _smooth_band(band_min_raw)
        cache[f'{gas}_ks_band_max'] = _smooth_band(band_max_raw)
        print(f" band", end='', flush=True)

        # S&W original embedded BIP implied ks
        gases_with_sw_embedded = {'CH4', 'N2', 'CO2', 'C2H6', 'C3H8',
                                  'iC4H10', 'nC4H10'}
        if gas in gases_with_sw_embedded:
            vle = SWBinaryVLE(gas, salinity_molal=0.0, framework='sw_original')
            ks_sw_emb = []
            P_Pa_main = P_BAR * 1e5
            for T_K in T_RANGE:
                if gas in ('CH4', 'C2H6', 'C3H8', 'nC4H10',
                           'iC4H10', 'nC5H12', 'iC5H12'):
                    kij_sw_fresh = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                                       COMPONENTS[gas].Tc, 0.0)
                    kij_sw_brine = kij_aq_hydrocarbon(T_K, COMPONENTS[gas].omega,
                                                       COMPONENTS[gas].Tc, M_BRINE)
                elif gas == 'CO2':
                    kij_sw_fresh = kij_aq_co2(T_K, 0.0)
                    kij_sw_brine = kij_aq_co2(T_K, M_BRINE)
                elif gas == 'N2':
                    kij_sw_fresh = kij_aq_n2(T_K, 0.0)
                    kij_sw_brine = kij_aq_n2(T_K, M_BRINE)
                else:
                    ks_sw_emb.append(np.nan)
                    continue
                try:
                    x_fw = vle._calc_x_with_kij(T_K, P_Pa_main, kij_sw_fresh)
                    x_br = vle._calc_x_with_kij(T_K, P_Pa_main, kij_sw_brine)
                    if x_fw > 0 and x_br > 0:
                        ks_sw_emb.append(np.log10(x_fw / x_br) / M_BRINE)
                    else:
                        ks_sw_emb.append(np.nan)
                except Exception:
                    ks_sw_emb.append(np.nan)
            cache[f'{gas}_ks_sw_emb'] = np.array(ks_sw_emb)
            print(f" sw_emb", end='', flush=True)

        print(f" done")

    np.savez_compressed(CACHE_PATH, **cache)
    print(f"\nSaved cache to {CACHE_PATH}")
    print(f"Keys: {list(cache.keys())}")


if __name__ == '__main__':
    main()
