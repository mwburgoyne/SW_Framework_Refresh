"""Brine solubility validation, models kept separate (see 10_validate_fresh_p1.py):
  PUB-P0 : published dropin (kij_fw + published Delta-kij), published dry-vapour solve, as published
  PUB-P1 : published dropin kij_fw + published Delta-kij, procedure 1
  SW-P1  : S&W 1992 kij(T, cs) (Eqs 11-15), S&W kij_NA; H2S Sechenov Eq 8 post-factor as the engine's
           sw_original path; procedure 1
  NEW-P1 : procedure-1 refit (kij_fw + Delta-kij from results/p1_params.json), procedure 1
Selection: the published 10-Validate brine set (QCd Data, single-gas rows, Sal_m > 0, 0-200 C,
exclusions, n >= 5). He: He extension brine rows (Use != exclude).
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from p1_common import E, HAVE_HE, HE_CSV, HERE, P1BinaryVLE, XLSX, kij_aq_p1, kij_na_p1, load_params, load_pub_script

warnings.filterwarnings('ignore')
v10 = load_pub_script('10-Validate_MARE_All_Gases.py')
SW_NA = {'CO2': 0.1896, 'N2': 0.4778, 'CH4': 0.4850, 'C2H6': 0.4920, 'C3H8': 0.5525}
GASES = ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8'] + (['He'] if HAVE_HE else [])


def delta_new(prm, gas, T, m, key='embedded'):
    e = prm[key][gas]
    c, Tr = e['coef'], T / e['Tc']
    d = (c[0] + c[1] * Tr + c[2] * Tr**2) * m
    if e['form'] == 'quadratic':
        d += (c[3] + c[4] * Tr) * m**2
    return d


def brine_rows(gas):
    if gas == 'He':
        d = pd.read_csv(HE_CSV, comment='#')
        d = d[(d.Sal_m > 0) & (d.Use != 'exclude')]
        return pd.DataFrame({'Source': d.Source, 'T_K': d.T_K, 'P_bar': d.P_bar, 'm': d.Sal_m, 'x_exp': d.x_Gas})
    x = pd.read_excel(XLSX, sheet_name='QCd Data', engine='openpyxl')
    xc, zc = v10.X_COL[gas], v10.Z_COL[gas]
    if zc not in x.columns:
        return pd.DataFrame()
    mk = ((x[zc] == 1.0) & x[xc].notna() & (x[xc] > 0) & (x['Sal_m'] > 0)
          & (x['T_K'] >= v10.T_MIN_K) & (x['T_K'] <= v10.T_MAX_K))
    b = x[mk]
    b = b[~b['Source'].isin(v10.EXCLUDE_SOURCES.get(gas, set()))]
    if len(b) < 5:
        return pd.DataFrame()
    return pd.DataFrame({'Source': b.Source, 'T_K': b.T_K, 'P_bar': b.P_bar, 'm': b.Sal_m, 'x_exp': b[xc]})


def run(gas):
    prm = load_params()
    rows = brine_rows(gas)
    if rows.empty:
        return pd.DataFrame()
    kfw_new, kna_new = kij_aq_p1(gas, prm), kij_na_p1(gas, prm)
    has_sw = gas in SW_NA or gas == 'H2S'
    out = []
    for r in rows.itertuples():
        T, P, m = r.T_K, r.P_bar * 1e5, r.m
        row = dict(Gas=gas, Source=r.Source, T_K=T, P_bar=r.P_bar, m=m, x_exp=r.x_exp)
        row['PUB_P0'] = E.SWBinaryVLE(gas, m, framework='dropin').calc_gas_solubility(T, P)
        kp = E.get_kij_aq(gas, T, 0.0, framework='dropin')
        if gas in E.EMBEDDED_SALINITY_PARAMS_DROPIN:
            kp += E.calc_embedded_delta_kij(gas, T, m, params=E.EMBEDDED_SALINITY_PARAMS_DROPIN[gas])
        row['PUB_P1'] = P1BinaryVLE(gas, m).solve(T, P, kp)[0]
        row['NEW_P1'] = P1BinaryVLE(gas, m, kij_na_func=kna_new).solve(T, P, kfw_new(T) + delta_new(prm, gas, T, m))[0]
        row['NEW_P1_B'] = (P1BinaryVLE(gas, m, kij_na_func=kna_new).solve(T, P, kfw_new(T) + delta_new(prm, gas, T, m, 'embedded_B'))[0]
                           if gas in prm.get('embedded_B', {}) else np.nan)
        if has_sw:
            swna = E.kij_na_h2s_sw_eq17 if gas == 'H2S' else (lambda t, v=SW_NA[gas]: v)
            ks = E.get_kij_aq(gas, T, m, framework='sw_original')
            x = P1BinaryVLE(gas, m, kij_na_func=swna).solve(T, P, ks)[0]
            if gas not in E._SW_GASES_WITH_EMBEDDED_SALINITY:
                x *= 10**(-E.get_sechenov_ks(gas, T, m, r.P_bar, framework='sw_original') * m)
            row['SW_P1'] = x
        else:
            row['SW_P1'] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def st(s, c):
    e = 100 * (s[c] / s.x_exp - 1)
    e = e[np.isfinite(e)]
    return f"{len(e):4d} {e.abs().mean():5.1f} {e.mean():+6.1f} {e.abs().max():5.0f}" if len(e) else f"{'-':>22s}"


if __name__ == '__main__':
    gases = sys.argv[1:] or GASES
    with Pool(len(gases)) as p:
        df = pd.concat(p.map(run, gases), ignore_index=True)
    df.to_csv(f'{HERE}/results/brine_points_p1.csv', index=False, float_format='%.8g')
    L = ['BRINE x_gas: n / MARE% / bias% / max%', '',
         f"{'Gas':5s} | {'PUB-P0 (published)':>22s} | {'PUB-P1':>22s} | {'SW-P1':>22s} | {'NEW-P1 (target A)':>22s} | {'NEW-P1 (target B)':>22s}"]
    for g in gases:
        s = df[df.Gas == g]
        if len(s):
            L.append(f"{g:5s} | " + ' | '.join(st(s, c) for c in ['PUB_P0', 'PUB_P1', 'SW_P1', 'NEW_P1', 'NEW_P1_B']))
    print('\n'.join(L))
    open(f'{HERE}/logs/10_brine_report.txt', 'w').write('\n'.join(L) + '\n')
