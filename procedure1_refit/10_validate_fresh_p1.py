"""Freshwater solubility validation, three models kept strictly separate:
  PUB-P0 : published dropin kij_AQ, published dry-vapour solve (engine calc_gas_solubility, as published)
  PUB-P1 : published dropin kij_AQ used in a procedure-1 calculation (what a S&W-method simulator gets)
  SW-P1  : S&W 1992 kij_AQ and S&W kij_NA (Table 5, Eq 17 for H2S), procedure 1 (as S&W fitted)
  NEW-P1 : procedure-1 refit (results/p1_params.json), procedure 1
Evaluation set = published 10-Validate set (H2 also drops Gillespie 1980); He = He fit set.
Output: logs/10_fresh_report.txt, results/fresh_points_p1.csv
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from p1_common import E, GASES, HERE, P1BinaryVLE, freshwater_points, kij_aq_p1, kij_na_p1, load_params

warnings.filterwarnings('ignore')
SW_NA = {'CO2': 0.1896, 'N2': 0.4778, 'CH4': 0.4850, 'C2H6': 0.4920, 'C3H8': 0.5525}


def run(gas):
    prm = load_params()
    pts = freshwater_points(gas)
    pts = pts[pts.in_pub_set].copy()
    if gas == 'H2':
        pts = pts[pts.Source != 'Gillespie 1980']
    pub = E.SWBinaryVLE(gas, 0.0, framework='dropin')
    v_pub1 = P1BinaryVLE(gas, 0.0)  # published kij_NA
    kA_new, kN_new = kij_aq_p1(gas, prm), kij_na_p1(gas, prm)
    v_new = P1BinaryVLE(gas, 0.0, kij_na_func=kN_new)
    has_sw = gas in ('CO2', 'H2S', 'CH4', 'N2', 'C2H6', 'C3H8')
    if has_sw:
        swna = E.kij_na_h2s_sw_eq17 if gas == 'H2S' else (lambda T, v=SW_NA[gas]: v)
        v_sw = P1BinaryVLE(gas, 0.0, kij_na_func=swna)
    out = []
    for r in pts.itertuples():
        T, P = r.T_K, r.P_bar * 1e5
        row = dict(Gas=gas, Source=r.Source, T_K=T, P_bar=r.P_bar, x_exp=r.x_exp)
        row['PUB_P0'] = pub.calc_gas_solubility(T, P)
        kp = E.get_kij_aq(gas, T, 0.0, framework='dropin')
        row['PUB_P1'] = v_pub1.solve(T, P, kp)[0]
        xn, yn, _ = v_new.solve(T, P, kA_new(T))
        row['NEW_P1'], row['yw_new'] = xn, yn
        row['SW_P1'] = v_sw.solve(T, P, E.get_kij_aq(gas, T, 0.0, framework='sw_original'))[0] if has_sw else np.nan
        out.append(row)
    return pd.DataFrame(out)


def stats(df, col):
    e = 100 * (df[col] / df.x_exp - 1)
    e = e[np.isfinite(e)]
    return len(e), e.abs().mean(), e.mean(), e.abs().max()


if __name__ == '__main__':
    gases = sys.argv[1:] or GASES
    with Pool(len(gases)) as p:
        df = pd.concat(p.map(run, gases), ignore_index=True)
    df.to_csv(f'{HERE}/results/fresh_points_p1.csv', index=False, float_format='%.8g')
    L = ['FRESHWATER x_gas: n / MARE% / bias% / max%', '']
    hdr = f"{'Gas':5s} {'N':>4s} | {'PUB-P0 (published)':>22s} | {'PUB-P1':>22s} | {'SW-P1':>22s} | {'NEW-P1':>22s}"
    L.append(hdr)
    for g in gases:
        s = df[df.Gas == g]
        cells = []
        for c in ['PUB_P0', 'PUB_P1', 'SW_P1', 'NEW_P1']:
            n, m, b, mx = stats(s, c)
            cells.append(f"{n:4d} {m:5.1f} {b:+6.1f} {mx:6.0f}" if n else f"{'-':>22s}")
        L.append(f"{g:5s} {len(s):4d} | " + ' | '.join(cells))
    L += ['', 'By temperature band, MARE% (bias%):']
    df['Tb'] = pd.cut(df.T_K - 273.15, [-1, 60, 100, 150, 201], labels=['<=60C', '60-100C', '100-150C', '150-200C'])
    for g in gases:
        for tb, s in df[df.Gas == g].groupby('Tb', observed=True):
            cells = []
            for c in ['PUB_P0', 'PUB_P1', 'SW_P1', 'NEW_P1']:
                n, m, b, mx = stats(s, c)
                cells.append(f"{m:5.1f} ({b:+5.1f})" if n else f"{'-':>13s}")
            L.append(f"{g:5s} {tb:9s} n={len(s):4d} | " + ' | '.join(cells))
    txt = '\n'.join(L)
    print(txt)
    open(f'{HERE}/logs/10_fresh_report.txt', 'w').write(txt + '\n')
