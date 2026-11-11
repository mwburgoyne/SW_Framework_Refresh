"""kij_NA on the procedure-1 basis, with the procedure-1 kij_AQ from the store.

Rows: exactly the published script-05 selection (its load_and_filter: converged,
non-bound-hitting, P >= 20 bar, C3H8 >= 10 bar, H2 >= 50 bar and 50-150 C).
Search: same two-stage grid (coarse 0.02, fine 0.001) and ranges as script 05.
Decision rule (the paper's): keep the published constant unless the optimum
improves y_H2O MARE by more than 1.1 pp (the largest gain the paper treated as
negligible). He: no water-content data, H2 surrogate 0.468 retained.
Writes results/p1_params.json['kij_na'] and logs/05_kna_report.txt.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np

from p1_common import CODE, E, HERE, P1BinaryVLE, kij_aq_p1, load_params, load_pub_script, save_params

warnings.filterwarnings('ignore')
s05 = load_pub_script('05-Regress_Kij_NA.py')
GASES = ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8']
THRESH_PP = 1.1


def mare_y(rows, vle, kA, kna):
    e = []
    for r in rows.itertuples():
        _, yw, ok = vle.solve(r.T_K, r.P_bar * 1e5, kA(r.T_K), kna(r.T_K) if callable(kna) else kna)
        if ok and np.isfinite(yw):
            e.append(abs(yw / r.y_H2O_exp - 1))
    return 100 * np.mean(e) if e else np.nan, len(e)


def run(gas):
    rows, st = s05.load_and_filter(gas, csv_path=f'{CODE}/../../shared/data/pointwise_kij_results.csv')
    prm = load_params()
    kA = kij_aq_p1(gas, prm)
    vle = P1BinaryVLE(gas, 0.0)
    pub = E.KIJ_NA[gas]
    sw = E.kij_na_h2s_sw_eq17 if gas == 'H2S' else s05.SW_KIJ_NA[gas]
    m_pub, n_pub = mare_y(rows, vle, kA, pub)
    m_sw, _ = mare_y(rows, vle, kA, sw)
    lo, hi = s05.SEARCH_RANGES[gas]
    best_k, best_m = None, np.inf
    for k in np.arange(lo, hi + 0.01, 0.02):
        m, _ = mare_y(rows, vle, kA, k)
        if np.isfinite(m) and m < best_m:
            best_k, best_m = k, m
    for k in np.arange(best_k - 0.02, best_k + 0.0205, 0.001):
        if lo <= k <= hi:
            m, _ = mare_y(rows, vle, kA, k)
            if np.isfinite(m) and m < best_m:
                best_k, best_m = k, m
    best_k = round(float(best_k), 3)
    adopt = (m_pub - best_m) > THRESH_PP
    return dict(gas=gas, n=len(rows), n_valid=n_pub, pub=pub, mare_pub=m_pub, mare_sw=m_sw,
                opt=best_k, mare_opt=best_m, adopt=bool(adopt), chosen=best_k if adopt else pub)


if __name__ == '__main__':
    with Pool(len(GASES)) as p:
        res = p.map(run, GASES)
    prm = load_params()
    prm['kij_na'] = {r['gas']: r['chosen'] for r in res}
    prm['kij_na']['He'] = 0.468
    prm['kij_na_detail'] = res
    save_params(prm)
    L = [f"kij_NA, procedure 1 (procedure-1 kij_AQ in the aqueous phase); y_H2O MARE %; adopt if gain > {THRESH_PP} pp", '',
         f"{'Gas':5s} {'n':>4s} {'pub kNA':>8s} {'MARE pub':>9s} {'MARE S&W':>9s} {'opt kNA':>8s} {'MARE opt':>9s} {'chosen':>8s}"]
    for r in res:
        L.append(f"{r['gas']:5s} {r['n']:4d} {r['pub']:8.4f} {r['mare_pub']:9.2f} {r['mare_sw']:9.2f} {r['opt']:8.3f} "
                 f"{r['mare_opt']:9.2f} {r['chosen']:8.4f}{'  CHANGED' if r['adopt'] else ''}")
    print('\n'.join(L))
    open(f'{HERE}/logs/05_kna_report.txt', 'w').write('\n'.join(L) + '\n')
