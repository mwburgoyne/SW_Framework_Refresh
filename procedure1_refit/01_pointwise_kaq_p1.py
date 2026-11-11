"""Pointwise kij_AQ on the procedure-1 basis (S&W water alpha, kij_NA in the vapour).

For each freshwater point, find kij_AQ with x_p1(T, P; kij_AQ, kij_NA) = x_exp.
kij_NA: current store value (results/p1_params.json), else the published constant.
Output: results/pointwise_kaq_p1.csv  (Gas, Source, T_K, P_bar, x_exp, in_pub_set,
        kij_p1, conv, x_back, yw_p1, kij_na_used)
Usage: python3 01_pointwise_kaq_p1.py [gas ...]
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from p1_common import GASES, HERE, P1BinaryVLE, freshwater_points, kij_na_p1, load_params

warnings.filterwarnings('ignore')
GRID = np.linspace(-1.6, 1.2, 57)


def regress_point(vle, T_K, P_Pa, x_exp):
    def f(k):
        x = vle._calc_x_with_kij(T_K, P_Pa, k)
        return np.log(x / x_exp) if np.isfinite(x) and x > 0 else np.nan
    vals = np.array([f(k) for k in GRID])
    # first sign change from the negative-kij side (x decreases with kij)
    for i in range(len(GRID) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            try:
                k = brentq(f, GRID[i], GRID[i + 1], xtol=1e-10, rtol=1e-12)
            except Exception:
                return np.nan, False
            return k, True
    # Edge refinement: near the mixture critical boundary x rises steeply as kij falls and
    # the procedure-1 solution vanishes; the root can sit between a no-solution grid node
    # and a finite one. Bisect towards the edge and bracket there.
    for i in range(len(GRID) - 1):
        a, b = vals[i], vals[i + 1]
        if not np.isfinite(a) and np.isfinite(b) and b < 0:
            lo, hi = GRID[i], GRID[i + 1]  # lo: no solution, hi: solution with x < x_exp
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if not np.isfinite(fm):
                    lo = mid
                elif fm > 0:
                    try:
                        return brentq(f, mid, hi, xtol=1e-12, rtol=1e-12), True
                    except Exception:
                        return np.nan, False
                else:
                    hi = mid
    return np.nan, False


def run_gas(gas):
    params = load_params()
    kna = kij_na_p1(gas, params)
    vle = P1BinaryVLE(gas, 0.0, kij_na_func=kna)
    pts = freshwater_points(gas)
    rows = []
    for r in pts.itertuples():
        P_Pa = r.P_bar * 1e5
        k, ok = regress_point(vle, r.T_K, P_Pa, r.x_exp)
        xb, yw = np.nan, np.nan
        if ok:
            xb, yw, _ = vle.solve(r.T_K, P_Pa, k)
            ok = abs(xb / r.x_exp - 1) < 1e-6
        rows.append(dict(Gas=gas, Source=r.Source, T_K=r.T_K, P_bar=r.P_bar, x_exp=r.x_exp,
                         in_pub_set=r.in_pub_set, kij_p1=k, conv=ok, x_back=xb, yw_p1=yw,
                         kij_na_used=kna(r.T_K)))
    df = pd.DataFrame(rows)
    print(f"{gas}: {len(df)} points, converged {df.conv.sum()}, pub-set {df.in_pub_set.sum()}, "
          f"pub-set converged {(df.conv & df.in_pub_set).sum()}", flush=True)
    return df


if __name__ == '__main__':
    gases = sys.argv[1:] or GASES
    with Pool(len(gases)) as pool:
        dfs = pool.map(run_gas, gases)
    out = pd.concat(dfs, ignore_index=True)
    path = f'{HERE}/results/pointwise_kaq_p1.csv'
    if len(gases) < len(GASES):
        try:
            old = pd.read_csv(path)
            out = pd.concat([old[~old.Gas.isin(gases)], out], ignore_index=True)
        except FileNotFoundError:
            pass
    out.to_csv(path, index=False, float_format='%.10g')
    print('saved', path)
