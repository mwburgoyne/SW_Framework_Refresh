"""Implementer check values for the procedure-1 refit (primary set: embedded target basis A).
kij_AQ(T) at fixed temperatures, and procedure-1 binary x_gas / y_H2O at fixed (T, P, m).
Output: results/check_values.csv and logs/11_check_values.txt
"""
import warnings

import numpy as np
import pandas as pd

from p1_common import E, GASES, HERE, P1BinaryVLE, kij_aq_p1, kij_na_p1, load_params

warnings.filterwarnings('ignore')
prm = load_params()


def delta(gas, T, m):
    e = prm['embedded'][gas]
    c, Tr = e['coef'], T / e['Tc']
    d = (c[0] + c[1] * Tr + c[2] * Tr**2) * m
    if e['form'] == 'quadratic':
        d += (c[3] + c[4] * Tr) * m**2
    return d


rows = []
L = ['kij_AQ (freshwater) at T [C]:  25 / 80 / 150 / 200', '']
for g in GASES:
    k = kij_aq_p1(g, prm)
    vals = [k(t + 273.15) for t in (25, 80, 150, 200)]
    L.append(f"{g:5s} " + '  '.join(f'{v:+.5f}' for v in vals) + f"   kij_NA {kij_na_p1(g, prm)(300.0):.4f}")
L += ['', 'Procedure-1 binary equilibrium: x_gas, y_H2O']
for g in GASES:
    kA, kN = kij_aq_p1(g, prm), kij_na_p1(g, prm)
    for (Tc, Pb, m) in [(80, 100, 0.0), (150, 300, 0.0), (80, 100, 2.0)]:
        T = Tc + 273.15
        x, y, ok = P1BinaryVLE(g, m, kij_na_func=kN).solve(T, Pb * 1e5, kA(T) + delta(g, T, m))
        rows.append(dict(Gas=g, T_C=Tc, P_bar=Pb, m_NaCl=m, kij_AQ=kA(T) + delta(g, T, m), kij_NA=kN(T), x_gas=x, y_H2O=y))
        L.append(f"{g:5s} {Tc:4d} C {Pb:4d} bar m={m:.0f}: kAQ {kA(T)+delta(g, T, m):+.5f}  x_gas {x:.5e}  y_H2O {y:.5e}")
pd.DataFrame(rows).to_csv(f'{HERE}/results/check_values.csv', index=False, float_format='%.6e')
open(f'{HERE}/logs/11_check_values.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
