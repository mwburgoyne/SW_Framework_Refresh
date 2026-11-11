"""Fit procedure-1 kij_AQ(T) correlations: same forms, same L1 fit routines (script 02a),
same data selection as the published dropin fit, kij from 01_pointwise_kaq_p1.py.
Fit set = published fit set AND procedure-1 pointwise converged.
Writes coefficients to results/p1_params.json['kij_aq'] and a report to logs/.
"""
import numpy as np
import pandas as pd

from p1_common import E, FORMS, GASES, HERE, PUB_CSV, kij_form, load_params, load_pub_script, save_params

sw02a = load_pub_script('02a-Fit_Dropin_Kij.py')

# Published dropin coefficients as starting points / comparison
PUB = {
    'CO2': None, 'H2S': None, 'N2': None,  # read numerically below from the engine
}
X0 = {'CO2': [-1.5, 1e-2, -2e-5, 2e-8], 'H2S': [-70.0, 1500.0, -4500.0, 0.2], 'CH4': [-2.0, 1.7, 0.2],
      'N2': [-1.5, 3e-3], 'H2': [-14.0, 3.5, 0.2], 'C2H6': [-1.3, 0.4, 1.3], 'C3H8': [-1.1, 0.6, 1.3],
      'He': [-71.6152, -0.394962, 0.505807]}


def fit(gas, T, k):
    form, Tc = FORMS[gas]
    if form == 'cubic':
        p, mae = sw02a.fit_l1_cubic(T, k, X0[gas])
    elif form == 'exp':
        p, mae = sw02a.fit_l1_exp(T, k, X0[gas])
    elif form == 'linear':
        p, mae = sw02a.fit_l1_linear(T, k, X0[gas])
    else:
        p, mae = sw02a.fit_l1_rational(T, k, Tc, X0[gas])
    # polish: restart Nelder-Mead from the solution (L1 surfaces are flat)
    for _ in range(3):
        if form == 'cubic':
            p, mae = sw02a.fit_l1_cubic(T, k, list(p))
        elif form == 'exp':
            p, mae = sw02a.fit_l1_exp(T, k, list(p))
        elif form == 'linear':
            p, mae = sw02a.fit_l1_linear(T, k, list(p))
        else:
            p, mae = sw02a.fit_l1_rational(T, k, Tc, list(p))
    return [float(v) for v in p], float(mae)


if __name__ == '__main__':
    d = pd.read_csv(f'{HERE}/results/pointwise_kaq_p1.csv')
    params = load_params()
    params.setdefault('kij_aq', {})
    lines = ['Procedure-1 kij_AQ correlations (S&W water alpha, L1, published forms and fit set)', '']
    for gas in GASES:
        g = d[(d.Gas == gas) & d.in_pub_set & d.conv]
        T, k = g.T_K.to_numpy(), g.kij_p1.to_numpy()
        p, mae = fit(gas, T, k)
        form, Tc = FORMS[gas]
        res = k - kij_form(form, Tc, p, T)
        pub = np.array([E.get_kij_aq(gas, t, 0.0, framework='dropin') for t in T])
        shift = kij_form(form, Tc, p, T) - pub
        params['kij_aq'][gas] = {'form': form, 'Tc': Tc, 'coef': p, 'n': int(len(g)), 'mae_kij': mae,
                                 'max_abs_res': float(np.abs(res).max()), 'T_range_K': [float(T.min()), float(T.max())],
                                 'n_dropped_unconverged': int((d[(d.Gas == gas) & d.in_pub_set]).conv.eq(False).sum())}
        s = (f"{gas:5s} {form:8s} n={len(g):4d} MAE={mae:.4f} max={np.abs(res).max():.4f} "
             f"T={T.min()-273.15:.0f}-{T.max()-273.15:.0f}C coef={['%.6g' % v for v in p]} "
             f"| corr minus published dropin: mean {shift.mean():+.4f}, range {shift.min():+.4f} to {shift.max():+.4f}")
        print(s, flush=True)
        lines.append(s)
    save_params(params)
    with open(f'{HERE}/logs/02_fit_report.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
