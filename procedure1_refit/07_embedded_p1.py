"""Embedded salinity Delta-kij on the procedure-1 basis.

Drives the published script 07a routines unchanged (same Sechenov targets: Dubessy
CO2, Akinfiev H2S, S&W Eq 8 others; same T/P/m grids; same Form D / Form E fits),
with two substitutions in 07a's namespace only:
  SWBinaryVLE  -> P1BinaryVLE (procedure-1 solve, brine S&W alpha in both phases)
  KIJ_FW_FUNCS -> procedure-1 freshwater kij_AQ from the store
He: grid of the He extension (H2 grid restricted to 25-100 C), Eq 8 with Tb 4.2238 K.
Writes results/p1_params.json['embedded'] and logs/07_embedded_report_{A,B}.txt (--basis=A primary -> 'embedded'; --basis=B sensitivity -> 'embedded_B').
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq, least_squares

from p1_common import E, HERE, P1BinaryVLE, kij_aq_p1, load_params, load_pub_script, save_params

warnings.filterwarnings('ignore')
s07 = load_pub_script('07a-Fit_Embedded_Dropin.py')
s07.GAS_CONFIGS['He'] = {'Tc': 5.1953, 'T_range_C': [25, 50, 75, 100], 'P_range_bar': [50, 100, 150, 200],
                         'm_range': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]}
GASES = ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8', 'nC4H10', 'He']


def _vle(gas, salinity_molal=0.0, framework='dropin'):
    return P1BinaryVLE(gas, salinity_molal)


def _find_kij_brine(vle_br, T_K, P_Pa, x_target, kij_fw):
    """Same root as 07a's _find_kij_brine (x_p1(kij) = x_target), but bracketed by a
    NaN-tolerant scan: 07a's fixed bracket (kij_fw - 3, kij_fw + 0.5) has no procedure-1
    solution at its lower end, which makes brentq raise and silently drops the point."""
    def f(k):
        x = vle_br._calc_x_with_kij(T_K, P_Pa, k)
        return np.log(x / x_target) if np.isfinite(x) and x > 0 else np.nan
    grid = kij_fw + np.arange(-0.5, 3.5001, 0.05)
    vals = [f(k) for k in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return brentq(f, a, b, xtol=1e-10)
    raise ValueError('no bracket')


def generate_gamma_targets(gas):
    """Basis B: Sechenov as an activity coefficient at equal gas fugacity (the paper's gamma
    definition, f_i^L = x_i gamma_i phi_i^L P with phi_i^L from the salt-free EOS).
    For each grid point find kij_eff such that, in the procedure-1 brine equilibrium (x_b),
    the salt-free aqueous phase holding gamma*x_b (gamma = 10^(ks m)) has the same gas
    fugacity as the brine aqueous phase. x_target = x_b(kij_eff). Same grid, ks and record
    layout as 07a.generate_synthetic_data, so 07a's fitting routines are used unchanged."""
    from p1_solver import _phi
    cfg = s07.GAS_CONFIGS[gas]
    fw = s07.KIJ_FW_FUNCS[gas]
    Tc = cfg['Tc']
    v_fw = P1BinaryVLE(gas, 0.0)
    data = []
    for T_C in cfg['T_range_C']:
        T_K = T_C + 273.15
        Tr = T_K / Tc
        k_fw = fw(T_K)
        for P_bar in cfg['P_range_bar']:
            P_Pa = P_bar * 1e5
            x_fresh = v_fw._calc_x_with_kij(T_K, P_Pa, k_fw)
            if not np.isfinite(x_fresh) or x_fresh <= 0 or x_fresh >= 0.5:
                continue
            for m in cfg['m_range']:
                base = dict(T_C=T_C, T_K=T_K, Tr=Tr, P_bar=P_bar, P_Pa=P_Pa, csw=m, x_fresh=x_fresh, kij_fw=k_fw)
                if m == 0:
                    data.append({**base, 'x_target': x_fresh, 'kij_eff': k_fw, 'delta_kij': 0.0})
                    continue
                gam = 10 ** (E.get_sechenov_ks(gas, T_K, m, P_bar) * m)
                v_br = P1BinaryVLE(gas, m)

                def resid(k):
                    xb = v_br._calc_x_with_kij(T_K, P_Pa, k)
                    if not np.isfinite(xb) or gam * xb >= 0.95:
                        return np.nan
                    fb = xb * _phi(v_br, T_K, P_Pa, np.array([1 - xb, xb]), k, 'liquid')[1]
                    xf = gam * xb
                    ff = xf * _phi(v_fw, T_K, P_Pa, np.array([1 - xf, xf]), k_fw, 'liquid')[1]
                    return np.log(ff / fb)
                # salting-out (ks > 0) means Delta-kij >= 0: scan upward from kij_fw first,
                # then downward; avoids a spurious near-critical root at kij < kij_fw
                up = k_fw + np.arange(-0.02, 3.5001, 0.05)
                dn = k_fw - np.arange(0.02, 0.5001, 0.05)
                k_eff = None
                pairs = []
                for grid in (up, dn):
                    vals = [resid(k) for k in grid]
                    pairs = list(zip(grid[:-1], grid[1:], vals[:-1], vals[1:]))
                    if any(np.isfinite(fa) and np.isfinite(fb_) and fa * fb_ < 0 for _, _, fa, fb_ in pairs):
                        break
                for a, b, fa, fb_ in pairs:
                    if np.isfinite(fa) and np.isfinite(fb_) and fa * fb_ < 0:
                        try:
                            k_eff = brentq(resid, a, b, xtol=1e-10)
                        except ValueError:  # NaN inside the bracket: bisect on finite values
                            lo, hi, flo = a, b, fa
                            for _ in range(60):
                                mid = 0.5 * (lo + hi)
                                fm = resid(mid)
                                if not np.isfinite(fm):
                                    break
                                if fm * flo < 0:
                                    hi = mid
                                else:
                                    lo, flo = mid, fm
                            else:
                                k_eff = 0.5 * (lo + hi)
                        break
                if k_eff is None:
                    continue
                xb = v_br._calc_x_with_kij(T_K, P_Pa, k_eff)
                data.append({**base, 'x_target': xb, 'kij_eff': k_eff, 'delta_kij': k_eff - k_fw})
    return data


BASIS = 'A'


def run(gas):
    prm = load_params()
    s07.SWBinaryVLE = _vle
    s07._find_kij_brine = _find_kij_brine
    if BASIS == 'B':
        s07.generate_synthetic_data = generate_gamma_targets
    if gas == 'nC4H10':  # no nC4 solubility data: S&W Eqs 11-12 at cs=0 (as published dropin)
        fw = lambda T, c=E.COMPONENTS['nC4H10']: E.kij_aq_hydrocarbon(T, c.omega, c.Tc, 0.0)
    else:
        fw = kij_aq_p1(gas, prm)
    s07.KIJ_FW_FUNCS[gas] = fw
    cfg = s07.GAS_CONFIGS[gas]
    Tc = cfg['Tc']
    data = s07.generate_synthetic_data(gas)
    brine = [d for d in data if d['csw'] > 0]
    ls = least_squares(lambda p: [d['delta_kij'] - (p[0] + p[1] * d['Tr'] + p[2] * d['Tr']**2) * d['csw'] for d in brine],
                       [0.0, 0.0, 0.0])
    pD, _ = s07.fit_solubility_minimizing(data, Tc, gas, fw, list(ls.x))
    accD = s07.evaluate_accuracy(data, gas, Tc, fw, pD, 'linear')
    out = dict(gas=gas, Tc=Tc, n_brine=len(brine), form='linear', coef=[float(v) for v in pD], acc=accD)
    if gas == 'CO2':
        pE, _ = s07.fit_quad_m_solubility(data, Tc, gas, fw, list(pD) + [0.0, 0.0])
        accE = s07.evaluate_accuracy(data, gas, Tc, fw, pE, 'quadratic')
        out['alt_quadratic'] = dict(coef=[float(v) for v in pE], acc=accE)
        if accE['mae_x_pct'] < accD['mae_x_pct'] * 0.8:  # 07a's rule
            out.update(form='quadratic', coef=[float(v) for v in pE], acc=accE)
    print(gas, out['form'], ['%.6g' % v for v in out['coef']], 'MAE x %.2f%% max %.2f%%' %
          (out['acc']['mae_x_pct'], out['acc']['max_x_pct']), flush=True)
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] in ('--basis=A', '--basis=B'):
        BASIS = args.pop(0)[-1]
    gases = args or GASES
    with Pool(len(gases)) as p:
        res = p.map(run, gases)
    prm = load_params()
    key = 'embedded' if BASIS == 'A' else 'embedded_B'
    prm.setdefault(key, {})
    for r in res:
        r['basis'] = BASIS
        prm[key][r['gas']] = r
    save_params(prm)
    L = ['Embedded Delta-kij, procedure 1: Delta = (a0 + a1 Tr + a2 Tr^2) m [+ (b0 + b1 Tr) m^2 for CO2]', '']
    for r in res:
        L.append(f"{r['gas']:7s} Tc={r['Tc']:<8} {r['form']:9s} coef={['%.6g' % v for v in r['coef']]} "
                 f"n_brine={r['n_brine']} MAE(x vs Sechenov target)={r['acc']['mae_x_pct']:.2f}% max={r['acc']['max_x_pct']:.2f}%")
    print('\n'.join(L))
    open(f'{HERE}/logs/07_embedded_report_{BASIS}.txt', 'w').write('\n'.join(L) + '\n')
