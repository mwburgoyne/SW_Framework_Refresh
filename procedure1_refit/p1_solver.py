"""
Procedure-1 (S&W 1992 Eqs. 10a/10b) binary gas-water equilibrium.

SEPARATE FROM THE PUBLISHED METHOD. The published paper (FPE 114824) regressed
kij_AQ with a pure, dry gas vapour at total pressure ("procedure 0",
SWBinaryVLE._calc_x_with_kij). This module solves ONE equilibrium in which the
aqueous phase uses kij_AQ and the non-aqueous phase uses kij_NA, with the
vapour carrying its equilibrium water (procedure 1, as S&W 1992 fitted).

Nothing in shared/vle_engine is modified. P1BinaryVLE subclasses SWBinaryVLE and
overrides only _calc_x_with_kij, so the published regression/fit routines
(scripts 01, 02a, 07a) can be driven unchanged with the procedure-1 solve.
All EOS quantities (alpha, a, b, cubic roots, fugacity coefficients) come from
the engine's own routines; only the phase-equilibrium iteration is new.

Water alpha: S&W original (framework='dropin'), salinity-dependent through
csw, applied to water in both phases (one pure-component parameter set, as in
S&W Eq. 9; the BIPs are the only phase-specific quantities).
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared', 'vle_engine'))
import _lib_vle_engine as E  # noqa: E402


def _phi(vle, T_K, P_Pa, z, kij, phase):
    return vle._calc_fugacity_coeff_with_kij(T_K, P_Pa, z, kij, phase=phase)


def solve_p1(vle, T_K, P_Pa, kij_aq, kij_na, x0=None, y0=None, tol=1e-12, max_iter=400):
    """Return (x_gas, y_H2O, converged) for the procedure-1 equilibrium.

    Successive substitution on K = phi_L(x; kAQ) / phi_V(y; kNA) with the
    closed-form binary Rachford-Rice; falls back to a nested bracketed solve
    (brentq on x_gas, inner brentq on y_H2O for water iso-fugacity).
    """
    if x0 is None or not np.isfinite(x0):
        x0 = E.SWBinaryVLE._calc_x_with_kij(vle, T_K, P_Pa, kij_aq)  # procedure-0 start
        if not np.isfinite(x0):
            x0 = 1e-3
    if y0 is None or not np.isfinite(y0):
        y0 = 0.02
    x = np.array([1.0 - x0, x0])
    y = np.array([y0, 1.0 - y0])
    for damp in (1.0, 0.5, 0.2):
        xx, yy = x.copy(), y.copy()
        ok = False
        for _ in range(max_iter):
            pL = _phi(vle, T_K, P_Pa, xx, kij_aq, 'liquid')
            pV = _phi(vle, T_K, P_Pa, yy, kij_na, 'vapor')
            if not (np.all(np.isfinite(pL)) and np.all(np.isfinite(pV))):
                break
            K = pL / pV
            if abs(K[1] - K[0]) < 1e-12:
                break
            xg = (1.0 - K[0]) / (K[1] - K[0])
            if not (0.0 < xg < 1.0):
                break
            xn = np.array([1.0 - xg, xg])
            yn = K * xn
            if not (0.0 < yn[0] < 1.0):
                break
            err = max(abs(xn[1] - xx[1]) / max(xn[1], 1e-14), abs(yn[0] - yy[0]) / max(yn[0], 1e-14))
            xx = xx + damp * (xn - xx)
            yy = yy + damp * (yn - yy)
            yy = yy / yy.sum()
            if err < tol:
                ok = True
                break
        if ok and _distinct(vle, T_K, P_Pa, xx, yy, kij_aq, kij_na):
            return xx[1], yy[0], True
    xg, yw = _nested(vle, T_K, P_Pa, kij_aq, kij_na, x0)
    return xg, yw, np.isfinite(xg)


def _distinct(vle, T_K, P_Pa, x, y, kAQ, kNA):
    """Reject the trivial solution (vapour collapsed onto the aqueous phase)."""
    return y[0] < 0.995 and x[0] > 0.5


def _water_y(vle, T_K, P_Pa, fwL, xg_guess, kNA):
    """Vapour water fraction y_w with f_w^V(y_w; kNA) = f_w^L."""
    def g(yw):
        y = np.array([yw, 1.0 - yw])
        return yw * _phi(vle, T_K, P_Pa, y, kNA, 'vapor')[0] * P_Pa - fwL
    grid = [1e-9, 1e-6, 1e-4, 1e-3, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.94]
    vals = [g(v) for v in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return brentq(g, a, b, xtol=1e-15, rtol=1e-13)
    return np.nan


def _nested(vle, T_K, P_Pa, kAQ, kNA, x0):
    def resid(xg):
        x = np.array([1.0 - xg, xg])
        pL = _phi(vle, T_K, P_Pa, x, kAQ, 'liquid')
        fwL, fgL = x[0] * pL[0] * P_Pa, xg * pL[1] * P_Pa
        yw = _water_y(vle, T_K, P_Pa, fwL, xg, kNA)
        if not np.isfinite(yw):
            return np.nan
        y = np.array([yw, 1.0 - yw])
        fgV = y[1] * _phi(vle, T_K, P_Pa, y, kNA, 'vapor')[1] * P_Pa
        return np.log(fgL / fgV)
    try:
        lo = max(x0 * 0.2, 1e-12)
        hi = min(x0 * 3.0, 0.6)
        pts = np.geomspace(lo, hi, 25)
        vals = [resid(p) for p in pts]
        for a, b, fa, fb in zip(pts[:-1], pts[1:], vals[:-1], vals[1:]):
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                xg = brentq(resid, a, b, xtol=1e-16, rtol=1e-12)
                x = np.array([1.0 - xg, xg])
                pL = _phi(vle, T_K, P_Pa, x, kAQ, 'liquid')
                yw = _water_y(vle, T_K, P_Pa, x[0] * pL[0] * P_Pa, xg, kNA)
                return xg, yw
    except Exception:
        pass
    return np.nan, np.nan


class P1BinaryVLE(E.SWBinaryVLE):
    """SWBinaryVLE with the procedure-1 solubility solve.

    kij_na_func(T_K) supplies kij_NA (default: the engine's recommended KIJ_NA).
    _calc_x_with_kij(T, P, kij) returns the procedure-1 x_gas with kij as kij_AQ,
    so published routines that take a vle object regress/fit on procedure 1.
    """

    def __init__(self, gas, salinity_molal=0.0, kij_na_func=None):
        super().__init__(gas, salinity_molal, framework='dropin')
        self.kij_na_func = kij_na_func or (lambda T_K: E.get_kij_na(gas, T_K))
        self._last = {}

    def _calc_x_with_kij(self, T_K, P_Pa, kij):
        key = (T_K, P_Pa)
        prev = self._last.get(key)
        x0, y0 = (prev if prev else (None, None))
        xg, yw, ok = solve_p1(self, T_K, P_Pa, kij, self.kij_na_func(T_K), x0, y0)
        if ok:
            self._last[key] = (xg, yw)
            return xg
        return np.nan

    def solve(self, T_K, P_Pa, kij_aq, kij_na=None):
        kna = self.kij_na_func(T_K) if kij_na is None else kij_na
        return solve_p1(self, T_K, P_Pa, kij_aq, kna)
