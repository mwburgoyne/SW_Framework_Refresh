"""Shared data selection, correlation forms and parameter store for the procedure-1 refit.

Data selection reuses the published definitions (02a/10 exclusions, T window,
published convergence flags) so the only change from the paper is the
equilibrium procedure. Parameters live in results/p1_params.json, never in
the engine.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..')
CODE = os.path.join(REPO, 'framework_refresh', 'code')
PUB_CSV = os.path.join(REPO, 'shared', 'data', 'pointwise_kij_results_sw_alpha.csv')
XLSX = os.path.join(REPO, 'shared', 'data', 'solubility_points.xlsx')
HE_CSV = os.path.join(REPO, 'shared', 'data', 'he_solubility.csv')  # post-publication He extension
HAVE_HE = os.path.exists(HE_CSV)
PARAMS = os.path.join(HERE, 'results', 'p1_params.json')

sys.path.insert(0, HERE)
from p1_solver import E, P1BinaryVLE, solve_p1  # noqa: E402,F401


def load_pub_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_').rstrip('.py'), os.path.join(CODE, name))
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(CODE)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


EXCLUDE_SOURCES = {
    'CH4': {'Blount 1982', 'McGee 1981'},
    'CO2': {'Prutton & Savage 1945'},
    'H2S': {'Barrett 1988'},
}
T_MIN_K, T_MAX_K = 273.15, 473.15
GASES = ['CO2', 'H2S', 'CH4', 'N2', 'H2', 'C2H6', 'C3H8'] + (['He'] if HAVE_HE else [])

# Functional forms: identical to the published dropin forms (02a); He as the He extension.
FORMS = {
    'CO2': ('cubic', None), 'H2S': ('exp', None), 'CH4': ('rational', 190.60),
    'N2': ('linear', None), 'H2': ('rational', 33.145), 'C2H6': ('rational', 305.40),
    'C3H8': ('rational', 369.80), 'He': ('rational', 5.1953),
}


def kij_form(form, Tc, p, T_K):
    T_K = np.asarray(T_K, dtype=float)
    if form == 'cubic':
        return p[0] + p[1] * T_K + p[2] * T_K**2 + p[3] * T_K**3
    if form == 'exp':
        return p[0] / T_K + p[1] * np.exp(p[2] / T_K) + p[3]
    if form == 'linear':
        return p[0] + p[1] * T_K
    if form == 'rational':
        Tr = T_K / Tc
        return (p[0] + Tr) / (p[1] + p[2] * Tr)
    raise ValueError(form)


def load_params():
    if os.path.exists(PARAMS):
        with open(PARAMS) as f:
            return json.load(f)
    return {}


def save_params(p):
    os.makedirs(os.path.dirname(PARAMS), exist_ok=True)
    with open(PARAMS, 'w') as f:
        json.dump(p, f, indent=2)


def kij_aq_p1(gas, params=None):
    """Freshwater procedure-1 kij_AQ(T) from the parameter store."""
    params = params or load_params()
    form, Tc = FORMS[gas]
    c = params['kij_aq'][gas]['coef']
    return lambda T_K: float(kij_form(form, Tc, c, T_K))


def kij_na_p1(gas, params=None):
    """Procedure-1 kij_NA (constant per gas) from the store; falls back to the published value."""
    params = params or load_params()
    val = params.get('kij_na', {}).get(gas)
    if val is None:
        val = E.KIJ_NA[gas]
    return lambda T_K: float(val)


def freshwater_points(gas):
    """All freshwater solubility points with the published fit-set flag.

    Columns: Source, T_K, P_bar, x_exp, in_pub_set (published 02a/10 selection).
    """
    if gas == 'He':
        d = pd.read_csv(HE_CSV, comment='#')
        d = d[(d.Sal_m == 0) & (d.Use != 'exclude')].copy()
        out = pd.DataFrame({'Source': d.Source, 'T_K': d.T_K, 'P_bar': d.P_bar, 'x_exp': d.x_Gas,
                            'Use': d.Use})
        out['in_pub_set'] = (out.Use == 'fit')
        return out.reset_index(drop=True)
    d = pd.read_csv(PUB_CSV)
    d['Source'] = d['Source'].str.replace('Chahab', 'Chabab', regex=False)
    d = d[(d.Gas == gas) & d.x_gas_exp.notna()].copy()
    sel = (d.kij_AQ_conv == True) & (d.T_K >= T_MIN_K) & (d.T_K <= T_MAX_K)  # noqa: E712
    sel &= ~d.Source.isin(EXCLUDE_SOURCES.get(gas, set()))
    out = pd.DataFrame({'Source': d.Source, 'T_K': d.T_K, 'P_bar': d.P_bar, 'x_exp': d.x_gas_exp,
                        'y_exp': d.y_H2O_exp})
    out['in_pub_set'] = sel.values
    return out.reset_index(drop=True)
