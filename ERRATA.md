# Errata

Corrections and reproducibility notes relative to the paper and earlier states of this repository.

## 1. CH4 embedded-salinity routing bug (fixed 2026-07-19, commit e1e2bbd)

`_SW_GASES_WITH_EMBEDDED_SALINITY` in `shared/vle_engine/_lib_vle_engine.py` omitted CH4. With `framework='sw_original'` and `salinity_method='auto'`, the engine applied the salinity-embedded S&W Eqs 11-12 k_ij and then multiplied by a Sechenov factor as well, double-counting salinity for CH4 in brine. The internal MARE report gave 46.5% for the original S&W framework on the CH4 brine data (O'Sullivan 1970, n = 32); the corrected value is 11.7%.

No number in the paper is affected. The parity figures construct the S&W-original brine comparison through a separate code path that did not carry the bug, and the 46.5% value appears nowhere in the manuscript. The 'proposed' and 'dropin' frameworks are unaffected.

## 2. Spycher & Pruess comparison values depend on pyrestoolbox version

The paper quotes Spycher & Pruess (2010) CO2 solubility MARE values computed with the `pyrestoolbox` implementation (`CO2_Brine_Mixture`): 11.7% on freshwater data (n = 611) and 9.0% on brine data (n = 109), with sub-breakdowns 17.2% (P < 50 bar), 6.4% (50-100 degC) and 7.3% (P >= 500 bar).

`CO2_Brine_Mixture` has since been revised upstream. pyrestoolbox 3.6+ returns 13.0% freshwater and 8.9% brine (sub-breakdowns 18.6%, 5.6% and 4.5%). Rerunning `framework_refresh/code/10-Validate_MARE_All_Gases.py` with a current pyrestoolbox therefore reproduces these values rather than the published ones. The comparison's conclusion is unchanged: the dedicated Spycher & Pruess model remains the most accurate option for pure CO2-brine solubility. If a further revision of the paper is submitted, these values will be updated.
