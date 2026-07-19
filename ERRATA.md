# Errata

Corrections and reproducibility notes relative to the paper and earlier states of this repository.

## 1. CH4 embedded-salinity routing bug (fixed 2026-07-19, commit e1e2bbd)

`_SW_GASES_WITH_EMBEDDED_SALINITY` in `shared/vle_engine/_lib_vle_engine.py` omitted CH4. With `framework='sw_original'` and `salinity_method='auto'`, the engine applied the salinity-embedded S&W Eqs 11-12 k_ij and then multiplied by a Sechenov factor as well, double-counting salinity for CH4 in brine. The internal MARE report gave 46.5% for the original S&W framework on the CH4 brine data (O'Sullivan 1970, n = 32); the corrected value is 11.7%.

No number in the paper is affected. The parity figures construct the S&W-original brine comparison through a separate code path that did not carry the bug, and the 46.5% value appears nowhere in the manuscript. The 'proposed' and 'dropin' frameworks are unaffected.

## 2. Spycher & Pruess comparison values depend on pyrestoolbox version

The paper quotes Spycher & Pruess (2010) CO2 solubility MARE values computed with the `pyrestoolbox` implementation (`CO2_Brine_Mixture`): 11.7% on freshwater data (n = 611) and 9.0% on brine data (n = 109), with sub-breakdowns 17.2% (P < 50 bar), 6.4% (50-100 degC) and 7.3% (P >= 500 bar).

`CO2_Brine_Mixture` has since been revised upstream. pyrestoolbox 3.6+ returns 13.0% freshwater and 8.9% brine (sub-breakdowns 18.6%, 5.6% and 4.5%). Rerunning `framework_refresh/code/10-Validate_MARE_All_Gases.py` with a current pyrestoolbox therefore reproduces these values rather than the published ones. The comparison's conclusion is unchanged: the dedicated Spycher & Pruess model remains the most accurate option for pure CO2-brine solubility. If a further revision of the paper is submitted, these values will be updated.

## 3. Parity-figure legends: S&W-original brine series (fixed 2026-07-19)

The parity figures in the paper (Figures with panels for CO2, H2S, N2, CH4 brine data) computed the "S&W original" brine series with the freshwater water-alpha (csw = 0) while using the salinity-embedded S&W k_ij, dropping the csw term of the S&W alpha function. The figure legends therefore disagree with the correctly computed S&W-original MARE values quoted in the paper's text and tables:

| Series (brine) | Figure legend | Correct |
|---|---|---|
| CO2, n = 109 | 17.5% | 20.1% |
| H2S, n = 100 | 57.8% | 63.0% |
| N2, n = 45 | 11.4% | 4.2% |
| CH4, n = 32 | 6.5% | 11.7% |

The error affects only the S&W-original comparison series; the "This work" and "Explicit Sechenov" series are correct as published. The direction is mixed: the figures understate the original S&W error for CO2, H2S and CH4, and overstate it for N2. The text's quantitative claims use the correct values throughout. Relatedly, the H2 freshwater parity panel included Gillespie (1980) data (n = 160, 9.7%) whereas the paper's stated basis excludes that source; with the exclusion applied the panel reads n = 148, 10.0% (measured-point basis; the text's 10.2% over n = 142 uses the converged-fit basis).

`12-Generate_Figures.py` in this repository is fixed (S&W-original brine series constructed with the brine alpha; Gillespie exclusion added for H2), and regenerated figures match the text values. The published PDFs retain the original legends unless a further revision is submitted.
