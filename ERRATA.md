# Errata

Corrections and reproducibility notes relative to the paper and earlier states of this repository.

## 1. CH4 embedded-salinity routing bug (fixed 2026-07-19, commit e1e2bbd)

`_SW_GASES_WITH_EMBEDDED_SALINITY` in `shared/vle_engine/_lib_vle_engine.py` omitted CH4. With `framework='sw_original'` and `salinity_method='auto'`, the engine applied the salinity-embedded S&W Eqs 11-12 k_ij and then multiplied by a Sechenov factor as well, double-counting salinity for CH4 in brine. The internal MARE report gave 46.5% for the original S&W framework on the CH4 brine data (O'Sullivan 1970, n = 32); the corrected value is 11.7%.

No number in the paper is affected. The parity figures construct the S&W-original brine comparison through a separate code path that did not carry the bug, and the 46.5% value appears nowhere in the manuscript. The 'proposed' and 'dropin' frameworks are unaffected.

## 2. Spycher & Pruess comparison values depend on pyrestoolbox version

The paper quotes Spycher & Pruess (2010) CO2 solubility MARE values computed with the `pyrestoolbox` implementation (`CO2_Brine_Mixture`): 11.7% on freshwater data (n = 611) and 9.0% on brine data (n = 109), with sub-breakdowns 17.2% (P < 50 bar), 6.4% (50-100 degC) and 7.3% (P >= 500 bar).

`CO2_Brine_Mixture` has since been revised upstream. pyrestoolbox 3.6+ returns 13.0% freshwater and 8.9% brine (sub-breakdowns 18.6%, 5.6% and 4.5%). Rerunning `framework_refresh/code/10-Validate_MARE_All_Gases.py` with a current pyrestoolbox therefore reproduces these values rather than the earlier ones. The comparison's conclusion is unchanged: the dedicated Spycher & Pruess model remains the most accurate option for pure CO2-brine solubility.

**Resolved:** the revised manuscript (R3, 2026-07-20, and the accepted final version) quotes the current pyrestoolbox 3.6+ values (13.0% / 8.9% and matching sub-breakdowns), so the paper and this repository now agree.

## 3. Parity-figure legends: S&W-original brine series (fixed 2026-07-19)

The parity figures in the paper (Figures with panels for CO2, H2S, N2, CH4 brine data) computed the "S&W original" brine series with the freshwater water-alpha (csw = 0) while using the salinity-embedded S&W k_ij, dropping the csw term of the S&W alpha function. The figure legends therefore disagree with the correctly computed S&W-original MARE values quoted in the paper's text and tables:

| Series (brine) | Figure legend | Correct |
|---|---|---|
| CO2, n = 109 | 17.5% | 20.1% |
| H2S, n = 100 | 57.8% | 63.0% |
| N2, n = 45 | 11.4% | 4.2% |
| CH4, n = 32 | 6.5% | 11.7% |

The error affects only the S&W-original comparison series; the "This work" and "Explicit Sechenov" series are correct as published. The direction is mixed: the figures understate the original S&W error for CO2, H2S and CH4, and overstate it for N2. The text's quantitative claims use the correct values throughout. Relatedly, the H2 freshwater parity panel included Gillespie (1980) data (n = 160, 9.7%) whereas the paper's stated basis excludes that source; with the exclusion applied the panel reads n = 148, 10.0% (measured-point basis; the text's 10.2% over n = 142 uses the converged-fit basis).

`12-Generate_Figures.py` in this repository is fixed (S&W-original brine series constructed with the brine alpha; Gillespie exclusion added for H2), and regenerated figures match the text values. **Resolved:** the accepted final version of the paper carries the regenerated figures with corrected legends.

## 4. Property standardisation and S&W Table 5 transcription corrections (2026-07-23)

A pre-submission source-verification audit of the accepted manuscript traced every S&W-attributed constant back to the original paper and corrected the following in this repository and in the final manuscript:

- **kij_NA transcription errors.** Earlier states of this repository (and the manuscript through R3) gave the "retained S&W Table 5" non-aqueous BIPs for C3H8 and nC4H10 as 0.5070 and 0.5080. S&W 1992 Table 5 actually prints **0.5525** and **0.5091**. The corrected C3H8 value is also effectively optimal against the water-content data (MARE 4.1% at 0.5525 vs 7.4% at the erroneous 0.5070; re-optimisation gains 0.0 pp). The engine, README and manuscript now carry the printed S&W values.
- **Acentric factors standardised on S&W Table 3.** The component database previously used omega = 0.0986 (C2H6) and 0.1524 (C3H8) from a different property lineage; these are now the S&W Table 3 values (0.0998, 0.1517), and the C2H6/C3H8 kij_AQ correlations were re-regressed and refitted against the standardised properties (function-level change below 0.005 in kij across the data range). A component-properties table was added to the README and the paper; the correlations are valid only with those properties.
- **C2H6 fit set n = 95.** The 12th point of the Rettich (1981) low-pressure family (275.45 K, 0.71 bar), missed in the January extraction, is restored; its 11 siblings were already in the fit set.
- **Water-content MAREs regenerated.** The kij_NA summary numbers in R3 were generated before the 2026-07-19 water-content solver hardening (erratum 1 era) and did not reproduce with the shipped code. All Table values (paper Table with recommended kij_NA) are regenerated with the current solver; the H2S recommendation is unchanged at 0.161 (MARE 11.7% for S&W Eq. 17 vs 8.6% at 0.161 over 162 filtered points).
- **kij_NA regression robustness.** The hardened water-content solver returns NaN where no valid root exists; NaN slipped past the objective guards in `01-Regress_Pointwise_Kij.py` and silently defeated the bounded minimiser for some high-temperature points. Fixed with an explicit isfinite guard and a valid-basin grid scan. Four C2H6 points (238-300 degC, above the framework's 200 degC ceiling) are genuinely unreachable under the hardened solver and are excluded; the single nC4H10 water-content point present in early data states is no longer available, and the nC4H10 kij_NA is adopted from S&W unchanged.
