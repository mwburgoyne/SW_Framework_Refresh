# Søreide-Whitson Framework Refresh

Code and data supporting a refresh of the Søreide-Whitson (S&W, 1992) Peng-Robinson framework for gas-water-brine phase equilibria: updated binary interaction parameter (BIP) correlations for eight gases (CO₂, H₂S, CH₄, N₂, H₂, C₂H₆, C₃H₈, nC₄H₁₀), developed from approximately 2,000 pointwise-regressed BIP values, including extension of the framework to hydrogen for underground storage. The correlations retain the original S&W water alpha function and are drop-in compatible with existing simulator implementations: only the BIP correlation coefficients need updating.

> **Coefficients updated November 2026.** The tables below are the current recommendation: the paper's correlations refitted on the coupled S&W equilibrium, in which each phase carries its own BIP set (Whitson & Brulé 2000, §9.3). An updated manuscript on this basis, not peer reviewed, is here: **[refit paper draft (PDF)](procedure1_refit/manuscript_REFIT/sw_framework_refresh_REFIT_draft.pdf)**; its Section 3 sets out the equilibrium procedures in use and why this one is adopted. The paper regressed its aqueous BIPs with the vapour treated as dry gas, so its printed $k_{ij}^{\mathrm{AQ}}$ and embedded salinity coefficients differ from these; they are kept in [PUBLISHED_COEFFICIENTS.md](PUBLISHED_COEFFICIENTS.md) for anyone checking the paper, and [ERRATA](ERRATA.md) item 5 explains the change. Non-aqueous BIPs, Sechenov models and component properties are as published. pyResToolbox 3.10 and later implement these coefficients.

The work is described in **Refreshed Søreide-Whitson framework for gas solubility in water and brine with extension to hydrogen** (Burgoyne & Nielsen, 2026, *Fluid Phase Equilibria*, 114824, [doi:10.1016/j.fluid.2026.114824](https://doi.org/10.1016/j.fluid.2026.114824)), referred to below as the paper. This README summarises the recommended relationships; derivations, data treatment, and validation detail are in the paper. This README gives the recommended relationships; derivations, data treatment and validation detail are in the paper, and the refit is documented in the [technical note](procedure1_refit/report/sw_procedure1_refit.pdf).

## Framework summary

The S&W framework models gas-water-brine equilibria with the Peng-Robinson EOS, a modified water alpha function carrying temperature and salinity dependence, and two sets of binary interaction parameters: aqueous-phase BIPs ($k_{ij}^{\mathrm{AQ}}$) for the water-rich liquid and non-aqueous BIPs ($k_{ij}^{\mathrm{NA}}$) for every other phase. Solubility and water content come from **one equilibrium** in which the aqueous phase uses $k_{ij}^{\mathrm{AQ}}$ and the vapour uses $k_{ij}^{\mathrm{NA}}$ and carries its equilibrium water: S&W's phase-specific mixture parameters (1992, Eqs. 10a/10b) applied simultaneously, as Whitson & Brulé (2000, §9.3) describe. The 1992 paper does not state how its equilibrium was solved; Equinor's NeqSim solves this coupled form. In a multicomponent flash, give a phase $k_{ij}^{\mathrm{AQ}}$ only when it is on the liquid root and mostly water; a gas-rich phase evaluated with both sets can split spuriously. The coefficients below were fitted this way and are valid only in this procedure. The original S&W water alpha is retained unchanged:

$$\sqrt{\alpha_{\mathrm{H_2O}}} = 1 + 0.4530\left[1 - T_r\left(1 - 0.0103 c_{\mathrm{sw}}^{1.1}\right)\right] + 0.0034\left(T_r^{-3} - 1\right)$$

with $T_r = T/647.3~\mathrm{K}$ and $c_{\mathrm{sw}}$ the NaCl molality (mol/kg H₂O). The paper shows (its Table 1) that replacing this alpha with a higher-accuracy Mathias-Copeman form, or moving to a modular gamma-phi salting-out implementation, leaves solubility predictions unchanged once the BIP is refitted: the regressed $k_{ij}$ absorbs upstream model differences, which is why the simpler drop-in formulation is recommended.

## Why these differ from the paper

Three ways of combining the two BIP sets appear in practice. They differ only in how much water the vapour carries when its gas fugacity is equated to the aqueous phase's:

| Procedure | Non-aqueous phase | Used by |
|---|---|---|
| 0, dry vapour | pure gas at total pressure | the paper (regression and every MARE); [published coefficients](PUBLISHED_COEFFICIENTS.md) |
| 1, coupled | one equilibrium; aqueous phase $k_{ij}^{\mathrm{AQ}}$, vapour $k_{ij}^{\mathrm{NA}}$ | Whitson & Brulé (2000, §9.3); NeqSim; **the coefficients on this page** |
| 2, dual flash | $k_{ij}^{\mathrm{AQ}}$ in every phase for the solubility flash | Whitson & Brulé (2000, §9.3) shortcut for single-BIP EOS software |

The difference grows with temperature and falls with pressure. In a procedure-1 calculation the published $k_{ij}^{\mathrm{AQ}}$ underpredict solubility by 12-29% (bias) at 150-200 °C for CO₂, H₂S, CH₄ and H₂; the refit removes that bias (−6 to +3%). Below 60 °C the two sets agree within 2 percentage points of MARE. Every pointwise $k_{ij}^{\mathrm{AQ}}$ moves down, by less than 0.001 at or below 60 °C and by 0.013-0.086 (gas mean) at 150-200 °C.

Only the equilibrium procedure changed: same data and exclusions, 0-200 °C window, functional forms, L1 routine and Sechenov targets as the paper. Five CO₂ points (Wang 2021, 160-200 °C at 5-10 bar, below the water vapour pressure) cannot be evaluated on procedure 1 and are dropped. B. Younus (Whitson AS) raised the procedure question; the procedure-1 solver reproduces his independent water contents to the printed digits.

## Component properties

All correlations in this work were regressed against the following properties and are valid only in combination with them ($T_c$, $P_c$, $\omega$, $T_b$ from S&W Table 3; H₂ and He, absent from S&W, from NIST; helium is a post-publication extension, see [Helium](#helium-post-publication-extension)). $T_b$ enters only the Sechenov correlation.

| Component | $T_c$ (K) | $P_c$ (bar) | $\omega$ | $T_b$ (K) |
|---|---|---|---|---|
| CO₂ | 304.2 | 73.8 | 0.2273 | 194.7 |
| H₂S | 373.2 | 89.4 | 0.1081 | 212.8 |
| N₂ | 126.1 | 34.0 | 0.0403 | 77.3 |
| CH₄ | 190.6 | 46.0 | 0.0108 | 111.6 |
| H₂ | 33.145 | 12.964 | −0.219 | 20.3 |
| C₂H₆ | 305.4 | 48.8 | 0.0998 | 184.6 |
| C₃H₈ | 369.8 | 42.5 | 0.1517 | 231.1 |
| nC₄H₁₀ | 425.2 | 38.0 | 0.1931 | 272.7 |
| H₂O | 647.3 | 221.2 | 0.3434 | --- |
| He | 5.1953 | 2.28323 | −0.38354 | 4.2238 |

## Recommended aqueous-phase BIP correlations, $k_{ij}^{\mathrm{AQ}}(T)$

Fitted to freshwater data only (T ≤ 200 °C) on procedure 1; $T$ in K, $T_r = T/T_c$. Use every digit: the parameters are strongly correlated. MAE is against pointwise-regressed $k_{ij}$ values; MARE is on predicted solubility, and S&W MARE scores the original S&W correlations on the same points in the same procedure.

| Gas | Form | Coefficients | n | MAE($k_{ij}$) | MARE (%) | S&W MARE (%) |
|---|---|---|---|---|---|---|
| CO₂ | $c_0 + c_1T + c_2T^2 + c_3T^3$ | $c_0 = -1.5502105835$, $c_1 = 9.3227195169\times10^{-3}$, $c_2 = -1.9777020578\times10^{-5}$, $c_3 = 1.5277320676\times10^{-8}$ | 606 | 0.0087 | 12.4 | 15.3 |
| H₂S | $a/T + b e^{c/T} + d$ | $a = -82.780980134$, $b = 1.7394905144\times10^{6}$, $c = -8585.0633295$, $d = 0.25183060821$ | 405 | 0.0070 | 31.9 (6.8*) | 30.8 (8.0*) |
| CH₄ | $(A + T_r)/(B + C T_r)$, $T_c$ 190.6 K | $A = -2.1976782374$, $B = 0.71501536698$, $C = 0.89450296392$ | 115 | 0.0076 | 3.9 | 3.9 |
| N₂ | $a + bT$ | $a = -1.6210963968$, $b = 3.2952725039\times10^{-3}$ | 127 | 0.0104 | 5.3 | 7.2 |
| H₂ | $(A + T_r)/(B + C T_r)$, $T_c$ 33.145 K | $A = -16.019825768$, $B = 0.73123628347$, $C = 0.68136196802$ | 154 | 0.0350 | 11.3 | n/a |
| C₂H₆ | $(A + T_r)/(B + C T_r)$, $T_c$ 305.4 K | $A = -1.2713478883$, $B = -0.029271366247$, $C = 1.6371834535$ | 95 | 0.0063 | 8.4 | 11.3 |
| C₃H₈ | $(A + T_r)/(B + C T_r)$, $T_c$ 369.8 K | $A = -1.1567778736$, $B = 0.20894596747$, $C = 1.8106628711$ | 59 | 0.0015 | 3.4 | 8.9 |
| nC₄H₁₀ | $-0.9354 + 1.2615 T_r - 0.3696 T_r^2$, $T_c$ 425.2 K | S&W form retained | 27 | --- | --- | --- |
| He | $(A + T_r)/(B + C T_r)$, $T_c$ 5.1953 K | $A = -71.615504934$, $B = -0.16440701914$, $C = 0.50173713895$ | 130 | 0.0184 | 6.3 | n/a |

\* Excluding 18 H₂S points (Selleck 1952 at 37.8 °C, where solubility falls as pressure rises, and Suleimenov 1994 at or below 60 °C) that every model misses by 250-1,200%. MARE n for H₂ is 142 (the kij fit uses 154). Helium is a post-publication extension, not peer reviewed. Full-precision values: [`procedure1_refit/results/p1_params.json`](procedure1_refit/results/p1_params.json). Check values ($x_{\mathrm{gas}}$, $y_{\mathrm{H_2O}}$ at three states per gas) for verifying an implementation: [`procedure1_refit/results/check_values.csv`](procedure1_refit/results/check_values.csv).

## Recommended non-aqueous-phase BIPs, $k_{ij}^{\mathrm{NA}}$

Unchanged by the procedure-1 refit (the largest re-optimisation gain on procedure 1 is 0.9 percentage points, below the paper's 1.1-point rule). Original S&W constants are retained for six gases; two values are new or changed:

| Gas | $k_{ij}^{\mathrm{NA}}$ | Status |
|---|---|---|
| H₂S | **0.161** (constant) | Replaces S&W Eq. 17 ($0.19031 - 0.05965 T_r$); water-content MARE 11.7% → 8.6% |
| H₂ | **0.468** (constant) | New; consistent with the CH₄ (0.485) / N₂ (0.478) volatility trend |
| CO₂ | 0.190 | Retained from S&W Table 5 |
| CH₄ | 0.485 | Retained |
| N₂ | 0.478 | Retained |
| C₂H₆ | 0.492 | Retained |
| C₃H₈ | 0.5525 | Retained (S&W Table 5 as printed; the 0.5070 in earlier states of this repository was a transcription error - see ERRATA) |
| nC₄H₁₀ | 0.5091 | Retained (S&W Table 5 as printed; no quality-filtered water-content data) |
| He | 0.468 | Surrogate (the H₂ value); no helium water-content data found, so unvalidated |

## Recommended Sechenov (salting-out) models

Gas solubility in brine follows $\log_{10}(x_{\mathrm{fresh}}/x_{\mathrm{brine}}) = k_s \cdot m$ with $k_s$ in kg/mol and $m$ the NaCl molality. The S&W generalised correlation (their Eq. 8), converted to SI units,

$$k_s = 1.3012 + 4.45\times10^{-4} T_b - 8.769\times10^{-3} T + 2.0293\times10^{-5} T^2 - 1.5233\times10^{-8} T^3$$

($T$, $T_b$ in K), remains adequate for the non-polar gases but not for the acid gases. Recommendations:

| Gas | Recommended model | $k_s$ MAE (kg/mol) |
|---|---|---|
| CO₂ | Dubessy et al. (2005) − 0.011 | 0.020 |
| H₂S | Akinfiev et al. (2016) + 0.019 | 0.013 |
| CH₄ | S&W Eq. 8 | 0.006 |
| N₂ | S&W Eq. 8 + 0.02 | 0.007 |
| H₂ | S&W Eq. 8 with $T_b = 20.3$ K | 0.009 |
| C₂H₆, C₃H₈, nC₄H₁₀ | S&W Eq. 8 (no reservoir-condition brine data; retained) | --- |
| He | S&W Eq. 8 with $T_b = 4.2238$ K | 0.0067 (n = 48) |

The Dubessy and Akinfiev model implementations are in `shared/vle_engine/_lib_salting_library.py`. H₂S had no salting-out support in the original framework.

## Embedded salinity BIP, $\Delta k_{ij}(T, m)$

For simulators where flash-routine modification is impractical, salinity is embedded directly in the aqueous BIP:

$$k_{ij}^{\mathrm{AQ}}(T, m) = k_{ij,\mathrm{fw}}(T) + \Delta k_{ij}(T, m)$$

with $\Delta k_{ij} = (a_0 + a_1 T_r + a_2 T_r^2) \cdot m$, and for CO₂ an added quadratic term $(b_0 + b_1 T_r) \cdot m^2$. Fitted on procedure 1 to the recommended Sechenov models; MAE and max are solubility deviations from those targets on the fitting grid:

| Gas | $T_c$ (K) | $a_0$ | $a_1$ | $a_2$ | $b_0$ | $b_1$ | MAE (%) | max (%) |
|---|---|---|---|---|---|---|---|---|
| CO₂ | 304.20 | 0.0408481 | −0.0789861 | 0.0502757 | 0.00676611 | −0.00735569 | 2.92 | 15.8 |
| H₂S | 373.20 | 0.0474380 | −0.0916139 | 0.0498240 | --- | --- | 2.35 | 19.1 |
| CH₄ | 190.60 | 0.134391 | −0.134258 | 0.0408428 | --- | --- | 0.98 | 3.7 |
| N₂ | 126.10 | 0.219484 | −0.148600 | 0.0306167 | --- | --- | 1.04 | 4.2 |
| H₂ | 33.145 | 0.376712 | −0.0647304 | 0.00315541 | --- | --- | 0.66 | 2.4 |
| C₂H₆ | 305.40 | 0.0777696 | −0.123360 | 0.0625866 | --- | --- | 1.09 | 3.6 |
| C₃H₈ | 369.80 | 0.0580567 | −0.111055 | 0.0692266 | --- | --- | 1.40 | 3.3 |
| nC₄H₁₀ | 425.20 | 0.0456416 | −0.100089 | 0.0729267 | --- | --- | 1.75 | 3.4 |
| He | 5.1953 | 0.363020 | −0.00963936 | 7.09302e-5 | --- | --- | 0.56 | 1.3 |

## Fit quality against the paper

Freshwater solubility MARE (%), bias in brackets: published coefficients in the paper's dry-vapour solve (Pub-P0, as published), published coefficients on procedure 1 (Pub-P1), S&W 1992 on procedure 1 (SW-P1), and the recommended coefficients (New-P1).

| Gas | n | Pub-P0 | Pub-P1 | SW-P1 | New-P1 |
|---|---|---|---|---|---|
| CO₂ | 611 (606) | 16.4 (+10.7) | 14.9 (−1.0) | 15.3 (−6.3) | **12.4** (+4.8) |
| H₂S | 405 | 38.5 (+30.4) | 38.2 (+13.9) | 30.8 (+20.6) | 31.9 (+25.7) |
| H₂S, excl. 18 suspect | 387 | 13.8 | 13.8 | 8.0 | **6.8** |
| CH₄ | 115 | 4.8 (−0.3) | 4.7 (−3.0) | 3.9 (−0.6) | **3.9** (−0.7) |
| N₂ | 127 | 5.7 (+1.6) | 5.6 (−0.4) | 7.2 (+3.7) | **5.3** (+1.2) |
| H₂ | 142 | **10.2** (+3.0) | 12.0 (−1.1) | --- | 11.3 (+2.3) |
| C₂H₆ | 95 | 12.9 (+4.6) | 9.2 (−2.4) | 11.3 (−5.5) | **8.4** (+0.8) |
| C₃H₈ | 59 | 5.0 (+1.4) | 4.8 (−1.2) | 8.9 (+4.8) | **3.4** (+1.2) |

H₂ is the one gas where the refit scores worse than the published solve (11.3% vs 10.2%), all of it at 100-200 °C where the data are sparse. The paper's S&W column (23.0, 51.3, 5.8% for CO₂, H₂S, CH₄) scored S&W's coefficients in the dry-vapour solve, which penalised them (ERRATA item 5).

## Helium (post-publication extension)

Helium was added after publication. **It is not part of the published paper** and was not peer reviewed with it.

- Component constants are NIST (Ortiz-Vega et al. 2019 equation of state). These are not the helium constants of the 5-component Z-factor model, whose helium Tc is a deliberately non-standard 6.35 °R tuned for viscosity.
- Aqueous BIP fitted L1 to 130 fugacity-targeted pointwise values from Gardiner & Smith (1972, doi:10.1021/j100652a019), Gerth (1983, doi:10.1007/BF00648669), Abrosimov & Lebedeva (2013, evaluated values, doi:10.1134/S0036023613070024), Pray et al. (1952, doi:10.1021/ie50509a058) and Potter & Clynne (1978, doi:10.1007/BF00650811), 20-163 °C and 1-1000 bar. Not validated above 200 °C. Per-source errors are in the technical note's helium section.
- S&W Eq. 8 matches measured helium Sechenov coefficients in NaCl to 0.0067 mean absolute deviation (n = 48, 25-100 °C).
- $k_{ij}^{\mathrm{NA}}$ = 0.468 is a surrogate, so the water content of helium-rich gas is unvalidated.
- Flash K-value initialisation uses a fitted Cross-form row (Wilson gives K < 1 for helium).

The helium data are in `shared/data/he_solubility.csv`. The October 2026 helium coefficients are in [PUBLISHED_COEFFICIENTS.md](PUBLISHED_COEFFICIENTS.md).

## Repository layout

```
shared/
  vle_engine/         the paper's VLE engine (SWBinaryVLE, multicomponent flash,
                      Sechenov model library, usage examples); procedure 0
  data/               experimental datasets and regressed results (see data/README.md),
                      including the helium data (he_solubility.csv)
framework_refresh/    the paper's analysis pipeline (code/00-12): pointwise kij regression,
                      correlation fitting, Sechenov analysis, embedded-salinity fits,
                      K-value initialisation, MARE validation, figures
procedure1_refit/     the recommended coefficients: refit on procedure 1, coupled (solver,
                      refit, validation, results, technical note); see its README
PUBLISHED_COEFFICIENTS.md   the paper's coefficients as printed
```

## Reproducing the results

**The paper's published numbers.** The pipeline and `shared/vle_engine/` reproduce the paper on its dry-vapour solve (procedure 0), by design. Do not drop the recommended coefficients into this engine: it would mismatch them the other way round. Run from the pipeline's `code/` directory:

```bash
cd framework_refresh/code
python 10-Validate_MARE_All_Gases.py     # the paper's MARE table, all gases
python 12-Generate_Figures.py            # the paper's figures
```

**The recommended coefficients.** They come from, and run in, `procedure1_refit/` (procedure-1 solver `p1_solver.py`). Run from that directory in order 01, 02, 05, 07, 10_validate_fresh, 10_validate_brine, 11 (see `procedure1_refit/README.md`).

Dependencies: numpy, scipy, pandas, matplotlib, openpyxl.

## Authors

Mark Burgoyne (Santos Ltd) and Markus H. Nielsen (Whitson AS).
