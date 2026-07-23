# Søreide-Whitson Framework Refresh

Code and data supporting a refresh of the Søreide-Whitson (S&W, 1992) Peng-Robinson framework for gas-water-brine phase equilibria: updated binary interaction parameter (BIP) correlations for eight gases (CO₂, H₂S, CH₄, N₂, H₂, C₂H₆, C₃H₈, nC₄H₁₀), developed from approximately 2,000 pointwise-regressed BIP values, including extension of the framework to hydrogen for underground storage. The correlations retain the original S&W water alpha function and are drop-in compatible with existing simulator implementations: only the BIP correlation coefficients need updating.

The work is described in **Refreshed Søreide-Whitson Framework for Gas Solubility in Water and Brine with Extension to Hydrogen** (Burgoyne & Nielsen, accepted for publication in *Fluid Phase Equilibria*), referred to below as the paper. This README summarises the recommended relationships; derivations, data treatment, and validation detail are in the paper.

## Framework summary

The S&W framework models gas-water-brine equilibria with the Peng-Robinson EOS, a modified water alpha function carrying temperature and salinity dependence, and a dual-flash scheme: aqueous-phase BIPs ($k_{ij}^{\mathrm{AQ}}$) control dissolved gas content, non-aqueous-phase BIPs ($k_{ij}^{\mathrm{NA}}$) control the water content of the gas phase. The original S&W water alpha is retained unchanged:

$$\sqrt{\alpha_{\mathrm{H_2O}}} = 1 + 0.4530\left[1 - T_r\left(1 - 0.0103 c_{\mathrm{sw}}^{1.1}\right)\right] + 0.0034\left(T_r^{-3} - 1\right)$$

with $T_r = T/647.3~\mathrm{K}$ and $c_{\mathrm{sw}}$ the NaCl molality (mol/kg H₂O). The paper shows (its Table 1) that replacing this alpha with a higher-accuracy Mathias-Copeman form, or moving to a modular gamma-phi salting-out implementation, leaves solubility predictions unchanged once the BIP is refitted: the regressed $k_{ij}$ absorbs upstream model differences, which is why the simpler drop-in formulation is recommended.

## Component properties

All correlations in this work were regressed against the following properties and are valid only in combination with them ($T_c$, $P_c$, $\omega$, $T_b$ from S&W Table 3; H₂, absent from S&W, from NIST). $T_b$ enters only the Sechenov correlation.

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

## Recommended aqueous-phase BIP correlations, $k_{ij}^{\mathrm{AQ}}(T)$

Fitted to freshwater data only (T ≤ 200 °C), with $T_r = T/T_c$ of the gas. MAE is against pointwise-regressed $k_{ij}$ values; MARE is on predicted solubility; S&W MARE uses the original correlation at zero salinity on the same points.

| Gas | $T_c$ (K) | Correlation | n | MAE | MARE (%) | S&W MARE (%) |
|---|---|---|---|---|---|---|
| CO₂ | 304.20 | $-1.5893 + 3.0077 T_r - 2.0532 T_r^2 + 0.5207 T_r^3$ | 611 | 0.0119 | 16.4 | 23.0 |
| H₂S | 373.20 | $-0.2001/T_r + 1348.96 e^{-12.071/T_r} + 0.2260$ | 405 | 0.0112 | 38.5 | 51.3 |
| CH₄ | 190.60 | $(-2.1756 + T_r) / (1.0388 + 0.6436 T_r)$ | 115 | 0.0089 | 4.8 | 5.8 |
| N₂ | 126.10 | $-1.6689 + 0.4340 T_r$ | 127 | 0.0114 | 5.7 | 7.7 |
| H₂ | 33.145 | $(-14.9412 + T_r) / (2.2832 + 0.3893 T_r)$ | 154 | 0.0300 | 10.2 | n/a |
| C₂H₆ | 305.40 | $(-1.2669 + T_r) / (0.1526 + 1.4335 T_r)$ | 95 | 0.0095 | 12.9 | 15.9 |
| C₃H₈ | 369.80 | $(-1.1460 + T_r) / (0.5760 + 1.3107 T_r)$ | 59 | 0.0023 | 5.0 | 11.7 |
| nC₄H₁₀ | 425.20 | $-0.9354 + 1.2615 T_r - 0.3696 T_r^2$ (S&W form retained) | 27 | --- | --- | --- |

The H₂S MARE is dominated by very dilute points; excluding $x < 0.003$ it drops to roughly 8%. H₂ was not in the original framework; the rational form reproduces the U-shaped solubility with its minimum near 50 °C (see the paper for source selection and validation).

## Recommended non-aqueous-phase BIPs, $k_{ij}^{\mathrm{NA}}$

Original S&W constants are retained for six gases (re-optimisation gains at most 1.1 percentage points on water-content MARE). Two values are new or changed:

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

The Dubessy and Akinfiev model implementations are in `shared/vle_engine/_lib_salting_library.py`. H₂S had no salting-out support in the original framework.

## Embedded salinity BIP, $\Delta k_{ij}(T, m)$

For simulators where flash-routine modification is impractical, salinity is embedded directly in the aqueous BIP:

$$k_{ij}^{\mathrm{AQ}}(T, m) = k_{ij,\mathrm{fw}}(T) + \Delta k_{ij}(T, m)$$

with $\Delta k_{ij} = (a_0 + a_1 T_r + a_2 T_r^2) \cdot m$ for seven gases, and for CO₂ an added quadratic term $(b_0 + b_1 T_r) \cdot m^2$. The fitted coefficients reproduce the recommended Sechenov models with mean solubility deviations below 2%:

| Gas | $T_c$ (K) | $a_0$ | $a_1$ | $a_2$ | $b_0$ | $b_1$ | MAE (%) |
|---|---|---|---|---|---|---|---|
| CO₂ | 304.20 | 0.0409 | −0.0807 | 0.0526 | 0.0079 | −0.0085 | 1.56 |
| H₂S | 373.20 | 0.0341 | −0.0655 | 0.0376 | --- | --- | 1.99 |
| CH₄ | 190.60 | 0.1304 | −0.1295 | 0.0394 | --- | --- | 0.93 |
| N₂ | 126.10 | 0.2173 | −0.1468 | 0.0302 | --- | --- | 0.97 |
| H₂ | 33.145 | 0.3658 | −0.0625 | 0.0030 | --- | --- | 0.63 |
| C₂H₆ | 305.40 | 0.0813 | −0.1287 | 0.0646 | --- | --- | 1.14 |
| C₃H₈ | 369.80 | 0.0606 | −0.1165 | 0.0772 | --- | --- | 0.31 |
| nC₄H₁₀ | 425.20 | 0.0488 | −0.1072 | 0.0836 | --- | --- | 0.26 |

## Repository layout

```
shared/
  vle_engine/         importable VLE engine (SWBinaryVLE, multicomponent flash,
                      Sechenov model library, usage examples)
  data/               experimental datasets and regressed results (see data/README.md)
framework_refresh/    analysis pipeline (code/00-12): pointwise kij regression,
                      correlation fitting, Sechenov analysis, embedded-salinity fits,
                      K-value initialisation, MARE validation, figures
```

## Reproducing the results

Scripts read shared data and import the engine via paths relative to their own directory, so run the pipeline from its `code/` directory:

```bash
cd framework_refresh/code
python 10-Validate_MARE_All_Gases.py     # headline MARE table, all gases
python 12-Generate_Figures.py            # all figures
```

Dependencies: numpy, scipy, pandas, matplotlib, openpyxl.

## Authors

Mark Burgoyne (Santos Ltd) and Markus H. Nielsen (Whitson AS).

## References

- Søreide, I., Whitson, C.H., 1992. Peng-Robinson predictions for hydrocarbons, CO₂, N₂, and H₂S with pure water and NaCl brine. *Fluid Phase Equilib.* 77, 217-240. [doi:10.1016/0378-3812(92)85105-H](https://doi.org/10.1016/0378-3812(92)85105-H)
- Dubessy, J., Tarantola, A., Sterpenich, J., 2005. Modelling of liquid-vapour equilibria in the H₂O-CO₂-NaCl and H₂O-H₂S-NaCl systems to 270 °C. *Oil Gas Sci. Technol. - Rev. IFP* 60, 339-355. [doi:10.2516/ogst:2005022](https://doi.org/10.2516/ogst:2005022)
- Akinfiev, N.N., Majer, V., Shvarov, Yu.V., 2016. Thermodynamic description of H₂S-H₂O-NaCl solutions at temperatures to 573 K and pressures to 40 MPa. *Chem. Geol.* 424, 1-11. [doi:10.1016/j.chemgeo.2016.01.006](https://doi.org/10.1016/j.chemgeo.2016.01.006)
- The paper listed at the top of this README (under review); full experimental-source citations are given there.
