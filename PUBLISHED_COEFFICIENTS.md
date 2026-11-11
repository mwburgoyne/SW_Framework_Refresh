# Published coefficients (Fluid Phase Equilibria 114824, as printed)

These are the aqueous-phase BIP correlations and embedded salinity coefficients as published in Burgoyne & Nielsen (2026), *Fluid Phase Equilibria* 114824, [doi:10.1016/j.fluid.2026.114824](https://doi.org/10.1016/j.fluid.2026.114824), kept here for anyone checking or reproducing the paper.

**Do not use them in a Søreide-Whitson calculation.** They were regressed, and every MARE below was computed, with the vapour treated as pure, dry gas at total pressure ("procedure 0"). They reproduce the paper only inside that solve. In S&W's own procedure (one equilibrium, $k_{ij}^{\mathrm{AQ}}$ in the aqueous phase, $k_{ij}^{\mathrm{NA}}$ in the water-carrying vapour) they underpredict solubility by 12-29% at 150-200 °C for CO₂, H₂S, CH₄ and H₂. The recommended coefficients are on the [front page](README.md); the reason for the change is [ERRATA](ERRATA.md) item 5. The non-aqueous BIPs, Sechenov models and component properties did not change and are only on the front page.

The S&W comparison column below scored S&W's coefficients in the same dry-vapour solve, which penalised them; on their own procedure S&W 1992 gives 15.3% (CO₂), 30.8% (H₂S) and 3.9% (CH₄).

The paper's pipeline in `framework_refresh/code/` and `shared/vle_engine/` reproduces these numbers.

## Aqueous-phase BIP correlations (published), $k_{ij}^{\mathrm{AQ}}(T)$

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

## Embedded salinity BIP, $\Delta k_{ij}(T, m)$ (published)

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

## Helium (October 2026 extension, before the procedure-1 refit)

Helium was never in the paper. Its October 2026 coefficients, superseded by the procedure-1 values on the front page (the curve moves by −0.0002 on average because the October fit already targeted the helium fugacity):

- Aqueous BIP (dropin, S&W water alpha): kij_AQ = (−71.6152 + Tr)/(−0.394962 + 0.505807 Tr), Tr = T/5.1953 K, fitted L1 to 130 fugacity-targeted pointwise values from Gardiner & Smith (1972, doi:10.1021/j100652a019), Gerth (1983, doi:10.1007/BF00648669), Abrosimov & Lebedeva (2013, evaluated values, doi:10.1134/S0036023613070024), Pray et al. (1952, doi:10.1021/ie50509a058) and Potter & Clynne (1978, doi:10.1007/BF00650811), 20-163 °C and 1-1000 bar. Solubility mean absolute error 2.4-6.8% by source below 150 °C, with the pressure bias inherent to a kij(T) form. Not validated above 200 °C. The same kij is mapped, unrefitted, into the proposed and sw_original dispatches.
- Embedded salinity BIP (October; procedure-1 values on the front page): Δkij = (0.359923 − 0.0095344 Tr + 7.00398e-5 Tr²) m, fitted to S&W Eq 8 Sechenov targets (Tb 4.2238 K) at 25-100 °C; Eq 8 matches measured helium Sechenov coefficients in NaCl to 0.007 mean absolute error.
