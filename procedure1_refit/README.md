# Procedure-1 refit (post-publication, November 2026)

This directory is separate from the published method on purpose. Nothing here modifies `shared/vle_engine/_lib_vle_engine.py`, the published coefficients, or the `framework_refresh/` pipeline.

## Why

The published paper (Fluid Phase Equilibria 114824) regressed kij_AQ, and computed every solubility MARE, with the vapour treated as pure dry gas at total pressure ("procedure 0", `SWBinaryVLE._calc_x_with_kij`). Procedure 1 is one equilibrium in which the aqueous phase uses kij_AQ and the non-aqueous phase uses kij_NA, with the vapour carrying its equilibrium water: S&W's phase-specific mixture parameters (1992, Eqs. 10a/10b) applied simultaneously, as Whitson & Brulé (2000, §9.3) describe and Equinor's NeqSim solves. The 1992 paper does not state how its regression was solved; see the refit paper draft (`manuscript_REFIT/`), Section 3. The difference is the water dilution of the gas in the vapour, which grows with temperature and falls with pressure. B. Younus (Whitson AS, November 2026) raised the procedure question.

## What

`p1_solver.py` solves procedure 1 for a gas-water binary. It reuses the engine's EOS routines and adds only the phase-equilibrium iteration. It reproduces the seven water contents and procedure-2/procedure-1 solubility ratios in Younus' Table 4 to the printed digits (independent implementation), with fugacity residuals near 1e-13.

| Script | Does |
|---|---|
| `01_pointwise_kaq_p1.py` | pointwise kij_AQ per freshwater point on procedure 1 |
| `02_fit_kaq_p1.py` | same forms, same L1 routine (script 02a), same fit set as the published dropin fit |
| `05_kna_p1.py` | kij_NA re-check on procedure 1 (script 05 rows, grid and ranges; paper's 1.1 pp rule) |
| `07_embedded_p1.py` | embedded Delta-kij via script 07a routines (same Sechenov targets and grids) |
| `10_validate_fresh_p1.py`, `10_validate_brine_p1.py` | MARE: published (as published), published-in-procedure-1, S&W 1992 in procedure 1, refit |

Coefficients: `results/p1_params.json` (single source; `embedded` = primary brine set on the paper's Sechenov target definition, ratio at equal T and total P; `embedded_B` = sensitivity, activity coefficient at equal gas fugacity, brine MAREs within 0.3 pp). Reports: `logs/`. Technical note: `report/sw_procedure1_refit.pdf`.

## Data selection

Identical to the published sets (same convergence flags, 0-200 C window, source exclusions, H2 Gillespie rule for MARE). Helium (post-publication extension) reads `shared/data/he_solubility.csv` with its fit/validate flags. One group of points cannot be evaluated on procedure 1 and is dropped from the fit and the procedure-1 MAREs: 5 CO2 points (Wang 2021, 160-200 C at 5-10 bar, below the water vapour pressure, so P must be CO2 partial pressure). Nine H2S points near the water-saturated mixture critical locus (Selleck 1952, Koschel 2013, 104-138 C, 155-300 bar) need the edge-refined root search in 01; all converge. Data QC flags (not removed): Selleck 1952 at 37.8 C (6 points, x falls with P) and Suleimenov 1994 at <= 60 C (12 points), 250-1,200% error for every model.

Run from this directory, in order: 01, 02, 05, 07 (`--basis=A`, default), 10_fresh, 10_brine, 11.
