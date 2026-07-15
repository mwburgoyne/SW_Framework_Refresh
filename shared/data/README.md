# Data files

## Experimental data

### solubility_points.xlsx
Compiled experimental gas solubility and water-content data for the main gases (CO₂, H₂S, CH₄, N₂, H₂) from the literature sources cited in the papers (sheet `QCd Data`). Columns include temperature, pressure, phase mole fractions (x_gas, y_H2O), NaCl molality (`Sal_m`), and source attribution. H₂ sources span Wiebe & Gaddy (1934) to Chabab et al. (2023); see the papers for the full source lists, quality assessment, and exclusions.

### extracted_solubility_data.csv
Hydrocarbon extension data (C₂H₆, C₃H₈, nC₄H₁₀), digitised from Reamer, Kobayashi and related sources; used by `02-Regress_HC_Kij.py`.

### extracted_ks_values.csv
Experimental Sechenov coefficients extracted from matched freshwater/brine pairs (573 pairs; log₁₀ convention, kg/mol). Output of `03-Extract_Ks_From_Data.py` and input to the salting-out analysis.

### blount_ch4_salinity_pair_ks.csv
Implied Sechenov coefficients from the Blount & Price (1982) CH₄ brine dataset. Retained for the exclusion diagnostics reported in the framework paper (Section 5.5); this source is excluded from validation.

## Regressed results

### pointwise_kij_results_sw_alpha.csv
Pointwise-regressed kij values per experimental point using the original S&W water alpha. This is the basis of the recommended (drop-in) correlations and the MAE(kij) values quoted in the framework paper. Columns: gas, source, T, P, experimental x_gas / y_H2O, regressed kij_AQ and kij_NA with convergence flags.

### pointwise_kij_results.csv
Same regression using the Mathias-Copeman refitted water alpha; used in the alpha-function sensitivity study (framework paper Table 1) and by the H₂ paper pipeline.

### embedded_salinity_bip_all_gases.csv
Fitted embedded salinity BIP coefficients (a0, a1, a2 and, for CO₂, b0, b1) per gas, as tabulated in the root README.

### embedded_ks_band_cache.npz
Precomputed implied-Sechenov bands for the embedded BIP figures (regenerate with `08-Build_Ks_Cache.py`).
