# framework_refresh - Søreide-Whitson framework refresh pipeline

Analysis pipeline for the framework paper: updated aqueous and non-aqueous binary interaction parameters, embedded-salinity BIP, and re-selected Sechenov (salting-out) models across eight gases (CO₂, H₂S, CH₄, N₂, H₂, C₂H₆, C₃H₈, nC₄H₁₀). The recommended correlations and coefficients are summarised in the repository root README; full derivations and validation are in the paper.

## Pipeline (`code/`)

Numbered scripts, roughly in dependency order. Run them from the `code/` directory; they import the engine from `../../shared/vle_engine/` and read/write data under `../../shared/data/`.

| Script | Purpose |
|---|---|
| `00-Refit_Water_Alpha.py` | Mathias-Copeman alpha refit to IAPWS-95 (sensitivity study) |
| `00a-Archer_Water_Activity.py` | brine water-activity comparison |
| `01-Regress_Pointwise_Kij.py` | pointwise kij regression from experimental solubility points |
| `02-Regress_HC_Kij.py` | pointwise kij for the C₂H₆/C₃H₈/nC₄H₁₀ extension data |
| `02a-Fit_Dropin_Kij.py` | fit the recommended (drop-in) kij correlations |
| `03-Extract_Ks_From_Data.py` | extract experimental Sechenov coefficients from matched fresh/brine pairs |
| `04-Analyze_Kij_All_Gases.py` | kij correlation analysis and reporting |
| `05-Regress_Kij_NA.py` | non-aqueous BIP regression from water-content data |
| `06-Analyze_Salting_All_Gases.py` | Sechenov model comparison per gas |
| `07-Fit_Embedded_Salinity_All_Gases.py` | embedded delta-kij(T, m) fits |
| `07a-Fit_Embedded_Dropin.py` | embedded delta-kij fits on the drop-in basis (recommended coefficients) |
| `08-Build_Ks_Cache.py` | build the ks-band cache used by the figure script |
| `09-Fit_Flash_KValues.py` | K-value initialisation correlations (replacement for Wilson in gas-water flash) |
| `10-Validate_MARE_All_Gases.py` | headline MARE validation table, all gases and frameworks |
| `12-Generate_Figures.py` | all paper figures (written to `manuscript/figures/`) |

Typical use:

```bash
cd code
python 10-Validate_MARE_All_Gases.py
python 12-Generate_Figures.py
```
