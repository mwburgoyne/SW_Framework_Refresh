# h2_solubility - H₂ solubility pipeline

Analysis pipeline for the H₂ paper: a new H₂-H₂O binary interaction parameter (rational form in reduced temperature) plus Sechenov salting-out treatment within the Søreide-Whitson framework, aimed at underground hydrogen storage. The recommended H₂ correlation and coefficients are summarised in the repository root README.

## Pipeline (`code/`)

Run scripts from the `code/` directory; they import the engine from `../../shared/vle_engine/` and read data from `../../shared/data/`.

| Script | Purpose |
|---|---|
| `01-Fit_Aqueous_BIP.py` | fit the H₂ aqueous BIP correlation to pointwise-regressed values |
| `02-Fit_Nonaqueous_BIP.py` | validate the constant non-aqueous BIP (0.468) against water-content data |
| `03-Fit_Embedded_Salinity_BIP.py` | fit the embedded salinity BIP for H₂ |
| `04-Validate_Errors.py` | MARE/MAE comparison against published H₂ correlations |
| `05-Validate_Manuscript_Claims.py` | numeric checks of values quoted in the paper |
| `06-Generate_Figures.py` | all paper figures |

Typical use:

```bash
cd code
python 06-Generate_Figures.py
```
