# shared - engine and data used by both pipelines

The VLE engine and datasets used by the analysis pipeline in `framework_refresh/`.

## `vle_engine/`

The importable VLE engine. Pipeline scripts add this directory to `sys.path`.

- `_lib_vle_engine.py` - core engine: `SWBinaryVLE` (binary gas-water flash in the S&W framework), `SWMultiComponentFlash`, `calc_gas_brine_equilibrium`, component property tables, and the framework toggles (original S&W, drop-in updated correlations, embedded salinity).
- `_lib_salting_library.py` - Sechenov model library: S&W Eq. 8, Dubessy et al. (2005), Akinfiev et al. (2016) and the recommended per-gas selections with offsets.
- `_lib_usage_example.py` - worked examples of calling the engine.

## `data/`

Experimental datasets and regressed results consumed by the pipelines. See `data/README.md` for a file-by-file description.
