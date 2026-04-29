# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BBPower is a power-spectrum-based component separation pipeline for constraining primordial B-modes from multi-frequency CMB polarization data. It is built on the [BBPipe](https://github.com/simonsobs/BBPipe) framework, which defines pipeline stages with typed inputs/outputs and connects them via YAML configuration.

## Build and Install

```bash
pip install -e .                  # core dependencies only
pip install -e ".[all]"           # includes healpy, pymaster, fgbuster, getdist
```

The package uses `pyproject.toml` (setuptools backend). There is no Makefile or CI configuration.

## Running Pipeline Stages

Each stage is invoked via `python -m bbpower <StageName>` with explicit `--input=path` and `--config=path` arguments. There is no single `bbpipe` orchestration command used in tests; instead, stages are run individually in shell scripts.

```bash
# Example: run the sampling test end-to-end
bash test/run_sampling_test.sh

# Example: run a single stage
python -m bbpower BBCompSep \
  --cells_coadded=./test/test_out/cls_coadd.fits \
  --cells_noise=./test/test_out/cls_noise.fits \
  --cells_fiducial=./test/test_out/cls_fid.fits \
  --cells_coadded_cov=./test/test_out/cls_coadd.fits \
  --output_dir=./test/test_out \
  --config_copy=./test/test_out/config_copy.yml \
  --config=./test/test_config_sampling.yml
```

## Tests

Tests are shell scripts in `test/` (no pytest). Each script generates synthetic data, runs one or more stages, checks for expected output files, then cleans up `test/test_out/`.

| Script | What it tests |
|---|---|
| `run_sampling_test.sh` | BBCompSep (maximum_likelihood) + BBPlotter |
| `run_power_specter_test.sh` | Full pipeline: BBPowerSpecter -> BBPowerSummarizer -> BBCompSep -> BBPlotter |
| `run_compsep_test.sh` | BBCompSep single-point chi2 check |
| `run_predicted_spectra_test.sh` | BBCompSep predicted spectra output |
| `run_polychord_test.sh` | Full pipeline with PolyChord sampler |

Most tests require generating 100 simulated maps first (`run_power_specter_test.sh`, `run_polychord_test.sh`), which is slow. The fastest test to validate basic functionality is `run_sampling_test.sh`.

## Architecture

### Pipeline Stage System

All stages inherit from `bbpipe.PipelineStage` and declare typed `inputs`, `outputs`, and `config_options`. The stage registry lives in `bbpower/_stages.py` which maps stage names to modules. Stages are lazy-loaded via `__init__.py`'s `__getattr__`. File types are defined in `bbpower/types.py` (FitsFile, TextFile, NpzFile, DirFile, etc.).

The four pipeline stages, in execution order:

1. **BBPowerSpecter** (`power_specter.py`) - Computes all cross-frequency/split/polarization power spectra from maps using NaMaster (`pymaster`). Produces SACC-format output with bandpower windows.
2. **BBPowerSummarizer** (`power_summarizer.py`) - Coadds split spectra, computes noise spectra (total minus cross-only), builds null tests, and estimates covariance matrices from simulations.
3. **BBCompSep** (`compsep.py`) - Foreground-cleaning likelihood analysis. Supports multiple samplers (emcee, PolyChord, Fisher, maximum_likelihood, single_point). Uses `ParameterManager` for prior handling and `FGModel` for foreground SEDs/spectra. Likelihood evaluation is in `likelihood.py` (`Likelihood` class) and sampler backends are in `samplers.py` (dispatched via `SAMPLERS` dict).
4. **BBPlotter** (`plotter.py`) - Generates an HTML page with diagnostic plots (bandpasses, coadded spectra, nulls, likelihood contours via getdist).

### Key Internal Modules

- **`likelihood.py`** (`Likelihood`) - Wraps the model function and data to compute chi-squared or Hamimeche & Lewis likelihood values. Used by all samplers.
- **`samplers.py`** - Standalone sampler backend functions (`run_emcee`, `run_polychord`, `run_minimizer`, `run_fisher`, `run_singlepoint`, `run_timing`, `run_predicted_spectra`). Registered in `SAMPLERS` dict and dispatched from `BBCompSep.run()`.
- **`param_manager.py`** (`ParameterManager`) - Parses YAML config to separate fixed vs. free parameters, builds prior functions (tophat/Gaussian), and maps flat parameter vectors back to named dictionaries.
- **`fg_model.py`** (`FGModel`) - Loads foreground SED models from `fgbuster` and power spectrum models from `fgcls.py`. Handles cross-component correlations, decorrelation, and moment expansion.
- **`fgcls.py`** - Symbolic power spectrum models using `sympy`. `ClAnalytic` parses string expressions into lambdified numpy functions. `ClPowerLaw` is the standard foreground Cl template.
- **`bandpasses.py`** (`Bandpass`) - Bandpass convolution with SED models, including systematics (frequency shift, gain, polarization angle rotation, frequency-dependent birefringence).

### Data Flow

All inter-stage data uses [SACC](https://github.com/LSSTDESC/sacc) format (`.fits` files) for power spectra, covariances, tracers (bandpasses + beams), and bandpower windows. Configuration is passed via YAML files with a `global` section and per-stage sections (e.g., `BBCompSep:`).

### Configuration Structure

Config YAML files have two layers:
- **Pipeline file** (e.g., `test/test_sampling.yml`): declares stages, inputs, and output directories for BBPipe orchestration.
- **Stage config file** (e.g., `test/test_config_sampling.yml`): contains `global` parameters (nside, compute_dell) and per-stage blocks defining the CMB model, foreground model (components with SEDs, Cl templates, priors), and sampler settings.

Parameter definitions in config follow the pattern: `param_name: ['internal_name', 'prior_type', [prior_args]]` where prior_type is `'fixed'`, `'tophat'`, or `'gaussian'`.
