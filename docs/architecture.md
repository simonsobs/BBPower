# Architecture

This document describes how BBPower's modules interact, how data flows through the pipeline, and the role of each key class.

## Pipeline Overview

BBPower is a four-stage pipeline. Each stage is a Python class that inherits from `bbpipe.PipelineStage` and declares typed `inputs`, `outputs`, and `config_options`. Stages communicate exclusively through files (SACC `.fits` for spectra, `.npz` for chains, `.txt` for file lists).

```
HEALPix Q/U maps          Config YAML
       |                       |
       v                       v
 +-----------------+     +-------------------+
 | BBPowerSpecter  |---->| BBPowerSummarizer |
 +-----------------+     +-------------------+
       |                       |
       | cells_all_splits      | cells_coadded
       | cells_all_sims        | cells_noise
       |                       | cells_null
       |                       | cells_coadded_total
       v                       v
                    +------------+
                    | BBCompSep  |
                    +------------+
                         |
                         | output_dir/ (chains, best-fit, etc.)
                         v
                    +------------+
                    | BBPlotter  |
                    +------------+
                         |
                         v
                    plots/ + HTML page
```

Users can enter at any stage. The most common entry point is **BBCompSep** with pre-computed bandpowers.

## Stage Registry and Lazy Loading

Stage classes are not imported at package load time. Instead:

1. `bbpower/_stages.py` defines `STAGE_MODULES`, a dict mapping stage names to module paths:
   ```python
   STAGE_MODULES = {
       "BBPowerSpecter": "bbpower.power_specter",
       "BBPowerSummarizer": "bbpower.power_summarizer",
       "BBCompSep": "bbpower.compsep",
       "BBPlotter": "bbpower.plotter",
   }
   ```

2. `bbpower/__init__.py` implements `__getattr__` to lazily import stages on first access. This avoids pulling in heavy dependencies (healpy, pymaster) when only running component separation.

3. `bbpower/__main__.py` provides the CLI entry point (`python -m bbpower <StageName> ...`), using `get_stage_class()` to look up and import the requested stage.

## Module Map

```
bbpower/
  __init__.py          # Lazy loader (imports stages on demand)
  __main__.py          # CLI: python -m bbpower <StageName>
  _stages.py           # Stage name -> module path registry
  types.py             # File type classes (FitsFile, TextFile, NpzFile, ...)

  power_specter.py     # Stage 1: Maps -> cross-frequency bandpowers
  power_summarizer.py  # Stage 2: Splits -> coadded spectra + covariances

  compsep.py           # Stage 3: Component separation (orchestrator)
  likelihood.py        #   Likelihood evaluation (chi2, H&L)
  samplers.py          #   Sampler backends (emcee, polychord, fisher, ...)
  param_manager.py     #   Parameter parsing, priors, p0 vector
  fg_model.py          #   Foreground model loading (SED + Cl config)
  fgcls.py             #   Symbolic Cl models (ClPowerLaw, ClAnalytic)
  bandpasses.py        #   Bandpass convolution + systematics

  plotter.py           # Stage 4: Diagnostic plots + HTML
```

## Stage 1: BBPowerSpecter

**File:** `power_specter.py`

Computes pseudo-C_l bandpowers from HEALPix Q/U maps using NaMaster.

### Inputs provided by the user

| Input | Type | Description |
|---|---|---|
| `splits_list` | Text | One HEALPix FITS map path per line (each containing Q/U for all bands in one data split) |
| `bandpasses_list` | Text | One bandpass file path per line (two columns: frequency in GHz, transmission) |
| `beams_list` | Text | One beam file path per line (two columns: ell, b_ell) |
| `masks_apodized` | FITS | Apodized sky mask (HEALPix format, any nside -- auto-resampled) |
| `sims_list` | Text | Directories containing simulated split maps |

### Internal workflow

```
init_params()          Set nside, npix, MCM prefix
read_bandpasses()      Load (nu, bnu, dnu) arrays from text files
read_beams()           Load and interpolate beam transfer functions to ell range [0, 3*nside-1]
read_masks()           Read HEALPix mask, ud_grade to working nside
get_bandpowers()       Build NaMaster NmtBin from edges file or uniform spacing
compute_workspaces()   Compute/load mode-coupling matrices for all (band1, band2) pairs
compute_cells_from_splits()
                       Create NaMaster spin-2 fields for each (band, split),
                       compute all cross-spectra, decouple using MCM
save_cell_to_file()    Write SACC file with tracers, spectra, optional bandpower windows
```

### Outputs

| Output | Type | Description |
|---|---|---|
| `cells_all_splits` | FITS | SACC file with all cross-split bandpowers (with windows) |
| `cells_all_sims` | Text | List of simulation bandpower file paths |
| `mcm` | Dummy | Mode-coupling matrix workspace files (prefix) |

## Stage 2: BBPowerSummarizer

**File:** `power_summarizer.py`

Coadds split power spectra, computes noise estimates, builds null tests, and estimates covariances.

### Internal workflow

```
init_params()                  Count splits, bands, compute null pairings
get_tracers()                  Build SACC tracers for coadded and null files
get_windows()                  Extract bandpower windows from data SACC
get_cl_indices()               Build (map1, map2, ell_bin) -> SACC index lookup
parse_splits_sacc_file()       Coadd splits (total + cross-only), compute noise,
                               build null test combinations
get_covariance_from_samples()  Estimate covariance from simulation ensemble
```

**Coadding logic:**
- **Total coadd**: Average all split pairs (including auto-correlations). This contains signal + noise.
- **Cross-only coadd**: Average only off-diagonal (cross-split) pairs. This is an unbiased signal estimate.
- **Noise estimate**: Total minus cross-only coadd.
- **Null tests**: Differences of split pairs that should be consistent with zero. Specifically, for splits (i, j, k, l): `(C_{ik} - C_{il} - C_{jk} + C_{jl})`.

### Iterators

`bands_pol_iterator()` and `bands_splits_pol_iterator()` yield all unique combinations of band/polarization/split indices. These are used throughout to ensure consistent ordering when building data vectors and covariance matrices.

## Stage 3: BBCompSep

**File:** `compsep.py` (orchestrator), `likelihood.py`, `samplers.py`, `param_manager.py`, `fg_model.py`, `fgcls.py`, `bandpasses.py`

This is the core analysis stage. It fits a parametric model to the observed bandpowers.

### How the pieces fit together

```
                    Config YAML
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
    ParameterManager  FGModel     Bandpass (one per freq)
    (fixed/free       (SED +       (nu, bnu, systematics)
     params, priors)   Cl models)
          |             |             |
          +------+------+------+------+
                 |
                 v
           BBCompSep.setup_compsep()
                 |
                 v
           Likelihood object
           (wraps model + data + covariance)
                 |
                 v
           Sampler dispatch
           (SAMPLERS dict in samplers.py)
```

### Data loading: `parse_sacc_file()`

1. Reads the coadded SACC file and its covariance
2. Removes unwanted polarization channels and applies ell cuts
3. Extracts bandpass info from SACC tracers, creates `Bandpass` objects
4. Reads bandpower windows
5. Reorganizes spectra into `(n_bpws, nmaps, nmaps)` arrays
6. Inverts the covariance matrix
7. If using H&L likelihood, also reads noise and fiducial spectra

### Model evaluation: `model(params)`

This is the core function called by the likelihood on every iteration. It builds the total model power spectrum:

```
model(params) returns shape (n_bpws, nmaps, nmaps)

1. CMB contribution:
   C_ell^CMB = r * C_tens + A_lens * C_lens + C_scal
   - Optionally rotated by birefringence angle

2. Foreground contribution (for each frequency pair f1, f2):
   For each component pair (c1, c2):
     - integrate_seds(params) -> frequency scaling F[c1,c2,f1,f2]
     - evaluate_power_spectra(params) -> C_ell^fg[c1,pol1,pol2]
     - Apply polarization rotation from complex bandpasses
     - C_fg[f1,f2] += F[c1,c2,f1,f2] * rotated(C_ell^fg)

3. Optional moment expansion terms (1x1 and 0x2)

4. Window convolution:
   For each (f1, p1, f2, p2):
     C_b = W @ C_ell   (bandpower windows x theory spectrum)

5. Polarization angle rotation (instrumental systematic)

Returns: (n_bpws, nmaps, nmaps) model bandpowers
```

### Foreground model: `FGModel` + `Bandpass`

**`FGModel`** (`fg_model.py`) parses the config to build a dictionary of foreground components. Each component has:
- An **SED function** from `fgbuster` (e.g., `Dust`, `Synchrotron`) with parameters (spectral index, temperature, reference frequency)
- **Power spectrum models** from `fgcls.py` (e.g., `ClPowerLaw`) for each polarization combination (EE, BB, EB)
- Optional **cross-correlations** with other components (a frequency-independent correlation coefficient)
- Optional **frequency decorrelation** (suppresses correlations between widely-separated bands)
- Optional **moment expansion** parameters for modeling SED spatial variation

**`Bandpass`** (`bandpasses.py`) wraps a single frequency channel. Its `convolve_sed(sed, params)` method integrates `sed(nu) * bandpass(nu) * nu^2` over the band, applying any systematics (frequency shift, gain, HWP phase, birefringence). For complex bandpasses (HWP or dphi1), it also returns a 2x2 polarization rotation matrix.

**`fgcls.py`** provides symbolic power spectrum models. `ClPowerLaw` implements `amp * (ell / ell0)^alpha`. More complex models can be defined by subclassing `ClAnalytic` with arbitrary SymPy expressions.

### Parameter management: `ParameterManager`

**`ParameterManager`** (`param_manager.py`) walks the entire config tree and collects every parameter definition. It separates them into:
- **Fixed parameters**: stored as `(name, value)` pairs
- **Free parameters**: sorted by name, with priors and initial values (`p0`)

Key methods:
- `build_params(par)`: Takes a flat numpy array of free-parameter values and returns a `{name: value}` dict including both free and fixed parameters. This is what `model(params)` receives.
- `lnprior(par)`: Evaluates the log-prior (Gaussian or tophat) for a free-parameter vector.

### Likelihood: `Likelihood`

**`Likelihood`** (`likelihood.py`) wraps the model function, observed data, noise, and inverse covariance into a single object. It provides:

- `lnlike(par)`: Evaluates `build_params(par)` -> `model(params)` -> residual -> `-0.5 * dx^T @ C^{-1} @ dx`
- `lnprob(par)`: `lnprior(par) + lnlike(par)` (the full log-posterior)

Two likelihood modes:
- **Chi-squared** (`chi2`): residual = `data - model`, flattened to upper-triangle vector
- **Hamimeche & Lewis** (`h&l`): applies a non-linear transform to handle the non-Gaussianity of power spectrum estimates. Requires fiducial and noise spectra.

### Sampler dispatch: `samplers.py`

Each sampler is a standalone function with signature:
```python
def run_XXX(likelihood: Likelihood, config: dict, output_dir: str) -> ...
```

They are registered in the `SAMPLERS` dict and dispatched from `BBCompSep.run()`:
```python
samplers.SAMPLERS[sampler_name](self.likelihood, self.config, output_dir)
```

The `predicted_spectra` sampler is special -- it also needs the `BBCompSep` object itself for model evaluation and SACC I/O, so it's called separately.

## Stage 4: BBPlotter

**File:** `plotter.py`

Reads all output spectra and chains, produces PNG plots and an HTML summary page using `dominate` for HTML generation and `matplotlib` for plotting. Optionally uses `getdist` for MCMC triangle plots.

## File Types

`types.py` defines file type classes used by BBPipe to manage I/O:

| Class | Suffix | Description |
|---|---|---|
| `FitsFile` | `.fits` | SACC power spectra, covariances, tracers |
| `TextFile` | `.txt` | File lists, bandpower edges |
| `NpzFile` | `.npz` | NumPy archives (chains, best-fit params) |
| `YamlFile` | `.yml` | Config copies |
| `DirFile` | `.dir` | Output directories |
| `HTMLFile` | `.html` | Plot summary pages |
| `HDFFile` | `.hdf` | HDF5 files (not currently used) |

## Data Formats

### SACC Files

All inter-stage power spectrum data uses the [SACC](https://github.com/LSSTDESC/sacc) format:

- **Tracers**: frequency channels with bandpass `(nu, bnu)`, beam `(ell, b_ell)`, and metadata
- **Data vector**: bandpowers organized as `(tracer1, tracer2, cl_type, ell_bin)` entries
- **Covariance**: full or block-diagonal covariance matrix over the data vector
- **Windows**: bandpower window functions mapping theory C_ell to observed bandpowers

### Bandpass Files

Plain text, two columns: `frequency_GHz  transmission`. Example:
```
20.0  1.0e-8
20.5  2.5e-7
21.0  1.0e-6
```

### Beam Files

Plain text, two columns: `ell  b_ell`. Example:
```
0  1.0
1  0.999
2  0.998
```

### CMB Template Files

CAMB output format, columns: `ell  D_TT  D_EE  D_BB  D_TE`. Two files are needed: one for the lensed spectrum with r=0, one with r=1.

### Bandpower Edges File

One ell value per line, defining bin edges:
```
2
12
22
32
```
