# BBPower

Power-spectrum-based component separation pipeline for constraining primordial B-modes from multi-frequency CMB polarization data.

BBPower performs a maps-to-parameters analysis: it computes cross-frequency bandpower spectra from HEALPix maps, coadds splits, estimates covariances from simulations, fits a parametric foreground + CMB model, and produces diagnostic plots. The pipeline is built on the [BBPipe](https://github.com/simonsobs/BBPipe) framework.

## Installation

Create and activate a conda environment first, then install the extra dependencies that match the stages you want to run.

```bash
conda create -n bbpower -c conda-forge python=3.13 pip setuptools wheel
conda activate bbpower
python -m pip install --upgrade pip
```

### Choose the right install

| Workflow | Stages covered | Install command |
|---|---|---|
| Shared code + lightweight utilities | Package import, config parsing, `BBPowerSummarizer` | `pip install -e .` |
| Spectra-to-parameters | `BBCompSep` | `pip install -e ".[compsep]"` |
| Spectra-to-parameters + plots | `BBCompSep`, `BBPlotter` | `pip install -e ".[compsep,plotting]"` |
| Moment models or Fisher runs | `BBCompSep` with `use_moments: true` or `sampler: fisher` | `pip install -e ".[compsep,plotting,sampling]"` |
| Full maps-to-parameters pipeline | All four stages | `pip install -e ".[all]"` |

Notes:
- `BBCompSep` always needs `fgbuster`, so `pip install -e .` by itself is **not** enough for component separation.
- Moment-expanded foreground models need `pyshtools` because `BBCompSep` uses Wigner 3-j symbols for the moment terms.
- `BBPlotter` can generate spectra plots without MCMC contours, but triangle plots require `getdist`.
- `BBPowerSpecter` and the full maps-to-parameters workflow need `healpy` and NaMaster (`pymaster`). Prefer conda-forge binaries for these heavy dependencies: `conda install -c conda-forge healpy namaster`.
- `sampler: polychord` requires a separate PolyChord installation; it is not installed by the package extras.

```bash
# Quick checks
python -m bbpower --help
python -c "import bbpower; print(bbpower.__file__)"
```

Requires Python >= 3.10. See [pyproject.toml](pyproject.toml) for the full dependency list.
See [docs/setup.md](docs/setup.md) for a setup checklist, stage-by-stage dependency guide, and troubleshooting notes.

## Quick Start

The fastest way to run BBPower is on pre-computed power spectra (skipping the map-level stages):

```bash
# 1. Generate synthetic Simons Observatory bandpowers
mkdir -p output
python examples/generate_SO_spectra.py output

# 2. Run component separation (maximum-likelihood fit)
python -m bbpower BBCompSep \
  --cells_coadded=output/cls_coadd.fits \
  --cells_noise=output/cls_noise.fits \
  --cells_fiducial=output/cls_fid.fits \
  --cells_coadded_cov=output/cls_coadd.fits \
  --output_dir=output \
  --config_copy=output/config_copy.yml \
  --config=test/test_config_sampling_legacy.yml

# 3. Generate diagnostic plots
python -m bbpower BBPlotter \
  --cells_coadded_total=output/cls_coadd.fits \
  --cells_coadded=output/cls_coadd.fits \
  --cells_noise=output/cls_noise.fits \
  --cells_null=output/cls_coadd.fits \
  --cells_fiducial=output/cls_fid.fits \
  --param_chains=output/chi2.npz \
  --plots=output/plots.dir \
  --plots_page=output/plots_page.html \
  --config=test/test_config_sampling_legacy.yml
```

## Pipeline Stages

BBPower has four stages that run in sequence. Each reads typed inputs and produces typed outputs in [SACC](https://github.com/LSSTDESC/sacc) format.

| Stage | Module | Purpose |
|---|---|---|
| **BBPowerSpecter** | `power_specter.py` | Compute cross-frequency bandpower spectra from HEALPix Q/U maps using NaMaster |
| **BBPowerSummarizer** | `power_summarizer.py` | Coadd splits, compute noise/null spectra, estimate covariances from simulations |
| **BBCompSep** | `compsep.py` | Fit a parametric CMB + foreground model to the bandpowers |
| **BBPlotter** | `plotter.py` | Generate diagnostic plots and an HTML summary page |

You can run the full pipeline (maps to parameters) or enter at any stage with pre-computed inputs. See [docs/architecture.md](docs/architecture.md) for the data flow and module interactions.

### Common entry points

| If you already have... | Start at | Required main files |
|---|---|---|
| HEALPix Q/U maps, mask, beams, bandpasses | `BBPowerSpecter` | `splits_list`, `masks_apodized`, `bandpasses_list`, `beams_list`, `sims_list` |
| Split-level SACC spectra from maps/sims | `BBPowerSummarizer` | `cells_all_splits`, `cells_all_sims`, `splits_list`, `bandpasses_list` |
| Coadded spectra + covariance | `BBCompSep` | `cells_coadded`, `cells_noise`, `cells_coadded_cov`, stage config |
| Existing BBPower outputs | `BBPlotter` | `cells_coadded*`, `cells_fiducial`, `param_chains`, plot paths |

For most users, the lowest-friction path is to start at `BBCompSep` with pre-computed SACC spectra instead of running the map-level stages.

## Configuration

BBPower uses two YAML files:

1. **Pipeline file** (e.g., `test/test_sampling_legacy.yml`) -- declares stages, input file paths, and output directories for BBPipe orchestration.
2. **Stage config file** (e.g., `test/test_config_sampling_legacy.yml`) -- defines the physical model (CMB templates, foreground components, priors) and sampler settings.

The stage config has a `global` section (shared by all stages) and per-stage sections:

```yaml
global:
  nside: 64
  compute_dell: true

BBCompSep:
  sampler: 'maximum_likelihood'     # emcee | polychord | fisher | single_point | timing
  likelihood_type: 'h&l'            # chi2 | h&l
  pol_channels: ['E', 'B']
  l_min: 30
  l_max: 300

  cmb_model:
    cmb_templates:
      - "./examples/data/camb_lens_nobb.dat"
      - "./examples/data/camb_lens_r1.dat"
    params:
      r_tensor: ['r_tensor', 'tophat', [-0.1, 0.0, 0.1]]
      A_lens:   ['A_lens',   'tophat', [0.0,  1.0, 2.0]]

  fg_model:
    component_1:
      name: Dust
      sed: Dust
      cl: { EE: ClPowerLaw, BB: ClPowerLaw }
      sed_parameters:
        beta_d: ['beta_d', 'Gaussian', [1.59, 0.11]]
        temp_d: ['temp',   'fixed',    [19.6]]
        nu0_d:  ['nu0',    'fixed',    [353.]]
      cl_parameters:
        BB:
          amp_d_bb:   ['amp',   'tophat', [0., 5., 10.]]
          alpha_d_bb: ['alpha', 'tophat', [-1., -0.2, 0.]]
          l0_d_bb:    ['ell0',  'fixed',  [80.]]
```

See [docs/configuration.md](docs/configuration.md) for the complete reference.

## What Each Stage Writes

These are the outputs you will typically inspect when wiring the pipeline together:

| Stage | Main outputs |
|---|---|
| `BBPowerSpecter` | `cells_all_splits.fits`, `cells_all_sims.txt`, `mcm*` |
| `BBPowerSummarizer` | `cells_coadded.fits`, `cells_coadded_total.fits`, `cells_noise.fits`, `cells_null.fits` |
| `BBCompSep` | `emcee.npz`, `chi2.npz`, `single_point.npz`, `fisher.npz`, `cells_model.npz`, `config_copy.yml` |
| `BBPlotter` | `plots.dir/`, `plots_page.html`, and optionally `triangle.png` |

`BBCompSep` always expects `--output_dir` and `--config_copy` to point to an existing writable directory. The common pattern is to create that directory before invoking the stage.

## Parameter Format

Every model parameter is defined as a three-element list:

```yaml
param_name: ['internal_name', 'prior_type', [prior_args]]
```

| Prior type | Args | Description |
|---|---|---|
| `fixed` | `[value]` | Not sampled; held constant |
| `tophat` | `[lower, center, upper]` | Uniform prior; `center` is the initial value |
| `Gaussian` | `[mean, sigma]` | Gaussian prior; `mean` is the initial value |

## Samplers

| Name | Config key | Extra options | Output |
|---|---|---|---|
| emcee MCMC | `emcee` | `nwalkers`, `n_iters` | `emcee.npz` (chain, names) |
| PolyChord nested sampling | `polychord` | `nlive`, `nrepeat` | `polychord/` directory |
| Maximum likelihood | `maximum_likelihood` | -- | `chi2.npz` (best-fit params) |
| Fisher matrix | `fisher` | -- | `fisher.npz` (params, fisher) |
| Single-point chi2 | `single_point` | -- | `single_point.npz` |
| Timing benchmark | `timing` | -- | `timing.npz` |

`emcee` runtime notes:

- `BBCompSep` uses a thread pool by default for `sampler: emcee`. This avoids pickling failures that can appear with process pools when the likelihood contains `fgbuster` bandpass wrappers.
- `BBPOWER_EMCEE_WORKERS` chooses the requested worker count, but BBPower caps the effective value to about half the walkers, `ceil(nwalkers / 2)`, because emcee's default stretch move updates one red-blue split at a time.
- `BBPOWER_EMCEE_POOL=thread` is the recommended default. `serial` is useful for debugging, and `process` should only be used when the likelihood is known to be fully picklable.
- Native math-library thread settings such as `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` are separate from the emcee worker count.
- Only one emcee process may write a given `output_dir/emcee.npz.h5` at a time. Resubmission is fine after the earlier process exits, but running two jobs against the same output directory concurrently can corrupt the backend.
- See [docs/threading.md](docs/threading.md) for the full explanation of parameter precedence, bash defaults, the worker cap, and cluster examples such as `32` CPUs with `40` walkers.

## Tests

Tests are shell scripts in `test/`. The fastest current smoke tests are:

```bash
bash test/run_compsep_test.sh
bash test/run_predicted_spectra_test.sh
```

See [docs/examples.md](docs/examples.md) for descriptions of all test scripts and example workflows.

For setup validation, the most useful smoke tests are:

```bash
# BBCompSep only
bash test/run_compsep_test.sh

# Legacy direct spectra -> BBCompSep + BBPlotter workflow
bash test/run_sampling_test.sh

# Predicted spectra output
bash test/run_predicted_spectra_test.sh
```

## Documentation

- [docs/setup.md](docs/setup.md) -- Installation by workflow, stage entry points, and troubleshooting
- [docs/threading.md](docs/threading.md) -- Detailed guide to `BBCompSep` emcee workers, BLAS/OpenMP threads, and cluster setup
- [docs/architecture.md](docs/architecture.md) -- Module interactions, data flow, and class relationships
- [docs/configuration.md](docs/configuration.md) -- Complete configuration reference
- [docs/examples.md](docs/examples.md) -- Step-by-step usage examples and test descriptions
- [docs/refactor_stack.md](docs/refactor_stack.md) -- Detailed branch-by-branch refactor integration notes

## Credits

Developed by the Simons Observatory BB Analysis Working Group. Questions and contributions welcome -- contact Max Abitbol (mabitbol), David Alonso (damonge), or open an issue.

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
