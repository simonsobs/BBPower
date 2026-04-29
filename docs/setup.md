# Setup and Entry Points

This page is the quickest way to get BBPower running with the fewest moving parts. It focuses on:

- which dependencies are needed for which stages
- what files each stage expects
- what to run first if something is missing

## 1. Create an environment

Use conda with Python 3.13 for new installs:

```bash
conda create -n bbpower -c conda-forge python=3.13 pip setuptools wheel
conda activate bbpower
python -m pip install --upgrade pip
```

BBPower requires Python >= 3.10, but Python 3.13 is the recommended default for new conda environments. A plain virtualenv still works if you already manage dependencies another way.

## 2. Install only what you need

The package is deliberately split into extras so you do not need heavy map-level dependencies unless you are running those stages.

| Goal | Stages | Install command | Extra notes |
|---|---|---|---|
| Inspect configs, use shared helpers, run lightweight pieces | Base package, `BBPowerSummarizer` | `pip install -e .` | Does **not** include `fgbuster`, `getdist`, `pymaster`, or `pyshtools` |
| Run component separation on pre-computed spectra | `BBCompSep` | `pip install -e ".[compsep]"` | This is the most common setup |
| Run component separation and make plots | `BBCompSep`, `BBPlotter` | `pip install -e ".[compsep,plotting]"` | Needed for triangle plots via `getdist` |
| Use moment-expanded foreground models or Fisher runs | `BBCompSep` | `pip install -e ".[compsep,plotting,sampling]"` | Adds `pyshtools` and `numdifftools` |
| Run the full maps-to-parameters pipeline | All four stages | `conda install -c conda-forge healpy namaster && pip install -e ".[all]"` | Prefer conda binaries for heavy compiled dependencies |

Practical notes:

- `BBCompSep` always needs `fgbuster`.
- `BBPlotter` only needs `getdist` if you want likelihood contours from `emcee` chains.
- `BBCompSep` with `fg_model.use_moments: true` needs `pyshtools` for Wigner 3-j calculations.
- `sampler: polychord` requires a separate PolyChord installation that is not provided by `pyproject.toml`.
- If `pip install -e ".[all]"` tries to compile NaMaster locally, install `namaster` from conda-forge first and use narrower BBPower extras for the remaining workflow.

## 3. Verify the install

```bash
python -m bbpower --help
python -c "import bbpower; print(bbpower.__file__)"
```

If you are working with multiple clones or installs, the second command is the fastest way to confirm which checkout Python is importing.

## 4. Pick the lowest-friction entry point

Most users do **not** need to start from maps. If you already have SACC spectra, start at `BBCompSep`.

| Starting point | When to use it | Required inputs |
|---|---|---|
| `BBPowerSpecter` | You only have HEALPix Q/U maps and simulations | `splits_list`, `masks_apodized`, `bandpasses_list`, `beams_list`, `sims_list` |
| `BBPowerSummarizer` | You already have split-level spectra from maps/sims | `cells_all_splits`, `cells_all_sims`, `splits_list`, `bandpasses_list` |
| `BBCompSep` | You already have coadded spectra and covariance | `cells_coadded`, `cells_noise`, `cells_coadded_cov`, config |
| `BBPlotter` | You already have BBPower outputs and want diagnostics | `cells_coadded*`, `cells_fiducial`, `param_chains`, plot paths |

## 5. Minimal file checklist for `BBCompSep`

To run:

```bash
python -m bbpower BBCompSep \
  --cells_coadded=... \
  --cells_noise=... \
  --cells_fiducial=... \
  --cells_coadded_cov=... \
  --output_dir=... \
  --config_copy=... \
  --config=...
```

You need:

- `cells_coadded`: coadded signal estimate in SACC format
- `cells_noise`: noise spectra in SACC format
- `cells_coadded_cov`: covariance in SACC format
- `config`: stage config YAML
- `output_dir`: an existing writable directory

Important detail:

- `cells_fiducial` is still a required CLI / stage input today for both likelihood modes.
- In practice, it is only used by `likelihood_type: h&l`.
- For `likelihood_type: chi2`, you still need to pass a path, even though the stage does not use the file contents at runtime.

## 6. What each stage writes

| Stage | Main outputs |
|---|---|
| `BBPowerSpecter` | `cells_all_splits.fits`, `cells_all_sims.txt`, workspace files under the `mcm` prefix |
| `BBPowerSummarizer` | `cells_coadded.fits`, `cells_coadded_total.fits`, `cells_noise.fits`, `cells_null.fits` |
| `BBCompSep` | sampler-specific files in `output_dir` such as `emcee.npz`, `chi2.npz`, `single_point.npz`, `fisher.npz`, `cells_model.npz`, plus `config_copy.yml` |
| `BBPlotter` | `plots.dir/`, `plots_page.html`, and optionally `triangle.png` |

## 7. `emcee` parallelism for `BBCompSep`

`BBCompSep` can parallelize `sampler: emcee`, but there are several interacting
runtime parameters and defaults. The details matter on clusters, especially if
you are comparing `nwalkers`, Slurm CPU requests, and BLAS/OpenMP thread counts.

Read [threading.md](threading.md) for the full guide. That page explains:

- what `BBPOWER_EMCEE_WORKERS`, `BBPOWER_EMCEE_POOL`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `SLURM_CPUS_PER_TASK` each do
- how bash defaults like `${VAR:-1}` work
- why emcee worker parallelism is capped to `ceil(nwalkers / 2)`
- why `thread` is the default emcee pool
- why `emcee.npz.h5` is single-writer only
- concrete examples such as `32` CPUs with `40` walkers

Recommended cluster settings for the standard `BBCompSep` likelihood:

```bash
export BBPOWER_EMCEE_POOL=thread
export BBPOWER_EMCEE_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Why keep BLAS threads at `1` here:

- emcee parallelism already spreads likelihood calls across workers
- enabling many BLAS threads inside every worker can oversubscribe the node and slow the run down

If you have more CPUs than the useful emcee worker count, there are only two realistic ways to use them:

- Increase `nwalkers`, if that is scientifically and operationally acceptable for your run.
- Try a hybrid setup with fewer emcee workers and more BLAS threads per worker, but benchmark it on your likelihood first. That is not the default because the best setting depends strongly on the model and machine.

## 8. Recommended smoke tests

These are the fastest ways to confirm a given install actually runs the stages you care about.

```bash
# Component separation only
bash test/run_compsep_test.sh

# Predicted spectra mode
bash test/run_predicted_spectra_test.sh

# Legacy direct spectra -> component separation + plot generation
bash test/run_sampling_test.sh
```

If you installed the full map-level stack:

```bash
bash test/run_power_specter_test.sh
```

## 9. Common setup failures

### `ModuleNotFoundError: fgbuster`

You installed the base package only. Reinstall with:

```bash
pip install -e ".[compsep]"
```

### `ModuleNotFoundError: getdist`

You are trying to make triangle plots without the plotting extra:

```bash
pip install -e ".[plotting]"
```

### `ModuleNotFoundError: pyshtools`

You are using moment-expanded foreground models:

```bash
pip install -e ".[sampling]"
```

### `ModuleNotFoundError: pymaster` or `healpy`

You are trying to run `BBPowerSpecter` without the map-level dependencies:

```bash
conda install -c conda-forge healpy namaster
pip install -e ".[power-spectra]"
```

### `BBCompSep` fails when copying `config_copy.yml`

Make sure `--output_dir` already exists. BBPower writes into that directory but does not create every intermediate parent path for you.

## 10. Where to go next

- [README.md](../README.md) for the main project overview
- [threading.md](threading.md) for the full `BBCompSep` emcee threading and environment guide
- [architecture.md](architecture.md) for stage internals and data flow
- [configuration.md](configuration.md) for all YAML options
- [examples.md](examples.md) for concrete workflows and commands
