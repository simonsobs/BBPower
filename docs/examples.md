# Examples and Usage

This page walks through the main ways to use BBPower, from the simplest component-separation-only run to a full maps-to-parameters pipeline.

## Example 1: Component Separation on Synthetic Spectra

The fastest way to try BBPower. No maps, no simulations -- just synthetic bandpowers.

### Step 1: Generate synthetic data

```bash
mkdir -p output
python examples/generate_SO_spectra.py output
```

This creates three SACC files in `output/`:
- `cls_coadd.fits` -- signal-only bandpowers (with noise-level covariance)
- `cls_fid.fits` -- fiducial (signal-only) bandpowers
- `cls_noise.fits` -- noise-only bandpowers

The script uses Simons Observatory V3 sensitivity curves (`examples/noise_calc.py`) and a dust+synchrotron+CMB model (`examples/utils.py`) to produce realistic multi-frequency polarization bandpowers across 6 frequency channels (27, 39, 93, 145, 225, 280 GHz).

To use the alternate SO 2023 forecast foreground parameters from Wolz et al. 2302.04276, add `--so_forecast`:

```bash
python examples/generate_SO_spectra.py output --so_forecast
```

### Step 2: Run the maximum-likelihood fit

```bash
python -m bbpower BBCompSep \
  --cells_coadded=output/cls_coadd.fits \
  --cells_noise=output/cls_noise.fits \
  --cells_fiducial=output/cls_fid.fits \
  --cells_coadded_cov=output/cls_coadd.fits \
  --output_dir=output \
  --config_copy=output/config_copy.yml \
  --config=test/test_config_sampling_legacy.yml
```

This reads the legacy direct-spectra config in `test/test_config_sampling_legacy.yml`, which sets `sampler: 'maximum_likelihood'`. The fit finds the best-fit values for all free parameters (r, A_lens, foreground amplitudes and tilts, spectral indices, dust-synchrotron correlation).

Output: `output/chi2.npz` containing `params` (best-fit vector), `names` (parameter names), `chi2`, and `ndof`.

### Step 3: Plot the results

```bash
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

Output: `output/plots.dir/` with PNG files and `output/plots_page.html`.

### Inspecting results in Python

```python
import numpy as np

# Maximum-likelihood result
data = np.load('output/chi2.npz')
for name, value in zip(data['names'], data['params']):
    print(f'{name:20s} = {value:.4f}')
print(f'chi2 = {data["chi2"]:.2f}, ndof = {data["ndof"]}')
```

---

## Example 2: MCMC Sampling with emcee

To run a full MCMC instead of just finding the MAP, change the sampler in the config:

```yaml
# In your config YAML:
BBCompSep:
  sampler: 'emcee'
  nwalkers: 24
  n_iters: 1000
```

Or use the provided `test/test_config_emcee.yml` which is set up this way:

```bash
python -m bbpower BBCompSep \
  --cells_coadded=output/cls_coadd.fits \
  --cells_noise=output/cls_noise.fits \
  --cells_fiducial=output/cls_fid.fits \
  --cells_coadded_cov=output/cls_coadd.fits \
  --output_dir=output \
  --config_copy=output/config_copy.yml \
  --config=test/test_config_emcee.yml
```

Output: `output/emcee.npz` with:
- `chain`: shape `(nwalkers, n_iters, n_params)` MCMC samples
- `names`: parameter names
- `time`: wall-clock time

### Analyzing the chain

```python
import numpy as np

data = np.load('output/emcee.npz')
chain = data['chain']  # (nwalkers, n_iters, n_params)
names = data['names']

# Discard first 25% as burn-in and flatten walkers
n_burn = chain.shape[1] // 4
flat_chain = chain[:, n_burn:, :].reshape(-1, chain.shape[2])

# Print posterior summary
for i, name in enumerate(names):
    median = np.median(flat_chain[:, i])
    lower = np.percentile(flat_chain[:, i], 16)
    upper = np.percentile(flat_chain[:, i], 84)
    print(f'{name:20s} = {median:.4f} (+{upper-median:.4f} / {lower-median:.4f})')
```

---

## Example 3: Full Pipeline (Maps to Parameters)

This runs all four stages: power spectrum estimation from maps, coadding, component separation, and plotting.

### Step 1: Generate simulated maps

```bash
mkdir -p output

# Generate fiducial spectra
python examples/generate_SO_spectra.py output

# Generate 100 simulated map realizations
for seed in $(seq 1001 1100); do
    mkdir -p output/s${seed}
    python examples/generate_SO_maps.py \
      --output-dir output/s${seed} \
      --seed ${seed} \
      --nside 64
done
```

Each simulation directory contains HEALPix Q/U maps for all frequency bands and data splits.

### Step 2: Compute power spectra

```bash
python -m bbpower BBPowerSpecter \
  --splits_list=./examples/test_data/splits_list.txt \
  --masks_apodized=./examples/test_data/masks_ones.fits.gz \
  --bandpasses_list=./examples/data/bpass_list.txt \
  --sims_list=./examples/test_data/sims_list.txt \
  --beams_list=./examples/data/beams_list.txt \
  --cells_all_splits=output/cells_all_splits.fits \
  --cells_all_sims=output/cells_all_sims.txt \
  --mcm=output/mcm.dum \
  --config=test/test_config_emcee.yml
```

### Step 3: Coadd and estimate covariances

```bash
python -m bbpower BBPowerSummarizer \
  --splits_list=./examples/test_data/splits_list.txt \
  --bandpasses_list=./examples/data/bpass_list.txt \
  --cells_all_splits=output/cells_all_splits.fits \
  --cells_all_sims=output/cells_all_sims.txt \
  --cells_coadded_total=output/cells_coadded_total.fits \
  --cells_coadded=output/cells_coadded.fits \
  --cells_noise=output/cells_noise.fits \
  --cells_null=output/cells_null.fits \
  --config=test/test_config_emcee.yml
```

### Step 4: Component separation

```bash
python -m bbpower BBCompSep \
  --cells_coadded=output/cells_coadded.fits \
  --cells_noise=output/cells_noise.fits \
  --cells_fiducial=output/cls_fid.fits \
  --cells_coadded_cov=output/cells_coadded.fits \
  --output_dir=output \
  --config_copy=output/config_copy.yml \
  --config=test/test_config_emcee.yml
```

### Step 5: Plot

```bash
python -m bbpower BBPlotter \
  --cells_coadded_total=output/cells_coadded_total.fits \
  --cells_coadded=output/cells_coadded.fits \
  --cells_noise=output/cells_noise.fits \
  --cells_null=output/cells_null.fits \
  --cells_fiducial=output/cls_fid.fits \
  --param_chains=output/emcee.npz \
  --plots=output/plots.dir \
  --plots_page=output/plots_page.html \
  --config=test/test_config_emcee.yml
```

---

## Example 4: Fisher Forecast

For a quick forecast without running an MCMC:

```yaml
BBCompSep:
  sampler: 'fisher'
```

```bash
python -m bbpower BBCompSep \
  --cells_coadded=output/cls_coadd.fits \
  --cells_noise=output/cls_noise.fits \
  --cells_fiducial=output/cls_fid.fits \
  --cells_coadded_cov=output/cls_coadd.fits \
  --output_dir=output \
  --config_copy=output/config_copy.yml \
  --config=my_fisher_config.yml
```

Output: `output/fisher.npz` with `params` (MAP), `fisher` (Fisher matrix), `names`. The Fisher matrix can be inverted to get parameter covariances:

```python
import numpy as np
data = np.load('output/fisher.npz')
cov = np.linalg.inv(data['fisher'])
for i, name in enumerate(data['names']):
    print(f'{name:20s} = {data["params"][i]:.4f} +/- {np.sqrt(cov[i,i]):.4f}')
```

---

## Example 5: Validating at Fiducial

To check that the chi-squared at the fiducial (input) parameter values is reasonable:

```yaml
BBCompSep:
  sampler: 'single_point'
```

Output: `output/single_point.npz` with `chi2` and `ndof`. For data generated at the fiducial, you expect `chi2 ~ ndof`.

---

## Input File Formats

### Splits list (`splits_list`)

Text file with one HEALPix FITS map path per line. Each map contains Q/U polarization data for all frequency bands in a single data split:

```
./data/obs_split1of4.fits
./data/obs_split2of4.fits
./data/obs_split3of4.fits
./data/obs_split4of4.fits
```

### Bandpass list (`bandpasses_list`)

Text file with one bandpass file path per line:

```
./data/bandpasses/band1.txt
./data/bandpasses/band2.txt
./data/bandpasses/band3.txt
```

Each bandpass file has two columns (no header):
```
# frequency_GHz  transmission
20.0   0.0
20.5   0.001
21.0   0.05
...
30.0   0.0
```

### Beam list (`beams_list`)

Text file with one beam transfer function file per line. Each beam file has two columns:
```
# ell  b_ell
0   1.0
1   0.9999
2   0.9995
...
```

### Simulations list (`sims_list`)

Text file with one simulation directory per line. Each directory must contain split map files named `obs_split{i}of{n}.fits`:

```
./sims/s1001
./sims/s1002
...
./sims/s1100
```

### CMB templates

CAMB output files with columns `ell  D_TT  D_EE  D_BB  D_TE`. You need two:
1. Lensed CMB with r=0 (lensing only)
2. Lensed CMB with r=1 (lensing + full tensor)

The tensor contribution is computed internally as `template_r1 - template_r0`, then scaled by the `r_tensor` parameter.

---

## Test Scripts

The `test/` directory contains integration tests:

| Script | What it tests | Runtime |
|---|---|---|
| `run_sampling_test.sh` | Legacy synthetic spectra -> BBCompSep (MAP) -> BBPlotter wrapper | ~1 min |
| `run_sampling_legacy_test.sh` | Legacy direct spectra workflow implementation | ~1 min |
| `run_compsep_test.sh` | BBCompSep single_point chi2 validation | ~30 sec |
| `run_predicted_spectra_test.sh` | BBCompSep predicted spectra output | ~30 sec |
| `run_power_specter_test.sh` | Full pipeline: maps -> spectra -> coadd -> MCMC -> plots | ~30 min |
| `run_polychord_test.sh` | Full pipeline with PolyChord sampler | ~1 hr |

Run the current lightweight tests to verify your installation:

```bash
bash test/run_compsep_test.sh
bash test/run_predicted_spectra_test.sh
```

The old direct sampling workflow is still available as a legacy smoke test:

```bash
bash test/run_sampling_test.sh
```

Each script creates `test/test_out/`, runs the pipeline, checks for expected output files, then cleans up.
