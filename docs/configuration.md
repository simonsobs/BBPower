# Configuration Reference

BBPower uses two YAML configuration files: a **pipeline file** that declares stages and file paths, and a **stage config file** that defines the physical model and analysis settings.

## Pipeline File

Used by BBPipe to orchestrate stage execution. Not needed when running stages individually via `python -m bbpower`.

```yaml
modules: bbpower

launcher: local

stages:
  - name: BBPowerSpecter
    nprocess: 1
  - name: BBPowerSummarizer
    nprocess: 1
  - name: BBCompSep
    nprocess: 1
  - name: BBPlotter
    nprocess: 1

inputs:
  splits_list: ./data/splits.txt
  bandpasses_list: ./data/bpass_list.txt
  beams_list: ./data/beams_list.txt
  masks_apodized: ./data/mask.fits
  sims_list: ./data/sims_list.txt
  cells_fiducial: ./data/cls_fid.fits

config: ./config.yml

output_dir: ./output
log_dir: ./output
pipeline_log: ./output/log.txt
resume: false
```

## Stage Config File

### Global Section

Shared by all stages:

```yaml
global:
  # HEALPix resolution (required by BBPowerSpecter, used for CMB template truncation)
  nside: 64

  # If true, power spectra are stored as D_ell = ell*(ell+1)/(2*pi) * C_ell.
  # If false, plain C_ell is used.
  compute_dell: true
```

---

### BBPowerSpecter

```yaml
BBPowerSpecter:
  # Path to a text file with ell bin edges (one per line),
  # OR an integer for uniform bin width.
  bpw_edges: "./data/bpw_edges.txt"

  # Enable B-mode purification in NaMaster.
  purify_B: true

  # Number of NaMaster purification iterations.
  n_iter: 3
```

---

### BBPowerSummarizer

```yaml
BBPowerSummarizer:
  # Covariance matrix structure for data spectra.
  # Options: "diagonal", "dense", "block_diagonal"
  data_covar_type: "block_diagonal"

  # For block_diagonal: how many off-diagonal blocks to keep.
  # 0 = strict block diagonal; higher = more off-diagonal coupling.
  data_covar_diag_order: 0

  # Covariance structure for null tests (usually simpler to save memory).
  nulls_covar_type: "diagonal"
  nulls_covar_diag_order: 0
```

---

### BBCompSep

This is the largest config section. It defines the sampler, likelihood, CMB model, foreground model, and optional systematics.

#### Top-level options

```yaml
BBCompSep:
  # Sampler backend. See Samplers section below.
  sampler: 'emcee'

  # Likelihood function.
  #   'chi2' - standard Gaussian chi-squared
  #   'h&l'  - Hamimeche & Lewis (recommended; handles non-Gaussianity)
  likelihood_type: 'h&l'

  # Which polarization channels to fit. Options:
  #   ['E', 'B']  - joint EE+BB+EB fit
  #   ['B']       - BB only
  #   ['E']       - EE only
  pol_channels: ['E', 'B']

  # Multipole range (applied uniformly to all frequency pairs).
  l_min: 30
  l_max: 300

  # Which frequency bands to use. 'all' uses all bands in the SACC file.
  # Alternatively, a list of tracer names: ['band1', 'band2', 'band3']
  bands: 'all'
```

#### Sampler options

```yaml
  # --- emcee ---
  nwalkers: 24       # Number of MCMC walkers
  n_iters: 1000      # Iterations per walker

  # --- polychord ---
  nlive: 50          # Number of live points
  nrepeat: 50        # Number of repeats per slice

  # --- predicted_spectra ---
  predict_at_minimum: true    # Evaluate at MAP (true) or fiducial (false)
  predict_to_sacc: false      # Save as SACC FITS (true) or NPZ (false)
```

Available samplers:

| `sampler` value | Description | Output file |
|---|---|---|
| `emcee` | Affine-invariant MCMC | `emcee.npz` |
| `polychord` | Nested sampling | `polychord/` directory |
| `maximum_likelihood` | Scipy Powell minimizer | `chi2.npz` |
| `fisher` | Fisher matrix at MAP | `fisher.npz` |
| `single_point` | Chi-squared at fiducial | `single_point.npz` |
| `timing` | Benchmark likelihood speed | `timing.npz` |
| `predicted_spectra` | Evaluate model at MAP/fiducial | `cells_model.npz` or `.fits` |

Notes:

- `polychord` requires a separate PolyChord installation; it is not installed by the standard package extras.
- `fisher` needs `numdifftools`.
- Moment-expanded foreground models (`fg_model.use_moments: true`) need `pyshtools`.
- `sampler: emcee` uses environment variables for runtime parallelism rather than YAML keys:
  - `BBPOWER_EMCEE_WORKERS` sets the worker count.
  - `BBPOWER_EMCEE_POOL` selects `thread`, `serial`, or `process`.
- The default pool is `thread`, which is the recommended mode for standard `BBCompSep` likelihoods because process pools can fail on non-picklable `fgbuster` helper objects.
- The useful emcee worker count is capped to `ceil(nwalkers / 2)` because the default stretch move updates one half of the walker ensemble at a time.
- `emcee.npz.h5` is a single-writer backend. Do not run two `BBCompSep` `emcee` jobs against the same output directory at once.
- If you want more CPU use than that cap allows, the next knob is usually a larger `nwalkers`, not a larger worker pool. Hybrid setups with fewer emcee workers and BLAS threads greater than `1` are possible, but they should be benchmarked explicitly to avoid oversubscription.
- See [threading.md](threading.md) for the full explanation of environment-variable precedence, bash defaults, worker caps, BLAS/OpenMP settings, and cluster examples.

#### CMB model

```yaml
  cmb_model:
    # Two CAMB-format template files:
    #   [0]: Lensed spectrum with r=0 (lensing-only)
    #   [1]: Lensed spectrum with r=1 (lensing + tensor with r=1)
    # The tensor contribution is computed as [1] - [0], scaled by r_tensor.
    cmb_templates:
      - "./examples/data/camb_lens_nobb.dat"
      - "./examples/data/camb_lens_r1.dat"

    # CMB parameters (see Parameter Format below)
    params:
      r_tensor: ['r_tensor', 'tophat', [-0.1, 0.0, 0.1]]
      A_lens:   ['A_lens',   'tophat', [0.0, 1.0, 2.0]]

    # Optional: enable isotropic cosmic birefringence rotation
    use_birefringence: false
    # If enabled, add a birefringence parameter:
    #   birefringence: ['birefringence', 'tophat', [-30., 0., 30.]]
```

#### Foreground model

```yaml
  fg_model:
    # Optional: enable moment expansion for SED spatial variation
    use_moments: false
    moments_lmax: 192      # Multipole cutoff for moment terms

    # Each component is named component_1, component_2, etc.
    component_1:
      name: Dust             # Human-readable label (for logging)

      # SED class from fgbuster.component_model
      # Common options: Dust, Synchrotron, CMB, FreeFree, AME
      sed: Dust

      # Power spectrum model for each polarization pair.
      # Class names from bbpower/fgcls.py (currently: ClPowerLaw).
      # Omitted pairs are set to zero.
      cl:
        EE: ClPowerLaw
        BB: ClPowerLaw

      # SED parameters
      sed_parameters:
        beta_d: ['beta_d', 'Gaussian', [1.59, 0.11]]
        temp_d: ['temp',   'fixed',    [19.6]]
        nu0_d:  ['nu0',    'fixed',    [353.]]
        # nu0 parameters MUST be fixed.

      # Power spectrum parameters, organized by polarization pair
      cl_parameters:
        EE:
          amp_d_ee:   ['amp',   'tophat', [0., 10., "inf"]]
          alpha_d_ee: ['alpha', 'tophat', [-1., -0.42, 0.]]
          l0_d_ee:    ['ell0',  'fixed',  [80.]]
          # ell0 (pivot scale) parameters MUST be fixed.
        BB:
          amp_d_bb:   ['amp',   'tophat', [0., 5., "inf"]]
          alpha_d_bb: ['alpha', 'tophat', [-1., -0.2, 0.]]
          l0_d_bb:    ['ell0',  'fixed',  [80.]]

      # Optional: cross-correlation with another component
      cross:
        # Format: ['target_component_name', 'prior_type', prior_args]
        # Creates a frequency-independent correlation coefficient.
        epsilon_ds: ['component_2', 'tophat', [-1., 0., 1.]]

      # Optional: frequency decorrelation
      decorr:
        decorr_amp:  ['decorr_amp',  'tophat', [0., 0.5, 1.]]
        decorr_nu01: ['decorr_nu01', 'fixed',  [100.]]
        decorr_nu02: ['decorr_nu02', 'fixed',  [200.]]

      # Optional: moment expansion parameters (requires use_moments: true)
      moments:
        gamma_d_beta: ['gamma_beta', 'tophat', [-6., -3.5, -2.]]
        amp_d_beta:   ['amp_beta',   'tophat', [0., 0., 1.]]

    component_2:
      name: Synchrotron
      sed: Synchrotron
      cl:
        EE: ClPowerLaw
        BB: ClPowerLaw
      sed_parameters:
        beta_s: ['beta_pl', 'Gaussian', [-3.0, 0.3]]
        nu0_s:  ['nu0',     'fixed',    [23.]]
      cl_parameters:
        EE:
          amp_s_ee:   ['amp',   'tophat', [0., 4., 8.]]
          alpha_s_ee: ['alpha', 'tophat', [-1., -0.6, 0.]]
          l0_s_ee:    ['ell0',  'fixed',  [80.]]
        BB:
          amp_s_bb:   ['amp',   'tophat', [0., 2., 4.]]
          alpha_s_bb: ['alpha', 'tophat', [-1., -0.4, 0.]]
          l0_s_bb:    ['ell0',  'fixed',  [80.]]
```

#### Bandpass systematics (optional)

```yaml
  systematics:
    bandpasses:
      bandpass_1:
        # Optional: file with frequency-dependent polarization phase
        # Two columns: frequency (GHz), phase angle (degrees)
        phase_nu: "./data/phase_nu_band1.txt"

        parameters:
          # Fractional frequency shift: delta_nu = shift * nu_mean
          shift_bp1: ['shift', 'tophat', [-0.01, 0., 0.01]]

          # Multiplicative gain calibration factor
          gain_bp1: ['gain', 'tophat', [0.9, 1.0, 1.1]]

          # Polarization angle rotation (degrees)
          angle_bp1: ['angle', 'tophat', [-1., 0., 1.]]

          # Frequency-dependent birefringence slope (degrees)
          dphi1_bp1: ['dphi1', 'tophat', [-5., 0., 5.]]

      bandpass_2:
        parameters:
          shift_bp2: ['shift', 'tophat', [-0.01, 0., 0.01]]
          gain_bp2:  ['gain',  'tophat', [0.9, 1.0, 1.1]]
```

---

### BBPlotter

```yaml
BBPlotter:
  # Maximum multipole for plots
  lmax_plot: 300

  # Include total (auto+cross) coadded spectra in plots
  plot_coadded_total: true

  # Include noise power spectra in plots
  plot_noise: true

  # Include null test plots
  plot_nulls: true

  # Include MCMC triangle plots (requires getdist and chain output)
  plot_likelihood: true
```

---

## Parameter Format

Every model parameter follows this convention:

```yaml
user_chosen_name: ['internal_name', 'prior_type', [prior_args]]
```

| Field | Description |
|---|---|
| `user_chosen_name` | Your label (must be unique across the entire config) |
| `internal_name` | Name used internally by fgbuster or fgcls (e.g., `beta_d`, `amp`, `ell0`) |
| `prior_type` | One of `fixed`, `tophat`, or `Gaussian` (case-insensitive) |
| `prior_args` | Depends on the prior type (see below) |

### Prior types

**`fixed`**: Parameter held constant.
```yaml
temp_d: ['temp', 'fixed', [19.6]]
#                          ^value
```

**`tophat`**: Uniform prior with hard bounds.
```yaml
r_tensor: ['r_tensor', 'tophat', [-0.1, 0.0, 0.1]]
#                                  ^min  ^p0  ^max
```
The center value is used as the initial guess (`p0`). Use `"inf"` for unbounded upper limits.

**`Gaussian`**: Gaussian prior.
```yaml
beta_d: ['beta_d', 'Gaussian', [1.59, 0.11]]
#                                ^mean ^sigma
```
The mean is used as the initial guess (`p0`).

### Constraints

- Parameters named `nu0` (reference frequencies) **must** be fixed.
- Parameters named `ell0` (pivot scales) **must** be fixed.
- Parameter names must be unique across all components, systematics, and CMB sections.
- The `internal_name` maps to the fgbuster SED parameter name or fgcls Cl parameter name. Check the respective model classes to see which names are expected.
