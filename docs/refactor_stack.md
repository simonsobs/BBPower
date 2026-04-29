# Refactor Stack Integration Notes

This document describes the clean refactor branch stack and what each stage is
intended to contribute. It is written for code review and integration planning,
not as an end-user tutorial.

The stack is based on `upstream/main` at commit `b5ac1ae` and is organized as:

```text
upstream/main
  -> refactor-stack/01-packaging-cli
  -> refactor-stack/02-model-helpers
  -> refactor-stack/03-likelihood-module
  -> refactor-stack/04-sampler-module
  -> refactor-stack/05-compsep-integration
  -> refactor-stack/06-power-specter
  -> refactor-stack/07-power-summarizer
  -> refactor-stack/08-plotter
  -> refactor-stack/09-examples-nopipe
  -> refactor-stack/10-docs-and-legacy-tests
```

The local `refactor` branch and `refactor-before-main-merge-20260428` branch are
useful historical context, but the numbered `refactor-stack/*` branches are the
best units for review because they separate packaging, shared model helpers,
likelihood extraction, sampler extraction, stage rewiring, map-level stages,
examples, and documentation.

## Catalogue

| Section | What to look for |
| --- | --- |
| [Push And Review Strategy](#push-and-review-strategy) | How to publish the stack and structure the discussion with collaborators. |
| [Branch 01: Packaging And CLI Entry Points](#branch-01-packaging-and-cli-entry-points) | `pyproject.toml`, lazy stage imports, `python -m bbpower`, file types, test mocks. |
| [Branch 02: Model And Parameter Helpers](#branch-02-model-and-parameter-helpers) | `ParameterManager`, `FGModel`, `fgcls`, bandpass systematics, rotations, decorrelation. |
| [Branch 03: Component-Separation Likelihood](#branch-03-component-separation-likelihood) | Extracted likelihood interface, chi-squared residuals, H&L transform, posterior evaluation. |
| [Branch 04: Sampler Backends](#branch-04-sampler-backends) | Sampler dispatch, emcee workers/pools, backend locking, PolyChord, Fisher, predicted spectra. |
| [Branch 05: BBCompSep Integration](#branch-05-bbcompsep-integration) | SACC parsing, CMB loading, SED integration, model assembly, sampler dispatch from the stage. |
| [Branch 06: Power Spectrum Stage](#branch-06-power-spectrum-stage) | NaMaster compatibility, workspace reuse, cross-split spectra, SACC output, simulations. |
| [Branch 07: Power Summarizer Stage](#branch-07-power-summarizer-stage) | SACC consistency checks, coadds, noise estimate, null tests, covariance modes. |
| [Branch 08: Plotter Stage](#branch-08-plotter-stage) | Optional input loading, likelihood plot gating, HTML contents, diagnostic plots. |
| [Branch 09: Examples And No-Pipe Workflow](#branch-09-examples-and-no-pipe-workflow) | Synthetic spectra/maps, shared example utilities, PolyChord plotting, shell workflow. |
| [Branch 10: Docs And Legacy Tests](#branch-10-docs-and-legacy-tests) | README/docs refresh, setup/config/examples/threading docs, legacy sampling tests. |
| [End-State Integration View](#end-state-integration-view) | Final responsibility split across modules. |
| [Suggested Review Checklist](#suggested-review-checklist) | High-risk review questions and recommended validation commands. |

Within each branch section, the `###` subsections document the major functional
code blocks changed by that branch.

## Push And Review Strategy

Yes, push these branches to your fork before asking others how to integrate the
work. Push to `origin`, not directly to `upstream`, and use a draft pull request
or discussion thread for review.

Recommended workflow:

```bash
git push origin refactor-stack/01-packaging-cli
git push origin refactor-stack/02-model-helpers
git push origin refactor-stack/03-likelihood-module
git push origin refactor-stack/04-sampler-module
git push origin refactor-stack/05-compsep-integration
git push origin refactor-stack/06-power-specter
git push origin refactor-stack/07-power-summarizer
git push origin refactor-stack/08-plotter
git push origin refactor-stack/09-examples-nopipe
git push origin refactor-stack/10-docs-and-legacy-tests
```

For discussion, the simplest review object is usually a draft PR from
`refactor-stack/10-docs-and-legacy-tests` into `upstream/main`, because that
shows the complete end state. The numbered branches still matter: reviewers can
compare each branch against the previous one if they want staged integration.

If maintainers prefer smaller PRs, the stack can be split into these review
groups:

| Review group | Branches | Main question |
| --- | --- | --- |
| Packaging and import model | `01` | Should BBPower become an installable package with lazy stage entry points? |
| Component-separation core | `02` to `05` | Should model, likelihood, and sampler responsibilities be separated this way? |
| Pipeline stages | `06` to `08` | Are the map, summarizer, and plotter hardening changes compatible with current workflows? |
| Examples and docs | `09` to `10` | Are the user-facing workflows and compatibility tests the right defaults? |

## Branch 01: Packaging And CLI Entry Points

- Branch: `refactor-stack/01-packaging-cli`
- Commit: `d579547 Add packaging and stage entry points`

This branch turns the repository into a normal installable Python package and
adds a consistent way to run stages from the command line without importing all
heavy stage dependencies up front.

Major files:

- `pyproject.toml`
- `bbpower/_stages.py`
- `bbpower/__init__.py`
- `bbpower/__main__.py`
- `bbpower/types.py`
- `tests/conftest.py`, `tests/test_stages.py`, `tests/test_types.py`

### Package Metadata And Dependency Extras

`pyproject.toml` defines the package build backend, runtime dependencies,
optional dependency groups, pytest config, Black config, and a console script.

The important functional change is the split between base dependencies and
stage-specific extras:

```toml
[project.optional-dependencies]
plotting = ["getdist>=1.7"]
compsep = ["fgbuster @ git+https://github.com/fgbuster/fgbuster.git"]
power-spectra = ["healpy>=1.19", "pymaster>=2.4"]
sampling = ["numdifftools>=0.9", "pyshtools>=4.10"]
all = [...]
```

Functional purpose:

- Users can install only what their workflow needs.
- `BBCompSep` users do not need map-level dependencies such as `healpy` and
  `pymaster`.
- Full maps-to-parameters users can still install everything with `.[all]`.
- Tests get centralized pytest settings and strict marker checking.

### Stage Registry

The new `bbpower/_stages.py` module centralizes stage names and module paths:

```python
STAGE_MODULES = {
    "BBPowerSpecter": "bbpower.power_specter",
    "BBPowerSummarizer": "bbpower.power_summarizer",
    "BBCompSep": "bbpower.compsep",
    "BBPlotter": "bbpower.plotter",
}

def get_stage_class(stage_name):
    module_name = STAGE_MODULES[stage_name]
    module = import_module(module_name)
    return getattr(module, stage_name)
```

Functional purpose:

- There is one canonical list of supported stage names.
- CLI dispatch and package lazy loading use the same source of truth.
- Unknown stage names produce a clear error listing known stages.

### Lazy Package Imports

`bbpower/__init__.py` now exposes stage classes through `__getattr__` instead of
eager imports:

```python
__all__ = ["PipelineStage", *STAGE_MODULES]

def __getattr__(name):
    module_name = STAGE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(...)
    module = import_module(module_name)
    return getattr(module, name)
```

Functional purpose:

- `import bbpower` no longer needs to import every pipeline stage.
- Component-separation-only environments avoid failures from missing map-level
  packages until `BBPowerSpecter` is actually requested.
- The package still supports direct access such as `bbpower.BBCompSep`.

### CLI Dispatch

`bbpower/__main__.py` adds this runtime pattern:

```python
python -m bbpower <stage_name> <stage_arguments>
```

The command reads `sys.argv[1]`, resolves it through `get_stage_class()`, then
delegates to the BBPipe stage `main()` method:

```python
stage_name = sys.argv[1]
stage_cls = get_stage_class(stage_name)
return stage_cls.main()
```

Functional purpose:

- Users can run stages without knowing module paths.
- `python -m bbpower --help` shows the available BBPower stages.
- Unknown stage names return a distinct usage error.

### File Type Helpers

`types.py` keeps the existing BBPipe-style file abstractions and clarifies their
runtime behavior:

- `DataFile.open()` uses standard Python `open()`.
- `HDFFile.open()` delegates to `h5py.File`.
- `FitsFile.open()` maps mode `"w"` to `"rw"` because `fitsio` does not support
  a pure write mode.
- `DummyFile.open()` always raises `NotImplementedError`.
- `DirFile`, `HTMLFile`, `NpzFile`, `TextFile`, and `YamlFile` supply typed
  suffixes used by stage input/output declarations.

This is mostly API hardening and test coverage, not a behavior redesign.

### Test Scaffolding

`tests/conftest.py` adds lightweight mock modules for optional dependencies:

```text
bbpipe.PipelineStage
fgbuster.component_model
sacc
healpy
pymaster
```

Functional purpose:

- Unit tests can import BBPower modules without installing all optional
  scientific packages.
- Stage class definitions can be collected by pytest in lightweight
  environments.
- Tests can focus on BBPower logic rather than dependency availability.

## Branch 02: Model And Parameter Helpers

- Branch: `refactor-stack/02-model-helpers`
- Commit: `8cd940b Refactor model and parameter helpers`

This branch isolates shared component-separation model logic: parameter parsing,
foreground model construction, symbolic power-spectrum models, bandpass
convolution, and instrumental systematics.

Major files:

- `bbpower/param_manager.py`
- `bbpower/fg_model.py`
- `bbpower/fgcls.py`
- `bbpower/bandpasses.py`
- `tests/test_param_manager.py`
- `tests/test_fg_model.py`
- `tests/test_fgcls.py`
- `tests/test_bandpasses.py`

### ParameterManager

`ParameterManager` turns a YAML config block into a stable sampling contract:

```python
self.p_free_names = []
self.p_free_priors = []
self.p_fixed = []
self.p0 = []
```

The central block is `_add_parameter()`:

```python
if p[1] == "fixed":
    self.p_fixed.append((p_name, float(p[2][0])))
    return

if p_name in self.p_free_names:
    raise KeyError("You have two parameters with the same name")

self.p_free_names.append(p_name)
self.p_free_priors.append(p)

if prior_kind == "tophat":
    p0 = float(p[2][1])
elif prior_kind == "gaussian":
    p0 = float(p[2][0])
else:
    raise ValueError(...)
self.p0.append(p0)
```

Functional purpose:

- Fixed parameters are stored once and never sampled.
- Free parameters get deterministic ordering through sorted config keys.
- Tophat priors use the config center value as the initial point.
- Gaussian priors use the mean as the initial point.
- Duplicate free parameter names are rejected early.
- Prior names are normalized case-insensitively for `tophat` and `gaussian`.

The constructor now explicitly gathers parameters from these config regions:

```text
cmb_model.params
fg_model.component_*.sed_parameters
fg_model.component_*.cross
fg_model.component_*.decorr
fg_model.component_*.cl_parameters, filtered by selected polarizations
fg_model.component_*.moments, only when fg_model.use_moments is true
systematics.bandpasses.bandpass_*.parameters
```

Two runtime methods define the interface used by likelihoods and samplers:

```python
def build_params(par):
    params = dict(self.p_fixed)
    params.update(dict(zip(self.p_free_names, par)))
    return params

def lnprior(par):
    ...
```

Functional purpose:

- Samplers only handle flat vectors.
- Model code receives a complete name-to-value dictionary.
- Prior evaluation stays independent of the model implementation.

### FGModel

`FGModel` normalizes the foreground section of the config into a structured
`components` dictionary. It now owns the interpretation of:

- component names and component order
- SED class lookup from `fgbuster.component_model`
- Cl model lookup from `bbpower.fgcls`
- SED parameter name mapping
- Cl parameter name mapping
- cross-component correlation parameter mapping
- frequency decorrelation parameter mapping
- moment-expansion parameter mapping

The component iterator filters only keys named `component_*`:

```python
for key, component in config["fg_model"].items():
    if key.startswith("component_"):
        yield key, component
```

Functional purpose:

- Non-component keys such as `use_moments` are ignored by component parsing.
- Component ordering is explicit and reusable by `BBCompSep.model()`.

Cross-correlation parameters are validated:

```python
if par[0] not in config["fg_model"].keys():
    raise KeyError(...)
if par[0] == key:
    raise KeyError(...)
comp["names_x_dict"][par[0]] = pn
```

Functional purpose:

- A component cannot cross-correlate with an unknown component.
- A component cannot cross-correlate with itself.
- Later model evaluation can map component-pair names to epsilon parameters
  without reparsing config.

SED construction distinguishes fixed and sampled parameters:

```python
if l[1] == "fixed":
    val = l[2][0]
else:
    val = None
params_fgc[l[0]] = val
comp["sed"] = sed_fnc(**params_fgc, units="K_RJ")
```

Functional purpose:

- Fixed SED parameters are passed directly to the SED class.
- Sampled SED parameters are left as `None`, matching the `fgbuster` API for
  free parameters.
- Reference frequencies `nu0` must remain fixed.

Cl construction follows the same fixed/free pattern and filters unused
polarization channels:

```python
if (p1 in config["pol_channels"]) and (p2 in config["pol_channels"]):
    comp["cl"][k] = cl_fnc(**params_fgl[k])
```

Functional purpose:

- B-only runs do not construct unused EE/EB Cl blocks.
- Reference multipoles `ell0` must remain fixed.

### Symbolic Cl Models

`fgcls.py` keeps the symbolic model mechanism but makes the responsibilities
clear:

```python
class ClAnalytic(ClGeneral):
    self._expr = parse_expr(expression).subs(self._fixed_params)
    self._params = sorted([str(s) for s in self._expr.free_symbols])
    ...
    self._lambda = sympy.lambdify(symbols, self._expr, "numpy")
```

Functional purpose:

- Configurable analytic expressions become NumPy-callable model functions.
- Fixed parameters are substituted before free-symbol discovery.
- `ell` is always the first argument internally, but it is not exposed as a
  sampled parameter.

`ClPowerLaw` is a specific analytic model:

```python
analytic_expr = "amp * (ell / ell0)**alpha"
super().__init__(analytic_expr, ell0=ell0, alpha=alpha)
```

Functional purpose:

- Foreground Cl amplitudes and tilts remain configurable.
- `ell0` is the fixed pivot scale.
- Defaults are available when symbolic parameters are free.

### Bandpass Systematics

`Bandpass` owns convolution of SEDs through an instrumental bandpass and now
tracks these systematic controls:

```text
shift: frequency shift, applied as dnu = parameter * nu_mean
gain: multiplicative calibration
angle: polarization angle rotation
dphi1: frequency-dependent phase term
phase_nu: external phase-vs-frequency file
```

The constructor computes the CMB normalization:

```python
self.bnu_dnu = bnu * dnu
cmbs = self.sed_CMB_RJ(self.nu)
self.nu_mean = sum(cmbs * bnu_dnu * nu**3) / sum(cmbs * bnu_dnu * nu**2)
self.cmb_norm = sum(cmbs * bnu_dnu * nu**2)
```

Functional purpose:

- All SED amplitudes are normalized consistently to CMB units.
- Frequency shifts can be expressed relative to the effective band center.

Complex bandpasses are created by `phase_nu` or `dphi1`:

```python
phase = cos(2 * phi_arr) + 1j * sin(2 * phi_arr)
self.bnu_dnu = self.bnu_dnu * phase
self.is_complex = True
```

Functional purpose:

- HWP-like phase effects and frequency-dependent birefringence can produce a
  complex bandpass response.
- Complex convolved amplitudes are converted into an amplitude plus a 2x2
  polarization rotation matrix.

The main convolution method applies shift, phase, CMB normalization, and gain:

```python
nu_prime = self.nu + dnu
conv_sed = sum(sed(nu_prime) * self.bnu_dnu * dphi1_phase * nu_prime**2)
conv_sed /= self.cmb_norm
if self.do_gain:
    conv_sed *= params[self.name_gain]
```

Functional purpose:

- SED integration is centralized.
- CMB and foreground components use the same bandpass convention.
- Systematic parameters enter through the same `params` dictionary used by the
  likelihood.

### Rotation And Decorrelation Helpers

`rotate_cells_mat()` applies left/right 2x2 rotations to spectra:

```python
if mat1 is not None:
    cls = np.einsum("ijk,lk", cls, mat1)
if mat2 is not None:
    cls = np.einsum("jk,ikl", mat2, cls)
```

Functional purpose:

- CMB, foreground, and final bandpower matrices can all use the same rotation
  primitive.
- Either side may be unrotated.

`decorrelated_bpass()` evaluates a decorrelated cross-bandpass scaling:

```text
decorrelation factor = decorr_delta ** (log(nu1 / nu2) ** 2)
decorrelated SED = bphi1^T * factor * bphi2 / (norm1 * norm2)
```

Functional purpose:

- Frequency decorrelation is applied inside bandpass integration instead of
  after collapsing each band to a single effective frequency.
- Shift and gain systematics are included in the decorrelated scaling.

## Branch 03: Component-Separation Likelihood

- Branch: `refactor-stack/03-likelihood-module`
- Commit: `6f497c8 Extract component-separation likelihood`

This branch adds `bbpower/likelihood.py` and moves likelihood evaluation out of
the pipeline stage. The resulting class is small enough to test independently.

Major files:

- `bbpower/likelihood.py`
- `tests/test_likelihood.py`

### Likelihood Interface

The new object is initialized with all runtime dependencies explicitly:

```python
Likelihood(
    model_func,
    param_manager,
    bbdata,
    bbnoise,
    invcov,
    matrix_to_vector,
    use_handl,
    bbfiducial=None,
)
```

Functional purpose:

- The likelihood no longer needs to know about BBPipe or stage I/O.
- The model function is injected, so tests can provide a small fake model.
- Parameter vector handling is delegated to `ParameterManager`.
- Matrix-to-vector ordering is injected by `BBCompSep`, where the map ordering
  is known.

### Chi-Squared Residual

The chi-squared mode computes:

```python
model_cls = self.model(params)
dx = matrix_to_vector(bbdata - model_cls).flatten()
loglike = -0.5 * dx.T @ invcov @ dx
```

Functional purpose:

- This preserves the standard Gaussian bandpower likelihood path.
- The residual vector uses the same upper-triangle ordering as the covariance.

### Hamimeche And Lewis Mode

When `use_handl` is true, setup precomputes:

```python
fiducial_noise = bbfiducial + bbnoise
Cfl_sqrt = sqrtm(fiducial_noise)
observed_cls = bbdata + bbnoise
```

For each bandpower, `h_and_l_dx()` evaluates the H&L transform:

```text
C = model + noise
Chat = observed data + noise
X = g(C^{-1/2} Chat C^{-1/2}) transformed back with fiducial sqrt covariance
dx = upper_triangle(X)
```

Functional purpose:

- H&L-specific linear algebra is isolated from `BBCompSep`.
- Numerical failures return `-inf` likelihood rather than crashing a sampler.
- Noise and fiducial spectra are only required in H&L mode.

### Posterior Evaluation

`lnlike()` converts the flat sampler vector into a full parameter dictionary:

```python
params = self.params.build_params(par)
```

`lnprob()` adds the prior:

```python
prior = self.params.lnprior(par)
if not np.isfinite(prior):
    return -np.inf
return prior + self.lnlike(par)
```

Functional purpose:

- Samplers call a single function, `lnprob()`.
- Priors are consistently enforced across emcee, minimizer, Fisher, timing,
  and single-point evaluations.

## Branch 04: Sampler Backends

- Branch: `refactor-stack/04-sampler-module`
- Commit: `601b9ce Extract sampler backends`

This branch adds `bbpower/samplers.py` and moves sampler-specific behavior out
of `BBCompSep`. It also adds `docs/threading.md`, which documents the new emcee
parallelism controls.

Major files:

- `bbpower/samplers.py`
- `docs/threading.md`
- `tests/test_samplers.py`

### Sampler Dispatch Table

Sampler names are registered in one table:

```python
SAMPLERS = {
    "emcee": run_emcee,
    "polychord": run_polychord,
    "maximum_likelihood": run_minimizer,
    "fisher": run_fisher,
    "single_point": run_singlepoint,
    "timing": run_timing,
}
```

Functional purpose:

- `BBCompSep` no longer needs a long sampler `if` block.
- Adding a new backend means adding one function and one registry entry.
- Unit tests can exercise sampler behavior without constructing a full stage.

### emcee Worker Selection

`_get_emcee_nworkers()` reads worker count from:

```text
1. BBPOWER_EMCEE_WORKERS
2. SLURM_CPUS_PER_TASK
3. os.cpu_count()
```

Then it caps the count:

```python
useful_limit = max(1, (nwalkers + 1) // 2)
nworkers = max(1, min(requested, useful_limit))
```

Functional purpose:

- The default stretch move only proposes about half the walkers at once.
- Asking for more workers than useful creates overhead rather than speedup.
- Cluster jobs can use `SLURM_CPUS_PER_TASK` without changing YAML configs.

`_get_emcee_pool_mode()` reads `BBPOWER_EMCEE_POOL`:

```text
serial
thread
process
```

Functional purpose:

- `thread` is the default because it avoids pickling failures that can happen
  when process pools receive complex likelihood objects.
- `serial` is available for debugging.
- `process` remains available for fully picklable workloads.

### emcee Backend Locking And Restart

`run_emcee()` writes an HDF5 backend at:

```text
<output_dir>/emcee.npz.h5
```

Before touching it, `_emcee_backend_lock()` acquires:

```text
<output_dir>/emcee.npz.h5.lock
```

Functional purpose:

- Two concurrent jobs cannot write the same emcee backend.
- A second writer fails immediately with a clear error.
- This protects a common cluster resubmission failure mode.

Restart behavior:

```text
if backend exists and is readable:
    resume from existing chain
    run max(n_iters - existing_steps, 0)
else:
    reset backend
    initialize walkers around p0
```

Functional purpose:

- Interrupted runs can resume.
- Completed runs with enough steps do not append extra samples.
- Corrupt or unreadable backends produce a targeted error message.

The final compatibility output remains:

```text
emcee.npz:
  chain
  names
  time
  chi2
  ndof
```

### Other Sampler Backends

`run_polychord()` maps the likelihood and priors into the PolyChord API:

```text
pc_likelihood(theta) -> likelihood.lnlike(theta)
pc_prior(unit_cube) -> tophat or Gaussian physical parameters
settings.base_dir = output_dir / "polychord"
```

Functional purpose:

- PolyChord receives a pure likelihood, while its own prior transform handles
  the unit hypercube.
- Output stays isolated under `output_dir/polychord`.

`run_minimizer()` performs a Powell minimization of:

```python
chi2(par) = -2 * likelihood.lnprob(par)
```

and writes:

```text
chi2.npz:
  params
  names
  chi2
  ndof
```

`run_fisher()` first minimizes, then computes:

```python
fisher = -Hessian(likelihood.lnprob)(best_fit)
cov = inv(fisher)
```

and writes `fisher.npz`.

`run_singlepoint()` evaluates the posterior at `p0` and writes
`single_point.npz`.

`run_timing()` repeatedly evaluates `likelihood.lnprob(p0)` and writes
`timing.npz`.

`run_predicted_spectra()` is separate from `SAMPLERS` because it needs the
`BBCompSep` stage object for SACC tracer/window I/O. It can write either:

```text
cells_model.npz
cells_model.fits
```

depending on config.

## Branch 05: BBCompSep Integration

- Branch: `refactor-stack/05-compsep-integration`
- Commit: `e9d09fe Wire BBCompSep to likelihood and samplers`

This branch rewires `BBCompSep` so it orchestrates data/model setup while
delegating parameter parsing, foreground model construction, likelihood
evaluation, and sampler execution to helper modules.

Major files:

- `bbpower/compsep.py`
- `tests/test_compsep.py`
- `test/run_predicted_spectra_test.sh`
- `test/test_config_predicted_spectra.yml`

### Setup Flow

`setup_compsep()` is now the central orchestration block:

```python
self.parse_sacc_file()
if self.config["fg_model"].get("use_moments"):
    self.precompute_w3j()
self.load_cmb()
self.fg_model = FGModel(self.config)
self.params = ParameterManager(self.config)
self.likelihood = Likelihood(
    model_func=self.model,
    param_manager=self.params,
    bbdata=self.bbdata,
    bbnoise=self.bbnoise,
    invcov=self.invcov,
    matrix_to_vector=self.matrix_to_vector,
    use_handl=self.use_handl,
    bbfiducial=getattr(self, "bbfiducial", None),
)
```

Functional purpose:

- Stage setup is explicit and testable.
- `BBCompSep` still owns SACC layout and physics model evaluation.
- Likelihood and sampler modules own their narrower responsibilities.

### Matrix-Vector Ordering

`matrix_to_vector()` and `vector_to_matrix()` define the covariance vector
ordering:

```text
matrix shape: (..., nmaps, nmaps)
vector shape: (..., ncross)
selected entries: upper triangle of the map-map matrix
```

Functional purpose:

- Data vectors, covariance matrices, and model residuals share one ordering.
- H&L and chi-squared modes do not duplicate ordering logic.

### Frequency/Polarization Iterator

`_freq_pol_iterator()` yields:

```text
b1, b2: frequency indices
p1, p2: polarization indices
m1, m2: flattened map indices
icl: running upper-triangle spectrum index
```

Functional purpose:

- SACC parsing, covariance reshaping, model writing, and predicted spectra use
  one consistent loop over unique spectra.
- Auto-frequency spectra only include the upper polarization triangle.

### SACC Parsing

`parse_sacc_file()` performs these major blocks:

```text
1. Select likelihood mode: chi2 or H&L.
2. Load coadded data SACC and covariance SACC.
3. Verify data/covariance ordering for cl_bb tracer pairs.
4. If H&L, load fiducial and noise SACC files.
5. Remove unrequested polarization channels.
6. Apply l_min and l_max cuts to all relevant SACC files.
7. Choose frequency tracers from config["bands"] or all tracers.
8. Build Bandpass objects from SACC NuMap tracers.
9. Extract bandpower windows and ell sampling.
10. Reorder data, noise, fiducial, and covariance into BBPower arrays.
11. Convert vectors back to symmetric matrices.
12. Solve for the inverse covariance.
```

Functional purpose:

- The likelihood receives dense NumPy arrays rather than SACC objects.
- Polarization and ell cuts are applied consistently to data, covariance,
  noise, and fiducial inputs.
- The covariance ordering is checked before inversion.
- Bandpasses come from the same SACC tracers used for the data.

The core data layout is:

```text
bbdata:     (n_bpws, nmaps, nmaps)
bbnoise:    (n_bpws, nmaps, nmaps), H&L only
bbfiducial: (n_bpws, nmaps, nmaps), H&L only
bbcovar:    (n_bpws * ncross, n_bpws * ncross)
invcov:     same as bbcovar
windows:    (ncross, n_bpws, n_ell)
```

### CMB Template Loading

`load_cmb()` reads the configured CMB template files and fills:

```text
cmb_tens[npol, npol, nell]
cmb_lens[npol, npol, nell]
cmb_scal[npol, npol, nell]
```

Functional purpose:

- `r_tensor` scales the tensor contribution.
- `A_lens` scales the lensing contribution.
- Scalar EE is included when E polarization is requested.
- B-only and E+B runs share the same template loading code.

### SED Integration

`integrate_seds()` computes:

```text
single_sed[n_components, nfreqs]
comp_scaling[n_components, nfreqs, nfreqs]
fg_scaling[n_components, n_components, nfreqs, nfreqs]
rot_matrices[n_components, nfreqs]
```

Functional purpose:

- Each component SED is convolved through each bandpass.
- Component auto scalings use either outer products or decorrelated bandpass
  integrals.
- Component cross scalings use the configured epsilon correlation parameter.
- Bandpass phase effects return per-component rotation matrices.

The cross-component block is:

```text
fg_scaling[c1, c2] = epsilon * outer(single_sed[c1], single_sed[c2])
fg_scaling[c2, c1] = epsilon * outer(single_sed[c2], single_sed[c1])
```

Functional purpose:

- Dust-synchrotron-like correlations are symmetric in component order.
- The model can handle multiple foreground components with explicit ordering.

### Foreground Power Spectra

`evaluate_power_spectra()` evaluates each configured Cl model:

```text
for component:
  for configured polarization pair:
    params = current values for that Cl function
    D_ell = clfunc.eval(bpw_l, *params)
    C_ell = D_ell * dl2cl
    fill foreground matrix, including symmetric transpose if needed
```

Functional purpose:

- Cl model evaluation is separated from SED scaling.
- D_ell-to-C_ell conversion happens once before model assembly.
- Only requested polarization channels are present.

### Full Model Assembly

`model(params)` now has a clear sequence:

```text
1. Build CMB C_ell from r_tensor, A_lens, and scalar templates.
2. Apply optional cosmic birefringence rotation.
3. Integrate SEDs through bandpasses.
4. Evaluate foreground Cl models.
5. For each frequency pair:
   a. Add rotated and scaled CMB.
   b. Add all foreground component auto terms.
   c. Add all foreground component cross terms.
6. Add moment-expansion terms when enabled.
7. Convolve theory C_ell with bandpower windows.
8. Apply instrumental polarization-angle rotations.
9. Return (n_bpws, nmaps, nmaps).
```

Functional purpose:

- All physical model pieces are still assembled inside `BBCompSep`, where
  bandpasses, windows, and map ordering are available.
- Likelihood code only asks for `model(params)`.
- Moment expansion remains opt-in through config.

Moment expansion blocks:

```text
precompute_w3j(): build squared Wigner-3j tensor
integrate_seds_der(): band-averaged SED beta derivatives
evaluate_1x1(): first-order moment correction
evaluate_0x2(): zeroth-by-second-order correction
```

Functional purpose:

- Expensive Wigner-3j values are precomputed once.
- Moment corrections are only evaluated when `fg_model.use_moments` is true.

### Stage Runtime Dispatch

`run()` now does only stage-level work:

```python
copyfile(self.get_input("config"), self.get_output("config_copy"))
self.setup_compsep()

sampler_name = self.config.get("sampler", "emcee")
if sampler_name == "predicted_spectra":
    samplers.run_predicted_spectra(...)
elif sampler_name in samplers.SAMPLERS:
    samplers.SAMPLERS[sampler_name](...)
else:
    raise ValueError(...)
```

Functional purpose:

- Config copying remains a stage responsibility.
- Sampler-specific code is no longer embedded in `compsep.py`.
- Predicted spectra remains special because it needs stage internals for SACC
  output.

## Branch 06: Power Spectrum Stage

- Branch: `refactor-stack/06-power-specter`
- Commit: `3600878 Harden power spectrum stage`

This branch hardens `BBPowerSpecter`, especially around NaMaster compatibility,
workspace reuse, split iteration, SACC window output, and simulation handling.

Major files:

- `bbpower/power_specter.py`
- `test/run_power_specter_test.sh`
- `tests/test_power_specter.py`

### Stage Inputs And Outputs

`BBPowerSpecter` remains the map-to-bandpowers stage:

```text
inputs:
  splits_list
  masks_apodized
  bandpasses_list
  sims_list
  beams_list

outputs:
  cells_all_splits
  cells_all_sims
  mcm
```

Functional purpose:

- Data split maps and simulation split maps are processed with the same
  measurement code.
- Data spectra include SACC bandpower windows.
- Simulation spectra are listed in `cells_all_sims` for the summarizer.

### Beam And Bandpass Reading

`read_beams()` validates the number of beam files and interpolates each beam
onto:

```python
self.larr_all = np.arange(3 * self.nside)
```

Functional purpose:

- Every frequency band must have a beam.
- Beam transfer functions match the multipole grid used by NaMaster fields.
- Values outside the input beam grid are filled safely.

`read_bandpasses()` builds:

```text
self.bpss["bandN"] = {"nu": nu, "dnu": dnu, "bnu": bnu}
```

Functional purpose:

- SACC tracers can carry bandpass and `dnu` metadata forward to later stages.

### NaMaster Version Compatibility

The branch adds compatibility helpers for NaMaster 1.x and 2.x constructor
differences:

```python
def _nmt_bin_uses_keyword_api():
    params = inspect.signature(nmt.NmtBin).parameters
    return "f_ell" in params and "is_Dell" not in params
```

Custom bin creation:

```text
NaMaster 2.x:
  NmtBin(bpws=..., ells=..., weights=..., f_ell=...)

NaMaster 1.x:
  NmtBin(nside, bpws=..., ells=..., weights=..., is_Dell=...)
```

Functional purpose:

- The same pipeline config works across installed NaMaster versions.
- Historical `compute_dell` behavior is preserved by passing `f_ell` when
  `is_Dell` is no longer available.

Workspace coupling matrix compatibility:

```text
if workspace.compute_coupling_matrix accepts n_iter:
    pass n_iter there
else:
    rely on n_iter passed to NmtField
```

Functional purpose:

- NaMaster 1.x and 2.x moved `n_iter` between APIs.
- The code avoids passing `n_iter` twice on newer installs.

### Workspace Reuse

Workspace filenames are canonicalized by sorted band pair:

```python
fname = f"{prefix_mcm}_{min(b1,b2)+1}_{max(b1,b2)+1}.fits"
```

Runtime behavior:

```text
if workspace file exists:
    read it
else:
    compute from dummy fields
    write it
```

Functional purpose:

- Expensive mode-coupling matrices are reused across runs.
- Band pair `(1, 2)` and `(2, 1)` map to the same file.
- Workspaces are computed only for unique upper-triangle band pairs.

### Cross-Split Spectra

`get_cell_iterator()` yields unique band/split pairs:

```text
for b1 <= b2:
  if same band:
    s2 starts at s1
  else:
    use all split pairs
```

Functional purpose:

- Auto-frequency spectra avoid duplicate split pairs.
- Cross-frequency spectra include all split combinations.
- The same iterator drives computation and SACC writing, avoiding ordering
  drift.

`compute_cells_from_splits()`:

```text
1. Build NmtField for every band/split Q/U map.
2. For every iterator pair, select the correct workspace.
3. Compute coupled cell.
4. Decouple with the workspace.
5. Store EE, EB, BE, BB spectra by map-label pair.
```

Functional purpose:

- Field construction is separated from cross-spectrum computation.
- Missing map files are checked, including `.gz` fallback.
- The output dictionary is structured exactly as the SACC writer expects.

### SACC Output

`get_sacc_tracers()` creates one `NuMap` tracer per band/split:

```text
name: bandN_splitM
quantity: cmb_polarization
spin: 2
bandpass, dnu, beam metadata included
```

`get_sacc_windows()` extracts EE, EB, BE, and BB windows from each workspace.

`save_cell_to_file()` writes:

```text
cl_ee
cl_eb
cl_be, except exact auto spectra where BE is symmetric with EB
cl_bb
```

Functional purpose:

- The summarizer receives all split-level spectra in SACC format.
- Data files carry bandpower windows needed by downstream stages.
- Simulation files can omit windows to reduce repeated metadata.

### Simulation Handling

The run method writes all expected simulation output filenames to
`cells_all_sims` before computing simulations:

```text
<cells_all_splits prefix>_sim0.fits
<cells_all_splits prefix>_sim1.fits
...
```

Functional purpose:

- Downstream stages can read one list file to find simulations.
- Existing simulation SACC outputs are skipped, making reruns cheaper.

## Branch 07: Power Summarizer Stage

- Branch: `refactor-stack/07-power-summarizer`
- Commit: `ae7e64a Harden power summarizer stage`

This branch hardens `BBPowerSummarizer`, which turns split-level SACC spectra
into coadded data products, noise estimates, null tests, and covariances from
simulations.

Major files:

- `bbpower/power_summarizer.py`
- `tests/test_power_summarizer.py`

### SACC Consistency Checks

`check_sacc_consistency()` validates:

```text
number of bands
number of splits
number of tracers == nbands * nsplits
number of tracer combinations
length of data vector == n_bpws * expected spectra
```

Functional purpose:

- The summarizer fails early on mismatched inputs.
- The downstream reshape logic is protected from silent ordering or size
  errors.

### Null Pairing Setup

`init_params()` computes null pairings of the form:

```text
(m_i - m_j) x (m_k - m_l)
```

with all split indices distinct. In spectra, each null is:

```text
C_ik - C_il - C_jk + C_jl
```

Functional purpose:

- Null tests compare independent split differences.
- Pairings are computed once and reused for data and simulations.

### Window And Tracer Construction

`get_windows()` extracts bandpower windows for each band pair and polarization:

```text
windows["band1_band2"]["ee"]
windows["band1_band2"]["eb"]
windows["band1_band2"]["be"]
windows["band1_band2"]["bb"]
```

`get_tracers()` creates:

```text
t_coadd: one tracer per frequency band
t_nulls: one tracer per band and null split-difference label
```

Functional purpose:

- Coadded SACC files use band-level tracers instead of band/split tracers.
- Null SACC files have explicit tracer names encoding the split difference.

### Index Lookup

`get_cl_indices()` builds a flattened lookup:

```text
inds[map1, map2, ell_bin] -> SACC data-vector index
```

where maps are ordered by:

```text
polarization + 2 * (band + nbands * split)
```

Functional purpose:

- SACC data can be reshaped into a dense split/band/pol tensor.
- Symmetric map pairs reuse the same index.
- Coadd and null calculations do not need repeated SACC lookups.

### Coadd, Noise, And Null Logic

`parse_splits_sacc_file()` reshapes the data vector into:

```text
spectra[split1, split2, band1, pol1, band2, pol2, ell]
```

Total coadd:

```text
weights = ones(nsplits) / nsplits
spectra_total = weights_i * spectra_ij * weights_j
```

Cross-only coadd:

```text
upper = mean(spectra over i < j)
lower = mean(spectra over i > j)
spectra_xcorr = 0.5 * (upper + lower)
```

Noise estimate:

```text
spectra_noise = spectra_total - spectra_xcorr
```

Nulls:

```text
spectra_null = spectra[i,k] - spectra[i,l] - spectra[j,k] + spectra[j,l]
```

Functional purpose:

- Total coadd includes autos and therefore signal plus noise.
- Cross-only coadd excludes auto-split noise bias and estimates signal.
- Noise is estimated as the difference between total and cross-only coadds.
- Null tests should be consistent with zero if split systematics are absent.

### Covariance Estimation

`get_covariance_from_samples()` supports:

```text
dense: full sample covariance
diagonal: only variance on the diagonal
block_diagonal: full covariance within ell-neighbor blocks, clipped by off_diagonal_cut
```

Dense covariance:

```text
cov = mean(v_i v_j) - mean(v_i) mean(v_j)
```

Block-diagonal covariance:

```text
reshape data dimension into (nblocks, n_bpws)
zero ell-bin blocks farther than off_diagonal_cut
reshape back to 2D covariance
```

Functional purpose:

- Data spectra can keep limited off-diagonal ell covariance.
- Null spectra can use diagonal covariance to avoid very large dense null
  matrices.
- Covariance policy is config-controlled.

## Branch 08: Plotter Stage

- Branch: `refactor-stack/08-plotter`
- Commit: `c567f92 Refactor plotter stage`

This branch refactors `BBPlotter` and fixes functional issues around optional
plot sections and non-MCMC chain files.

Major files:

- `bbpower/plotter.py`
- `tests/test_plotter.py`

### Input Loading Based On Plot Flags

`read_inputs()` always loads fiducial and cross-coadded spectra:

```text
cells_fiducial
cells_coadded
```

and conditionally loads:

```text
cells_coadded_total, only if plot_coadded_total
cells_noise, only if plot_noise
cells_null, only if plot_nulls
param_chains, only if plot_likelihood
```

Functional purpose:

- Plotter does not read unused optional inputs unnecessarily.
- The runtime state mirrors enabled plot sections.

### Likelihood Plot Availability

The plotter now checks whether the `param_chains` file actually contains MCMC
samples:

```python
self.can_plot_likelihood = (
    "chain" in self.chain.files and "names" in self.chain.files
)
```

Functional purpose:

- Maximum-likelihood outputs such as `chi2.npz` do not cause triangle-plot
  crashes.
- The HTML contents only link to likelihood plots when they can be generated.

### HTML Page Creation

`create_page()` builds the table of contents from enabled/available sections:

```text
Bandpasses
Coadded power spectra
Null tests, if plot_nulls
Likelihood, if can_plot_likelihood
```

Functional purpose:

- The HTML page reflects what was actually plotted.
- Users do not get dead likelihood links when the chain file is not an MCMC
  file.

### Diagnostic Plot Blocks

`add_bandpasses()` writes:

```text
bpass_summary.png
bpass_<tracer>.png
```

Functional purpose:

- Reviewers can inspect bandpass shapes and effective frequencies.

`add_coadded()` writes one plot per tracer pair and polarization pair:

```text
fiducial model
total coadd, optional
noise, optional
cross coadd
positive values as filled markers
negative values as open shifted markers
```

Functional purpose:

- Total, cross-only, noise, and fiducial spectra can be compared on one page.
- `D_ell` versus `C_ell` axis labeling follows `compute_dell`.

`add_nulls()` plots:

```text
C_ell / sigma_ell
```

for each null tracer pair and polarization.

Functional purpose:

- Nulls are shown in significance units rather than raw spectra.
- The branch fixes the save keyword to `bbox_inches`, so null plots are
  actually saved with the intended bounding box behavior.

`add_contours()` uses `getdist` only for available MCMC chains:

```text
discard first quarter as burn-in
flatten walkers
select known labeled parameters
export triangle.png
```

Functional purpose:

- Existing diagnostic contour behavior remains available for emcee chains.
- Non-MCMC outputs are skipped gracefully.

## Branch 09: Examples And No-Pipe Workflow

- Branch: `refactor-stack/09-examples-nopipe`
- Commit: `94b2fcd Update examples and no-pipe workflow`

This branch updates examples so they are more callable, package-aware, and
aligned with the refactored stage entry points.

Major files:

- `examples/generate_SO_spectra.py`
- `examples/generate_SO_maps.py`
- `examples/utils.py`
- `examples/noise_calc.py`
- `examples/polychord_plot_triangle.py`
- `examples/config_nopipe.yml`
- `test/run_polychord_test.sh`

### Synthetic Spectra Script

`generate_SO_spectra.py` is reorganized around:

```python
def main(prefix_out: str, so_forecast: bool = False) -> None:
    ...

if __name__ == "__main__":
    main(sys.argv[1], so_forecast="--so_forecast" in sys.argv)
```

Functional purpose:

- The script can still be run from the command line.
- The same logic can be imported and called from tests or notebooks.
- `--so_forecast` remains available for the alternate SO 2023 forecast
  foreground parameters.

The major data-generation blocks are:

```text
1. Load bandpasses.
2. Build bandpower windows.
3. Load SO beams.
4. Build CMB, synchrotron, and dust component spectra.
5. Convert D_ell to C_ell.
6. Convolve components with bandpower windows.
7. Convolve components with bandpasses.
8. Add SO noise spectra.
9. Write signal, fiducial, and noise SACC files.
10. Add analytic covariance to the coadded data file.
```

Functional purpose:

- Produces the quick component-separation test inputs:
  `cls_coadd.fits`, `cls_fid.fits`, and `cls_noise.fits`.
- Keeps examples independent of the map-level stage when users only want to
  test `BBCompSep`.

### Synthetic Map Script

`generate_SO_maps.py` now uses `argparse` and a `main()` function:

```text
--output-dir
--seed
--nside
```

Functional blocks:

```text
1. Seed NumPy RNG.
2. Generate CMB, synchrotron, and dust Q/U maps with healpy.synfast.
3. Convolve component maps into observing frequencies using bandpass SEDs.
4. Generate split-dependent noise maps.
5. Smooth signal maps by SO beam FWHM.
6. Write obs_splitNof4.fits.gz maps.
```

Functional purpose:

- The map generator is usable in shell scripts and importable contexts.
- The output file naming matches `BBPowerSpecter` expectations.

### Shared Example Utilities

`examples/utils.py` keeps the example physical model:

```text
fcmb()
comp_sed()
dl_plaw()
read_camb()
Bpass
get_component_spectra()
get_convolved_seds()
```

Functional changes worth noting:

- `get_component_spectra(..., so_forecast=True)` selects the Wolz et al.
  forecast foreground parameter set.
- The default remains the earlier SO forecast-style parameters.
- `get_convolved_seds()` uses the same `so_forecast` switch for dust spectral
  index and temperature.

`examples/noise_calc.py` remains the SO noise-curve source used by spectra and
map generation. Most branch changes there are cleanup, but the file remains a
functional dependency for example data generation.

### PolyChord Plot Script

`polychord_plot_triangle.py` now imports the installed package helper:

```python
from bbpower.param_manager import ParameterManager
```

and reads:

```text
test/test_out/config_copy.yml
test/test_out/param_chains/pch
```

Functional purpose:

- The plotting script uses the same parameter parsing code as BBPower itself.
- It no longer depends on inserting `bbpower/` directly into `sys.path`.
- GetDist labels are generated only for parameters present in the config.

### PolyChord Test Script

`test/run_polychord_test.sh` now runs a fuller workflow:

```text
1. Generate fiducial spectra.
2. Generate 100 simulation map directories.
3. Run BBPowerSpecter.
4. Run BBPowerSummarizer.
5. Run BBCompSep with PolyChord config.
6. Run PolyChord triangle plot script.
7. Check expected PolyChord chain output.
8. Clean up test output.
```

Functional purpose:

- Exercises the refactored stage entry points end to end.
- Uses the full simulation list rather than a short commented-out block.

## Branch 10: Docs And Legacy Tests

- Branch: `refactor-stack/10-docs-and-legacy-tests`
- Commit: `07bb86b Update docs and preserve legacy sampling test`

This branch documents the refactored project and preserves legacy sampling
coverage.

Major files:

- `README.md`
- `docs/setup.md`
- `docs/architecture.md`
- `docs/configuration.md`
- `docs/examples.md`
- `test/run_sampling_legacy_test.sh`
- `test/run_sampling_test.sh`
- `test/test_config_sampling_legacy.yml`
- `test/test_sampling_legacy.yml`

### README Refresh

The README now presents:

```text
installation by workflow
quick start from synthetic spectra
four pipeline stages
common entry points
configuration shape
stage outputs
parameter format
sampler table
test commands
documentation links
```

Functional purpose:

- New users can start at `BBCompSep` without reading map-level docs.
- Optional dependency installation is tied to stage needs.
- The README points to detailed docs instead of trying to contain everything.

### Setup Documentation

`docs/setup.md` explains:

```text
environment creation
dependency extras
install verification
lowest-friction stage entry points
minimal BBCompSep file checklist
stage outputs
emcee threading summary
smoke tests
common setup failures
```

Functional purpose:

- Users can choose a small dependency set.
- Common missing-package errors map to install commands.
- Cluster threading guidance is discoverable from setup docs.

### Architecture Documentation

`docs/architecture.md` records the refactored module map and runtime data flow:

```text
BBPowerSpecter -> BBPowerSummarizer -> BBCompSep -> BBPlotter
```

It also describes the helper-module split:

```text
compsep.py: orchestration and model assembly
likelihood.py: residuals and posterior values
samplers.py: backend execution
param_manager.py: parameter vector contract
fg_model.py / fgcls.py / bandpasses.py: physical model helpers
```

Functional purpose:

- Reviewers can see why code moved out of `compsep.py`.
- Users can identify the right module for a bug or extension.

### Configuration Documentation

`docs/configuration.md` documents:

```text
pipeline YAML versus stage config YAML
global options
BBCompSep options
CMB model block
foreground component block
systematics block
sampler options
predicted spectra mode
```

Functional purpose:

- The config contract becomes explicit.
- Examples of parameter list structure and priors are centralized.

### Examples Documentation

`docs/examples.md` provides runnable workflows:

```text
synthetic spectra -> BBCompSep -> BBPlotter
emcee sampling
full maps-to-parameters pipeline
Fisher forecast
predicted spectra output
test script descriptions
```

Functional purpose:

- Users can test each entry point without reverse-engineering shell scripts.
- Component-separation-only usage is presented as the fastest path.

### Threading Documentation

`docs/threading.md` explains the new emcee runtime controls:

```text
BBPOWER_EMCEE_WORKERS
BBPOWER_EMCEE_POOL
SLURM_CPUS_PER_TASK
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
NUMEXPR_NUM_THREADS
```

Functional purpose:

- Users can avoid CPU oversubscription on clusters.
- The worker cap and default thread pool are justified in one place.
- HDF backend single-writer behavior is documented.

### Legacy Sampling Test Preservation

The branch adds a legacy sampling config and wrapper script so the old direct
spectra sampling path remains testable after the refactor.

Functional purpose:

- Refactoring does not only validate new workflows.
- The historical `BBCompSep + BBPlotter` smoke path remains available for
  regression checks.

## End-State Integration View

The stack moves BBPower toward this responsibility split:

```text
bbpower.__main__       CLI stage dispatch
bbpower._stages        stage registry
bbpower.types          BBPipe file type declarations

power_specter.py       maps -> split-level SACC spectra
power_summarizer.py    split-level spectra -> coadds, nulls, covariances
compsep.py             SACC parsing + physical model assembly + orchestration
likelihood.py          chi2 / H&L likelihood evaluation
samplers.py            inference and evaluation backends
plotter.py             PNG diagnostics + HTML page

param_manager.py       fixed/free parameter vector contract
fg_model.py            foreground config normalization
fgcls.py               symbolic Cl models
bandpasses.py          SED convolution and bandpass systematics
```

The main architectural goal is not to change BBPower's scientific model. The
goal is to make the existing pipeline easier to install, test, review, and
extend by giving each code block a narrower responsibility.

## Suggested Review Checklist

Reviewers should focus on these higher-risk integration points:

- Does lazy importing preserve all existing stage access patterns?
- Are dependency extras acceptable for the environments used by collaborators?
- Does `ParameterManager` preserve parameter ordering expected by old outputs?
- Does `parse_sacc_file()` preserve the exact data-vector and covariance
  ordering used by previous BBCompSep runs?
- Does the H&L likelihood match the previous numerical behavior?
- Do sampler output filenames and keys remain compatible with existing scripts?
- Does NaMaster compatibility logic work on the versions used by the group?
- Are coadd, noise, null, and covariance conventions scientifically unchanged?
- Should the examples use generated synthetic data as defaults, or should they
  point more strongly to existing production inputs?

Suggested checks before proposing integration:

```bash
python -m pytest
bash test/run_sampling_test.sh
bash test/run_predicted_spectra_test.sh
```

For full map-level validation, run:

```bash
bash test/run_power_specter_test.sh
```

That last test is slower and needs the heavier map-level optional dependencies.
