# Cross-calibration against Planck

Gain, polarization efficiency and polarization-angle fit for the SATs, using
Planck as the absolute reference, on SOOPERCOOL cross-Planck SACC files.

## The model

For every pair of map sets `(i, j)` the observed spectra are

```
C^obs_{l,ij} = R_i(psi_i) P_i C^sky_{l,ij} P_j R_j^T(psi_j)
```

with, in the `(T, E, B)` basis,

```
P_i = g_i * diag(1, eps_i, eps_i)          gain g_i on T,Q,U; pol. eff. eps_i on Q,U

             ( 1      0            0        )
R_i(psi_i) = ( 0   cos 2psi_i   sin 2psi_i  )
             ( 0  -sin 2psi_i   cos 2psi_i  )
```

and the sky

```
C^sky_{l,ij} = C^CMB_l + C^dust_{l,ij}

D^{d,XX}_{l,ij} = A_d^XX (l/l0)^{alpha_d^XX} f_i^d f_j^d
```

`f^d` is a modified black body integrated over each bandpass and converted to
K_CMB, normalised to unity at `nu0 = 353 GHz`:

```
f_i = g^CMB_RJ(nu0) * Int dnu B_i(nu) nu^2 s_RJ(nu) / Int dnu B_i(nu) nu^2 g^CMB_RJ(nu)
s_RJ(nu) = (nu/nu0)^(1+beta_d) * (exp(h nu0/k T_d) - 1)/(exp(h nu/k T_d) - 1)
```

which is the same normalisation BBPower uses (`fgbuster` `Dust` in `K_RJ`
times `CMB('K_RJ').eval(nu0)`).

The likelihood is Gaussian with the SACC covariance held fixed:

```
-2 ln L = [d - m(theta)]^T C^-1 [d - m(theta)]
```

### Sign convention for the angle

With `R` written exactly as above, an auto-spectrum picks up

```
C^EB_obs = (g^2 eps^2 / 2) sin(4 psi) (C^BB - C^EE)
```

i.e. a *positive* `psi` gives a *negative* EB when EE > BB. Flip the sign if
you want the opposite convention; nothing else in the fit depends on it.

## Relationship to `BBCompSep`

This is a separate likelihood, not a mode of `BBCompSep`, because the stock
stage cannot express the model:

* it is E/B-only -- it strips every `cl_0*` / `cl_*0` spectrum from the SACC
  (`bbpower/compsep_nopipe.py:161`), its CMB templates carry only EE/BB, and
  its rotation matrices are 2x2;
* it has a single scalar `gain` per bandpass, with no separate polarization
  efficiency;
* it assumes the *complete* `N_map x N_map` spectrum matrix at every
  bandpower, so it cannot drop BB or use EB-only for SAT x SAT.

Supporting T, `eps` and per-pair spectrum selection would mean rewriting
`parse_sacc_file` and `model`, i.e. most of the file. So `compsep_calib.py`
sits alongside `compsep_nopipe.py` as a second standalone entry point: same
config grammar, same prior syntax, same SACC / bandpass / window conventions,
same sampler options, but a 3x3 T/E/B model. Like `compsep_nopipe.py` it is
run as a script and does not go through `bbpipe`, so it is not exported from
`bbpower/__init__.py`.

One `BBCompSep` bug worth knowing if you go back to it: it looks its
per-bandpass systematics up under the keys `bandpass_1`, `bandpass_2`, ... --
both in `Bandpass.__init__` and in `ParameterManager` -- so the
map-set-named `systematics:` blocks in `examples/config_nopipe.yml` are
silently ignored. `compsep_calib.py` keys them by map-set name instead.

## Conventions taken from the data

Checked directly on the SACC file, not assumed:

* **Spectra are `D_l`.** The bandpower windows already contain the
  `l(l+1)/2pi` factor, so they map a model `C_l` onto a binned `D_l`. The code
  builds `D_l`, multiplies by `2pi/(l(l+1))`, then applies the window — the
  same thing BBPower does with `compute_dell: True`.
* **Beams are unity** in every tracer, i.e. already deconvolved. No beam is
  applied in the model.
* **Auto-pairs have redundant transposes.** For a single map set, `cl_e0` is
  bit-for-bit `cl_0e` (and `cl_be` is `cl_eb`). Keeping both makes the
  covariance singular, so `compsep_calib.py` drops `ET`/`BE`/`BT` whenever
  the two tracers are the same. This is why the Planck auto-spectra
  contribute `TE, EE, EB` rather than the four types requested for them, and
  why SAT f090 x f090 contributes only `EB`.
* **The Planck TT auto-spectra are excluded.** With Planck's `g`, `eps` and
  `psi` pinned and no dust TT term, 100x100 and 143x143 TT contain no free
  parameter at all — the model cannot move to fit them — and they were the
  two largest chi2 contributors in every run. Dropping them is a scope
  choice, not a fix: it removes a consistency check of the fiducial CMB
  against Planck on this mask. Add `'TT'` back to `planck_x_planck` to
  restore it.
* **`l_min = 105`, `l_max = 400`**, which keeps the 20 bandpowers centred
  from 112 to 397. The low cut matches the SATp1 null tests (which run
  `105..495`); the upper cut here is tighter. The low cut is
  deliberate: at `l_min = 60` the bins centred at 67 and 82 come in, and
  SAT x Planck TT is not trustworthy there — the SAT filtering drives the
  data/theory ratio to 0.42 at f090 and to **-1.19** at f150 near `l = 67`,
  and those bins pull directly on the gains. Pushing lower also costs
  positive-definiteness in the covariance; the code raises rather than
  inverting a bad matrix.

## Degeneracies

* The CMB is **fixed**, so it is what sets the absolute scale. If you let the
  CMB amplitude float, every gain floats with it.
* Planck's `g`, `eps`, `psi` are **fixed** (1, 1, 0). That is what makes this a
  cross-calibration: the SAT parameters are measured against Planck through
  the SAT x Planck cross-spectra. Letting them float instead measures each
  experiment against the fiducial CMB, which is a different (weaker) question.
* `g_353` is degenerate with the dust amplitudes, because `f_353 = 1` by
  construction. Keep it fixed even if you free the other Planck gains.
* `g` and `eps` separate because TT constrains `g_i g_j`, EE constrains
  `g_i g_j eps_i eps_j`, and TE/ET constrain the mixed products.
* `psi` comes from EB/BE: from SAT x Planck (giving `psi_SAT - psi_Planck`) and
  from SAT x SAT (giving `sin 4 psi_SAT`). Dust EB is set to zero; if you free
  it, it will fight the angles.
* **Dust BB is fixed, not fitted.** Constraining it would need SAT x SAT BB,
  which is deliberately out of the data selection. With no BB spectrum in the
  fit, dust BB only reaches the likelihood through the EB leakage
  `sin(4 psi) (BB - EE)`: moving `A_d_BB` from 0 to 20 muK^2 changes the total
  chi2 by less than 0.32. Left free it walks into a prior wall. Dust TT and TE
  *are* fitted -- Planck 353 x 353 TT carries ~1000 muK^2 of dust at `l = 80`,
  and TT/TE are in the data vector.

## Files

```
bbpower/
  compsep_calib.py    the likelihood + samplers
  plotter_calib.py    data vs model, pulls, triangle plot
examples/calibration_cross-planck/
  README.md           this file
  satp1/
    config_satp1.yml    the run configuration
    run_satp1.sh        driver
    outputs/fiducial/   chains, best fit, plots (gitignored)
```

### Which CMB template

The fit uses `examples/data/camb_lens_nobb_planck2018.dat` — lensed scalar
spectra with `r = 0`, i.e. lensing BB and no primordial BB, in the
`ell D_TT D_EE D_BB D_TE` column order `load_cmb` expects.

It was generated with CAMB at the **Planck 2018 VI Table 2
TT,TE,EE+lowE+lensing** marginalised means:

| | | | |
|---|---|---|---|
| `H0` 67.36 | `ombh2` 0.02237 | `omch2` 0.1200 | `tau` 0.0544 |
| `As` 2.100e-9 | `ns` 0.9649 | `mnu` 0.06 eV | `omk` 0 |

The full parameter set is in the file's own header, so the cosmology never
has to be guessed later. Rows run `ell = 1..1950`, matching the other
`camb_lens_*.dat` files one for one — worth preserving if you ever pair it
with `camb_lens_r1.dat` under stock `BBCompSep`, whose `load_cmb` builds a
mask from one template and applies it *positionally* to the other.

The CMB is the absolute calibrator, so this choice sets the gains directly
and **two runs are only comparable if they use the same template.**

The other file in that directory, `camb_lens_nobb.dat`, is at a different
and unrecorded cosmology: its `D_TT` runs about 4.5–5% high across the
fitted range (ratio 1.052 at `l = 100`, 1.045 at `l = 220`, 1.029 at
`l = 500`), and its low-`l` EE differs by far more, which is a different
optical depth.

Running the identical fit with both shows what that costs. **The
calibration parameters barely move; the chi2 gets much worse.**

| | `camb_lens_nobb` | Planck 2018 | shift |
|---|---|---|---|
| `g_90` | 0.8587 ± 0.0081 | 0.8587 ± 0.0085 | 0.0σ |
| `eps_90` | 0.8479 ± 0.0127 | 0.8536 ± 0.0133 | 0.4σ |
| `g_150` | 0.9325 ± 0.0141 | 0.9321 ± 0.0148 | 0.0σ |
| `eps_150` | 0.9154 ± 0.0169 | 0.9222 ± 0.0177 | 0.4σ |
| chi2 / ndof | 1299.7 / 1141 | 1231.5 / 1141 | |
| PTE | 0.0007 | 0.031 | |

It is tempting to expect the gains to absorb a rescaled theory, since the
SAT x Planck TT model is `g_SAT * C^TT` and therefore linear in `g_SAT`.
They do not. A cosmology error is common to *every* spectrum while the
gains are not free to follow it: Planck's `g` and `eps` are pinned at 1, so
the Planck auto-spectra have no freedom at all, and each SAT gain is
constrained simultaneously through TT and TE (which go as `g`) and through
ET and EE (which go as `g*eps`). No single rescaling of `g` satisfies all of
them, so the mismatch surfaces as residual rather than as a parameter
shift.

That is a useful property: the calibration is robust against a
percent-level error in the assumed cosmology, and it is the chi2 — not the
best-fit parameters — that tells you whether the fiducial spectra are
right. This is why the config uses Planck 2018: the data clearly prefers
it, and under the other template every badly-fitting spectrum involved T.

To measure that dependence, point `cmb.cl_file` at any other CAMB output in
the same `ell D_TT D_EE D_BB D_TE` column order — `load_cmb` indexes by the
`ell` column, so the starting multipole and `lmax` do not have to match, as
long as `lmax` covers the bandpower windows.

## Running

```bash
cd examples/calibration_cross-planck/satp1
./run_satp1.sh single_point   # one likelihood evaluation, seconds
./run_satp1.sh minimize       # best fit
./run_satp1.sh fisher         # best fit + Fisher errors
./run_satp1.sh                # full emcee run, then plots
```

Or directly, in the style of `examples/run_compsep_nopipe.sh`:

```bash
python -u bbpower/compsep_calib.py --config examples/calibration_cross-planck/satp1/config_satp1.yml
python -u bbpower/plotter_calib.py --dir  examples/calibration_cross-planck/satp1/outputs/fiducial
```

Set `OMP_NUM_THREADS=1` (the driver does): the likelihood is small, and BLAS
threading on a 112-core node costs far more than it saves. emcee parallelises
over walkers via `n_threads` in the config.

## Doing another array

Copy `satp1/`, change `data.sacc_file`, and rename the entries under
`map_sets` to the tracer names in that file (`python -c "import sacc;
print(list(sacc.Sacc.load_fits(F).tracers))"`). Nothing else is SATp1-specific.
