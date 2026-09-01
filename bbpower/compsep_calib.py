#!/usr/bin/env python
"""
BBPower-style multi-frequency calibration likelihood (T, E, B).

Fits, per bandpass i:
    a gain            g_i     (multiplies T, Q, U)
    a pol. efficiency eps_i   (multiplies Q, U only)
    a rotation angle  psi_i   (rotation in the E-B plane)
together with a power-law thermal-dust foreground

    D_ell^{d,XX}_{ij} = A_d^XX (ell/ell0)^{alpha_d^XX} f_i^d f_j^d

where f^d is the modified-black-body SED integrated over the bandpass and
normalised to unity at nu0 in CMB thermodynamic units.

The observed spectra are

    C^obs_{ell,ij} = R_i(psi_i) P_i C^sky_{ell,ij} P_j R_j^T(psi_j)

    P_i = g_i diag(1, eps_i, eps_i)

                 ( 1     0            0        )
    R_i(psi_i) = ( 0   cos 2psi_i   sin 2psi_i )
                 ( 0  -sin 2psi_i   cos 2psi_i )

    C^sky_{ell,ij} = C^CMB_ell + C^dust_{ell,ij}

and the likelihood is the Gaussian

    -2 ln L = [d - m(theta)]^T C^-1 [d - m(theta)]

Conventions follow BBPower / SOOPERCOOL:
  * spectra in the SACC file are D_ell = ell(ell+1)C_ell/2pi (the bandpower
    windows carry the ell(ell+1)/2pi factor, so they map C_ell -> D_ell),
  * beams are already deconvolved (tracer beams are unity),
  * bandpasses are read from the SACC NuMap tracers.

Usage:
    python bbcalib.py --config config.yml
"""
import argparse
import os
import sys
import time

import numpy as np
import sacc
import yaml
from scipy.linalg import cho_factor, cho_solve

# --------------------------------------------------------------- constants
H_OVER_K = 0.0479924466            # GHz^-1 K
T_CMB = 2.72548                    # K
X_CMB = H_OVER_K / T_CMB           # GHz^-1, so x = X_CMB * nu

POLS = ('T', 'E', 'B')
POL_IDX = {'T': 0, 'E': 1, 'B': 2}

SACC_DT = {('T', 'T'): 'cl_00', ('T', 'E'): 'cl_0e', ('T', 'B'): 'cl_0b',
           ('E', 'T'): 'cl_e0', ('E', 'E'): 'cl_ee', ('E', 'B'): 'cl_eb',
           ('B', 'T'): 'cl_b0', ('B', 'E'): 'cl_be', ('B', 'B'): 'cl_bb'}
DT_POLS = {v: k for k, v in SACC_DT.items()}

# Spectra that are the exact transpose of another one. For an auto-pair
# (same tracer twice) these carry *identical* numbers, so keeping both makes
# the covariance singular. They are dropped automatically in that case.
TRANSPOSE_OF = {'cl_e0': 'cl_0e', 'cl_b0': 'cl_0b', 'cl_be': 'cl_eb'}

# Sky spectra that the dust model can populate.
DUST_SPECTRA = ('TT', 'TE', 'EE', 'BB', 'EB', 'TB')


# ----------------------------------------------------------------- helpers
def _yaml_loader(fname):
    """YAML loader supporting the BBPower `!path [a, b]` join constructor."""
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(fname, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def _tofloat(x):
    """float(), understanding 'inf' / '-inf' strings from YAML."""
    if isinstance(x, str):
        return float(x)
    return float(x)


def sed_cmb_rj(nu):
    """CMB blackbody derivative, i.e. the K_CMB -> K_RJ conversion factor."""
    x = X_CMB * nu
    ex = np.exp(x)
    return ex * (x / (ex - 1.))**2


def sed_dust_rj(nu, beta_d, temp_d, nu0):
    """Modified black body in K_RJ, normalised to 1 at nu0."""
    return ((nu / nu0)**(1. + beta_d) *
            (np.expm1(H_OVER_K * nu0 / temp_d) /
             np.expm1(H_OVER_K * nu / temp_d)))


class Bandpass(object):
    """Bandpass of one map set, read from a SACC NuMap tracer."""

    def __init__(self, nu, bandpass):
        self.nu = np.atleast_1d(np.asarray(nu, dtype=float))
        dnu = np.zeros_like(self.nu)
        if len(self.nu) > 2:
            dnu[1:-1] = 0.5 * (self.nu[2:] - self.nu[:-2])
            dnu[0] = self.nu[1] - self.nu[0]
            dnu[-1] = self.nu[-1] - self.nu[-2]
        elif len(self.nu) == 2:
            dnu[:] = self.nu[1] - self.nu[0]
        else:
            dnu[:] = 1.
        self.w = np.asarray(bandpass, dtype=float) * dnu * self.nu**2
        # Normalisation turning a K_RJ SED into a K_CMB one.
        self.cmb_norm = np.sum(sed_cmb_rj(self.nu) * self.w)
        self.nu_mean = (np.sum(sed_cmb_rj(self.nu) * self.w * self.nu) /
                        self.cmb_norm)

    def convolve_dust(self, beta_d, temp_d, nu0):
        """Bandpass-integrated dust SED in K_CMB, unity at nu0."""
        sed = sed_dust_rj(self.nu, beta_d, temp_d, nu0)
        return sed_cmb_rj(nu0) * np.sum(sed * self.w) / self.cmb_norm


# ------------------------------------------------------------- parameters
class ParameterManager(object):
    """Priors and free/fixed bookkeeping, BBPower-style.

    A parameter is declared as one of
        ['fixed',    [value]]
        ['tophat',   [low, start, high]]
        ['Gaussian', [mean, sigma]]
    """

    def __init__(self):
        self.free_names = []
        self.free_priors = []
        self.fixed = {}
        self.p0 = []

    def add(self, name, spec):
        if spec is None:
            raise ValueError("No prior given for parameter '%s'" % name)
        kind = str(spec[0]).lower()
        vals = [_tofloat(v) for v in spec[1]]
        if kind == 'fixed':
            self.fixed[name] = vals[0]
            return
        if name in self.free_names:
            raise KeyError("Duplicate parameter name '%s'" % name)
        if kind == 'tophat':
            if not len(vals) == 3:
                raise ValueError("tophat prior on '%s' needs "
                                 "[low, start, high]" % name)
            start = vals[1]
        elif kind == 'gaussian':
            if not len(vals) == 2:
                raise ValueError("Gaussian prior on '%s' needs "
                                 "[mean, sigma]" % name)
            start = vals[0]
        else:
            raise ValueError("Unknown prior type '%s' for '%s'" % (kind, name))
        self.free_names.append(name)
        self.free_priors.append((kind, vals))
        self.p0.append(start)

    def finalize(self):
        self.p0 = np.array(self.p0, dtype=float)
        self.ndim = len(self.p0)
        # Characteristic scale of each parameter, used to precondition the
        # minimiser and to seed the emcee walkers. Parameters here span
        # ~1e-2 (angles in rad-equivalent) to ~1e3 (dust TT amplitude), so
        # an unscaled finite-difference gradient is meaningless.
        scales = []
        for kind, pr in self.free_priors:
            if kind == 'gaussian':
                scales.append(abs(pr[1]))
            else:
                lo, hi = pr[0], pr[2]
                if np.isfinite(lo) and np.isfinite(hi):
                    s = 0.1 * (hi - lo)
                else:
                    s = max(abs(pr[1]), 1.) * 0.1
                scales.append(s)
        self.scales = np.array(scales, dtype=float)

    def build(self, vec):
        params = dict(self.fixed)
        params.update(zip(self.free_names, vec))
        return params

    def lnprior(self, vec):
        lnp = 0.
        for v, (kind, pr) in zip(vec, self.free_priors):
            if kind == 'gaussian':
                lnp += -0.5 * ((v - pr[0]) / pr[1])**2
            else:
                if not (pr[0] <= v <= pr[2]):
                    return -np.inf
        return lnp


# ------------------------------------------------------------- likelihood
class CalibLikelihood(object):

    def __init__(self, config_file):
        self.config_file = config_file
        cfg = _yaml_loader(config_file)
        self.cfg = cfg
        self.out_dir = cfg['output_dir']
        os.makedirs(self.out_dir, exist_ok=True)

        self.parse_map_sets()
        self.parse_sacc()
        self.load_cmb()
        self.setup_params()
        self.setup_dust_cache()

    # ---------------------------------------------------------- map sets
    def parse_map_sets(self):
        ms = self.cfg['map_sets']
        self.tracer_names = list(ms.keys())
        self.n_tr = len(self.tracer_names)
        self.ms_cfg = ms
        self.alias = {}
        self.groups = {}
        for name, d in ms.items():
            self.alias[name] = d.get('alias', name)
            self.groups[name] = d.get('group', 'other')

    # ------------------------------------------------------------- data
    def parse_sacc(self):
        s = sacc.Sacc.load_fits(self.cfg['data']['sacc_file'])

        missing = [t for t in self.tracer_names if t not in s.tracers]
        if missing:
            raise KeyError("Tracers not in SACC file: %s\nAvailable: %s"
                           % (missing, list(s.tracers)))
        self.s = s

        # Bandpasses
        self.bpss = []
        for t in self.tracer_names:
            tr = s.tracers[t]
            self.bpss.append(Bandpass(tr.nu, tr.bandpass))

        l_min = float(self.cfg['l_min'])
        l_max = float(self.cfg['l_max'])
        sel_cfg = self.cfg['spectra_selection']

        # Which sky-spectrum types are requested for each pair class.
        pair_types = {k: [str(v).upper() for v in vv]
                      for k, vv in sel_cfg['pair_types'].items()}
        include_cross_within = sel_cfg.get('include_cross_within_group', {})

        self.specs = []
        data_idx = []
        for i1, t1 in enumerate(self.tracer_names):
            for i2 in range(i1, self.n_tr):
                t2 = self.tracer_names[i2]
                g1, g2 = self.groups[t1], self.groups[t2]
                key = '_x_'.join(sorted([g1, g2]))
                if key not in pair_types:
                    continue
                if g1 == g2 and t1 != t2:
                    if not include_cross_within.get(g1, True):
                        continue
                for xx in pair_types[key]:
                    p1, p2 = xx[0], xx[1]
                    dt = SACC_DT[(p1, p2)]
                    # An auto-pair's transposed spectrum is the same number.
                    if t1 == t2 and dt in TRANSPOSE_OF:
                        continue
                    try:
                        ind = np.array(s.indices(dt, (t1, t2)))
                    except Exception:
                        ind = np.array([], dtype=int)
                    if len(ind) == 0:
                        print("  [warn] %s not found for (%s, %s), skipping"
                              % (dt, t1, t2))
                        continue
                    ells = np.array([s.data[k]['ell'] for k in ind])
                    order = np.argsort(ells)
                    ind, ells = ind[order], ells[order]
                    keep = (ells >= l_min) & (ells <= l_max)
                    ind, ells = ind[keep], ells[keep]
                    if len(ind) == 0:
                        continue
                    win = s.get_bandpower_windows(ind)
                    self.specs.append({
                        'i1': i1, 'i2': i2, 't1': t1, 't2': t2, 'dt': dt,
                        'xx': xx, 'p1': POL_IDX[p1], 'p2': POL_IDX[p2],
                        'ell': ells, 'n_bpw': len(ind),
                        'win_values': win.values, 'win_weight': win.weight,
                        'slice': slice(len(data_idx),
                                       len(data_idx) + len(ind))})
                    data_idx.extend(ind)

        if not self.specs:
            raise RuntimeError("No spectra selected -- check "
                               "`spectra_selection` and `l_min`/`l_max`.")
        self.data_idx = np.array(data_idx)
        self.n_data = len(self.data_idx)
        self.data_vec = np.asarray(s.mean)[self.data_idx]

        # Common model ell grid (avoid ell<2 where ell(ell+1)/2pi is singular)
        lmax_win = int(max(sp['win_values'].max() for sp in self.specs))
        self.ls = np.arange(2, lmax_win + 1)
        self.dl2cl = 2. * np.pi / (self.ls * (self.ls + 1.))

        # Windows restricted to that grid: (n_bpw, n_ell), maps C_ell -> D_ell
        for sp in self.specs:
            v = sp['win_values'].astype(int)
            m = v >= 2
            w = np.zeros((sp['n_bpw'], len(self.ls)))
            cols = v[m] - 2
            w[:, cols] = sp['win_weight'][m, :].T
            sp['W'] = w
            del sp['win_values'], sp['win_weight']

        # Covariance
        cov = np.asarray(self.s.covariance.covmat)[
            np.ix_(self.data_idx, self.data_idx)]
        self.cov = cov
        ev = np.linalg.eigvalsh(cov)
        print("Covariance: N = %d, eig_min = %.4e, eig_max = %.4e"
              % (self.n_data, ev.min(), ev.max()))
        if ev.min() <= 0:
            raise RuntimeError(
                "Selected covariance is not positive definite (min eigenvalue "
                "%.4e). Raise l_min, or check for duplicated spectra."
                % ev.min())
        self.cho = cho_factor(cov, lower=True)

        print("Selected %d spectra / %d data points"
              % (len(self.specs), self.n_data))
        for sp in self.specs:
            print("   %-6s %-28s x %-28s  %2d bpws (%.0f-%.0f)"
                  % (sp['xx'], self.alias[sp['t1']], self.alias[sp['t2']],
                     sp['n_bpw'], sp['ell'][0], sp['ell'][-1]))

        # Group spectra by tracer pair so the model is built once per pair.
        self.pairs = {}
        for sp in self.specs:
            self.pairs.setdefault((sp['i1'], sp['i2']), []).append(sp)

    # -------------------------------------------------------------- CMB
    def load_cmb(self):
        """Load fiducial lensed CMB D_ell (columns: ell TT EE BB TE)."""
        d = np.loadtxt(self.cfg['cmb']['cl_file'])
        lt = d[:, 0].astype(int)
        table = np.zeros((4, self.ls.max() + 1))
        m = (lt >= 2) & (lt <= self.ls.max())
        for k in range(4):
            table[k, lt[m]] = d[m, k + 1]
        tt, ee, bb, te = (table[0][self.ls], table[1][self.ls],
                          table[2][self.ls], table[3][self.ls])
        self.cmb_tt, self.cmb_ee, self.cmb_te = tt, ee, te
        self.cmb_bb = bb
        print("CMB template: %s (lmax = %d)"
              % (self.cfg['cmb']['cl_file'], lt.max()))

    def cmb_matrix(self, params):
        """CMB sky D_ell as a (n_ell, 3, 3) matrix."""
        a_lens = params.get('A_lens', 1.)
        c = np.zeros((len(self.ls), 3, 3))
        c[:, 0, 0] = self.cmb_tt
        c[:, 0, 1] = c[:, 1, 0] = self.cmb_te
        c[:, 1, 1] = self.cmb_ee
        c[:, 2, 2] = a_lens * self.cmb_bb
        return c

    # ------------------------------------------------------------- dust
    def setup_dust_cache(self):
        self._dust_key = None
        self._dust_f = None

    def dust_f(self, beta_d, temp_d):
        key = (beta_d, temp_d)
        if key != self._dust_key:
            nu0 = self.dust_nu0
            self._dust_f = np.array(
                [bp.convolve_dust(beta_d, temp_d, nu0) for bp in self.bpss])
            self._dust_key = key
        return self._dust_f

    def dust_matrix(self, params):
        """Dust sky D_ell shape as a (n_ell, 3, 3) matrix (no f_i f_j yet)."""
        x = self.ls / self.dust_l0
        c = np.zeros((len(self.ls), 3, 3))
        for xx in self.dust_active:
            amp = params['A_d_%s' % xx]
            alpha = params['alpha_d_%s' % xx]
            shape = amp * x**alpha
            a, b = POL_IDX[xx[0]], POL_IDX[xx[1]]
            c[:, a, b] = shape
            if a != b:
                c[:, b, a] = shape
        return c

    # ------------------------------------------------------- parameters
    def setup_params(self):
        pm = ParameterManager()

        cmb_cfg = self.cfg.get('cmb', {})
        if 'A_lens' in cmb_cfg:
            pm.add('A_lens', cmb_cfg['A_lens'])

        # Per-map-set calibration parameters
        self.par_names = []
        for name in self.tracer_names:
            a = self.alias[name]
            d = self.ms_cfg[name]
            names = {}
            for kind, default in (('gain', ['fixed', [1.0]]),
                                  ('poleff', ['fixed', [1.0]]),
                                  ('angle', ['fixed', [0.0]])):
                pname = '%s_%s' % (kind, a)
                pm.add(pname, d.get(kind, default))
                names[kind] = pname
            self.par_names.append(names)

        # Dust
        dcfg = self.cfg['dust']
        self.dust_nu0 = float(dcfg.get('nu0', 353.))
        self.dust_l0 = float(dcfg.get('ell0', 80.))
        pm.add('beta_d', dcfg.get('beta_d', ['fixed', [1.54]]))
        pm.add('temp_d', dcfg.get('temp_d', ['fixed', [20.]]))
        self.dust_active = []
        for xx, spec in dcfg.get('spectra', {}).items():
            xx = xx.upper()
            if xx not in DUST_SPECTRA:
                raise ValueError("Unknown dust spectrum '%s'" % xx)
            pm.add('A_d_%s' % xx, spec['amp'])
            pm.add('alpha_d_%s' % xx, spec['alpha'])
            self.dust_active.append(xx)

        pm.finalize()
        self.pm = pm
        print("Free parameters (%d): %s" % (pm.ndim, ', '.join(pm.free_names)))
        if pm.fixed:
            print("Fixed parameters: %s"
                  % ', '.join('%s=%g' % (k, v)
                              for k, v in sorted(pm.fixed.items())))

    # ------------------------------------------------------------ model
    def instrument_matrices(self, params):
        """M_i = R_i(psi_i) P_i, shape (n_tracers, 3, 3)."""
        M = np.zeros((self.n_tr, 3, 3))
        for i, names in enumerate(self.par_names):
            g = params[names['gain']]
            e = params[names['poleff']]
            psi = np.radians(params[names['angle']])
            c, s = np.cos(2 * psi), np.sin(2 * psi)
            R = np.array([[1., 0., 0.],
                          [0., c, s],
                          [0., -s, c]])
            P = g * np.diag([1., e, e])
            M[i] = R @ P
        return M

    def model_vector(self, params):
        M = self.instrument_matrices(params)
        f = self.dust_f(params['beta_d'], params['temp_d'])
        cmb = self.cmb_matrix(params)
        dust = self.dust_matrix(params)

        out = np.empty(self.n_data)
        for (i1, i2), specs in self.pairs.items():
            sky = cmb + f[i1] * f[i2] * dust
            obs = np.einsum('ab,lbc,dc->lad', M[i1], sky, M[i2],
                            optimize=True)
            for sp in specs:
                cl = obs[:, sp['p1'], sp['p2']] * self.dl2cl
                out[sp['slice']] = sp['W'] @ cl
        return out

    # ------------------------------------------------------- likelihood
    def chi2(self, vec):
        params = self.pm.build(vec)
        r = self.data_vec - self.model_vector(params)
        return float(r @ cho_solve(self.cho, r))

    def lnprob(self, vec):
        lp = self.pm.lnprior(vec)
        if not np.isfinite(lp):
            return -np.inf
        try:
            c2 = self.chi2(vec)
        except (ValueError, FloatingPointError):
            return -np.inf
        if not np.isfinite(c2):
            return -np.inf
        return lp - 0.5 * c2

    # ----------------------------------------------------------- output
    def save_best_fit(self, vec, tag='bestfit'):
        params = self.pm.build(vec)
        model = self.model_vector(params)
        np.savez(os.path.join(self.out_dir, '%s.npz' % tag),
                 free_names=np.array(self.pm.free_names),
                 free_values=np.asarray(vec, dtype=float),
                 fixed_names=np.array(list(self.pm.fixed.keys())),
                 fixed_values=np.array(list(self.pm.fixed.values())),
                 data=self.data_vec, model=model,
                 chi2=self.chi2(vec), n_data=self.n_data,
                 ndim=self.pm.ndim,
                 spec_labels=np.array(
                     ['%s %s x %s' % (sp['xx'], self.alias[sp['t1']],
                                      self.alias[sp['t2']])
                      for sp in self.specs]),
                 spec_starts=np.array([sp['slice'].start
                                       for sp in self.specs]),
                 spec_ells=np.concatenate([sp['ell'] for sp in self.specs]),
                 cov_diag=np.diag(self.cov))
        return params, model


# ------------------------------------------------------------- samplers
def run_single_point(lk):
    p0 = lk.pm.p0
    t0 = time.time()
    c2 = lk.chi2(p0)
    dt = time.time() - t0
    ndof = lk.n_data - lk.pm.ndim
    print("\nchi2 at the prior centre = %.2f for %d data points "
          "(ndof = %d)  ->  chi2/ndof = %.3f"
          % (c2, lk.n_data, ndof, c2 / ndof))
    print("one likelihood call: %.1f ms" % (1e3 * dt))
    lk.save_best_fit(p0, tag='single_point')


def run_timing(lk, n=200):
    p0 = lk.pm.p0
    lk.chi2(p0)
    t0 = time.time()
    for _ in range(n):
        lk.chi2(p0)
    print("mean likelihood evaluation: %.2f ms"
          % (1e3 * (time.time() - t0) / n))


def run_minimize(lk, verbose=True):
    """Best fit, minimised in units of each parameter's prior width."""
    from scipy.optimize import minimize

    sc = lk.pm.scales
    x0 = lk.pm.p0 / sc
    bounds = []
    for (kind, pr), s in zip(lk.pm.free_priors, sc):
        if kind == 'tophat':
            bounds.append((pr[0] / s, pr[2] / s))
        else:
            bounds.append((None, None))

    def nll(x):
        v = x * sc
        lp = lk.pm.lnprior(v)
        if not np.isfinite(lp):
            return 1e30
        return lk.chi2(v) - 2. * lp

    t0 = time.time()
    res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 5000, 'maxfun': 100000,
                            'ftol': 1e-12, 'gtol': 1e-10, 'eps': 1e-5})
    # Polish with a derivative-free step; the tophat walls make the objective
    # non-smooth at the edges and L-BFGS-B can stall there.
    res2 = minimize(nll, res.x, method='Nelder-Mead',
                    options={'maxiter': 20000, 'maxfev': 40000,
                             'xatol': 1e-8, 'fatol': 1e-8})
    best = res2.x if res2.fun < res.fun else res.x
    p_best = best * sc

    if verbose:
        ndof = lk.n_data - lk.pm.ndim
        c2 = lk.chi2(p_best)
        print("\nminimisation: L-BFGS-B %s + Nelder-Mead, %.1f s, %d evals"
              % ('ok' if res.success else 'stalled',
                 time.time() - t0, res.nfev + res2.nfev))
        print("chi2 = %.2f  ndof = %d  chi2/ndof = %.3f" % (c2, ndof, c2 / ndof))
        print("\nbest fit:")
        for n_, v in zip(lk.pm.free_names, p_best):
            print("   %-24s = %12.6f" % (n_, v))
    lk.save_best_fit(p_best, tag='minimize')
    return p_best


def run_fisher(lk, centre=None):
    p0 = lk.pm.p0 if centre is None else np.asarray(centre)
    n = lk.pm.ndim
    steps = np.array([max(1e-4, 1e-3 * abs(v)) for v in p0])
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = steps[i]
            ej = np.zeros(n); ej[j] = steps[j]
            H[i, j] = H[j, i] = (
                lk.chi2(p0 + ei + ej) - lk.chi2(p0 + ei - ej)
                - lk.chi2(p0 - ei + ej) + lk.chi2(p0 - ei - ej)
            ) / (4. * steps[i] * steps[j])
    F = 0.5 * H
    C = np.linalg.inv(F)
    sig = np.sqrt(np.diag(C))
    print("\nFisher forecast around the evaluation point:")
    for n_, v, s in zip(lk.pm.free_names, p0, sig):
        print("   %-24s = %12.6f +/- %.6f" % (n_, v, s))
    np.savez(os.path.join(lk.out_dir, 'fisher.npz'),
             names=np.array(lk.pm.free_names), centre=p0,
             fisher=F, cov=C, sigma=sig)
    return C


def run_emcee(lk):
    import emcee
    cfg = lk.cfg
    nwalkers = int(cfg.get('nwalkers', 4 * lk.pm.ndim))
    nsteps = int(cfg.get('n_iters', 5000))
    nthreads = int(cfg.get('n_threads', 1))
    ndim = lk.pm.ndim

    # Start from a small ball around the prior centre, inside the priors.
    rng = np.random.default_rng(int(cfg.get('seed', 1234)))
    centre = np.asarray(cfg.get('_start', lk.pm.p0), dtype=float)
    scatter = 0.02 * lk.pm.scales
    p0 = np.empty((nwalkers, ndim))
    for k in range(nwalkers):
        while True:
            trial = centre + scatter * rng.normal(size=ndim)
            if np.isfinite(lk.pm.lnprior(trial)) \
                    and np.isfinite(lk.lnprob(trial)):
                p0[k] = trial
                break

    chain_file = os.path.join(lk.out_dir, 'chain.h5')
    backend = None
    try:
        backend = emcee.backends.HDFBackend(chain_file)
        if not bool(cfg.get('resume', False)):
            backend.reset(nwalkers, ndim)
    except Exception as e:
        print("  [warn] HDF backend unavailable (%s); chain kept in memory"
              % e)
        backend = None

    if nthreads > 1:
        from multiprocessing import Pool
        pool = Pool(nthreads)
    else:
        pool = None
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lk.lnprob,
                                    pool=pool, backend=backend)
    start = p0
    if backend is not None and bool(cfg.get('resume', False)) \
            and backend.iteration > 0:
        start = None
        print("resuming from iteration %d" % backend.iteration)
    t0 = time.time()
    sampler.run_mcmc(start, nsteps, progress=True)
    if pool is not None:
        pool.close()
    print("emcee: %d steps x %d walkers in %.1f s"
          % (nsteps, nwalkers, time.time() - t0))

    try:
        tau = sampler.get_autocorr_time(tol=0)
        print("autocorrelation times: %s" % np.array2string(tau, precision=1))
        burn = int(3 * np.nanmax(tau))
        thin = max(1, int(0.5 * np.nanmin(tau)))
    except Exception:
        burn, thin = nsteps // 4, 1
    burn = min(burn, nsteps // 2)
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    lnp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    np.savez(os.path.join(lk.out_dir, 'chain.npz'),
             names=np.array(lk.pm.free_names), chain=chain, lnprob=lnp,
             burn=burn, thin=thin,
             acceptance=np.mean(sampler.acceptance_fraction))
    print("\nposterior (mean +/- std, %d samples):" % len(chain))
    for i, n_ in enumerate(lk.pm.free_names):
        q = np.percentile(chain[:, i], [16, 50, 84])
        print("   %-24s = %10.5f  +%.5f -%.5f   (mean %10.5f +/- %.5f)"
              % (n_, q[1], q[2] - q[1], q[1] - q[0],
                 chain[:, i].mean(), chain[:, i].std()))
    best = chain[np.argmax(lnp)]
    lk.save_best_fit(best, tag='bestfit')
    ndof = lk.n_data - lk.pm.ndim
    print("\nchi2 at max posterior = %.2f  ndof = %d  chi2/ndof = %.3f"
          % (lk.chi2(best), ndof, lk.chi2(best) / ndof))
    return chain


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--sampler', default=None,
                    help="override the sampler in the config")
    args = ap.parse_args()

    lk = CalibLikelihood(args.config)
    import shutil
    shutil.copy(args.config, os.path.join(lk.out_dir, 'config_copy.yml'))

    sampler = args.sampler or lk.cfg.get('sampler', 'emcee')
    print("\n=== sampler: %s ===" % sampler)
    if sampler == 'single_point':
        run_single_point(lk)
    elif sampler == 'timing':
        run_timing(lk)
    elif sampler == 'minimize':
        run_minimize(lk)
    elif sampler == 'fisher':
        run_fisher(lk, centre=run_minimize(lk))
    elif sampler == 'emcee':
        if lk.cfg.get('start_from_bestfit', True):
            print("locating the best fit to seed the walkers...")
            lk.cfg['_start'] = run_minimize(lk)
        run_emcee(lk)
    else:
        raise ValueError("Unknown sampler '%s'" % sampler)


if __name__ == '__main__':
    sys.exit(main())
