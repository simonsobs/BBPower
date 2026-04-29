"""Sampler backends for BBCompSep.

Each function takes a ``Likelihood`` object (and configuration) and runs a
specific inference or evaluation strategy.  They are registered in
``SAMPLERS`` and dispatched by name from ``BBCompSep.run()``.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .likelihood import Likelihood


def _get_emcee_nworkers(nwalkers: int) -> int:
    """Choose a worker count for emcee from the runtime environment.

    Check ``BBPOWER_EMCEE_WORKERS`` then ``SLURM_CPUS_PER_TASK``, falling
    back to ``os.cpu_count()``.  The result is capped so that it never
    exceeds half the walkers (the stretch-move concurrency limit).

    Parameters
    ----------
    nwalkers : int
        Number of emcee walkers (used to compute the useful cap).

    Returns
    -------
    int
        Number of workers to use (always >= 1).
    """
    useful_limit = max(1, (nwalkers + 1) // 2)

    def clip_workers(requested: int) -> int:
        if requested > useful_limit:
            print(
                "Capping emcee workers to "
                f"{useful_limit}; the default stretch move only proposes about "
                "half of the walkers at a time."
            )
        return max(1, min(requested, useful_limit))

    env_value = os.environ.get("BBPOWER_EMCEE_WORKERS")
    if env_value is None:
        env_value = os.environ.get("SLURM_CPUS_PER_TASK")

    if env_value is not None:
        try:
            requested = int(env_value)
        except ValueError:
            print(f"Ignoring invalid worker count {env_value!r}")
        else:
            return clip_workers(requested)

    detected = os.cpu_count() or 1
    return clip_workers(detected)


def _get_emcee_pool_mode() -> str:
    """Choose the emcee parallel backend from the runtime environment.

    Read ``BBPOWER_EMCEE_POOL`` and return one of ``"serial"``,
    ``"thread"``, or ``"process"``.  Default to ``"thread"`` when the
    variable is absent or invalid.

    Returns
    -------
    str
        One of ``"serial"``, ``"thread"``, ``"process"``.
    """
    mode = os.environ.get("BBPOWER_EMCEE_POOL", "thread").strip().lower()
    if mode in {"serial", "thread", "process"}:
        return mode
    print(f"Ignoring invalid pool mode {mode!r}")
    return "thread"


@contextmanager
def _emcee_backend_lock(filename: str) -> Generator[None, None, None]:
    """Protect an emcee HDF5 backend from concurrent writers.

    Acquire an exclusive advisory lock on ``<filename>.lock``.  If
    another process already holds the lock, raise ``RuntimeError``
    immediately instead of blocking.

    Parameters
    ----------
    filename : str
        Path to the HDF5 backend file (the lock file is ``<filename>.lock``).

    Yields
    ------
    None

    Raises
    ------
    RuntimeError
        If another process already holds the lock.
    """
    import fcntl

    lock_path = f"{filename}.lock"
    with open(lock_path, "a+", encoding="ascii") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another BBCompSep emcee run is already using "
                f"{filename}. Wait for that run to finish or use a different "
                "output directory."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _reference_chi2(likelihood: Likelihood) -> tuple[float, int]:
    """Return a best-effort reference chi2 and number of data degrees.

    Older BBPower scripts expect ``emcee.npz`` to include ``chi2`` and
    ``ndof``.  Keep those compatibility fields outside ``BBCompSep`` so the
    stage can continue delegating sampler behavior to this module.
    """
    from scipy.optimize import minimize

    def chi2(par: np.ndarray) -> float:
        return -2 * likelihood.lnprob(par)

    try:
        result = minimize(chi2, likelihood.params.p0, method="Powell")
        par = result.x
    except Exception:
        par = likelihood.params.p0

    ndof = len(getattr(likelihood, "invcov", []))
    return chi2(par), ndof


def run_emcee(likelihood: Likelihood, config: dict, output_dir: str) -> dict:
    """Run an MCMC using emcee.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration (must contain ``nwalkers``, ``n_iters``).
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict
        Keys: ``chain``, ``names``, ``time``.
    """
    import emcee
    from multiprocessing import Pool as ProcessPool
    from multiprocessing.pool import ThreadPool

    fname_temp = os.path.join(output_dir, "emcee.npz.h5")
    with _emcee_backend_lock(fname_temp):
        backend = emcee.backends.HDFBackend(fname_temp)

        nwalkers = config["nwalkers"]
        n_iters = config["n_iters"]
        ndim = len(likelihood.params.p0)
        found_file = os.path.isfile(fname_temp)

        try:
            nchain = len(backend.get_chain())
        except AttributeError:
            found_file = False
        except (OSError, IOError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Existing emcee backend {fname_temp} is unreadable. "
                "This usually means a previous run was interrupted while "
                "writing, or another process touched the same backend. Move "
                "the file aside or use a fresh output directory."
            ) from exc

        if not found_file:
            backend.reset(nwalkers, ndim)
            pos = [
                likelihood.params.p0 + 1.0e-3 * np.random.randn(ndim)
                for _ in range(nwalkers)
            ]
            nsteps_use = n_iters
        else:
            print("Restarting from previous run")
            pos = None
            nsteps_use = max(n_iters - nchain, 0)

        nworkers = _get_emcee_nworkers(nwalkers)
        pool_mode = _get_emcee_pool_mode()
        print(f"Using {nworkers} emcee worker(s) with {pool_mode} pool")

        start = time.time()
        try:
            if nworkers == 1 or pool_mode == "serial":
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, likelihood.lnprob, backend=backend
                )
                if nsteps_use > 0:
                    sampler.run_mcmc(pos, nsteps_use, store=True, progress=False)
            else:
                pool_factory = ThreadPool if pool_mode == "thread" else ProcessPool
                with pool_factory(processes=nworkers) as pool:
                    sampler = emcee.EnsembleSampler(
                        nwalkers,
                        ndim,
                        likelihood.lnprob,
                        pool=pool,
                        backend=backend,
                    )
                    if nsteps_use > 0:
                        sampler.run_mcmc(pos, nsteps_use, store=True, progress=False)
        except OSError as exc:
            raise RuntimeError(
                f"emcee backend {fname_temp} became unreadable during sampling. "
                "This usually happens when two runs write the same output "
                "directory/backend concurrently, or when a previous write left "
                "the HDF5 file corrupted."
            ) from exc
        elapsed = time.time() - start

    chi2, ndof = _reference_chi2(likelihood)
    out_path = os.path.join(output_dir, "emcee.npz")
    np.savez(
        out_path,
        chain=sampler.chain,
        names=likelihood.params.p_free_names,
        time=elapsed,
        chi2=chi2,
        ndof=ndof,
    )
    print(f"Finished sampling {elapsed}")
    return {
        "chain": sampler.chain,
        "names": likelihood.params.p_free_names,
        "time": elapsed,
        "chi2": chi2,
        "ndof": ndof,
    }


def run_polychord(likelihood: Likelihood, config: dict, output_dir: str) -> Any:
    """Run nested sampling using PolyChord.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration (must contain ``nlive``, ``nrepeat``).
    output_dir : str
        Directory for output files.

    Returns
    -------
    object
        PolyChord output object.
    """
    import pypolychord
    from pypolychord.settings import PolyChordSettings
    from pypolychord.priors import UniformPrior, GaussianPrior

    ndim = len(likelihood.params.p0)
    nder = 0

    def pc_likelihood(theta: np.ndarray) -> tuple[float, list[int]]:
        """Evaluate the log-likelihood for PolyChord."""
        return likelihood.lnlike(theta), [0]

    def pc_prior(hypercube: list[float]) -> list[float]:
        """Map the unit hypercube to the physical prior."""
        prior = []
        for h, pr in zip(hypercube, likelihood.params.p_free_priors):
            if pr[1] == "Gaussian":
                prior.append(GaussianPrior(float(pr[2][0]), float(pr[2][1]))(h))
            else:
                prior.append(UniformPrior(float(pr[2][0]), float(pr[2][2]))(h))
        return prior

    def dumper(
        live: np.ndarray,
        dead: np.ndarray,
        logweights: np.ndarray,
        logZ: float,
        logZerr: float,
    ) -> None:
        """Print the last dead point during PolyChord sampling."""
        print("Last dead point:", dead[-1])

    settings = PolyChordSettings(ndim, nder)
    settings.base_dir = os.path.join(output_dir, "polychord")
    settings.file_root = "pch"
    settings.nlive = config["nlive"]
    settings.num_repeats = config["nrepeat"]
    settings.do_clustering = False
    settings.boost_posterior = 10
    settings.nprior = 200
    settings.maximise = True
    settings.read_resume = False
    settings.feedback = 2

    output = pypolychord.run_polychord(
        pc_likelihood, ndim, nder, settings, pc_prior, dumper
    )
    print("Finished sampling")
    return output


def run_minimizer(likelihood: Likelihood, config: dict, output_dir: str) -> np.ndarray:
    """Find the maximum-likelihood point.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.

    Returns
    -------
    np.ndarray
        Best-fit parameter vector.
    """
    from scipy.optimize import minimize

    def chi2(par: np.ndarray) -> float:
        """Return negative-two-log-posterior for the minimizer."""
        return -2 * likelihood.lnprob(par)

    res = minimize(chi2, likelihood.params.p0, method="Powell")
    best_fit = res.x

    chi2_val = -2 * likelihood.lnprob(best_fit)
    out_path = os.path.join(output_dir, "chi2.npz")
    np.savez(
        out_path,
        params=best_fit,
        names=likelihood.params.p_free_names,
        chi2=chi2_val,
        ndof=len(likelihood.invcov),
    )

    print("Best fit:")
    for n, p in zip(likelihood.params.p_free_names, best_fit):
        print(f"{n} = {p:.3E}")
    print(f"Chi2: {chi2_val:.3E}")
    return best_fit


def run_fisher(
    likelihood: Likelihood, config: dict, output_dir: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Fisher matrix at the best-fit point.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.

    Returns
    -------
    tuple
        ``(best_fit_params, fisher_matrix)``.
    """
    import numdifftools as nd
    from scipy.optimize import minimize

    def chi2(par: np.ndarray) -> float:
        """Return negative-two-log-posterior for the minimizer."""
        return -2 * likelihood.lnprob(par)

    res = minimize(chi2, likelihood.params.p0, method="Powell")
    best_fit = res.x

    def lnprobd(p: np.ndarray) -> float:
        """Clamped log-posterior for numerical differentiation."""
        val = likelihood.lnprob(p)
        if val == -np.inf:
            val = -1e100
        return val

    fisher = -nd.Hessian(lnprobd)(best_fit)
    cov = np.linalg.inv(fisher)

    for i, (n, p) in enumerate(zip(likelihood.params.p_free_names, best_fit)):
        print(f"{n} = {p:.3E} +- {np.sqrt(cov[i, i]):.3E}")

    out_path = os.path.join(output_dir, "fisher.npz")
    np.savez(
        out_path, params=best_fit, fisher=fisher, names=likelihood.params.p_free_names
    )
    return best_fit, fisher


def run_singlepoint(likelihood: Likelihood, config: dict, output_dir: str) -> float:
    """Evaluate the chi-squared at the fiducial point.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.

    Returns
    -------
    float
        Chi-squared value.
    """
    chi2 = -2 * likelihood.lnprob(likelihood.params.p0)
    out_path = os.path.join(output_dir, "single_point.npz")
    np.savez(
        out_path,
        chi2=chi2,
        ndof=len(likelihood.invcov),
        names=likelihood.params.p_free_names,
    )
    print("Chi2:", chi2, len(likelihood.invcov))
    return chi2


def run_timing(
    likelihood: Likelihood, config: dict, output_dir: str, n_eval: int = 300
) -> tuple[float, float]:
    """Benchmark likelihood evaluation speed.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.
    n_eval : int
        Number of evaluations to run.

    Returns
    -------
    tuple
        ``(total_time, time_per_eval)``.
    """
    start = time.time()
    for _ in range(n_eval):
        likelihood.lnprob(likelihood.params.p0)
    elapsed = time.time() - start

    out_path = os.path.join(output_dir, "timing.npz")
    np.savez(out_path, timing=elapsed / n_eval, names=likelihood.params.p_free_names)
    print("Total time:", elapsed)
    print("Time per eval:", elapsed / n_eval)
    return elapsed, elapsed / n_eval


def run_predicted_spectra(
    likelihood: Likelihood, compsep: Any, config: dict, output_dir: str
) -> None:
    """Evaluate model at the MAP and save predicted spectra.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    compsep : BBCompSep
        The pipeline stage (needed for model evaluation and SACC I/O).
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.
    """
    import sacc

    at_min = config.get("predict_at_minimum", True)
    save_npz = not config.get("predict_to_sacc", False)

    if at_min:
        from scipy.optimize import minimize

        def chi2(par: np.ndarray) -> float:
            """Return negative-two-log-posterior for the minimizer."""
            return -2 * likelihood.lnprob(par)

        res = minimize(chi2, likelihood.params.p0, method="Powell")
        p = np.array(res.x)
    else:
        p = likelihood.params.p0

    pars = likelihood.params.build_params(p)
    print(pars)
    model_cls = compsep.model(pars)

    if config["bands"] == "all":
        tr_names = sorted(list(compsep.s.tracers.keys()))
    else:
        tr_names = config["bands"]

    if save_npz:
        np.savez(
            os.path.join(output_dir, "cells_model.npz"),
            tracers=tr_names,
            ls=compsep.ell_b,
            dls=model_cls,
        )
        print("Predicted spectra saved")
        return

    s = sacc.Sacc()
    for tn in tr_names:
        t = compsep.s.tracers[tn]
        s.add_tracer(
            "NuMap",
            tn,
            quantity="cmb_polarization",
            spin=2,
            nu=t.nu,
            bandpass=t.bandpass,
            ell=t.ell,
            beam=t.beam,
            nu_unit="GHz",
            map_unit="uK_CMB",
        )
    for b1, b2, p1, p2, m1, m2, ind in compsep._freq_pol_iterator():
        cl = model_cls[:, m1, m2]
        t1 = tr_names[b1]
        t2 = tr_names[b2]
        pol1 = compsep.pols[p1].lower()
        pol2 = compsep.pols[p2].lower()
        cltyp = f"cl_{pol1}{pol2}"
        win = sacc.BandpowerWindow(compsep.bpw_l, compsep.windows[ind].T)
        s.add_ell_cl(cltyp, t1, t2, compsep.ell_b, cl, window=win)
    s.add_covariance(compsep.bbcovar)
    s.save_fits(os.path.join(output_dir, "cells_model.fits"), overwrite=True)
    print("Predicted spectra saved")


SAMPLERS = {
    "emcee": run_emcee,
    "polychord": run_polychord,
    "maximum_likelihood": run_minimizer,
    "fisher": run_fisher,
    "single_point": run_singlepoint,
    "timing": run_timing,
}
