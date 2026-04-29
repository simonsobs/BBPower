"""Tests for bbpower.samplers — sampler dispatch and registry."""

from __future__ import annotations

import fcntl
import sys
import types

import numpy as np
import pytest

import bbpower.samplers as samplers
from bbpower.samplers import (
    SAMPLERS,
    run_fisher,
    run_minimizer,
    run_singlepoint,
    run_timing,
)


class TestSamplersDict:
    """Tests for the SAMPLERS registry."""

    def test_keys(self):
        """All expected sampler backends are registered."""
        expected = {
            "emcee",
            "polychord",
            "maximum_likelihood",
            "fisher",
            "single_point",
            "timing",
        }
        assert expected == set(SAMPLERS.keys())

    def test_callable(self):
        """All registered values are callable."""
        for name, func in SAMPLERS.items():
            assert callable(func), f"{name} is not callable"

    def test_unknown_not_in_dict(self):
        """Nonexistent sampler is not registered."""
        assert "nonexistent" not in SAMPLERS


class TestRunSinglepoint:
    """Test the single-point chi-squared evaluation."""

    def test_writes_output(self, tmp_path):
        """run_singlepoint writes an npz file with chi2 and ndof."""

        class MockParams:
            p0 = np.array([0.0])
            p_free_names = ["r"]

            def lnprior(self, par):
                return 0.0

            def build_params(self, par):
                return {"r": par[0]}

        class MockLikelihood:
            params = MockParams()
            invcov = np.eye(3)

            def lnprob(self, par):
                return -5.0

        lik = MockLikelihood()
        chi2 = run_singlepoint(lik, {}, str(tmp_path))
        assert chi2 == pytest.approx(10.0)  # -2 * (-5.0)
        out = np.load(tmp_path / "single_point.npz")
        assert "chi2" in out
        assert "ndof" in out


class TestRunTiming:
    """Test the timing benchmark."""

    def test_returns_positive_time(self, tmp_path):
        """run_timing returns positive elapsed times."""

        class MockParams:
            p0 = np.array([0.0])
            p_free_names = ["r"]

            def lnprior(self, par):
                return 0.0

            def build_params(self, par):
                return {"r": par[0]}

        class MockLikelihood:
            params = MockParams()

            def lnprob(self, par):
                return -1.0

        lik = MockLikelihood()
        total, per_eval = run_timing(lik, {}, str(tmp_path), n_eval=5)
        assert total > 0
        assert per_eval > 0
        out = np.load(tmp_path / "timing.npz")
        assert "timing" in out


class TestRunEmcee:
    """Test the emcee backend wiring."""

    def test_backend_lock_raises_for_concurrent_writer(self, monkeypatch, tmp_path):
        """A second writer gets a clear error before touching the HDF backend."""

        def fail_lock(fd, flags):
            if flags & fcntl.LOCK_UN:
                return None
            raise BlockingIOError

        monkeypatch.setattr(fcntl, "flock", fail_lock)

        with pytest.raises(RuntimeError, match="already using"):
            with samplers._emcee_backend_lock(str(tmp_path / "emcee.npz.h5")):
                pass

    def test_worker_count_uses_env_and_caps_to_walkers(self, monkeypatch):
        """Worker count respects the useful parallel limit."""
        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "32")
        assert samplers._get_emcee_nworkers(40) == 20
        assert samplers._get_emcee_nworkers(8) == 4

        monkeypatch.delenv("BBPOWER_EMCEE_WORKERS")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
        assert samplers._get_emcee_nworkers(40) == 6

    def test_default_pool_mode_is_thread(self, monkeypatch):
        """Thread pools are the safe default for BBCompSep likelihoods."""
        monkeypatch.delenv("BBPOWER_EMCEE_POOL", raising=False)
        assert samplers._get_emcee_pool_mode() == "thread"

        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "process")
        assert samplers._get_emcee_pool_mode() == "process"

    def test_passes_thread_pool_to_ensemble_sampler(self, monkeypatch, tmp_path):
        """run_emcee wires the thread pool into emcee."""

        class MockParams:
            p0 = np.array([0.0, 1.0])
            p_free_names = ["r", "A_lens"]

        class MockLikelihood:
            params = MockParams()

            def lnprob(self, par):
                return -1.0

        calls: dict[str, object] = {}

        class FakeBackend:
            def __init__(self, path):
                calls["backend_path"] = path

            def get_chain(self):
                raise AttributeError

            def reset(self, nwalkers, ndim):
                calls["reset"] = (nwalkers, ndim)

        class FakeSampler:
            def __init__(
                self,
                nwalkers,
                ndim,
                log_prob_fn,
                pool=None,
                backend=None,
                **kwargs,
            ):
                calls["pool"] = pool
                calls["backend"] = backend
                calls["kwargs"] = kwargs
                self.chain = np.zeros((nwalkers, 1, ndim))

            def run_mcmc(self, pos, nsteps, store=True, progress=False):
                calls["run_mcmc"] = (len(pos), nsteps, store, progress)

        class FakePool:
            def __init__(self, processes=None):
                calls["processes"] = processes

            def __enter__(self):
                calls["pool_obj"] = self
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_emcee = types.SimpleNamespace(
            backends=types.SimpleNamespace(HDFBackend=FakeBackend),
            EnsembleSampler=FakeSampler,
        )

        monkeypatch.setitem(sys.modules, "emcee", fake_emcee)
        monkeypatch.setattr("multiprocessing.pool.ThreadPool", FakePool)
        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "3")
        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "thread")

        out = samplers.run_emcee(
            MockLikelihood(),
            {"nwalkers": 4, "n_iters": 2},
            str(tmp_path),
        )

        assert calls["processes"] == 2
        assert calls["pool"] is calls["pool_obj"]
        assert calls["reset"] == (4, 2)
        assert calls["run_mcmc"] == (4, 2, True, False)
        assert out["chain"].shape == (4, 1, 2)

        saved = np.load(tmp_path / "emcee.npz")
        assert saved["chain"].shape == (4, 1, 2)
        assert saved["names"].tolist() == ["r", "A_lens"]

    def test_thread_pool_handles_local_likelihood(self, monkeypatch, tmp_path):
        """Thread parallelism works with local likelihood objects."""

        class MockParams:
            p0 = np.array([0.1, -0.2])
            p_free_names = ["r", "A_lens"]

        class LocalLikelihood:
            params = MockParams()

            def lnprob(self, par):
                return -0.5 * np.dot(par, par)

        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "2")
        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "thread")

        out = samplers.run_emcee(
            LocalLikelihood(),
            {"nwalkers": 6, "n_iters": 3},
            str(tmp_path),
        )

        assert out["chain"].shape == (6, 3, 2)


class TestGetEmceeNworkers:
    """Edge-case tests for _get_emcee_nworkers."""

    def test_invalid_env_falls_back_to_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-integer env value is ignored and CPU count is used."""
        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "not_a_number")
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        result = samplers._get_emcee_nworkers(100)
        assert result >= 1

    def test_no_env_falls_back_to_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no env vars set, fall back to os.cpu_count()."""
        monkeypatch.delenv("BBPOWER_EMCEE_WORKERS", raising=False)
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        result = samplers._get_emcee_nworkers(100)
        assert result >= 1

    def test_caps_at_half_walkers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Worker count never exceeds (nwalkers + 1) // 2."""
        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "100")
        # With 4 walkers, cap is (4+1)//2 = 2
        assert samplers._get_emcee_nworkers(4) == 2

    def test_minimum_one_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Worker count is at least 1 even with 1 walker."""
        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "1")
        assert samplers._get_emcee_nworkers(1) == 1


class TestGetEmceePoolMode:
    """Edge-case tests for _get_emcee_pool_mode."""

    def test_invalid_mode_falls_back_to_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid pool mode string defaults to 'thread'."""
        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "mpi")
        assert samplers._get_emcee_pool_mode() == "thread"

    def test_serial_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'serial' is a valid pool mode."""
        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "serial")
        assert samplers._get_emcee_pool_mode() == "serial"

    def test_whitespace_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Leading/trailing whitespace in the env value is stripped."""
        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "  process  ")
        assert samplers._get_emcee_pool_mode() == "process"


class TestRunMinimizer:
    """Test the maximum-likelihood minimizer."""

    def test_finds_minimum_of_quadratic(self, tmp_path: str) -> None:
        """Minimize a simple quadratic likelihood and verify output."""

        class MockParams:
            p0 = np.array([5.0, -3.0])
            p_free_names = ["a", "b"]

            def lnprior(self, par: np.ndarray) -> float:
                return 0.0

            def build_params(self, par: np.ndarray) -> dict:
                return {"a": par[0], "b": par[1]}

        class QuadraticLikelihood:
            params = MockParams()
            invcov = np.eye(2)

            def lnprob(self, par: np.ndarray) -> float:
                # Maximum at (0, 0)
                return -0.5 * np.dot(par, par)

        best_fit = run_minimizer(QuadraticLikelihood(), {}, str(tmp_path))
        np.testing.assert_allclose(best_fit, [0.0, 0.0], atol=1e-4)

        out = np.load(tmp_path / "chi2.npz")
        assert "params" in out
        assert "names" in out
        assert "chi2" in out
        assert "ndof" in out
        assert out["chi2"] == pytest.approx(0.0, abs=1e-4)

    def test_output_names_match(self, tmp_path: str) -> None:
        """Saved parameter names match the free parameter list."""

        class MockParams:
            p0 = np.array([1.0])
            p_free_names = ["r_tensor"]

            def lnprior(self, par: np.ndarray) -> float:
                return 0.0

            def build_params(self, par: np.ndarray) -> dict:
                return {"r_tensor": par[0]}

        class SimpleLikelihood:
            params = MockParams()
            invcov = np.eye(1)

            def lnprob(self, par: np.ndarray) -> float:
                return -0.5 * par[0] ** 2

        run_minimizer(SimpleLikelihood(), {}, str(tmp_path))
        out = np.load(tmp_path / "chi2.npz")
        assert out["names"].tolist() == ["r_tensor"]


class TestRunFisher:
    """Test the Fisher matrix computation."""

    def test_fisher_of_gaussian(self, tmp_path: str) -> None:
        """Fisher matrix of a Gaussian likelihood matches the precision matrix."""

        sigma = np.array([2.0, 0.5])
        precision = np.diag(1.0 / sigma**2)

        class MockParams:
            p0 = np.array([0.0, 0.0])
            p_free_names = ["x", "y"]

            def lnprior(self, par: np.ndarray) -> float:
                return 0.0

            def build_params(self, par: np.ndarray) -> dict:
                return {"x": par[0], "y": par[1]}

        class GaussianLikelihood:
            params = MockParams()

            def lnprob(self, par: np.ndarray) -> float:
                return -0.5 * par @ precision @ par

        best_fit, fisher = run_fisher(GaussianLikelihood(), {}, str(tmp_path))
        np.testing.assert_allclose(best_fit, [0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(fisher, precision, rtol=1e-3)

        out = np.load(tmp_path / "fisher.npz")
        assert "params" in out
        assert "fisher" in out
        assert "names" in out

    def test_fisher_output_shape(self, tmp_path: str) -> None:
        """Fisher matrix has the right shape for 3 parameters."""

        class MockParams:
            p0 = np.array([0.0, 0.0, 0.0])
            p_free_names = ["a", "b", "c"]

            def lnprior(self, par: np.ndarray) -> float:
                return 0.0

            def build_params(self, par: np.ndarray) -> dict:
                return {"a": par[0], "b": par[1], "c": par[2]}

        class SimpleLikelihood:
            params = MockParams()

            def lnprob(self, par: np.ndarray) -> float:
                return -0.5 * np.dot(par, par)

        _, fisher = run_fisher(SimpleLikelihood(), {}, str(tmp_path))
        assert fisher.shape == (3, 3)


class TestRunTimingEdgeCases:
    """Additional edge cases for run_timing."""

    def test_timing_output_keys(self, tmp_path: str) -> None:
        """Saved npz contains 'timing' and 'names' keys."""

        class MockParams:
            p0 = np.array([1.0, 2.0])
            p_free_names = ["alpha", "beta"]

            def lnprior(self, par: np.ndarray) -> float:
                return 0.0

            def build_params(self, par: np.ndarray) -> dict:
                return {"alpha": par[0], "beta": par[1]}

        class MockLikelihood:
            params = MockParams()

            def lnprob(self, par: np.ndarray) -> float:
                return -1.0

        total, per_eval = run_timing(MockLikelihood(), {}, str(tmp_path), n_eval=10)
        assert per_eval == pytest.approx(total / 10)
        out = np.load(tmp_path / "timing.npz")
        assert out["names"].tolist() == ["alpha", "beta"]


class TestRunSinglepointEdgeCases:
    """Additional edge cases for run_singlepoint."""

    def test_large_chi2(self, tmp_path: str) -> None:
        """Very negative lnprob yields large chi2."""

        class MockParams:
            p0 = np.array([0.0])
            p_free_names = ["r"]

            def lnprior(self, par: np.ndarray) -> float:
                return 0.0

            def build_params(self, par: np.ndarray) -> dict:
                return {"r": par[0]}

        class MockLikelihood:
            params = MockParams()
            invcov = np.eye(5)

            def lnprob(self, par: np.ndarray) -> float:
                return -500.0

        chi2 = run_singlepoint(MockLikelihood(), {}, str(tmp_path))
        assert chi2 == pytest.approx(1000.0)
        out = np.load(tmp_path / "single_point.npz")
        assert out["ndof"] == 5
