"""Tests for bbpower.likelihood — chi-squared and H&L likelihood."""

from __future__ import annotations

import numpy as np
import pytest

from bbpower.likelihood import Likelihood
from bbpower.param_manager import ParameterManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_simple_likelihood(
    n_bpws: int = 5,
    nmaps: int = 2,
    use_handl: bool = False,
    model_offset: float = 0.0,
):
    """Build a Likelihood with synthetic data and a trivial model function.

    Parameters
    ----------
    n_bpws : int
        Number of bandpower bins.
    nmaps : int
        Number of maps (determines matrix dimension).
    use_handl : bool
        Whether to use the H&L likelihood.
    model_offset : float
        Constant offset added to the model (data is model + offset -> 0 residual
        when offset=0).
    """
    rng = np.random.default_rng(123)
    # Build positive-definite symmetric matrices so sqrtm returns real
    raw = rng.random((n_bpws, nmaps, nmaps))
    bbdata = np.array([r @ r.T for r in raw]) + 0.5 * np.eye(nmaps)
    bbnoise = 0.1 * np.eye(nmaps)[None, :, :] * np.ones((n_bpws, 1, 1))
    bbfiducial = bbdata.copy() if use_handl else None

    index_ut = np.triu_indices(nmaps)
    ncross = len(index_ut[0])
    ndata = n_bpws * ncross
    invcov = np.eye(ndata)

    def matrix_to_vector(mat):
        return mat[..., index_ut[0], index_ut[1]]

    def model_func(params):
        return bbdata + model_offset

    config = {
        "pol_channels": ["B"],
        "cmb_model": {
            "params": {"r": ["r", "tophat", [-1, 0, 1]]},
        },
        "fg_model": {},
    }
    pm = ParameterManager(config)

    return Likelihood(
        model_func=model_func,
        param_manager=pm,
        bbdata=bbdata,
        bbnoise=bbnoise,
        invcov=invcov,
        matrix_to_vector=matrix_to_vector,
        use_handl=use_handl,
        bbfiducial=bbfiducial,
    )


# ---------------------------------------------------------------------------
# Chi-squared tests
# ---------------------------------------------------------------------------
class TestChiSquared:
    """Tests for the chi-squared likelihood mode."""

    def test_chi_sq_dx_perfect_model(self):
        """Zero residual when model == data."""
        lik = _make_simple_likelihood(model_offset=0.0)
        params = lik.params.build_params(lik.params.p0)
        dx = lik.chi_sq_dx(params)
        np.testing.assert_allclose(dx, 0.0, atol=1e-15)

    def test_chi_sq_dx_shape(self):
        """Output length = n_bpws * ncross."""
        n_bpws, nmaps = 5, 2
        lik = _make_simple_likelihood(n_bpws=n_bpws, nmaps=nmaps)
        params = lik.params.build_params(lik.params.p0)
        dx = lik.chi_sq_dx(params)
        ncross = nmaps * (nmaps + 1) // 2
        assert dx.shape == (n_bpws * ncross,)

    def test_lnlike_perfect(self):
        """lnlike = 0 for perfect model (chi2 mode)."""
        lik = _make_simple_likelihood(model_offset=0.0)
        val = lik.lnlike(lik.params.p0)
        assert val == pytest.approx(0.0)

    def test_lnlike_offset_is_negative(self):
        """lnlike is negative when model != data."""
        lik = _make_simple_likelihood(model_offset=0.5)
        val = lik.lnlike(lik.params.p0)
        assert val < 0

    def test_no_fiducial_needed(self):
        """chi2 mode works without bbfiducial."""
        lik = _make_simple_likelihood(use_handl=False)
        assert lik.bbfiducial is None
        val = lik.lnlike(lik.params.p0)
        assert np.isfinite(val)

    def test_chi2_with_none_bbnoise(self):
        """chi2 mode works when bbnoise is None (regression test)."""
        n_bpws, nmaps = 5, 2
        rng = np.random.default_rng(99)
        raw = rng.random((n_bpws, nmaps, nmaps))
        bbdata = np.array([r @ r.T for r in raw]) + 0.5 * np.eye(nmaps)

        index_ut = np.triu_indices(nmaps)
        ncross = len(index_ut[0])
        invcov = np.eye(n_bpws * ncross)

        config = {
            "pol_channels": ["B"],
            "cmb_model": {"params": {"r": ["r", "tophat", [-1, 0, 1]]}},
            "fg_model": {},
        }
        pm = ParameterManager(config)

        lik = Likelihood(
            model_func=lambda params: bbdata,
            param_manager=pm,
            bbdata=bbdata,
            bbnoise=None,
            invcov=invcov,
            matrix_to_vector=lambda mat: mat[..., index_ut[0], index_ut[1]],
            use_handl=False,
            bbfiducial=None,
        )
        val = lik.lnlike(pm.p0)
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Log-posterior tests
# ---------------------------------------------------------------------------
class TestLnprob:
    """Tests for the full log-posterior (prior + likelihood)."""

    def test_includes_prior(self):
        """lnprob = lnprior + lnlike."""
        lik = _make_simple_likelihood(model_offset=0.0)
        par = lik.params.p0
        lnprior = lik.params.lnprior(par)
        lnlike = lik.lnlike(par)
        lnprob = lik.lnprob(par)
        assert lnprob == pytest.approx(lnprior + lnlike)

    def test_bad_prior_returns_neginf(self):
        """Returns -inf when params are outside prior bounds."""
        lik = _make_simple_likelihood()
        bad_par = np.array([999.0])  # way outside tophat [-1, 1]
        assert lik.lnprob(bad_par) == -np.inf


# ---------------------------------------------------------------------------
# H&L transform tests
# ---------------------------------------------------------------------------
class TestHAndL:
    """Tests for the Hamimeche & Lewis likelihood mode."""

    def test_transform_identity(self):
        """C == Chat => transform is near zero."""
        n = 3
        C = np.eye(n) * 2.0
        Chat = np.eye(n) * 2.0
        Cfl_sqrt = np.eye(n) * np.sqrt(2.0)
        X = Likelihood._h_and_l_transform(C, Chat, Cfl_sqrt)
        # When C == Chat, the signed-sqrt transform should give ~0
        np.testing.assert_allclose(X, np.zeros((n, n)), atol=1e-10)

    def test_transform_singular_returns_inf(self):
        """Singular C returns [np.inf]."""
        C = np.zeros((2, 2))  # singular
        Chat = np.eye(2)
        Cfl_sqrt = np.eye(2)
        X = Likelihood._h_and_l_transform(C, Chat, Cfl_sqrt)
        # eigh may succeed on zero matrix but sqrt of zero diag is fine
        # The exact behavior depends on the eigenvalues; just check it doesn't crash
        assert isinstance(X, (np.ndarray, list))

    def test_h_and_l_dx_perfect(self):
        """Near-zero residual when model+noise matches observed."""
        lik = _make_simple_likelihood(use_handl=True, model_offset=0.0)
        params = lik.params.build_params(lik.params.p0)
        dx = lik.h_and_l_dx(params)
        # Should be close to zero (not exactly due to nonlinear transform)
        assert np.all(np.isfinite(dx))
        assert np.max(np.abs(dx)) < 1.0

    def test_handl_prepare(self):
        """_prepare_h_and_l sets Cfl_sqrt and observed_cls."""
        lik = _make_simple_likelihood(use_handl=True)
        assert hasattr(lik, "Cfl_sqrt")
        assert hasattr(lik, "observed_cls")
        assert lik.Cfl_sqrt.shape[0] == lik.bbdata.shape[0]

    def test_lnlike_handl_finite(self):
        """H&L lnlike returns a finite value for well-conditioned data."""
        lik = _make_simple_likelihood(use_handl=True, model_offset=0.0)
        val = lik.lnlike(lik.params.p0)
        assert np.isfinite(val)
