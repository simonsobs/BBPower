"""Tests for bbpower.fgcls — symbolic power spectrum models."""

from __future__ import annotations

import numpy as np
import pytest

from bbpower.fgcls import ClAnalytic, ClGeneral, ClPowerLaw


class TestClPowerLaw:
    """Tests for the ClPowerLaw model: amp * (ell / ell0)**alpha."""

    def test_free_params(self):
        """Fully free ClPowerLaw has ['alpha', 'amp'] (sorted)."""
        cl = ClPowerLaw(ell0=80.0)
        assert cl.params == ["alpha", "amp"]
        assert cl.n_par == 2

    def test_eval_at_pivot(self):
        """At ell = ell0 the power law evaluates to amp."""
        cl = ClPowerLaw(ell0=80.0)
        result = cl.eval(np.array([80.0]), -0.5, 3.0)
        np.testing.assert_allclose(result, 3.0)

    def test_eval_scaling(self):
        """(160/80)**(-1) * 1.0 == 0.5."""
        cl = ClPowerLaw(ell0=80.0)
        result = cl.eval(np.array([160.0]), -1.0, 1.0)
        np.testing.assert_allclose(result, 0.5)

    def test_eval_vectorized(self):
        """Array input returns correct shape and values."""
        cl = ClPowerLaw(ell0=80.0)
        ells = np.array([40.0, 80.0, 160.0])
        result = cl.eval(ells, -1.0, 2.0)
        expected = 2.0 * (ells / 80.0) ** (-1.0)
        assert result.shape == ells.shape
        np.testing.assert_allclose(result, expected)

    def test_fixed_alpha(self):
        """Fixing alpha leaves only 'amp' free."""
        cl = ClPowerLaw(ell0=80.0, alpha=-0.5)
        assert cl.params == ["amp"]
        assert cl.n_par == 1

    def test_amp_always_free(self):
        """amp is always free in ClPowerLaw (not passed to ClAnalytic)."""
        cl = ClPowerLaw(ell0=80.0, amp=2.0)
        assert "amp" in cl.params

    def test_defaults_fully_free(self):
        """Default values match class reference constants."""
        cl = ClPowerLaw(ell0=80.0)
        assert cl.defaults == [ClPowerLaw._REF_ALPHA, ClPowerLaw._REF_AMP]

    def test_defaults_partial_fix(self):
        """Fixing alpha leaves defaults for amp only."""
        cl = ClPowerLaw(ell0=80.0, alpha=-0.3)
        assert cl.defaults == [ClPowerLaw._REF_AMP]

    def test_eval_wrong_nparams_raises(self):
        """Passing wrong number of params raises AssertionError."""
        cl = ClPowerLaw(ell0=80.0)
        with pytest.raises(AssertionError):
            cl.eval(np.arange(10), 1.0)  # needs 2 params, got 1

    def test_repr(self):
        """repr returns a non-empty string."""
        cl = ClPowerLaw(ell0=80.0)
        assert isinstance(repr(cl), str)
        assert len(repr(cl)) > 0

    def test_eval_ell_zero(self):
        """eval at ell=0 does not raise (result may be 0, inf, or nan)."""
        cl = ClPowerLaw(ell0=80.0)
        result = cl.eval(np.array([0.0]), 0.5, 1.0)
        assert result.shape == (1,)


class TestClAnalytic:
    """Tests for the ClAnalytic model with custom expressions."""

    def test_custom_expression_free_params(self):
        """Both A and n are free in 'A * ell**n'."""
        cl = ClAnalytic("A * ell**n")
        assert "A" in cl.params
        assert "n" in cl.params
        assert cl.n_par == 2

    def test_fixed_substitution(self):
        """Fixing L0 removes it from params."""
        cl = ClAnalytic("A * (ell / L0)**n", L0=100.0)
        assert "L0" not in cl.params
        assert "A" in cl.params
        assert "n" in cl.params

    def test_none_stays_free(self):
        """Passing None for a kwarg keeps the parameter free."""
        cl = ClAnalytic("A * ell", A=None)
        assert "A" in cl.params

    def test_eval_custom(self):
        """Evaluate a custom expression at a known point."""
        cl = ClAnalytic("A * (ell / L0)**n", L0=100.0)
        # params are sorted: ['A', 'n']
        result = cl.eval(np.array([100.0]), 5.0, -1.0)
        np.testing.assert_allclose(result, 5.0)  # A * (100/100)**(-1) = 5


class TestClGeneralDefaults:
    """Test the defaults fallback in ClGeneral."""

    def test_defaults_uninitialized(self):
        """Subclass without _defaults returns list of ones."""

        class Bare(ClGeneral):
            _params = ["a", "b"]
            _lambda = lambda self, ell, a, b: a * ell + b

        bare = Bare()
        assert bare.defaults == [1.0, 1.0]
