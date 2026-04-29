"""Tests for bbpower.param_manager — parameter parsing and priors."""

from __future__ import annotations

import numpy as np
import pytest

from bbpower.param_manager import ParameterManager


class TestParameterManagerInit:
    """Verify that ParameterManager correctly separates fixed/free params."""

    def test_separates_fixed_and_free(self, bb_only_config):
        """Free params are detected and fixed params stored."""
        pm = ParameterManager(bb_only_config)
        # CMB free: r_tensor, A_lens
        # FG free: alpha_d_bb, amp_d_bb, beta_d, epsilon_ds,
        #          alpha_s_bb, amp_s_bb, beta_s
        assert "r_tensor" in pm.p_free_names
        assert "A_lens" in pm.p_free_names
        assert "amp_d_bb" in pm.p_free_names
        assert "beta_d" in pm.p_free_names
        assert "beta_s" in pm.p_free_names
        # Fixed params: temp_d=19.6, nu0_d=353, nu0_s=23, ell0s
        fixed_names = [name for name, _ in pm.p_fixed]
        assert "temp_d" in fixed_names
        assert "nu0_d" in fixed_names
        assert "nu0_s" in fixed_names

    def test_p0_tophat_uses_center(self, bb_only_config):
        """Tophat prior p0 is the center (second) element."""
        pm = ParameterManager(bb_only_config)
        idx = pm.p_free_names.index("r_tensor")
        assert pm.p0[idx] == 0.0  # center of [-0.1, 0.0, 0.1]

    def test_p0_gaussian_uses_mean(self, bb_only_config):
        """Gaussian prior p0 is the mean (first) element."""
        pm = ParameterManager(bb_only_config)
        idx = pm.p_free_names.index("beta_d")
        assert pm.p0[idx] == 1.59  # mean of Gaussian [1.59, 0.11]

    def test_p0_is_numpy_array(self, bb_only_config):
        """p0 is converted to a numpy array."""
        pm = ParameterManager(bb_only_config)
        assert isinstance(pm.p0, np.ndarray)


class TestBuildParams:
    """Verify build_params round-trips correctly."""

    def test_roundtrip(self, bb_only_config):
        """build_params(p0) returns all param names."""
        pm = ParameterManager(bb_only_config)
        params = pm.build_params(pm.p0)
        for name in pm.p_free_names:
            assert name in params
        for name, val in pm.p_fixed:
            assert name in params
            assert params[name] == val

    def test_custom_values(self, bb_only_config):
        """Passing custom array sets free params to those values."""
        pm = ParameterManager(bb_only_config)
        custom = np.ones(len(pm.p_free_names)) * 42.0
        params = pm.build_params(custom)
        for name in pm.p_free_names:
            assert params[name] == 42.0


class TestLnPrior:
    """Verify prior evaluation logic."""

    def test_at_fiducial_finite(self, bb_only_config):
        """lnprior at p0 is finite."""
        pm = ParameterManager(bb_only_config)
        assert np.isfinite(pm.lnprior(pm.p0))

    def test_tophat_out_of_bounds(self, bb_only_config):
        """Setting a tophat param beyond upper edge returns -inf."""
        pm = ParameterManager(bb_only_config)
        par = pm.p0.copy()
        idx = pm.p_free_names.index("r_tensor")
        par[idx] = 999.0  # way beyond tophat [−0.1, 0.1]
        assert pm.lnprior(par) == -np.inf

    def test_gaussian_value(self):
        """Gaussian prior gives expected log-prob."""
        config = {
            "pol_channels": ["B"],
            "cmb_model": {
                "params": {
                    "r_tensor": ["r_tensor", "Gaussian", [0.0, 1.0]],
                }
            },
            "fg_model": {},
        }
        pm = ParameterManager(config)
        # At p=2.0: lnp = -0.5 * (2/1)**2 = -2.0
        assert pm.lnprior(np.array([2.0])) == pytest.approx(-2.0)


class TestPriorKind:
    """Verify _prior_kind normalisation."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Gaussian", "gaussian"),
            ("TOPHAT", "tophat"),
            (" Fixed ", "fixed"),
            ("gaussian", "gaussian"),
        ],
    )
    def test_normalization(self, raw, expected):
        """Prior strings are normalized to lowercase."""
        assert ParameterManager._prior_kind(raw) == expected


class TestEdgeCases:
    """Error handling and edge cases."""

    def test_duplicate_name_raises(self):
        """Duplicate free parameter names (dict keys) raise KeyError."""
        # ParameterManager checks p_name (the dict key) for duplicates.
        # To trigger this, we need the same key appearing twice which
        # can't happen in a Python dict literal. Instead, test by calling
        # _add_parameter directly with a duplicate name.
        pm = ParameterManager.__new__(ParameterManager)
        pm.p_free_names = ["already_exists"]
        pm.p_free_priors = [["x", "tophat", [0, 1, 2]]]
        pm.p_fixed = []
        pm.p0 = [1.0]
        with pytest.raises(KeyError, match="same name"):
            pm._add_parameter("already_exists", ["x", "tophat", [0, 1, 2]])

    def test_unknown_prior_raises(self):
        """Unknown prior type raises ValueError."""
        config = {
            "pol_channels": ["B"],
            "cmb_model": {
                "params": {
                    "x": ["x", "loguniform", [0.1, 10]],
                }
            },
            "fg_model": {},
        }
        with pytest.raises(ValueError, match="Unknown prior"):
            ParameterManager(config)


class TestPolChannelFiltering:
    """Verify cl_parameters are filtered by pol_channels."""

    def test_eb_includes_ee_params(self, eb_config):
        """EE cl_parameters appear when pol_channels includes E."""
        pm = ParameterManager(eb_config)
        assert "amp_d_ee" in pm.p_free_names
        assert "amp_d_bb" in pm.p_free_names

    def test_bb_excludes_ee_params(self, bb_only_config):
        """EE cl_parameters are absent when pol_channels is B-only."""
        pm = ParameterManager(bb_only_config)
        assert "amp_d_ee" not in pm.p_free_names
        assert "amp_d_bb" in pm.p_free_names


class TestGetComponentNames:
    """Verify get_component_names helper."""

    def test_returns_sorted(self, bb_only_config):
        """Returns sorted component keys."""
        pm = ParameterManager(bb_only_config)
        names = pm.get_component_names(bb_only_config)
        assert names == ["component_1", "component_2"]


class TestMomentsParams:
    """Verify moment parameters are included/excluded correctly."""

    def test_moments_when_enabled(self, bb_only_config):
        """Moment params appear when use_moments is True."""
        config = dict(bb_only_config)
        config["fg_model"] = dict(config["fg_model"])
        config["fg_model"]["use_moments"] = True
        config["fg_model"]["component_1"] = dict(config["fg_model"]["component_1"])
        config["fg_model"]["component_1"]["moments"] = {
            "gamma_d_beta": ["gamma_beta", "tophat", [-6.0, -3.5, -2.0]],
        }
        pm = ParameterManager(config)
        assert "gamma_d_beta" in pm.p_free_names

    def test_moments_when_disabled(self, bb_only_config):
        """Moment params are absent when use_moments is not set."""
        pm = ParameterManager(bb_only_config)
        assert "gamma_d_beta" not in pm.p_free_names
