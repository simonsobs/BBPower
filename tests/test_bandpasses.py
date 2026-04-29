"""Tests for bbpower.bandpasses — bandpass convolution and rotation."""

from __future__ import annotations

import numpy as np
import pytest

from bbpower.bandpasses import (
    Bandpass,
    decorrelated_bpass,
    rotate_cells,
    rotate_cells_mat,
)


class TestSedCmbRj:
    """Tests for the CMB SED in Rayleigh-Jeans units."""

    def test_low_freq_limit(self, make_bandpass):
        """At very low frequency the CMB SED approaches 1."""
        bp = make_bandpass()
        nu_low = np.array([0.1, 0.5, 1.0])
        sed = bp.sed_CMB_RJ(nu_low)
        np.testing.assert_allclose(sed, np.ones(3), atol=1e-3)

    def test_vectorized(self, make_bandpass):
        """Output shape matches input shape."""
        bp = make_bandpass()
        nu = np.linspace(30, 300, 50)
        sed = bp.sed_CMB_RJ(nu)
        assert sed.shape == nu.shape

    def test_positive(self, make_bandpass):
        """SED values are positive for physical frequencies."""
        bp = make_bandpass()
        nu = np.linspace(10, 500, 100)
        sed = bp.sed_CMB_RJ(nu)
        assert np.all(sed > 0)


class TestBandpassInit:
    """Tests for Bandpass construction."""

    def test_nu_mean_in_range(self, make_bandpass):
        """nu_mean is within the frequency range."""
        bp = make_bandpass(nu_center=150.0, bandwidth=30.0)
        assert 135.0 <= bp.nu_mean <= 165.0

    def test_cmb_norm_positive(self, make_bandpass):
        """CMB normalization is positive."""
        bp = make_bandpass()
        assert bp.cmb_norm > 0

    def test_not_complex_by_default(self, make_bandpass):
        """Default bandpass is not complex."""
        bp = make_bandpass()
        assert not bp.is_complex


class TestConvolveSed:
    """Tests for convolve_sed method."""

    def test_cmb_near_unity(self, make_bandpass):
        """Convolving CMB SED (sed=None) returns ~1.0."""
        bp = make_bandpass()
        amp, rot = bp.convolve_sed(None, {})
        assert rot is None
        np.testing.assert_allclose(amp, 1.0, atol=0.05)

    def test_custom_sed(self, make_bandpass):
        """Custom SED convolution returns a finite value."""
        bp = make_bandpass(nu_center=150.0)
        sed_func = lambda nu: (nu / 150.0) ** 2.0
        amp, rot = bp.convolve_sed(sed_func, {})
        assert np.isfinite(amp)
        assert rot is None  # not complex

    def test_shift_changes_result(self, make_bandpass):
        """Frequency shift systematic changes the convolution result."""
        config = {
            "systematics": {
                "bandpasses": {
                    "bandpass_1": {
                        "parameters": {
                            "shift_1": ["shift", "tophat", [-0.01, 0.0, 0.01]]
                        }
                    }
                }
            }
        }
        bp = make_bandpass(bp_number=1, config=config)
        amp_no_shift, _ = bp.convolve_sed(None, {"shift_1": 0.0})
        amp_with_shift, _ = bp.convolve_sed(None, {"shift_1": 0.05})
        assert amp_no_shift != amp_with_shift

    def test_gain_scales_result(self, make_bandpass):
        """Gain systematic scales the output by the gain value."""
        config = {
            "systematics": {
                "bandpasses": {
                    "bandpass_1": {
                        "parameters": {"gain_1": ["gain", "tophat", [0.9, 1.0, 1.1]]}
                    }
                }
            }
        }
        bp = make_bandpass(bp_number=1, config=config)
        amp_base, _ = bp.convolve_sed(None, {"gain_1": 1.0})
        amp_scaled, _ = bp.convolve_sed(None, {"gain_1": 2.0})
        np.testing.assert_allclose(amp_scaled, amp_base * 2.0, rtol=1e-10)


class TestRotationMatrix:
    """Tests for get_rotation_matrix."""

    def test_none_without_angle(self, make_bandpass):
        """Returns None when angle systematics are disabled."""
        bp = make_bandpass()
        assert bp.get_rotation_matrix({}) is None

    def test_with_angle(self, make_bandpass):
        """Returns correct 2x2 rotation matrix."""
        config = {
            "systematics": {
                "bandpasses": {
                    "bandpass_1": {
                        "parameters": {"angle_1": ["angle", "tophat", [-1.0, 0.0, 1.0]]}
                    }
                }
            }
        }
        bp = make_bandpass(bp_number=1, config=config)
        mat = bp.get_rotation_matrix({"angle_1": 45.0})
        assert mat.shape == (2, 2)
        phi = np.radians(45.0)
        expected = np.array(
            [[np.cos(2 * phi), np.sin(2 * phi)], [-np.sin(2 * phi), np.cos(2 * phi)]]
        )
        np.testing.assert_allclose(mat, expected)


class TestRotateCellsMat:
    """Tests for the rotate_cells_mat free function."""

    def test_both_none(self):
        """No rotation when both matrices are None."""
        # cls shape is (n_ell, npol, npol) in actual usage
        cls = np.random.default_rng(0).random((10, 2, 2))
        result = rotate_cells_mat(None, None, cls)
        np.testing.assert_array_equal(result, cls)

    def test_identity_rotation(self):
        """Identity matrix leaves cls unchanged."""
        eye = np.eye(2)
        cls = np.random.default_rng(0).random((10, 2, 2))
        result = rotate_cells_mat(eye, eye, cls)
        np.testing.assert_allclose(result, cls, atol=1e-14)

    def test_known_rotation(self):
        """Negative-identity rotation on one side leaves cls unchanged."""
        # -I applied on one side: result[i,j,l] = sum_k cls[i,j,k]*(-I)[l,k]
        # = -cls[i,j,l]. So result = -cls.
        mat = np.array([[-1.0, 0], [0, -1.0]])
        cls = np.ones((5, 2, 2))
        result = rotate_cells_mat(mat, None, cls)
        np.testing.assert_allclose(result, -cls)


class TestRotateCells:
    """Tests for the rotate_cells convenience wrapper."""

    def test_no_rotation(self, make_bandpass):
        """Without angle params, rotate_cells returns unchanged spectra."""
        bp1 = make_bandpass(nu_center=90.0)
        bp2 = make_bandpass(nu_center=150.0)
        cls = np.array([[1.0, 0.5], [0.5, 2.0]])
        result = rotate_cells(bp1, bp2, cls, {})
        np.testing.assert_allclose(result, cls)

    def test_with_angle(self, make_bandpass):
        """With angle params, rotate_cells changes the spectrum."""
        config = {
            "systematics": {
                "bandpasses": {
                    "bandpass_1": {
                        "parameters": {
                            "alpha_1": ["angle", "tophat", [-10.0, 0.0, 10.0]]
                        }
                    },
                    "bandpass_2": {
                        "parameters": {
                            "alpha_2": ["angle", "tophat", [-10.0, 0.0, 10.0]]
                        }
                    },
                }
            }
        }
        bp1 = make_bandpass(nu_center=90.0, bp_number=1, config=config)
        bp2 = make_bandpass(nu_center=150.0, bp_number=2, config=config)
        # cls must be (n_ell, npol, npol)
        cls = np.zeros((5, 2, 2))
        cls[:, 0, 0] = 1.0  # EE = 1
        cls[:, 1, 1] = 1.0  # BB = 1
        params = {"alpha_1": 10.0, "alpha_2": 5.0}
        result = rotate_cells(bp1, bp2, cls, params)
        assert result.shape == cls.shape
        # With angles, off-diagonal (EB/BE) should be nonzero
        assert not np.allclose(result[:, 0, 1], 0.0)


class TestDecorrelatedBpass:
    """Tests for decorrelated_bpass function."""

    def test_same_freq_matches_product(self, make_bandpass):
        """Same bandpass with delta=1 gives same result as simple product."""
        bp = make_bandpass(nu_center=150.0)
        sed_func = lambda nu: np.ones_like(nu)
        # decorr_delta=1 means no decorrelation: delta**(log(1)**2) = 1**0 = 1
        result = decorrelated_bpass(bp, bp, sed_func, {}, decorr_delta=1.0)
        # Compare to plain convolution squared / (cmb_norm1 * cmb_norm2)
        conv, _ = bp.convolve_sed(sed_func, {})
        expected = conv**2
        np.testing.assert_allclose(result, expected, rtol=1e-6)
