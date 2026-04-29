"""Tests for bbpower.compsep — component separation stage.

This is the primary test focus. Tests construct minimal BBCompSep instances
by calling ``object.__new__(BBCompSep)`` and setting attributes directly,
avoiding the need for real SACC files or bbpipe initialization.
"""

from __future__ import annotations

import numpy as np
import pytest

from bbpower.compsep import BBCompSep
from bbpower.bandpasses import Bandpass
from bbpower.fg_model import FGModel
from bbpower.fgcls import ClPowerLaw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_compsep(nmaps: int = 2, nfreqs: int = 2, npol: int = 1) -> BBCompSep:
    """Build a minimal BBCompSep with required attributes set."""
    obj = object.__new__(BBCompSep)
    obj.nmaps = nmaps
    obj.nfreqs = nfreqs
    obj.npol = npol
    obj.ncross = nmaps * (nmaps + 1) // 2
    obj.index_ut = np.triu_indices(nmaps)
    obj._configs = {}
    return obj


# ---------------------------------------------------------------------------
# matrix_to_vector / vector_to_matrix
# ---------------------------------------------------------------------------
class TestMatrixToVector:
    """Tests for upper-triangle extraction."""

    def test_2x2(self):
        """2x2 symmetric matrix -> 3 upper-triangle elements."""
        cs = _make_compsep(nmaps=2)
        mat = np.array([[1.0, 2.0], [2.0, 3.0]])
        vec = cs.matrix_to_vector(mat)
        np.testing.assert_array_equal(vec, [1.0, 2.0, 3.0])

    def test_3x3(self):
        """3x3 symmetric matrix -> 6 upper-triangle elements."""
        cs = _make_compsep(nmaps=3)
        mat = np.arange(9).reshape(3, 3).astype(float)
        mat = (mat + mat.T) / 2  # symmetrize
        vec = cs.matrix_to_vector(mat)
        assert vec.shape == (6,)

    def test_batched(self):
        """Batched (5, 3, 3) -> (5, 6)."""
        cs = _make_compsep(nmaps=3)
        mats = np.random.default_rng(0).random((5, 3, 3))
        mats = 0.5 * (mats + mats.transpose(0, 2, 1))
        vecs = cs.matrix_to_vector(mats)
        assert vecs.shape == (5, 6)


class TestVectorToMatrix:
    """Tests for symmetric matrix reconstruction."""

    def test_1d_roundtrip(self):
        """vec -> matrix -> vec round-trips."""
        cs = _make_compsep(nmaps=3)
        mat_orig = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
        vec = cs.matrix_to_vector(mat_orig)
        mat_rec = cs.vector_to_matrix(vec)
        np.testing.assert_allclose(mat_rec, mat_orig)

    def test_2d_roundtrip(self):
        """Batched vec -> matrix -> vec round-trips."""
        cs = _make_compsep(nmaps=2)
        rng = np.random.default_rng(1)
        mats = rng.random((7, 2, 2))
        mats = 0.5 * (mats + mats.transpose(0, 2, 1))
        vecs = cs.matrix_to_vector(mats)
        mats_rec = cs.vector_to_matrix(vecs)
        np.testing.assert_allclose(mats_rec, mats)

    def test_symmetry(self):
        """Reconstructed matrix is symmetric."""
        cs = _make_compsep(nmaps=3)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = cs.vector_to_matrix(vec)
        np.testing.assert_array_equal(mat, mat.T)

    def test_3d_raises(self):
        """3-D input raises ValueError."""
        cs = _make_compsep(nmaps=2)
        with pytest.raises(ValueError, match="1- or 2-D"):
            cs.vector_to_matrix(np.zeros((2, 3, 4)))

    def test_inverse_relationship(self):
        """mat -> vec -> mat recovers original symmetric matrix."""
        cs = _make_compsep(nmaps=4)
        rng = np.random.default_rng(2)
        mat = rng.random((4, 4))
        mat = (mat + mat.T) / 2
        vec = cs.matrix_to_vector(mat)
        mat_rec = cs.vector_to_matrix(vec)
        np.testing.assert_allclose(mat_rec, mat)


# ---------------------------------------------------------------------------
# _freq_pol_iterator
# ---------------------------------------------------------------------------
class TestFreqPolIterator:
    """Tests for the frequency-polarization index iterator."""

    def test_bb_2freq(self):
        """B-only, 2 frequencies: 3 unique pairs."""
        cs = _make_compsep(nmaps=2, nfreqs=2, npol=1)
        tuples = list(cs._freq_pol_iterator())
        assert len(tuples) == 3
        # Cross-spectrum indices should be 0, 1, 2
        icls = [t[-1] for t in tuples]
        assert icls == [0, 1, 2]

    def test_eb_2freq(self):
        """E+B, 2 frequencies: 10 unique pairs from 4 maps."""
        cs = _make_compsep(nmaps=4, nfreqs=2, npol=2)
        tuples = list(cs._freq_pol_iterator())
        assert len(tuples) == 10

    def test_bb_3freq(self):
        """B-only, 3 frequencies: 6 unique pairs."""
        cs = _make_compsep(nmaps=3, nfreqs=3, npol=1)
        tuples = list(cs._freq_pol_iterator())
        assert len(tuples) == 6

    def test_m_indices(self):
        """Map indices m1, m2 follow m = p + npol * b."""
        cs = _make_compsep(nmaps=4, nfreqs=2, npol=2)
        for b1, b2, p1, p2, m1, m2, icl in cs._freq_pol_iterator():
            assert m1 == p1 + cs.npol * b1
            assert m2 == p2 + cs.npol * b2

    def test_count_equals_ncross(self):
        """Iterator length matches ncross."""
        for nmaps, nfreqs, npol in [(2, 2, 1), (4, 2, 2), (3, 3, 1), (6, 3, 2)]:
            cs = _make_compsep(nmaps=nmaps, nfreqs=nfreqs, npol=npol)
            count = sum(1 for _ in cs._freq_pol_iterator())
            assert count == cs.ncross


# ---------------------------------------------------------------------------
# bcls (moment expansion helper)
# ---------------------------------------------------------------------------
class TestBcls:
    """Tests for the bcls power-law helper."""

    def test_shape(self):
        """Output length matches lmax."""
        cs = _make_compsep()
        result = cs.bcls(100, -3.5, 1e-6)
        assert len(result) == 100

    def test_first_two_zero(self):
        """ell=0 and ell=1 are zero."""
        cs = _make_compsep()
        result = cs.bcls(50, -3.5, 1e-6)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_scaling(self):
        """Correct power-law value at ell=160."""
        cs = _make_compsep()
        amp = 2.0
        gamma = -2.0
        result = cs.bcls(200, gamma, amp)
        # At ell=160: amp * (160/80)**gamma = 2 * (2)**(-2) = 0.5
        np.testing.assert_allclose(result[160], amp * (160 / 80) ** gamma)

    def test_zero_amplitude(self):
        """All zeros when amp=0."""
        cs = _make_compsep()
        result = cs.bcls(50, -3.5, 0.0)
        np.testing.assert_array_equal(result, np.zeros(50))


# ---------------------------------------------------------------------------
# get_moments_lmax
# ---------------------------------------------------------------------------
class TestGetMomentsLmax:
    """Tests for get_moments_lmax config reading."""

    def test_default(self):
        """Returns 384 when not in config."""
        cs = _make_compsep()
        cs._configs = {"fg_model": {}}
        assert cs.get_moments_lmax() == 384

    def test_custom(self):
        """Returns custom value from config."""
        cs = _make_compsep()
        cs._configs = {"fg_model": {"moments_lmax": 192}}
        assert cs.get_moments_lmax() == 192


# ---------------------------------------------------------------------------
# integrate_seds
# ---------------------------------------------------------------------------
def _make_compsep_with_fg(
    bb_only_config: dict,
    make_bandpass,
) -> tuple[BBCompSep, dict]:
    """Build a BBCompSep wired with FGModel and Bandpasses for SED tests."""
    nfreqs = 2
    npol = 1
    nmaps = nfreqs * npol
    cs = _make_compsep(nmaps=nmaps, nfreqs=nfreqs, npol=npol)
    cs._configs = bb_only_config

    # Build real FGModel from the config
    cs.fg_model = FGModel(bb_only_config)

    # Build real Bandpass objects at two different frequencies
    cs.bpss = [
        make_bandpass(nu_center=90.0, bp_number=0, config=bb_only_config),
        make_bandpass(nu_center=150.0, bp_number=1, config=bb_only_config),
    ]

    # Build parameter dict at fiducial values
    from bbpower.param_manager import ParameterManager

    pm = ParameterManager(bb_only_config)
    params = pm.build_params(pm.p0)
    return cs, params


class TestIntegrateSeds:
    """Tests for BBCompSep.integrate_seds."""

    def test_output_shapes(self, bb_only_config, make_bandpass) -> None:
        """fg_scaling has shape (nc, nc, nf, nf)."""
        cs, params = _make_compsep_with_fg(bb_only_config, make_bandpass)
        fg_scaling, rot = cs.integrate_seds(params)
        nc = cs.fg_model.n_components
        nf = cs.nfreqs
        assert fg_scaling.shape == (nc, nc, nf, nf)

    def test_diagonal_positive(self, bb_only_config, make_bandpass) -> None:
        """Auto-component scaling (diagonal) is non-negative."""
        cs, params = _make_compsep_with_fg(bb_only_config, make_bandpass)
        fg_scaling, _ = cs.integrate_seds(params)
        for ic in range(cs.fg_model.n_components):
            diag = fg_scaling[ic, ic]
            assert np.all(diag >= 0)


class TestEvaluatePowerSpectra:
    """Tests for BBCompSep.evaluate_power_spectra."""

    def test_output_shape(self, bb_only_config, make_bandpass) -> None:
        """Output shape matches (n_components, npol, npol, n_ell)."""
        cs, params = _make_compsep_with_fg(bb_only_config, make_bandpass)
        n_ell = 10
        cs.n_ell = n_ell
        cs.bpw_l = np.arange(2, 2 + n_ell)
        cs.dl2cl = 1.0 / (cs.bpw_l * (cs.bpw_l + 1) / (2 * np.pi))
        cs.pol_order = {"B": 0}
        result = cs.evaluate_power_spectra(params)
        assert result.shape == (cs.fg_model.n_components, 1, 1, n_ell)

    def test_nonzero(self, bb_only_config, make_bandpass) -> None:
        """At fiducial parameters, power spectra are nonzero."""
        cs, params = _make_compsep_with_fg(bb_only_config, make_bandpass)
        n_ell = 10
        cs.n_ell = n_ell
        cs.bpw_l = np.arange(2, 2 + n_ell)
        cs.dl2cl = 1.0 / (cs.bpw_l * (cs.bpw_l + 1) / (2 * np.pi))
        cs.pol_order = {"B": 0}
        result = cs.evaluate_power_spectra(params)
        assert np.any(result != 0)
