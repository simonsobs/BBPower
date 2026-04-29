"""Tests for pure helper methods in bbpower.power_summarizer."""
from __future__ import annotations

import numpy as np
import pytest

from bbpower.power_summarizer import BBPowerSummarizer


def _make_summarizer(**attrs: object) -> BBPowerSummarizer:
    """Create a BBPowerSummarizer without calling __init__."""
    obj = object.__new__(BBPowerSummarizer)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class TestBandsPolIterator:
    """Test BBPowerSummarizer.bands_pol_iterator."""

    def test_count_1band_half(self) -> None:
        """1 band, half=True, with_windows=False -> 3 (EE, EB, BB)."""
        ps = _make_summarizer(nbands=1, n_bpws=5)
        items = list(ps.bands_pol_iterator(half=True, with_windows=False))
        assert len(items) == 3

    def test_count_2band_half(self) -> None:
        """2 bands, half=True, with_windows=False."""
        ps = _make_summarizer(nbands=2, n_bpws=5)
        items = list(ps.bands_pol_iterator(half=True, with_windows=False))
        # band combos: (0,0), (0,1), (1,1) = 3 pairs
        # for each: 4 pol combos (EE,EB,BE,BB) except auto-bands
        # (0,0): EE,EB,BB = 3; (0,1): 4; (1,1): 3 -> total 10
        assert len(items) == 10

    def test_tuple_structure(self) -> None:
        """Each yielded tuple has 8 elements."""
        ps = _make_summarizer(nbands=1, n_bpws=5)
        items = list(ps.bands_pol_iterator(half=True, with_windows=False))
        for item in items:
            assert len(item) == 8
            b1, ip1, b2, ip2, l1, l2, x, win = item
            assert isinstance(l1, str)
            assert isinstance(x, str)
            assert win is None  # with_windows=False


class TestBandsSplitsPolIterator:
    """Test BBPowerSummarizer.bands_splits_pol_iterator."""

    def test_count_1band_2splits(self) -> None:
        """1 band, 2 splits -> upper triangle of (band,split,pol) combos."""
        ps = _make_summarizer(nbands=1, nsplits=2, n_bpws=5, pol_names=["E", "B"])
        items = list(ps.bands_splits_pol_iterator())
        # 1 band pair (0,0); splits: (0,0),(0,1),(1,1) = 3
        # for (0,0) same-split-same-band: pols upper triangle: 3
        # for (0,1) diff-split-same-band: 4 pol combos
        # for (1,1): 3
        assert len(items) == 10

    def test_tuple_has_9_elements(self) -> None:
        """Each yielded tuple is a 9-element tuple."""
        ps = _make_summarizer(nbands=1, nsplits=1, n_bpws=5, pol_names=["E", "B"])
        items = list(ps.bands_splits_pol_iterator())
        for item in items:
            assert len(item) == 9
            s1, s2, b1, b2, p1, p2, m1, m2, cl_name = item
            assert cl_name.startswith("cl_")


class TestGetCovarianceFromSamples:
    """Test BBPowerSummarizer.get_covariance_from_samples."""

    def test_diagonal(self) -> None:
        """Diagonal covariance returns a diagonal matrix."""
        import sacc

        ps = _make_summarizer(n_bpws=3)
        rng = np.random.default_rng(42)
        v = rng.normal(size=(50, 6))
        s = sacc.Sacc()
        ps.get_covariance_from_samples(v, s, covar_type="diagonal")
        cov = s.covariance.covmat
        # Off-diagonal should be zero
        np.testing.assert_array_equal(cov - np.diag(np.diag(cov)), 0)

    def test_dense(self) -> None:
        """Dense covariance is a full matrix."""
        import sacc

        ps = _make_summarizer(n_bpws=3)
        rng = np.random.default_rng(42)
        v = rng.normal(size=(50, 6))
        s = sacc.Sacc()
        ps.get_covariance_from_samples(v, s, covar_type="dense")
        cov = s.covariance.covmat
        assert cov.shape == (6, 6)
