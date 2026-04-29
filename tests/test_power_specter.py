"""Tests for pure helper methods in bbpower.power_specter."""

from __future__ import annotations

import numpy as np
import pytest

import bbpower.power_specter as power_specter
from bbpower.power_specter import BBPowerSpecter


def _make_specter(**attrs: object) -> BBPowerSpecter:
    """Create a BBPowerSpecter without calling __init__."""
    obj = object.__new__(BBPowerSpecter)
    for k, v in attrs.items():
        if k == "config":
            setattr(obj, "_configs", v)
        else:
            setattr(obj, k, v)
    return obj


class TestGetMapLabel:
    """Test BBPowerSpecter.get_map_label."""

    def test_basic(self) -> None:
        """Verify 1-indexed band/split naming."""
        ps = _make_specter()
        assert ps.get_map_label(0, 0) == "band1_split1"

    def test_multidigit(self) -> None:
        """Verify correct numbering for higher indices."""
        ps = _make_specter()
        assert ps.get_map_label(2, 3) == "band3_split4"


class TestGetWorkspaceLabel:
    """Test BBPowerSpecter.get_workspace_label."""

    def test_ordered(self) -> None:
        """Canonical ordering puts the smaller index first."""
        ps = _make_specter()
        assert ps.get_workspace_label(0, 1) == "b1_b2"

    def test_reversed(self) -> None:
        """Reversing band order still produces canonical label."""
        ps = _make_specter()
        assert ps.get_workspace_label(2, 0) == "b1_b3"

    def test_same_band(self) -> None:
        """Auto-pair produces repeated index."""
        ps = _make_specter()
        assert ps.get_workspace_label(1, 1) == "b2_b2"


class TestGetFnameWorkspace:
    """Test BBPowerSpecter.get_fname_workspace."""

    def test_contains_prefix(self) -> None:
        """Result is based on prefix_mcm."""
        ps = _make_specter(prefix_mcm="/tmp/mcm_prefix")
        fname = ps.get_fname_workspace(0, 1)
        assert fname.startswith("/tmp/mcm_prefix")
        assert fname.endswith(".fits")


class TestGetCellIterator:
    """Test BBPowerSpecter.get_cell_iterator."""

    def test_count_1band_2splits(self) -> None:
        """1 band, 2 splits -> upper triangle: (0,0), (0,1), (1,1) = 3 pairs."""
        ps = _make_specter(n_bpss=1, nsplits=2)
        items = list(ps.get_cell_iterator())
        assert len(items) == 3

    def test_count_2band_1split(self) -> None:
        """2 bands, 1 split -> band pairs (0,0), (0,1), (1,1) = 3."""
        ps = _make_specter(n_bpss=2, nsplits=1)
        items = list(ps.get_cell_iterator())
        assert len(items) == 3

    def test_tuple_structure(self) -> None:
        """Each yielded item is a 6-tuple of (b1, b2, s1, s2, l1, l2)."""
        ps = _make_specter(n_bpss=1, nsplits=1)
        items = list(ps.get_cell_iterator())
        assert len(items) == 1
        b1, b2, s1, s2, l1, l2 = items[0]
        assert (b1, b2, s1, s2) == (0, 0, 0, 0)
        assert l1 == "band1_split1"
        assert l2 == "band1_split1"


class TestGetBandpowers:
    """Test NaMaster bandpower construction compatibility helpers."""

    def test_custom_bins_use_namaster2_keyword_api(self, monkeypatch, tmp_path) -> None:
        """NaMaster >=2 custom bins use f_ell instead of the removed is_Dell."""
        calls = []

        class FakeNmtBin:
            def __init__(self, *, bpws, ells, lmax=None, weights=None, f_ell=None):
                calls.append(
                    {
                        "bpws": bpws,
                        "ells": ells,
                        "weights": weights,
                        "f_ell": f_ell,
                        "lmax": lmax,
                    }
                )

            @classmethod
            def from_nside_linear(cls, nside, nlb, is_Dell=False, f_ell=None):
                raise AssertionError("custom bin test should not use linear bins")

        monkeypatch.setattr(power_specter.nmt, "NmtBin", FakeNmtBin)
        edges = tmp_path / "edges.txt"
        np.savetxt(edges, np.array([2, 4, 6]))
        ps = _make_specter(
            config={"bpw_edges": str(edges), "compute_dell": True},
            nside=8,
            larr_all=np.arange(24),
        )

        ps.get_bandpowers()

        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0]["ells"], ps.larr_all)
        np.testing.assert_allclose(
            calls[0]["f_ell"],
            ps.larr_all * (ps.larr_all + 1) / (2 * np.pi),
        )

    def test_custom_bins_keep_namaster1_is_dell(self, monkeypatch, tmp_path) -> None:
        """NaMaster 1 custom bins keep the historical positional constructor."""
        calls = []

        class FakeNmtBin:
            def __init__(
                self, nside, bpws=None, ells=None, weights=None, is_Dell=False
            ):
                calls.append(
                    {
                        "nside": nside,
                        "bpws": bpws,
                        "ells": ells,
                        "weights": weights,
                        "is_Dell": is_Dell,
                    }
                )

        monkeypatch.setattr(power_specter.nmt, "NmtBin", FakeNmtBin)
        edges = tmp_path / "edges.txt"
        np.savetxt(edges, np.array([2, 4, 6]))
        ps = _make_specter(
            config={"bpw_edges": str(edges), "compute_dell": True},
            nside=8,
            larr_all=np.arange(24),
        )

        ps.get_bandpowers()

        assert len(calls) == 1
        assert calls[0]["nside"] == 8
        assert calls[0]["is_Dell"] is True
        np.testing.assert_array_equal(calls[0]["ells"], ps.larr_all)

    def test_linear_bins_use_namaster2_constructor(self, monkeypatch) -> None:
        """NaMaster >=2 integer-width bins use from_nside_linear."""
        calls = []

        class FakeNmtBin:
            def __init__(self, *, bpws, ells, lmax=None, weights=None, f_ell=None):
                raise AssertionError("linear bin test should use from_nside_linear")

            @classmethod
            def from_nside_linear(cls, nside, nlb, is_Dell=False, f_ell=None):
                calls.append(
                    {
                        "nside": nside,
                        "nlb": nlb,
                        "is_Dell": is_Dell,
                        "f_ell": f_ell,
                    }
                )
                return "bins"

        monkeypatch.setattr(power_specter.nmt, "NmtBin", FakeNmtBin)
        ps = _make_specter(config={"bpw_edges": 20}, nside=8)

        ps.get_bandpowers()

        assert ps.bins == "bins"
        assert calls == [{"nside": 8, "nlb": 20, "is_Dell": False, "f_ell": None}]


class TestComputeCouplingMatrix:
    """Test NaMaster workspace API compatibility."""

    def test_passes_n_iter_when_supported(self) -> None:
        """NaMaster 1-style workspaces receive n_iter on the workspace call."""
        calls = []

        class Workspace:
            def compute_coupling_matrix(self, field_1, field_2, bins, n_iter=None):
                calls.append(
                    {
                        "field_1": field_1,
                        "field_2": field_2,
                        "bins": bins,
                        "n_iter": n_iter,
                    }
                )

        BBPowerSpecter._compute_coupling_matrix(
            Workspace(),
            "f1",
            "f2",
            "bins",
            n_iter=3,
        )

        assert calls == [
            {"field_1": "f1", "field_2": "f2", "bins": "bins", "n_iter": 3}
        ]

    def test_omits_n_iter_when_not_supported(self) -> None:
        """NaMaster 2-style workspaces do not receive the removed n_iter kwarg."""
        calls = []

        class Workspace:
            def compute_coupling_matrix(self, field_1, field_2, bins):
                calls.append(
                    {
                        "field_1": field_1,
                        "field_2": field_2,
                        "bins": bins,
                    }
                )

        BBPowerSpecter._compute_coupling_matrix(
            Workspace(),
            "f1",
            "f2",
            "bins",
            n_iter=3,
        )

        assert calls == [{"field_1": "f1", "field_2": "f2", "bins": "bins"}]
