"""Structural tests for bbpower.plotter.BBPlotter."""
from __future__ import annotations

import pytest

try:
    from bbpower.plotter import BBPlotter

    HAS_PLOTTER = True
except ImportError:
    HAS_PLOTTER = False

pytestmark = pytest.mark.skipif(
    not HAS_PLOTTER, reason="plotter dependencies not installed"
)


class TestBBPlotterAttributes:
    """Verify pipeline stage class attributes."""

    def test_name(self) -> None:
        """Stage name matches expected string."""
        assert BBPlotter.name == "BBPlotter"

    def test_inputs_list(self) -> None:
        """inputs is a non-empty list of (tag, type) tuples."""
        assert isinstance(BBPlotter.inputs, list)
        assert len(BBPlotter.inputs) > 0
        for tag, ftype in BBPlotter.inputs:
            assert isinstance(tag, str)

    def test_outputs_list(self) -> None:
        """outputs is a non-empty list of (tag, type) tuples."""
        assert isinstance(BBPlotter.outputs, list)
        assert len(BBPlotter.outputs) > 0

    def test_config_options(self) -> None:
        """config_options has expected keys."""
        assert "lmax_plot" in BBPlotter.config_options
        assert "plot_likelihood" in BBPlotter.config_options
