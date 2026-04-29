"""Tests for bbpower.fg_model — foreground model loading."""

from __future__ import annotations

import pytest

import bbpower.fgcls as fgl
from bbpower.fg_model import FGModel, get_function


class TestGetFunction:
    """Tests for the module attribute lookup helper."""

    def test_valid(self):
        """Finds ClPowerLaw in the fgcls module."""
        result = get_function(fgl, "ClPowerLaw")
        assert result is fgl.ClPowerLaw

    def test_invalid_raises(self):
        """Raises KeyError for a nonexistent class."""
        with pytest.raises(KeyError, match="cannot be found"):
            get_function(fgl, "NonExistentClass")


class TestComponentIterator:
    """Tests for FGModel.component_iterator."""

    def test_yields_components(self, bb_only_config):
        """Yields only entries starting with 'component_'."""
        fg = object.__new__(FGModel)
        names = [name for name, _ in fg.component_iterator(bb_only_config)]
        assert "component_1" in names
        assert "component_2" in names
        assert len(names) == 2

    def test_skips_non_component_keys(self):
        """Non-component keys like 'use_moments' are skipped."""
        config = {
            "fg_model": {
                "use_moments": True,
                "moments_lmax": 192,
                "component_dust": {"name": "Dust"},
            }
        }
        fg = object.__new__(FGModel)
        names = [name for name, _ in fg.component_iterator(config)]
        assert names == ["component_dust"]


class TestFGModelInit:
    """Tests for full FGModel initialization (requires mock fgbuster)."""

    def test_loads_components(self, bb_only_config):
        """FGModel sets n_components and component_names."""
        fg = FGModel(bb_only_config)
        assert fg.n_components == 2
        assert "component_1" in fg.component_names
        assert "component_2" in fg.component_names

    def test_cross_correlation_setup(self, bb_only_config):
        """Cross-correlation dict is populated for component_1."""
        fg = FGModel(bb_only_config)
        comp1 = fg.components["component_1"]
        assert "component_2" in comp1["names_x_dict"]

    def test_nu0_must_be_fixed(self):
        """Varying nu0 raises ValueError."""
        config = {
            "pol_channels": ["B"],
            "fg_model": {
                "component_1": {
                    "name": "Dust",
                    "sed": "Dust",
                    "cl": {("B", "B"): "ClPowerLaw"},
                    "sed_parameters": {
                        "nu0_d": ["nu0", "tophat", [100, 353, 500]],
                    },
                    "cl_parameters": {
                        ("B", "B"): {
                            "amp_d": ["amp", "fixed", [1.0]],
                            "l0_d": ["ell0", "fixed", [80.0]],
                        }
                    },
                }
            },
        }
        with pytest.raises(ValueError, match="reference frequencies"):
            FGModel(config)
