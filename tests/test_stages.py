"""Tests for bbpower._stages — stage registry and lazy loading."""

from __future__ import annotations

import importlib

import pytest

from bbpower._stages import STAGE_MODULES, get_stage_class


class TestStageModules:
    """Tests for the STAGE_MODULES registry."""

    def test_has_four_entries(self):
        """Registry contains exactly 4 pipeline stages."""
        assert len(STAGE_MODULES) == 4

    def test_expected_names(self):
        """All expected stage names are present."""
        expected = {"BBPowerSpecter", "BBPowerSummarizer", "BBCompSep", "BBPlotter"}
        assert set(STAGE_MODULES.keys()) == expected

    def test_module_paths_importable(self):
        """All module path strings are valid Python module paths."""
        for name, module_path in STAGE_MODULES.items():
            parts = module_path.split(".")
            assert len(parts) >= 2, f"{name}: path too short: {module_path}"
            assert parts[0] == "bbpower", f"{name}: must start with 'bbpower'"


class TestGetStageClass:
    """Tests for get_stage_class lookup."""

    def test_valid_name(self):
        """get_stage_class returns a class for a valid stage name."""
        cls = get_stage_class("BBCompSep")
        assert hasattr(cls, "name")
        assert cls.name == "BBCompSep"

    def test_invalid_raises(self):
        """Unknown stage name raises an error."""
        with pytest.raises((KeyError, ValueError)):
            get_stage_class("NonExistentStage")
