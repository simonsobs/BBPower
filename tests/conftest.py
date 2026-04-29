"""Shared fixtures for the BBPower test suite.

Provides mock ``bbpipe`` and ``fgbuster`` modules so that all ``bbpower``
submodules can be imported even when those packages are not installed.
Also provides reusable configuration dictionaries and synthetic data factories.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock bbpipe.PipelineStage
# ---------------------------------------------------------------------------
class _MockPipelineStage:
    """Minimal stand-in for ``bbpipe.PipelineStage``.

    Provides enough of the interface so that ``BBCompSep`` (and the other
    stage classes) can be *defined* (class body + ``__init_subclass__``)
    and instantiated without the real bbpipe.
    """

    pipeline_stages: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls.pipeline_stages[cls.name] = (cls, None)

    def __init__(self, args=None):
        self._configs = {}
        self._inputs = {}
        self._outputs = {}

    @property
    def config(self):
        return self._configs

    def get_input(self, tag):
        return self._inputs.get(tag)

    def get_output(self, tag):
        return self._outputs.get(tag)


# ---------------------------------------------------------------------------
# Mock fgbuster.component_model
# ---------------------------------------------------------------------------
class _MockSEDBase:
    """Trivial SED that returns nu**power."""

    _power = 0.0

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @property
    def params(self):
        """Return names of free (None-valued) parameters, matching fgbuster API."""
        return [k for k, v in self._kwargs.items() if v is None and k != "units"]

    def eval(self, nu, *args):
        return np.ones_like(np.asarray(nu, dtype=float))


class _MockCMB(_MockSEDBase):
    def __init__(self, units="K_RJ"):
        super().__init__(units=units)
        self._units = units

    def eval(self, nu, *args):
        return np.ones_like(np.asarray(nu, dtype=float))


class _MockDust(_MockSEDBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, nu, *args):
        return (np.asarray(nu, dtype=float) / 353.0) ** 1.5


class _MockSynchrotron(_MockSEDBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, nu, *args):
        return (np.asarray(nu, dtype=float) / 23.0) ** (-3.0)


# ---------------------------------------------------------------------------
# Inject mocks into sys.modules (before any bbpower import)
# ---------------------------------------------------------------------------
def _inject_mock_bbpipe():
    if "bbpipe" not in sys.modules:
        mod = types.ModuleType("bbpipe")
        mod.PipelineStage = _MockPipelineStage
        sys.modules["bbpipe"] = mod


def _inject_mock_fgbuster():
    if "fgbuster" not in sys.modules:
        fgb = types.ModuleType("fgbuster")
        fgc = types.ModuleType("fgbuster.component_model")
        fgc.CMB = _MockCMB
        fgc.Dust = _MockDust
        fgc.Synchrotron = _MockSynchrotron
        fgb.component_model = fgc
        sys.modules["fgbuster"] = fgb
        sys.modules["fgbuster.component_model"] = fgc


# ---------------------------------------------------------------------------
# Mock sacc
# ---------------------------------------------------------------------------
class _MockDataPoint:
    def __init__(self, data_type: str = "cl_bb", tracers: tuple = ("t1", "t2")):
        self.data_type = data_type
        self.tracers = tracers


class _MockBandpowerWindow:
    def __init__(self, ells: np.ndarray, weight: np.ndarray):
        self.values = weight
        self.ells = ells


class _MockCovariance:
    def __init__(self, covmat: np.ndarray | None = None):
        self.covmat = covmat if covmat is not None else np.array([[]])


class _MockSacc:
    """Minimal stand-in for ``sacc.Sacc``."""

    def __init__(self):
        self.tracers = {}
        self.mean = np.array([])
        self.data = []
        self.covariance = _MockCovariance()

    @classmethod
    def load_fits(cls, path: str) -> "_MockSacc":
        return cls()

    def get_ell_cl(self, *args, **kwargs):
        if kwargs.get("return_ind"):
            return np.array([]), np.array([]), np.array([], dtype=int)
        if kwargs.get("return_cov"):
            return np.array([]), np.array([]), np.array([[]])
        return np.array([]), np.array([])

    def get_tracer_combinations(self):
        return []

    def add_covariance(self, cov):
        self.covariance = _MockCovariance(cov)

    def indices(self, *args, **kwargs):
        return np.array([], dtype=int)

    def get_bandpower_windows(self, indices=None):
        return _MockBandpowerWindow(np.array([]), np.array([[]]))

    def add_tracer(self, *args, **kwargs):
        pass

    def add_ell_cl(self, *args, **kwargs):
        pass

    def save_fits(self, path, overwrite=False):
        pass


class _MockBaseTracer:
    @staticmethod
    def make(*args, **kwargs):
        return type(
            "Tracer",
            (),
            {
                "nu": np.array([]),
                "bandpass": np.array([]),
                "ell": np.array([]),
                "beam": np.array([]),
                "bandpass_extra": {},
            },
        )()


def _inject_mock_sacc():
    if "sacc" not in sys.modules:
        mod = types.ModuleType("sacc")
        mod.Sacc = _MockSacc
        mod.BandpowerWindow = _MockBandpowerWindow
        mod.BaseTracer = _MockBaseTracer
        sys.modules["sacc"] = mod


# ---------------------------------------------------------------------------
# Mock healpy
# ---------------------------------------------------------------------------
def _inject_mock_healpy():
    if "healpy" not in sys.modules:
        mod = types.ModuleType("healpy")
        mod.nside2npix = lambda nside: 12 * nside**2
        mod.read_map = lambda *a, **kw: np.zeros(12)
        mod.ud_grade = lambda m, nside_out: np.zeros(12 * nside_out**2)
        sys.modules["healpy"] = mod


# ---------------------------------------------------------------------------
# Mock pymaster
# ---------------------------------------------------------------------------
def _inject_mock_pymaster():
    if "pymaster" not in sys.modules:
        mod = types.ModuleType("pymaster")

        class _NmtField:
            def __init__(self, *args, **kwargs):
                pass

        class _NmtWorkspace:
            def __init__(self):
                pass

            def read_from(self, *a):
                pass

            def write_to(self, *a):
                pass

            def compute_coupling_matrix(self, *a, **kw):
                pass

            def decouple_cell(self, cl):
                return cl

            def get_bandpower_windows(self):
                return np.zeros((4, 10, 4, 10))

        class _NmtBin:
            def __init__(self, *args, **kwargs):
                self.leff = np.array([])

            def get_effective_ells(self):
                return self.leff

        mod.NmtField = _NmtField
        mod.NmtWorkspace = _NmtWorkspace
        mod.NmtBin = _NmtBin
        mod.compute_coupled_cell = lambda f1, f2: np.zeros((4, 10))
        sys.modules["pymaster"] = mod


# Run injections at import time so they are available before collection
_inject_mock_bbpipe()
_inject_mock_fgbuster()
_inject_mock_sacc()
_inject_mock_healpy()
_inject_mock_pymaster()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def bb_only_config():
    """Minimal B-only config with dust + synchrotron components."""
    return {
        "pol_channels": ["B"],
        "l_min": 30,
        "l_max": 120,
        "bands": "all",
        "likelihood_type": "h&l",
        "sampler": "maximum_likelihood",
        "nwalkers": 8,
        "n_iters": 10,
        "cmb_model": {
            "params": {
                "r_tensor": ["r_tensor", "tophat", [-0.1, 0.0, 0.1]],
                "A_lens": ["A_lens", "tophat", [0.0, 1.0, 2.0]],
            }
        },
        "fg_model": {
            "component_1": {
                "name": "Dust",
                "sed": "Dust",
                "cl": {("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_d": ["beta_d", "Gaussian", [1.59, 0.11]],
                    "temp_d": ["temp", "fixed", [19.6]],
                    "nu0_d": ["nu0", "fixed", [353.0]],
                },
                "cl_parameters": {
                    ("B", "B"): {
                        "amp_d_bb": ["amp", "tophat", [0.0, 5.0, 100.0]],
                        "alpha_d_bb": ["alpha", "tophat", [-1.0, -0.2, 0.0]],
                        "l0_d_bb": ["ell0", "fixed", [80.0]],
                    }
                },
                "cross": {"epsilon_ds": ["component_2", "tophat", [-1.0, 0.0, 1.0]]},
            },
            "component_2": {
                "name": "Synchrotron",
                "sed": "Synchrotron",
                "cl": {("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_s": ["beta_pl", "Gaussian", [-3.0, 0.3]],
                    "nu0_s": ["nu0", "fixed", [23.0]],
                },
                "cl_parameters": {
                    ("B", "B"): {
                        "amp_s_bb": ["amp", "tophat", [0.0, 2.0, 10.0]],
                        "alpha_s_bb": ["alpha", "tophat", [-1.0, -0.4, 0.0]],
                        "l0_s_bb": ["ell0", "fixed", [80.0]],
                    }
                },
            },
        },
    }


@pytest.fixture(scope="session")
def eb_config():
    """Minimal E+B config with dust + synchrotron components."""
    return {
        "pol_channels": ["E", "B"],
        "l_min": 30,
        "l_max": 120,
        "bands": "all",
        "likelihood_type": "chi2",
        "sampler": "maximum_likelihood",
        "nwalkers": 8,
        "n_iters": 10,
        "cmb_model": {
            "params": {
                "r_tensor": ["r_tensor", "tophat", [-0.1, 0.0, 0.1]],
                "A_lens": ["A_lens", "tophat", [0.0, 1.0, 2.0]],
            }
        },
        "fg_model": {
            "component_1": {
                "name": "Dust",
                "sed": "Dust",
                "cl": {("E", "E"): "ClPowerLaw", ("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_d": ["beta_d", "Gaussian", [1.59, 0.11]],
                    "temp_d": ["temp", "fixed", [19.6]],
                    "nu0_d": ["nu0", "fixed", [353.0]],
                },
                "cl_parameters": {
                    ("E", "E"): {
                        "amp_d_ee": ["amp", "tophat", [0.0, 10.0, 100.0]],
                        "alpha_d_ee": ["alpha", "tophat", [-1.0, -0.42, 0.0]],
                        "l0_d_ee": ["ell0", "fixed", [80.0]],
                    },
                    ("B", "B"): {
                        "amp_d_bb": ["amp", "tophat", [0.0, 5.0, 100.0]],
                        "alpha_d_bb": ["alpha", "tophat", [-1.0, -0.2, 0.0]],
                        "l0_d_bb": ["ell0", "fixed", [80.0]],
                    },
                },
                "cross": {"epsilon_ds": ["component_2", "tophat", [-1.0, 0.0, 1.0]]},
            },
            "component_2": {
                "name": "Synchrotron",
                "sed": "Synchrotron",
                "cl": {("E", "E"): "ClPowerLaw", ("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_s": ["beta_pl", "Gaussian", [-3.0, 0.3]],
                    "nu0_s": ["nu0", "fixed", [23.0]],
                },
                "cl_parameters": {
                    ("E", "E"): {
                        "amp_s_ee": ["amp", "tophat", [0.0, 4.0, 20.0]],
                        "alpha_s_ee": ["alpha", "tophat", [-1.0, -0.6, 0.0]],
                        "l0_s_ee": ["ell0", "fixed", [80.0]],
                    },
                    ("B", "B"): {
                        "amp_s_bb": ["amp", "tophat", [0.0, 2.0, 10.0]],
                        "alpha_s_bb": ["alpha", "tophat", [-1.0, -0.4, 0.0]],
                        "l0_s_bb": ["ell0", "fixed", [80.0]],
                    },
                },
            },
        },
    }


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def make_bandpass():
    """Factory fixture for creating ``Bandpass`` objects with synthetic data."""
    from bbpower.bandpasses import Bandpass

    def _factory(
        nu_center: float = 150.0,
        bandwidth: float = 30.0,
        n_points: int = 11,
        bp_number: int = 1,
        config: dict | None = None,
    ) -> Bandpass:
        if config is None:
            config = {}
        nu = np.linspace(nu_center - bandwidth / 2, nu_center + bandwidth / 2, n_points)
        bnu = np.ones_like(nu)
        dnu = np.ones_like(nu) * (nu[1] - nu[0])
        return Bandpass(nu, dnu, bnu, bp_number, config)

    return _factory
