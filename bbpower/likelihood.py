from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import sqrtm

from .param_manager import ParameterManager


class Likelihood:
    """Likelihood evaluator for component separation.

    Wraps the model function and data to compute chi-squared or
    Hamimeche & Lewis likelihood values.

    Parameters
    ----------
    model_func : callable
        Function mapping a parameter dict to model power spectra
        with shape ``(n_bpws, nmaps, nmaps)``.
    param_manager : ParameterManager
        Manages free/fixed parameters and priors.
    bbdata : np.ndarray
        Observed data power spectra, shape ``(n_bpws, nmaps, nmaps)``.
    bbnoise : np.ndarray or None
        Noise power spectra, shape ``(n_bpws, nmaps, nmaps)``.
        Required when ``use_handl`` is True; may be None for chi-squared mode.
    invcov : np.ndarray
        Inverse covariance matrix.
    matrix_to_vector : callable
        Converts ``(nmaps, nmaps)`` matrices to upper-triangle vectors.
    use_handl : bool
        If True, use the Hamimeche & Lewis likelihood instead of chi-squared.
    bbfiducial : np.ndarray or None
        Fiducial power spectra (required if ``use_handl`` is True).
    """

    def __init__(
        self,
        model_func: Callable[[dict], np.ndarray],
        param_manager: ParameterManager,
        bbdata: np.ndarray,
        bbnoise: np.ndarray | None,
        invcov: np.ndarray,
        matrix_to_vector: Callable[[np.ndarray], np.ndarray],
        use_handl: bool,
        bbfiducial: np.ndarray | None = None,
    ) -> None:
        self.model = model_func
        self.params = param_manager
        self.bbdata = bbdata
        self.bbnoise = bbnoise
        self.invcov = invcov
        self.matrix_to_vector = matrix_to_vector
        self.use_handl = use_handl
        self.bbfiducial = bbfiducial

        if self.use_handl:
            self._prepare_h_and_l()

    def _prepare_h_and_l(self) -> None:
        """Pre-compute quantities needed for the H&L likelihood."""
        fiducial_noise = self.bbfiducial + self.bbnoise
        self.Cfl_sqrt = np.array([sqrtm(f) for f in fiducial_noise])
        self.observed_cls = self.bbdata + self.bbnoise

    def chi_sq_dx(self, params: dict) -> np.ndarray:
        """Compute the chi-squared residual vector.

        Parameters
        ----------
        params : dict
            Named parameter dictionary.

        Returns
        -------
        np.ndarray
            Flattened residual vector ``(data - model)``.
        """
        model_cls = self.model(params)
        return self.matrix_to_vector(self.bbdata - model_cls).flatten()

    def h_and_l_dx(self, params: dict) -> np.ndarray | list:
        """Compute the Hamimeche & Lewis residual vector.

        Parameters
        ----------
        params : dict
            Named parameter dictionary.

        Returns
        -------
        np.ndarray
            Flattened H&L transformed residual vector.
        """
        model_cls = self.model(params)
        dx_vec = []
        for k in range(model_cls.shape[0]):
            C = model_cls[k] + self.bbnoise[k]
            X = self._h_and_l_transform(C, self.observed_cls[k], self.Cfl_sqrt[k])
            if np.any(np.isinf(X)):
                return [np.inf]
            dx = self.matrix_to_vector(X).flatten()
            dx_vec = np.concatenate([dx_vec, dx])
        return dx_vec

    @staticmethod
    def _h_and_l_transform(
        C: np.ndarray, Chat: np.ndarray, Cfl_sqrt: np.ndarray
    ) -> np.ndarray | list:
        """Hamimeche & Lewis likelihood transform.

        Taken from Cobaya written by Hamimeche, Lewis and Torrado.

        Parameters
        ----------
        C : np.ndarray
            Model covariance matrix for a single bandpower.
        Chat : np.ndarray
            Observed covariance matrix for a single bandpower.
        Cfl_sqrt : np.ndarray
            Square root of fiducial+noise covariance.

        Returns
        -------
        np.ndarray
            Transformed matrix, or ``[np.inf]`` on numerical failure.
        """
        try:
            diag, U = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            return [np.inf]
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        try:
            diag, rot = np.linalg.eigh(rot)
        except np.linalg.LinAlgError:
            return [np.inf]
        diag = np.sign(diag - 1) * np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1))
        Cfl_sqrt.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        return rot.dot(U.T)

    def lnlike(self, par: np.ndarray) -> float:
        """Log-likelihood without priors.

        Parameters
        ----------
        par : np.ndarray
            Free parameter vector.

        Returns
        -------
        float
            Log-likelihood value.
        """
        params = self.params.build_params(par)
        if self.use_handl:
            dx = self.h_and_l_dx(params)
            if np.any(np.isinf(dx)):
                return -np.inf
        else:
            dx = self.chi_sq_dx(params)
        return -0.5 * np.dot(dx, np.dot(self.invcov, dx))

    def lnprob(self, par: np.ndarray) -> float:
        """Log-posterior: log-likelihood plus log-prior.

        Parameters
        ----------
        par : np.ndarray
            Free parameter vector.

        Returns
        -------
        float
            Log-posterior value.
        """
        prior = self.params.lnprior(par)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.lnlike(par)
