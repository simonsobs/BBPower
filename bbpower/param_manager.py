from __future__ import annotations

import numpy as np


class ParameterManager:
    """Parse a YAML config dict to manage fixed and free parameters.

    Separates parameters into fixed values and free (sampled) values,
    builds prior functions (tophat or Gaussian) for free parameters,
    and maps flat parameter vectors back to named dictionaries.

    Attributes
    ----------
    p_free_names : list of str
        Names of the free parameters, in sorted order.
    p_free_priors : list
        Prior specifications for each free parameter.
    p_fixed : list of tuple
        (name, value) pairs for fixed parameters.
    p0 : numpy.ndarray
        Initial/fiducial values for the free parameters.
    """

    @staticmethod
    def _prior_kind(prior: str) -> str:
        return str(prior).strip().lower()

    def _add_parameter(self, p_name: str, p: list) -> None:
        """Register a single parameter as fixed or free.

        Parameters
        ----------
        p_name : str
            The internal name of the parameter.
        p : list
            Parameter specification ``[internal_name, prior_type, prior_args]``
            where *prior_type* is ``'fixed'``, ``'tophat'``, or ``'gaussian'``.

        Raises
        ------
        KeyError
            If a free parameter with the same name already exists.
        ValueError
            If *prior_type* is not recognised.
        """
        # If fixed parameter, just add its name and value
        if p[1] == "fixed":
            self.p_fixed.append((p_name, float(p[2][0])))
            return  # Then move on

        # Otherwise it's free
        # Check for duplicate names
        if p_name in self.p_free_names:
            raise KeyError("You have two parameters with the same name")
        # Add name and prior to list
        self.p_free_names.append(p_name)
        self.p_free_priors.append(p)
        # Add fiducial value to initial vector
        prior_kind = self._prior_kind(p[1])
        if prior_kind == "tophat":
            p0 = float(p[2][1])
        elif prior_kind == "gaussian":
            p0 = float(p[2][0])
        else:
            raise ValueError(f"Unknown prior type {p[1]}")
        self.p0.append(p0)

    def _add_parameters(self, params: dict) -> None:
        """Register multiple parameters from a dictionary.

        Parameters
        ----------
        params : dict
            Mapping of parameter names to their specifications.
            Each value has the format expected by ``_add_parameter``.
        """
        for p_name in sorted(params.keys()):
            p = params[p_name]
            self._add_parameter(p_name, p)

    def get_component_names(self, config: dict) -> list[str]:
        """Return sorted list of foreground component names from the config.

        Parameters
        ----------
        config : dict
            Full configuration dictionary containing an ``'fg_model'`` key.

        Returns
        -------
        list of str
            Sorted names of entries whose keys start with ``'component_'``.
        """
        comps = []
        for c_name in config["fg_model"].keys():
            if c_name.startswith("component_"):
                comps.append(c_name)
        return sorted(comps)

    def __init__(self, config: dict) -> None:
        """Initialise the parameter manager from a configuration dictionary.

        Reads CMB parameters, foreground component parameters
        (sed_parameters, cross, decorr, cl_parameters, moments), and
        systematics (bandpass shifts/gains/angles), splitting each into
        fixed or free categories and constructing priors for the free ones.

        Parameters
        ----------
        config : dict
            Full YAML configuration dictionary. Expected top-level keys
            include ``'cmb_model'``, ``'fg_model'``, ``'pol_channels'``,
            and optionally ``'systematics'``.
        """
        self.p_free_names = []
        self.p_free_priors = []
        self.p_fixed = []
        self.p0 = []

        # CMB parameters
        d = config.get("cmb_model")
        if d:
            self._add_parameters(d["params"])

        # Loop through FG components
        comp_names = self.get_component_names(config)
        for c_name in comp_names:
            c = config["fg_model"][c_name]
            for tag in ["sed_parameters", "cross", "decorr"]:
                d = c.get(tag)
                if d:
                    self._add_parameters(d)
            dc = c.get("cl_parameters")
            if dc:  # Power spectra
                for cl_name, d in dc.items():
                    p1, p2 = cl_name
                    # Add parameters only if we're using both
                    # polarization channels
                    if (p1 in config["pol_channels"]) and (
                        p2 in config["pol_channels"]
                    ):
                        self._add_parameters(d)

            dm = c.get("moments")
            if dm and config["fg_model"].get("use_moments"):  # Moments
                self._add_parameters(dm)

        # Loop through different systematics
        if "systematics" in config.keys():
            cnf_sys = config["systematics"]
            # Bandpasses
            if "bandpasses" in cnf_sys.keys():
                cnf_bps = cnf_sys["bandpasses"]
                i_bps = 1
                while f"bandpass_{i_bps}" in cnf_bps:
                    if cnf_bps[f"bandpass_{i_bps}"].get("parameters"):
                        self._add_parameters(cnf_bps[f"bandpass_{i_bps}"]["parameters"])
                    i_bps += 1

        self.p0 = np.array(self.p0)

    def build_params(self, par: np.ndarray) -> dict[str, float]:
        """Map a flat free-parameter vector to a full name-to-value dict.

        Combines the free parameter values in *par* with the stored fixed
        parameter values into a single dictionary.

        Parameters
        ----------
        par : array_like
            Values for the free parameters, in the same order as
            ``p_free_names``.

        Returns
        -------
        dict
            Mapping of all parameter names (fixed and free) to their
            values.
        """
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        return params

    def lnprior(self, par: np.ndarray) -> float:
        """Evaluate the log-prior for a free-parameter vector.

        Gaussian priors contribute ``-0.5 * ((x - mu) / sigma)**2``.
        Tophat priors contribute 0 inside bounds and ``-inf`` outside.

        Parameters
        ----------
        par : array_like
            Values for the free parameters, in the same order as
            ``p_free_names``.

        Returns
        -------
        float
            Log-prior probability. Returns ``-numpy.inf`` if any
            parameter lies outside its tophat bounds.
        """
        lnp = 0
        for p, pr in zip(par, self.p_free_priors):
            if self._prior_kind(pr[1]) == "gaussian":  # Gaussian prior
                lnp += -0.5 * ((p - pr[2][0]) / pr[2][1]) ** 2
            else:  # Only other option is top-hat
                if not (float(pr[2][0]) <= p <= float(pr[2][2])):
                    return -np.inf
        return lnp
