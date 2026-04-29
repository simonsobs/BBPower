from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import fgbuster.component_model as fgc
import bbpower.fgcls as fgl


class FGModel:
    """
    FGModel loads the foreground models and prepares the unit conversions
    to K_CMB units. This creates a class that has an components attribute.
    The components attribute is a dictionary of foreground models. Each
    foreground model is also a dictionary containing the SED function,
    SED parameters, SED nu0, CMB nu0 normalization, and the foreground
    power spectrum parameters.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the foreground model from a pipeline configuration.

        Parameters
        ----------
        config : dict
            Pipeline configuration containing an ``fg_model`` section with
            component definitions (SEDs, Cl templates, parameters).
        """
        self.load_foregrounds(config)

    def component_iterator(self, config: dict) -> Iterator[tuple[str, dict]]:
        """Yield ``(name, component_dict)`` for each foreground component.

        Parameters
        ----------
        config : dict
            Pipeline configuration containing an ``fg_model`` section.

        Yields
        ------
        name : str
            Component key (starts with ``'component_'``).
        component : dict
            Component specification from the config.
        """
        for key, component in config["fg_model"].items():
            if key.startswith("component_"):
                yield key, component

    def load_foregrounds(self, config: dict) -> None:
        """Parse the config and build foreground component models.

        Populates ``self.components``, ``self.component_names``,
        ``self.component_order``, and ``self.n_components``.

        Parameters
        ----------
        config : dict
            Pipeline configuration.  Must contain ``fg_model`` with one or
            more ``component_*`` entries, each specifying an SED, Cl model,
            and optionally cross-correlation, decorrelation, and moment
            parameters.
        """
        self.component_names = []
        self.components = {}
        self.component_order = {}

        i_comp = 0
        for key, component in self.component_iterator(config):
            comp = {}

            decorr = component.get("decorr")
            comp["decorr"] = False
            if decorr:
                comp["decorr"] = True
                comp["decorr_param_names"] = {}
                for k, l in decorr.items():
                    comp["decorr_param_names"][l[0]] = k

            comp["names_x_dict"] = {}
            d_x = component.get("cross")
            if d_x:
                for pn, par in d_x.items():
                    if par[0] not in config["fg_model"].keys():
                        raise KeyError(
                            f"Component {par[0]} is not a valid component "
                            f"to correlate {key} with"
                        )
                    if par[0] == key:
                        raise KeyError(f"{par[0]} is cross correlated with itself.")
                    comp["names_x_dict"][par[0]] = pn

            # Loop through SED parameters.
            # Find nu0 if it exists
            # Make a list of all parameters ready to pass to fgc
            comp["sed_parameters"] = component["sed_parameters"]
            nu0 = None
            params_fgc = {}
            comp["names_sed_dict"] = {}
            for k, l in comp["sed_parameters"].items():
                comp["names_sed_dict"][l[0]] = k

                # nu0
                if l[0] == "nu0":
                    if l[1] != "fixed":
                        raise ValueError("You can't vary reference" " frequencies!")
                    nu0 = l[2][0]

                # SED parameter
                if l[1] == "fixed":
                    val = l[2][0]
                else:
                    val = None
                params_fgc[l[0]] = val

            # Set units normalization
            if nu0 is not None:
                comp["cmb_n0_norm"] = fgc.CMB("K_RJ").eval(nu0)
            else:
                comp["cmb_n0_norm"] = 1.0

            # Set SED function
            sed_fnc = get_function(fgc, component["sed"])
            comp["sed"] = sed_fnc(**params_fgc, units="K_RJ")

            # Same thing for C_ell parameters
            comp["names_cl_dict"] = {}
            params_fgl = {}
            for k, d in component["cl_parameters"].items():
                p1, p2 = k
                # Add parameters only if we're using both polarization channels
                if (p1 in config["pol_channels"]) and (p2 in config["pol_channels"]):
                    comp["names_cl_dict"][k] = {}
                    params_fgl[k] = {}
                    for n, l in d.items():
                        comp["names_cl_dict"][k][l[0]] = n
                        if l[0] == "ell0":
                            if l[1] != "fixed":
                                raise ValueError("You can't vary " "reference scales!")
                        if l[1] == "fixed":
                            val = l[2][0]
                        else:
                            val = None
                        params_fgl[k][l[0]] = val

            # Moment parameters
            comp["names_moments_dict"] = {}
            d = component.get("moments")
            if d and config["fg_model"].get("use_moments"):
                comp["moments_parameters"] = component["moments"]
                for k, l in component["moments"].items():
                    comp["names_moments_dict"][l[0]] = k

            # Set Cl functions
            comp["cl"] = {}
            for k, c in component["cl"].items():
                p1, p2 = k
                # Add parameters only if we're using both polarization channels
                if (p1 in config["pol_channels"]) and (p2 in config["pol_channels"]):
                    cl_fnc = get_function(fgl, c)
                    comp["cl"][k] = cl_fnc(**(params_fgl[k]))
            self.components[key] = comp
            self.component_names.append(key)
            self.component_order[key] = i_comp
            i_comp += 1
        self.n_components = len(self.component_names)


def get_function(mod: ModuleType, sed_name: str) -> type:
    """Look up a function by name from a module.

    Parameters
    ----------
    mod : module
        Module to search (e.g. ``fgbuster.component_model`` or ``bbpower.fgcls``).
    sed_name : str
        Name of the function or class to retrieve.

    Returns
    -------
    callable
        The requested function or class.

    Raises
    ------
    KeyError
        If the function is not found in the module.
    """
    try:
        return getattr(mod, sed_name)
    except AttributeError:
        raise KeyError(f"Function named {sed_name} cannot be found")
