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

    # Attributes:
    #
    #   component_names: list of names of the components of the model, taken from the config file
    #           (e.g. if dust is labelled component1 and synchtron component2, then component_names will be ["component1","component2"]
    #
    #   components: DICT of:
    #
    #       {COMPONENT: PARAM_DICT} pairs where PARAM_DICT is a dictionary containing the following elements
    #           - {"decorr": BOOL}
    #           - {"deccor_param_names": DECCOR_DICT} 
    #                   Ommitted if decorr = False
    #           - {"names_x_dict": X_DICT}
    #                   where X_DICT contains {PARAM: COMPONENT2} pairs (where PARAM is the parameter name expressing correlation with foreground component "COMPONENT2")
    #                   ommitted if there is no cross terms within "COMPONENT" in the config file
    #           - {"sed_parameters": PARAM_DICT}
    #                   where PARAM_DICT contains {PARAM:p} pairs, with p a list taking one of the following forms:
    #                       [LABEL, "fixed", [VAL]]
    #                       [LABEL, "Gaussian", [MEAN, STDEV]]
    #                       [LABEL, "tophat", [MIN, INITIAL_VAL, MAX]]
    #           - {"names_sed_dict": NAMES_DICT}
    #                   where NAMES_DICT is a dictionary of {LABEL:NAME} where LABEL is the label used to describe a given part of the sed model (e.g. nu_0) and NAME is the associated parameter's name (e.g. nu_0_d)
    #           - {"cmb_n0_norm": NORMALISATION} 
    #           - {"nu0: VAL"}
    #           - {"sed", SED_FUNC}
    #                   where SED_FUNCTION gives the function for the component's sed, evaluated at the position of the fixed params
    #           - {"names_cl_dict": NAMES_DICT}
    #                   where NAMES_DICT is a dictionary of {LABEL:NAME} pairs where LABEL is the label used to describe a part of the Cl model (e.g. amp) and NAME is the parameter's name (e.g. amp_d_ee)
    #           - {"names_moment_dict": DICT}
    #           - {"cl": CL_FUNC_DICT}
    #                   where CL_FUNC_DICT stores {POLARIZATION_MODE: CL_FUNC} pairs, where CL_FUNC is the cl function evaluated at the position of the fixed params
    #
    #
    #   component_order: dictionary of {COMPONENT: ORDER} pairs, where COMPONENT is the name of the component and ORDER is an index describing the order in which components appear in the config file
    #
    #   n_components: the number of components present in the config file
    #

    
    def __init__(self, config):
        self.load_foregrounds(config)
        return

    def component_iterator(self, config):
        for key, component in config['fg_model'].items():
            if key.startswith('component_'):
                yield key, component

    def load_foregrounds(self, config):
        # Initialises the foreground-related attributes
        #
        # Input:
        #   config: the configuration file
        #
        # Output: None
        
        self.component_names = []
        self.components = {}
        self.component_order = {}

        i_comp = 0
        for key, component in self.component_iterator(config):
            comp = {}

            decorr = component.get('decorr')
            comp['decorr'] = False
            if decorr:
                comp['decorr'] = True
                comp['decorr_param_names'] = {}
                for k, l in decorr.items():
                    comp['decorr_param_names'][l[0]] = k

            comp['names_x_dict'] = {}
            d_x = component.get('cross')
            if d_x:
                for pn, par in d_x.items():
                    if par[0] not in config['fg_model'].keys():
                        raise KeyError("Component %s " % (par[0]) +
                                       "is not a valid component" +
                                       "to correlate %s with" % key)
                    if par[0] == key:
                        raise KeyError("%s is cross correlated with itself." % par[0])
                    comp['names_x_dict'][par[0]] = pn

            # Loop through SED parameters.
            # Find nu0 if it exists
            # Make a list of all parameters ready to pass to fgc
            comp['sed_parameters'] = component['sed_parameters']
            nu0 = None
            params_fgc = {}
            comp['names_sed_dict'] = {}
            for k, l in comp['sed_parameters'].items():
                comp['names_sed_dict'][l[0]] = k

                # nu0
                if l[0] == 'nu0':
                    if l[1] != 'fixed':
                        raise ValueError("You can't vary reference"
                                         " frequencies!")
                    nu0 = l[2][0]

                # SED parameter
                if l[1] == 'fixed':
                    val = l[2][0]
                else:
                    val = None
                params_fgc[l[0]] = val

            # Set units normalization
            if nu0 is not None:
                comp['cmb_n0_norm'] = fgc.CMB('K_RJ').eval(nu0)
                comp['nu0'] = nu0
            else:
                comp['cmb_n0_norm'] = 1.

            # Set SED function
            sed_fnc = get_function(fgc, component['sed'])
            comp['sed'] = sed_fnc(**params_fgc, units='K_RJ')

            # Same thing for C_ell parameters
            comp['names_cl_dict'] = {}
            params_fgl = {}
            for k, d in component['cl_parameters'].items():
                p1, p2 = k
                # Add parameters only if we're using both polarization channels
                if ((p1 in config['pol_channels']) and
                        (p2 in config['pol_channels'])):
                    comp['names_cl_dict'][k] = {}
                    params_fgl[k] = {}
                    for n, l in d.items():
                        comp['names_cl_dict'][k][l[0]] = n
                        if l[0] == 'ell0':
                            if l[1] != 'fixed':
                                raise ValueError("You can't vary "
                                                 "reference scales!")
                        if l[1] == 'fixed':
                            val = l[2][0]
                        else:
                            val = None
                        params_fgl[k][l[0]] = val

            # Moment parameters
            comp['names_moments_dict'] = {}
            d = component.get('moments')
            if d and config['fg_model'].get('use_moments'):
                comp['moments_pameters'] = component['moments']
                for k, l in component['moments'].items():
                    comp['names_moments_dict'][l[0]] = k

            # Set Cl functions
            comp['cl'] = {}
            for k, c in component['cl'].items():
                p1, p2 = k
                # Add parameters only if we're using both polarization channels
                if ((p1 in config['pol_channels']) and
                        (p2 in config['pol_channels'])):
                    cl_fnc = get_function(fgl, c)
                    comp['cl'][k] = cl_fnc(**(params_fgl[k]))

            self.components[key] = comp
            self.component_names.append(key)
            self.component_order[key] = i_comp
            i_comp += 1
        self.n_components = len(self.component_names)
        return


def get_function(mod, sed_name):
    try:
        return getattr(mod, sed_name)
    except AttributeError:
        raise KeyError("Function named %s cannot be found" % (sed_name))