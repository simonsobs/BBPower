import numpy as np

class ParameterManager(object):
    # Class containing details of the model's parameters. 
    #
    # Init inputs:
    #   config: <TODO: not sure what this is>
    #
    # Attributes:
    #
    #   p_free_names: List of free parameter names
    #
    #   p_free_priors: List of free parameters priors, [p1,p2, ... ] with each pi in one of the following forms:
    #
    #       [<LABEL>, "fixed", [<VAL>]]
    #               for fixed params, where VAL is the value the param is fixed as
    #
    #       [<LABEL>, "tophat", [<MIN>, <FIDUCIAL_VAL>, <MAX>]] 
    #               for params with tophat priorswhere MIN and MAX are the min and max values in the tophat prior and FIDUCIAL_VAL the fiducial value
    #
    #       [<LABEL>, "Gaussian", [<MEAN>, <STDEV>]]
    #               for params with gaussian priors where MEAN is the gaussian's mean and STDEV the standard deviation
    #
    #   p_fixed: List of (NAME, VAL) pairs of all fixed parameters
    #
    #   p0: numpy array of starting values of the free parameters

    def _add_parameter(self, p_name, p):
        # Adds a (single) new parameter to the current class
        #
        # Input:
        #
        #   p_name: parameter's name
        #
        #   p: list, taking one of the following forms:
        #
        #       [<LABEL>, "fixed", [<VAL>]]
        #               for fixed params, where VAL is the value the param is fixed as
        #
        #       [<LABEL>, "tophat", [<MIN>, <FIDUCIAL_VAL>, <MAX>]] 
        #               for params with tophat priorswhere MIN and MAX are the min and max values in the tophat prior and FIDUCIAL_VAL the fiducial value
        #
        #       [<LABEL>, "Gaussian", [<MEAN>, <STDEV>]]
        #               for params with gaussian priors where MEAN is the gaussian's mean and STDEV the standard deviation
        #
        # No output

        # If fixed parameter, just add its name and value
        if p[1] == 'fixed':
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
        if np.char.lower(p[1]) == 'tophat':
            p0 = float(p[2][1])
        elif np.char.lower(p[1]) == 'gaussian':
            p0 = float(p[2][0])
        elif np.char.lower(p[1]) == "jeffreys":
            p0 = float(p[2][0])
        else:
            raise ValueError("Unknown prior type %s" % p[1])
        self.p0.append(p0)

    def _add_parameters(self, params):
        # Adds a collection of parameters to the current class. 
        #
        # Input:
        #
        #   params: a dictionary of  {<NAME>: <p>} pairs, where NAME is the parameter's name, and <p> is a list of one of the following forms:
        #
        #       [<LABEL>, "fixed", [<VAL>]]
        #               for fixed params, where VAL is the value the param is fixed as
        #
        #       [<LABEL>, "tophat", [<MIN>, <FIDUCIAL_VAL>, <MAX>]] 
        #               for params with tophat priorswhere MIN and MAX are the min and max values in the tophat prior and FIDUCIAL_VAL the fiducial value
        #
        #       [<LABEL>, "Gaussian", [<MEAN>, <STDEV>]]
        #               for params with gaussian priors where MEAN is the gaussian's mean and STDEV the standard deviation
        #
        # No output
        
        for p_name in sorted(params.keys()):
            p = params[p_name]
            self._add_parameter(p_name, p)

    def get_component_names(self, config):
        # Input:
        #   config: the configuration file
        #
        # Output: 
        #   a list of component names, sorted in alphabetical order
        comps = []
        for c_name in config['fg_model'].keys():
            if c_name.startswith('component_'):
                comps.append(c_name)
        return sorted(comps)

    def __init__(self, config):
        # Initialises the atributes of the current object from the config file. 
        #
        # Input:
        #   configuration file
        # 
        # Output:
        #   None
        self.p_free_names = []
        self.p_free_priors = []
        self.p_fixed = []
        self.p0 = []

        # CMB parameters
        d = config.get('cmb_model')
        if d:
            self._add_parameters(d['params'])

        # Loop through FG components
        comp_names = self.get_component_names(config)
        for c_name in comp_names:
            c = config['fg_model'][c_name]
            for tag in ['sed_parameters', 'cross', 'decorr']:
                d = c.get(tag)
                if d:
                    self._add_parameters(d)
            dc = c.get('cl_parameters')
            if dc:  # Power spectra
                for cl_name, d in dc.items():
                    p1, p2 = cl_name
                    # Add parameters only if we're using both
                    # polarization channels
                    if ((p1 in config['pol_channels']) and
                            (p2 in config['pol_channels'])):
                        self._add_parameters(d)

            dm = c.get('moments')
            if dm and config['fg_model'].get('use_moments'):  # Moments
                self._add_parameters(dm)

        # Loop through different systematics
        if 'systematics' in config.keys():
            cnf_sys = config['systematics']
            # Bandpasses
            if 'bandpasses' in cnf_sys.keys():
                cnf_bps = cnf_sys['bandpasses']
                i_bps = 1
                while 'bandpass_%d' % i_bps in cnf_bps:
                    if cnf_bps['bandpass_%d' % i_bps].get('parameters'):
                        self._add_parameters(cnf_bps['bandpass_%d' % i_bps]['parameters'])
                    i_bps += 1

        self.p0 = np.array(self.p0)

    def build_params(self, par):
        #
        # Input:
        #   par: list of values of the free parameters
        #
        # Output:
        #   params: a dictionary of {NAME: VAL} pairs for all parameters
        #
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        return params

    def lnprior(self, get_jeffreys, par):
        # Evaluates the log prior
        #
        # Input: 
        #   par: list of values of the free parameters (where we are evaluating the prior at)
        #
        # Output: 
        #   lnp: the log of the prior, evaluated when the free parameters take the values given in par
        #
        lnp = 0
        jeffreys_params = []
        for p, pr, name in zip(par, self.p_free_priors, self.p_free_names):
            if np.char.lower(pr[1]) == 'gaussian':  # Gaussian prior
                lnp += -0.5 * ((p - pr[2][0])/pr[2][1])**2
            elif np.char.lower(pr[1]) == 'jeffreys': #Jeffreys prior
                jeffreys_params.append(name)
            else:  # Only other option is top-hat
                if not(float(pr[2][0]) <= p <= float(pr[2][2])):
                    return -np.inf
                
        if jeffreys_params:
            lnp += np.log(get_jeffreys(jeffreys_params, par))
        return lnp